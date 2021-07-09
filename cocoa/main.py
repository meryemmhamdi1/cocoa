'''
Load data, learn model and evaluate
'''

import argparse
import random
import os
import time
import tensorflow as tf
from itertools import chain
from src.basic.util import read_json, write_json, read_pickle, write_pickle
from src.basic.dataset import add_dataset_arguments, read_dataset
from src.basic.schema import Schema
from src.basic.scenario_db import ScenarioDB, add_scenario_arguments
from src.basic.lexicon import Lexicon, add_lexicon_arguments
from src.model.preprocess import DataGenerator, Preprocessor, add_preprocess_arguments
from src.model.encdec import add_model_arguments, build_model, build_model_MUSE, build_model_BERT, build_model_HME
from src.model.learner import add_learner_arguments, Learner
from src.model.evaluate import Evaluator
from src.model.graph import Graph, GraphMetadata, add_graph_arguments
from src.model.graph_embedder import add_graph_embed_arguments
from src.lib import logstats
from src.model.preprocess import tokenize_wp, tokenize_type_based
import sys

#add_scenario_arguments(parser)
#add_lexicon_arguments(parser)
#add_dataset_arguments(parser)
#add_preprocess_arguments(parser)
#add_model_arguments(parser)
#add_graph_arguments(parser)
#add_graph_embed_arguments(parser)
#add_learner_arguments(parser)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-seed', help='Random seed', type=int, default=1)
    parser.add_argument('--stats-file', help='Path to save json statistics (dataset, training etc.) file')
    parser.add_argument('--test', default=False, action='store_true', help='Test mode')
    parser.add_argument('--best', default=False, action='store_true', help='Test using the best model on dev set')
    parser.add_argument('--verbose', default=False, action='store_true', help='More prints')
    parser.add_argument('--domain', type=str, choices=['MutualFriends', 'Matchmaking'])

    # Scenario arguments
    parser.add_argument('--schema-path', help='Input path that describes the schema of the domain', required=True)
    parser.add_argument('--scenarios-path', help='Output path for the scenarios generated', required=True)

    # Lexicon arguments
    parser.add_argument('--stop-words', type=str, default='data/common_words.txt', help='Path to stop words list')
    parser.add_argument('--learned-lex', default=False, action='store_true', help='if true have entity linking in lexicon use learned system')
    parser.add_argument('--inverse-lexicon', help='Path to inverse lexicon data')

    # Dataset arguments
    parser.add_argument('--train-examples-paths', help='Input training examples', nargs='*', default=[])
    parser.add_argument('--test-examples-paths', help='Input test examples', nargs='*', default=[])
    parser.add_argument('--train-max-examples', help='Maximum number of training examples', type=int)
    parser.add_argument('--test-max-examples', help='Maximum number of test examples', type=int)

    # Preprocess arguments
    parser.add_argument('--entity-encoding-form', choices=['type', 'canonical'], default='canonical', help='Input entity form to the encoder')
    parser.add_argument('--entity-decoding-form', choices=['canonical', 'type'], default='canonical', help='Input entity form to the decoder')
    parser.add_argument('--entity-target-form', choices=['canonical', 'type', 'graph'], default='canonical', help='Output entity form to the decoder')

    # Graph arguments
    parser.add_argument('--num-items', type=int, default=10, help='Maximum number of items in each KB')
    parser.add_argument('--entity-hist-len', type=int, default=2, help='Number of most recent utterances to consider when updating entity node embeddings')
    parser.add_argument('--max-num-entities', type=int, default=30, help='Estimate of maximum number of entities in a dialogue')
    parser.add_argument('--max-degree', type=int, default=10, help='Maximum degree of a node in the graph')

    # Model arguments
    parser.add_argument('--model', default='encdec', help='Model name {encdec}')
    parser.add_argument('--rnn-size', type=int, default=20, help='Dimension of hidden units of RNN')
    parser.add_argument('--rnn-type', default='lstm', help='Type of RNN unit {rnn, gru, lstm}')
    parser.add_argument('--num-layers', type=int, default=1, help='Number of RNN layers')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of examples per batch')
    parser.add_argument('--word-embed-size', type=int, default=20, help='Word embedding size')
    parser.add_argument('--word-embed-type', type=str, default="RANDOM", help='Word embedding type')
    parser.add_argument('--bow-utterance', default=False, action='store_true', help='Use sum of word embeddings as utterance embedding')
    parser.add_argument('--decoding', nargs='+', default=['sample', 0, 'select'], help='Decoding method')
    parser.add_argument('--node-embed-in-rnn-inputs', default=False, action='store_true', help='Add node embedding of entities as inputs to the RNN')
    parser.add_argument('--no-graph-update', default=False, action='store_true', help='Do not update the KB graph during the dialogue')

    parser.add_argument('--attn-scoring', default='linear', help='How to compute scores between hidden state and context {bilinear, linear}')
    parser.add_argument('--attn-output', default='project', help='How to combine rnn output and attention {concat, project}')
    parser.add_argument('--no-checklist', default=False, action='store_true', help='Whether to include checklist at each RNN step')

    # Graph embedding arguments
    parser.add_argument('--node-embed-size', type=int, default=10, help='Knowledge graph node/subgraph embedding size')
    parser.add_argument('--edge-embed-size', type=int, default=10, help='Knowledge graph edge label embedding size')
    parser.add_argument('--entity-embed-size', type=int, default=10, help='Knowledge graph entity embedding size')
    parser.add_argument('--entity-cache-size', type=int, default=2, help='Number of entities to remember (this is more of a performance concern; ideally we can remember all entities within the history)')
    parser.add_argument('--use-entity-embedding', action='store_true', default=False, help='Whether to use entity embedding when compute node embeddings')
    parser.add_argument('--mp-iters', type=int, default=2, help='Number of iterations of message passing on the graph')
    parser.add_argument('--utterance-decay', type=float, default=1, help='Decay of old utterance embedding over time')
    parser.add_argument('--learned-utterance-decay', default=False, action='store_true', help='Learning weight to combine old and new utterances')
    parser.add_argument('--msg-aggregation', default='sum', choices=['sum', 'max', 'avg'], help='How to aggregate messages from neighbors')

    # Learner arguments
    parser.add_argument('--optimizer', default='sgd', help='Optimization method')
    parser.add_argument('--grad-clip', type=int, default=5, help='Min and max values of gradients')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--min-epochs', type=int, default=10, help='Number of training epochs to run before checking for early stop')
    parser.add_argument('--max-epochs', type=int, default=50, help='Maximum number of training epochs')
    parser.add_argument('--num-per-epoch', type=int, default=None, help='Number of examples per epoch')
    parser.add_argument('--print-every', type=int, default=1, help='Number of examples between printing training loss')
    parser.add_argument('--init-from', help='Initial parameters')
    parser.add_argument('--checkpoint', default='.', help='Directory to save learned models')
    parser.add_argument('--gpu', type=int, default=0, help='Use GPU or not')
    parser.add_argument('--log-file', type=str, default="", help='Full path to log file')

    # HME arguments
    parser.add_argument('--mode', type=str, default='attn_sum', help='attn_sum or concat or linear')

    parser.add_argument('--add-emb', action="store_true", help='add trainable emb')
    parser.add_argument('--no-word-emb', action="store_true", help='no word embedding')
    parser.add_argument('--add-char-emb', action="store_true", help='add trainable char emb')
    parser.add_argument('--no_projection', action='store_true', help='without projection matrix')

    parser.add_argument('--char-hidden-size', type=int, default=100, help='add trainable char emb')
    parser.add_argument('--embedding_size_char_per_word', type=int, default=100, help='embedding size char per word')
    parser.add_argument('--embedding_size_char', type=int, default=300, help='embedding size char')

    parser.add_argument('--bpe-lang-list', nargs='+', default=['en', 'es', 'hi'])
    parser.add_argument('--bpe-dim', type=int, default=300)
    parser.add_argument('--bpe-vocab', type=int, default=5000)
    parser.add_argument('--bpe-hidden-size', type=int, default=100)
    parser.add_argument('--bpe-emb-size', type=int, default=300)

    parser.add_argument('--dim_key', type=int, default=0, help='attention key channels')
    parser.add_argument('--dim_value', type=int, default=0, help='attention value channels')
    parser.add_argument('--filter_size', type=int, default=128, help='filter size')

    parser.add_argument('--drop', type=float, default=0, help='dropout')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')
    parser.add_argument('--input_dropout', type=float, default=0.2, help='input dropout')
    parser.add_argument('--attn_dropout', type=float, default=0.2, help='attention dropout')
    parser.add_argument('--relu_dropout', type=float, default=0.2, help='relu dropout')

    parser.add_argument('--max_length', type=int, default=256, help='maximum length')

    args = parser.parse_args()

    #stdoutOrigin = sys.stdout
    #print("args.log_file:", args.log_file)
    #print("args.init_from:", args.init_from)
    #sys.stdout = open(args.log_file, "w")

    random.seed(args.random_seed)
    #logstats.init(args.stats_file)
    #logstats.add_args('config', args)

    print("INIT FROM:", args.init_from)
    # Save or load models
    if args.init_from:
        start = time.time()
        print('Load model (config, vocab, checkpoint) from', args.init_from)
        config_path = os.path.join(args.init_from, 'config.json')
        vocab_path = os.path.join(args.init_from, 'vocab.pkl')
        saved_config = read_json(config_path)
        saved_config['decoding'] = args.decoding
        saved_config['batch_size'] = args.batch_size
        model_args = argparse.Namespace(**saved_config)

        # Checkpoint
        if args.test and args.best:
            ckpt = tf.train.get_checkpoint_state(args.init_from+'-best')
        else:
            ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt, 'No checkpoint found'
        assert ckpt.model_checkpoint_path, 'No model path found in checkpoint'

        # Load vocab
        mappings = read_pickle(vocab_path)
        print ('Done [%fs]' % (time.time() - start))
    else:
        # Save config
        if not os.path.isdir(args.checkpoint):
            os.makedirs(args.checkpoint)
        config_path = os.path.join(args.checkpoint, 'config.json')
        write_json(vars(args), config_path)
        model_args = args
        mappings = None
        ckpt = None

    schema = Schema(model_args.schema_path, model_args.domain)
    scenario_db = ScenarioDB.from_dict(schema, read_json(args.scenarios_path))
    dataset = read_dataset(scenario_db, args)
    print('Building lexicon...')
    start = time.time()
    lexicon = Lexicon(schema, args.learned_lex, stop_words=args.stop_words)
    print('%.2f s'% (time.time() - start))

    # Dataset
    use_kb = False if model_args.model == 'encdec' else True
    copy = True if model_args.model == 'attn-copy-encdec' else False
    if model_args.model == 'attn-copy-encdec':
        model_args.entity_target_form = 'graph'
    preprocessor = Preprocessor(args,
                                schema,
                                lexicon,
                                model_args.entity_encoding_form,
                                model_args.entity_decoding_form,
                                model_args.entity_target_form)
    if args.word_embed_type == "BERT":
        tokenizer = tokenize_wp
    else:
        tokenizer = tokenize_type_based
    if args.test:
        model_args.dropout = 0
        data_generator = DataGenerator(args,
                                       None,
                                       None,
                                       dataset.test_examples,
                                       preprocessor,
                                       schema,
                                       model_args.num_items,
                                       mappings,
                                       use_kb,
                                       copy,
                                       tokenizer)
    else:
        data_generator = DataGenerator(args,
                                       dataset.train_examples,
                                       dataset.test_examples,
                                       None,
                                       preprocessor,
                                       schema,
                                       model_args.num_items,
                                       mappings,
                                       use_kb,
                                       copy,
                                       tokenizer)

    for d, n in data_generator.num_examples.items():
        logstats.add('data', d, 'num_dialogues', n)
    # Save mappings
    if not mappings:
        mappings = data_generator.mappings
        vocab_path = os.path.join(args.checkpoint, 'vocab.pkl')
        write_pickle(mappings, vocab_path)
    for name, m in mappings.items():
        logstats.add('mappings', name)

    # Build the model
    logstats.add_args('model_args', model_args)
    if args.word_embed_type == "RANDOM":
        print("Checking schema:", schema.get_attributes())
        print("Checking mappings:", mappings)
        model = build_model(schema, mappings, model_args)
    elif args.word_embed_type == "MUSE":
        model = build_model_MUSE(schema, mappings, model_args)
    elif args.word_embed_type == "HME":
        model = build_model_HME(schema, mappings, model_args)
    elif args.word_embed_type == "BERT":
        model = build_model_BERT(schema, mappings, model_args)

    # Tensorflow config
    if args.gpu == 0:
        print('GPU is disabled')
        config = tf.ConfigProto(device_count={'GPU': 0})
    else:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True)
        config = tf.ConfigProto(device_count={'GPU': 1}, gpu_options=gpu_options)

    if args.test:
        assert args.init_from and ckpt, 'No model to test'
        evaluator = Evaluator(data_generator, model, splits=('test',), batch_size=args.batch_size, verbose=args.verbose)
        learner = Learner(data_generator, model, evaluator, batch_size=args.batch_size, verbose=args.verbose)

    else:
        print("******** CALLING EVALUATOR ***********")
        evaluator = Evaluator(data_generator, model, splits=('dev',), batch_size=args.batch_size, verbose=args.verbose)
        print("******** INITALIZING LEARNER ***********")
        learner = Learner(data_generator, model, evaluator, batch_size=args.batch_size, verbose=args.verbose)

        print("******** CALLING LEARNER ***********")
        learner.learn(args, config, args.stats_file, ckpt)

    print("ckpt.model_checkpoint_path:")
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print('Load TF model', ckpt.model_checkpoint_path)
        start = time.time()
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Done [%fs]' % (time.time() - start))

        for split, test_data, num_batches in evaluator.dataset():
            print('================== Eval %s ==================' % split)
            print('================== Sampling ==================')
            start_time = time.time()
            bleu, (ent_prec, ent_recall, ent_f1), lu, success_rate = evaluator.test_bleu(sess, test_data, num_batches)
            print('bleu=%.4f/%.4f/%.4f entity_f1=%.4f/%.4f/%.4f time(s)=%.4f' % (bleu[0], bleu[1], bleu[2], ent_prec, ent_recall, ent_f1, time.time() - start_time))
            print('lu=%.4f, success_rate=%.4f' % (lu, success_rate))
            print('================== Perplexity ==================')
            start_time = time.time()
            loss = learner.test_loss(sess, test_data, num_batches)
            print('loss=%.4f time(s)=%.4f' % (loss, time.time() - start_time))
            logstats.add(split, {'bleu-4': bleu[0], 'bleu-3': bleu[1], 'bleu-2': bleu[2], 'entity_precision': ent_prec, 'entity_recall': ent_recall, 'entity_f1': ent_f1, 'loss': loss})

    #sys.stdout.close()
    #sys.stdout = stdoutOrigin
