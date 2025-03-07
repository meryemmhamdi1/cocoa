'''
Encoder-decoder models.
'''

import tensorflow as tf
import numpy as np
from tensorflow.python.util import nest
from src.model.rnn_cell import AttnRNNCell, add_attention_arguments, build_rnn_cell
from src.model.graph import Graph, GraphMetadata
from src.model.graph_embedder import GraphEmbedder
from src.model.graph_embedder_config import GraphEmbedderConfig
from src.model.word_embedder import WordEmbedder, MUSEWordEmbedder, HMEEmbedder
from src.model.util import transpose_first_two_dims, batch_linear, batch_embedding_lookup, EPS
from src.model.preprocess import markers, item_to_entity
import io
from bert import modeling

import os
if "/nas/clear/users/meryem/" in os.getcwd():
    vocab_file = "/nas/clear/users/meryem/Embeddings/BERT/multi_cased_L-12_H-768_A-12/vocab.txt"
    muse_es = "/nas/clear/users/meryem/Embeddings/MUSE/wiki.multi.es.vec"
    muse_en = "/nas/clear/users/meryem/Embeddings/MUSE/wiki.multi.en.vec"
    muse_hi = "/nas/clear/users/meryem/Embeddings/MUSE/hi-en-sup/5eibit537k/vectors-hi.txt"
    glove_en = "/home1/mmhamdi/Models/Glove/glove.840B.300d.txt"
    fasttext_en = "/home1/mmhamdi/Models/Fasttext/"
    fasttext_es = "/home1/mmhamdi/Models/Fasttext/"
    fasttext_hi = "/home1/mmhamdi/Models/Fasttext/"
    mbert_config_file = "/nas/clear/users/meryem/Embeddings/BERT/multi_cased_L-12_H-768_A-12/bert_config.json"
elif "/home1/mmhamdi" in os.getcwd():
    vocab_file = "/home1/mmhamdi/Models/bert-base-multilingual-cased/vocab.txt"
    muse_es = "/home1/mmhamdi/Models/MUSE/wiki.multi.es.vec"
    muse_en = "/home1/mmhamdi/Models/MUSE/wiki.multi.en.vec"
    muse_hi = "/home1/mmhamdi/Models/MUSE/vectors-hi.txt"
    glove_en = "/home1/mmhamdi/Models/Glove/glove.840B.300d.txt"
    fasttext_en = "/home1/mmhamdi/Models/Fasttext/cc.en.300.vec"
    fasttext_es = "/home1/mmhamdi/Models/Fasttext/cc.es.300.vec"
    fasttext_hi = "/home1/mmhamdi/Models/Fasttext/cc.hi.300.vec"
    mbert_config_file = "/home1/mmhamdi/Models/bert-base-multilingual-cased/config.json"

else:
    vocab_file = "/Users/d22admin/USCGDrive/ISI/EventExtraction/4Embeddings/Multilingual/multi_cased_L-12_H-768_A-12-2/vocab.txt"
    muse_es = "/Users/d22admin/USCGDrive/ISI/BETTER/3.Datasets/Embeddings/MUSE/wiki.multi.es.vec.txt"
    muse_en = "/Users/d22admin/USCGDrive/ISI/BETTER/3.Datasets/Embeddings/MUSE/wiki.multi.en.vec.txt"
    mbert_config_file = "/Users/d22admin/USCGDrive/ISI/EventExtraction/4Embeddings/Multilingual/multi_cased_L-12_H-768_A-12-2/bert_config.json"


def add_model_arguments(parser):
    parser.add_argument('--model', default='encdec', help='Model name {encdec}')
    parser.add_argument('--rnn-size', type=int, default=20, help='Dimension of hidden units of RNN')
    parser.add_argument('--rnn-type', default='lstm', help='Type of RNN unit {rnn, gru, lstm}')
    parser.add_argument('--num-layers', type=int, default=1, help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of examples per batch')
    parser.add_argument('--word-embed-size', type=int, default=20, help='Word embedding size')
    parser.add_argument('--bow-utterance', default=False, action='store_true', help='Use sum of word embeddings as utterance embedding')
    parser.add_argument('--decoding', nargs='+', default=['sample', 0, 'select'], help='Decoding method')
    parser.add_argument('--node-embed-in-rnn-inputs', default=False, action='store_true', help='Add node embedding of entities as inputs to the RNN')
    parser.add_argument('--no-graph-update', default=False, action='store_true', help='Do not update the KB graph during the dialogue')

    add_attention_arguments(parser)


def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id


# MUSE
def build_model_MUSE(schema, mappings, args):
    tf.reset_default_graph()
    tf.set_random_seed(args.random_seed)

    vocab = mappings['vocab']
    pad = vocab.to_ind_word(markers.PAD)
    select = vocab.to_ind_word(markers.SELECT)

    spanish_embed, id2word_es, word2id_es = load_vec(muse_es)
    hindi_embed, id2word_hi, word2id_hi = load_vec(muse_hi)
    english_embed, id2word_en, word2id_en = load_vec(muse_en)

    W = np.zeros((vocab.size_word, args.word_embed_size))
    for word in vocab.word_to_ind:
        if word in word2id_en:
            W[vocab.word_to_ind[word]] = english_embed[word2id_en[word]]
        if word in word2id_es:
            W[vocab.word_to_ind[word]] = spanish_embed[word2id_es[word]]
        if word in word2id_hi:
            W[vocab.word_to_ind[word]] = hindi_embed[word2id_hi[word]]

    with tf.variable_scope('EncoderWordEmbedder'):
        print("vocab.size_word:", vocab.size_word)
        print("args.word_embed_size:", args.word_embed_size)
        encoder_word_embedder = MUSEWordEmbedder(vocab.size_word, args.word_embed_size, W, pad)
    with tf.variable_scope('DecoderWordEmbedder'):
        decoder_word_embedder = MUSEWordEmbedder(vocab.size_word, args.word_embed_size, W, pad)

    if args.decoding[0] == 'sample':
        sample_t = float(args.decoding[1])
        sample_select = None if len(args.decoding) < 3 or args.decoding[2] != 'select' else select
    else:
        raise('Unknown decoding method')

    update_graph = (not args.no_graph_update)
    node_embed_in_rnn_inputs = args.node_embed_in_rnn_inputs

    if args.model == 'encdec':
        encoder = BasicEncoder(args.rnn_size, args.rnn_type, args.num_layers, args.dropout)
        decoder = BasicDecoder(args.rnn_size, vocab.size_word, args.rnn_type, args.num_layers, args.dropout, sample_t, sample_select)
        model = BasicEncoderDecoder(encoder_word_embedder, decoder_word_embedder, encoder, decoder, pad, select)
    elif args.model == 'attn-encdec' or args.model == 'attn-copy-encdec':
        max_degree = args.num_items + len(schema.attributes)
        utterance_size = args.word_embed_size if args.bow_utterance else args.rnn_size
        graph_metadata = GraphMetadata(schema, mappings['entity'], mappings['relation'], utterance_size, args.max_num_entities, max_degree=max_degree, entity_hist_len=args.entity_hist_len, max_num_items=args.num_items)
        graph_embedder_config = GraphEmbedderConfig(args.node_embed_size, args.edge_embed_size, graph_metadata, entity_embed_size=args.entity_embed_size, use_entity_embedding=args.use_entity_embedding, mp_iters=args.mp_iters, decay=args.utterance_decay, msg_agg=args.msg_aggregation, learned_decay=args.learned_utterance_decay)
        Graph.metadata = graph_metadata
        graph_embedder = GraphEmbedder(graph_embedder_config)
        encoder = GraphEncoder(args.rnn_size, graph_embedder, rnn_type=args.rnn_type, num_layers=args.num_layers, bow_utterance=args.bow_utterance, dropout=args.dropout, update_graph=update_graph, node_embed_in_rnn_inputs=node_embed_in_rnn_inputs)
        if args.model == 'attn-encdec':
            decoder = GraphDecoder(args.rnn_size, vocab.size_word, graph_embedder, rnn_type=args.rnn_type, num_layers=args.num_layers, bow_utterance=args.bow_utterance, checklist=(not args.no_checklist), dropout=args.dropout, sample_t=sample_t, sample_select=sample_select, update_graph=update_graph, node_embed_in_rnn_inputs=node_embed_in_rnn_inputs)
        elif args.model == 'attn-copy-encdec':
            decoder = CopyGraphDecoder(args.rnn_size, vocab.size_word, graph_embedder, rnn_type=args.rnn_type, num_layers=args.num_layers, bow_utterance=args.bow_utterance, checklist=(not args.no_checklist), dropout=args.dropout, sample_t=sample_t, sample_select=sample_select, update_graph=update_graph, node_embed_in_rnn_inputs=node_embed_in_rnn_inputs)
        model = GraphEncoderDecoder(encoder_word_embedder, decoder_word_embedder, graph_embedder, encoder, decoder, pad, select)
    else:
        raise ValueError('Unknown model')
    return model


def build_model(schema, mappings, args):
    tf.reset_default_graph()
    tf.set_random_seed(args.random_seed)

    vocab = mappings['vocab']
    pad = vocab.to_ind_word(markers.PAD)
    select = vocab.to_ind_word(markers.SELECT)

    with tf.variable_scope('EncoderWordEmbedder'):
        print("vocab.size_word:", vocab.size_word)
        print("args.word_embed_size:", args.word_embed_size)

        encoder_word_embedder = WordEmbedder(vocab.size_word, args.word_embed_size, pad)
    with tf.variable_scope('DecoderWordEmbedder'):
        decoder_word_embedder = WordEmbedder(vocab.size_word, args.word_embed_size, pad)

    if args.decoding[0] == 'sample':
        sample_t = float(args.decoding[1])
        sample_select = None if len(args.decoding) < 3 or args.decoding[2] != 'select' else select
    else:
        raise('Unknown decoding method')

    update_graph = (not args.no_graph_update)
    node_embed_in_rnn_inputs = args.node_embed_in_rnn_inputs

    print("args.model:", args.model)

    if args.model == 'encdec':
        encoder = BasicEncoder(args.rnn_size, args.rnn_type, args.num_layers, args.dropout)
        decoder = BasicDecoder(args.rnn_size, vocab.size_word, args.rnn_type, args.num_layers, args.dropout, sample_t, sample_select)
        model = BasicEncoderDecoder(encoder_word_embedder, decoder_word_embedder, encoder, decoder, pad, select)
    elif args.model == 'attn-encdec' or args.model == 'attn-copy-encdec':
        max_degree = args.num_items + len(schema.attributes)
        utterance_size = args.word_embed_size if args.bow_utterance else args.rnn_size
        graph_metadata = GraphMetadata(schema, mappings['entity'], mappings['relation'], utterance_size, args.max_num_entities, max_degree=max_degree, entity_hist_len=args.entity_hist_len, max_num_items=args.num_items)
        Graph.metadata = graph_metadata
        graph_embedder_config = GraphEmbedderConfig(args.node_embed_size, args.edge_embed_size, graph_metadata, entity_embed_size=args.entity_embed_size, use_entity_embedding=args.use_entity_embedding, mp_iters=args.mp_iters, decay=args.utterance_decay, msg_agg=args.msg_aggregation, learned_decay=args.learned_utterance_decay)
        graph_embedder = GraphEmbedder(graph_embedder_config)
        encoder = GraphEncoder(args.rnn_size, graph_embedder, rnn_type=args.rnn_type, num_layers=args.num_layers, bow_utterance=args.bow_utterance, dropout=args.dropout, update_graph=update_graph, node_embed_in_rnn_inputs=node_embed_in_rnn_inputs)
        if args.model == 'attn-encdec':
            decoder = GraphDecoder(args.rnn_size, vocab.size_word, graph_embedder, rnn_type=args.rnn_type, num_layers=args.num_layers, bow_utterance=args.bow_utterance, checklist=(not args.no_checklist), dropout=args.dropout, sample_t=sample_t, sample_select=sample_select, update_graph=update_graph, node_embed_in_rnn_inputs=node_embed_in_rnn_inputs)
        elif args.model == 'attn-copy-encdec':
            decoder = CopyGraphDecoder(args.rnn_size, vocab.size_word, graph_embedder, rnn_type=args.rnn_type, num_layers=args.num_layers, bow_utterance=args.bow_utterance, checklist=(not args.no_checklist), dropout=args.dropout, sample_t=sample_t, sample_select=sample_select, update_graph=update_graph, node_embed_in_rnn_inputs=node_embed_in_rnn_inputs)
        model = GraphEncoderDecoder(encoder_word_embedder, decoder_word_embedder, graph_embedder, encoder, decoder, pad, select)
    else:
        raise ValueError('Unknown model')
    return model


def build_model_BERT(schema, mappings, args):
    tf.reset_default_graph()
    tf.set_random_seed(args.random_seed)

    vocab = mappings['vocab']
    pad = vocab.to_ind_word(markers.PAD)
    select = vocab.to_ind_word(markers.SELECT)

    with tf.variable_scope('EncoderWordEmbedder'):
        print("vocab.size_word:", vocab.size_word)
        print("args.word_embed_size:", args.word_embed_size)

        encoder_word_embedder = WordEmbedder(vocab.size_word, args.word_embed_size, pad)
    with tf.variable_scope('DecoderWordEmbedder'):
        decoder_word_embedder = WordEmbedder(vocab.size_word, args.word_embed_size, pad)

    if args.decoding[0] == 'sample':
        sample_t = float(args.decoding[1])
        sample_select = None if len(args.decoding) < 3 or args.decoding[2] != 'select' else select
    else:
        raise('Unknown decoding method')

    update_graph = (not args.no_graph_update)
    node_embed_in_rnn_inputs = args.node_embed_in_rnn_inputs

    print("args.model:", args.model)

    if args.model == 'encdec':
        encoder = BasicEncoder(args.rnn_size, args.rnn_type, args.num_layers, args.dropout)
        decoder = BasicDecoder(args.rnn_size, vocab.size_word, args.rnn_type, args.num_layers, args.dropout, sample_t, sample_select)
        model = BasicEncoderDecoder(encoder_word_embedder, decoder_word_embedder, encoder, decoder, pad, select)
    elif args.model == 'attn-encdec' or args.model == 'attn-copy-encdec':
        max_degree = args.num_items + len(schema.attributes)
        utterance_size = args.word_embed_size if args.bow_utterance else args.rnn_size
        graph_metadata = GraphMetadata(schema, mappings['entity'], mappings['relation'], utterance_size, args.max_num_entities, max_degree=max_degree, entity_hist_len=args.entity_hist_len, max_num_items=args.num_items)
        graph_embedder_config = GraphEmbedderConfig(args.node_embed_size, args.edge_embed_size, graph_metadata, entity_embed_size=args.entity_embed_size, use_entity_embedding=args.use_entity_embedding, mp_iters=args.mp_iters, decay=args.utterance_decay, msg_agg=args.msg_aggregation, learned_decay=args.learned_utterance_decay)
        Graph.metadata = graph_metadata
        graph_embedder = GraphEmbedder(graph_embedder_config)
        encoder = GraphEncoder(args.rnn_size, graph_embedder, rnn_type=args.rnn_type, num_layers=args.num_layers, bow_utterance=args.bow_utterance, dropout=args.dropout, update_graph=update_graph, node_embed_in_rnn_inputs=node_embed_in_rnn_inputs)
        if args.model == 'attn-encdec':
            decoder = GraphDecoder(args.rnn_size, vocab.size_word, graph_embedder, rnn_type=args.rnn_type, num_layers=args.num_layers, bow_utterance=args.bow_utterance, checklist=(not args.no_checklist), dropout=args.dropout, sample_t=sample_t, sample_select=sample_select, update_graph=update_graph, node_embed_in_rnn_inputs=node_embed_in_rnn_inputs)
        elif args.model == 'attn-copy-encdec':
            decoder = CopyGraphDecoder(args.rnn_size, vocab.size_word, graph_embedder, rnn_type=args.rnn_type, num_layers=args.num_layers, bow_utterance=args.bow_utterance, checklist=(not args.no_checklist), dropout=args.dropout, sample_t=sample_t, sample_select=sample_select, update_graph=update_graph, node_embed_in_rnn_inputs=node_embed_in_rnn_inputs)
        model = GraphEncoderDecoder(encoder_word_embedder, decoder_word_embedder, graph_embedder, encoder, decoder, pad, select)
    else:
        raise ValueError('Unknown model')
    return model


def build_model_HME(schema, mappings, args):
    tf.reset_default_graph()
    tf.set_random_seed(args.random_seed)

    vocab = mappings['vocab']
    pad = vocab.to_ind_word(markers.PAD)
    select = vocab.to_ind_word(markers.SELECT)

    es_fasttext_embed, id2word_fasttext_es, word2id_fasttext_es = load_vec(fasttext_es)
    hi_fasttext_embed, id2word_fasttext_hi, word2id_fasttext_hi = load_vec(fasttext_hi)
    en_fasttext_embed, id2word_fasttext_en, word2id_fasttext_en = load_vec(fasttext_en)
    en_glove_embed, id2word_glove_en, word2id_glove_en = load_vec(glove_en)

    W_fasttext_es = np.zeros((vocab.size_word, args.word_embed_size))
    W_fasttext_hi = np.zeros((vocab.size_word, args.word_embed_size))
    W_fasttext_en = np.zeros((vocab.size_word, args.word_embed_size))
    W_glove_en = np.zeros((vocab.size_word, args.word_embed_size))

    for word in vocab.word_to_ind:
        if word in word2id_fasttext_es:
            W_fasttext_es[vocab.word_to_ind[word]] = es_fasttext_embed[word2id_fasttext_es[word]]

        if word in word2id_fasttext_hi:
            W_fasttext_hi[vocab.word_to_ind[word]] = hi_fasttext_embed[word2id_fasttext_hi[word]]

        if word in word2id_fasttext_en:
            W_fasttext_en[vocab.word_to_ind[word]] = en_fasttext_embed[word2id_fasttext_en[word]]

        if word in word2id_glove_en:
            W_glove_en[vocab.word_to_ind[word]] = en_glove_embed[word2id_glove_en[word]]

    W = [W_fasttext_es, W_fasttext_hi, W_fasttext_en, W_glove_en]

    with tf.variable_scope('EncoderWordEmbedder'):
        encoder_word_embedder = HMEEmbedder(vocab.word_to_ind, vocab.ind_to_word,
                                            vocab.char_to_ind, vocab.ind_to_char,
                                            args.word_embed_size,
                                            args.embedding_size_char,
                                            args.char_hidden_size,
                                            args.dim_key,
                                            args.dim_value,
                                            args.filter_size,
                                            args.max_length,
                                            args.input_dropout,
                                            args.drop,
                                            args.attn_dropout,
                                            args.relu_dropout,
                                            args.bpe_lang_list,
                                            args.bpe_dim,
                                            args.bpe_vocab,
                                            args.bpe_hidden_size,
                                            args.bpe_emb_size,
                                            mappings['bpe_embs'],
                                            args.no_word_emb,
                                            args.mode,
                                            args.add_char_emb,
                                            W,
                                            args.no_projection,
                                            pad=pad)

    with tf.variable_scope('DecoderWordEmbedder'):
        decoder_word_embedder = HMEEmbedder(vocab.word_to_ind, vocab.ind_to_word,
                                            vocab.char_to_ind, vocab.ind_to_char,
                                            args.word_embed_size,
                                            args.embedding_size_char,
                                            args.char_hidden_size,
                                            args.dim_key,
                                            args.dim_value,
                                            args.filter_size,
                                            args.max_length,
                                            args.input_dropout,
                                            args.drop,
                                            args.attn_dropout,
                                            args.relu_dropout,
                                            args.bpe_lang_list,
                                            args.bpe_dim,
                                            args.bpe_vocab,
                                            args.bpe_hidden_size,
                                            args.bpe_emb_size,
                                            mappings['bpe_embs'],
                                            args.no_word_emb,
                                            args.mode,
                                            args.add_char_emb,
                                            W,
                                            args.no_projection,
                                            pad=pad)

    if args.decoding[0] == 'sample':
        sample_t = float(args.decoding[1])
        sample_select = None if len(args.decoding) < 3 or args.decoding[2] != 'select' else select
    else:
        raise('Unknown decoding method')

    update_graph = (not args.no_graph_update)
    node_embed_in_rnn_inputs = args.node_embed_in_rnn_inputs

    if args.model == 'encdec':
        encoder = BasicEncoder(args.rnn_size,
                               args.rnn_type,
                               args.num_layers,
                               args.dropout,
                               mode=args.word_embed_type)

        decoder = BasicDecoder(args.rnn_size,
                               vocab.size_word,
                               args.rnn_type,
                               args.num_layers,
                               args.dropout,
                               sample_t,
                               sample_select,
                               mode=args.word_embed_type)

        model = BasicEncoderDecoder(encoder_word_embedder,
                                    decoder_word_embedder,
                                    encoder,
                                    decoder,
                                    pad,
                                    select)

    elif args.model == 'attn-encdec' or args.model == 'attn-copy-encdec':
        max_degree = args.num_items + len(schema.attributes)
        utterance_size = args.word_embed_size if args.bow_utterance else args.rnn_size

        graph_metadata = GraphMetadata(schema, mappings['entity'],
                                       mappings['relation'],
                                       utterance_size,
                                       args.max_num_entities,
                                       max_degree=max_degree,
                                       entity_hist_len=args.entity_hist_len,
                                       max_num_items=args.num_items)

        graph_embedder_config = GraphEmbedderConfig(args.node_embed_size,
                                                    args.edge_embed_size,
                                                    graph_metadata,
                                                    entity_embed_size=args.entity_embed_size,
                                                    use_entity_embedding=args.use_entity_embedding,
                                                    mp_iters=args.mp_iters,
                                                    decay=args.utterance_decay,
                                                    msg_agg=args.msg_aggregation,
                                                    learned_decay=args.learned_utterance_decay)

        Graph.metadata = graph_metadata
        graph_embedder = GraphEmbedder(graph_embedder_config)

        encoder = GraphEncoder(args.rnn_size,
                               graph_embedder,
                               rnn_type=args.rnn_type,
                               num_layers=args.num_layers,
                               bow_utterance=args.bow_utterance,
                               dropout=args.dropout,
                               update_graph=update_graph,
                               node_embed_in_rnn_inputs=node_embed_in_rnn_inputs,
                               mode=args.word_embed_type)

        if args.model == 'attn-encdec':
            decoder = GraphDecoder(args.rnn_size,
                                   vocab.size_word,
                                   graph_embedder,
                                   rnn_type=args.rnn_type,
                                   num_layers=args.num_layers,
                                   bow_utterance=args.bow_utterance,
                                   checklist=(not args.no_checklist),
                                   dropout=args.dropout,
                                   sample_t=sample_t,
                                   sample_select=sample_select,
                                   update_graph=update_graph,
                                   node_embed_in_rnn_inputs=node_embed_in_rnn_inputs,
                                   mode=args.word_embed_type)

        elif args.model == 'attn-copy-encdec':
            decoder = CopyGraphDecoder(args.rnn_size,
                                       vocab.size_word,
                                       graph_embedder,
                                       rnn_type=args.rnn_type,
                                       num_layers=args.num_layers,
                                       bow_utterance=args.bow_utterance,
                                       checklist=(not args.no_checklist),
                                       dropout=args.dropout,
                                       sample_t=sample_t,
                                       sample_select=sample_select,
                                       update_graph=update_graph,
                                       node_embed_in_rnn_inputs=node_embed_in_rnn_inputs)

        model = GraphEncoderDecoder(encoder_word_embedder,
                                    decoder_word_embedder,
                                    graph_embedder,
                                    encoder,
                                    decoder,
                                    pad,
                                    select)
    else:
        raise ValueError('Unknown model')
    return model


def optional_add(feed_dict, key, value):
    if value is not None:
        feed_dict[key] = value


class Sampler(object):
    '''
    Return a symbol from output/logits (batch_size, seq_len, vocab_size).
    '''
    def __init__(self, t, select=None):
        self.t = t  # Temperature
        # If select is not None, we will down weight <select> during sampling
        self.select = select

    def sample(self, logits, prev_words=None, masked_words=None):
        assert logits.shape[1] == 1
        if prev_words is not None:
            prev_words = np.expand_dims(prev_words, 1)
            logits = np.where(prev_words == 1, logits - np.log(2), logits)

        if masked_words is not None:
            for i, words in enumerate(masked_words):
                for word_id in words:
                    logits[i][0][word_id] = float('-inf')

        if self.select is not None:
            logits[:, 0, self.select] -= np.log(2)

        # Greedy
        if self.t == 0:
            return np.argmax(logits, axis=2)
        # Multinomial sample
        else:
            p = self.softmax(logits, self.t)
            batch_size, seq_len, num_symbols = logits.shape
            preds = np.zeros([batch_size, seq_len], dtype=np.int32)
            for i in range(batch_size):
                for j in range(seq_len):
                    try:
                        preds[i][j] = np.random.choice(num_symbols, 1, p=p[i][j])[0]
                    # p[i][j] do not sum to 1
                    except ValueError:
                        preds[i][j] = np.argmax(p[i][j])
            return preds

    def softmax(self, logits, t=1):
        exp_x = np.exp(logits / t)
        return exp_x / np.sum(exp_x, axis=2, keepdims=True)


bert_config = modeling.BertConfig.from_json_file(mbert_config_file)


class BasicEncoder(object):
    '''
    A basic RNN encoder.
    '''
    def __init__(self, rnn_size, rnn_type='lstm', num_layers=1, dropout=0, mode="RANDOM"):
        self.rnn_size = rnn_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.keep_prob = tf.placeholder(tf.float32)
        self.output_dict = {}
        self.mode = mode

    def _build_inputs(self, input_dict):
        with tf.name_scope(type(self).__name__+'/inputs'):
            self.inputs = tf.placeholder(tf.int32, shape=[None, None], name='inputs')
            self.last_inds = tf.placeholder(tf.int32, shape=[None], name='last_inds')
        if self.mode == "HME":
            with tf.name_scope(type(self).__name__+'/char_inputs'):
                self.char_inputs = tf.placeholder(tf.int32, shape=[None, None], name='char_inputs')

            with tf.name_scope(type(self).__name__+'/bpe_inputs'):
                self.bpe_inputs = tf.placeholder(tf.int32, shape=[None, None, None], name='bpe_inputs')

    def _build_rnn_cell(self):
        return build_rnn_cell(self.rnn_type, self.rnn_size, self.num_layers, self.keep_prob)

    def _build_rnn_inputs(self, word_embedder, time_major):
        if self.mode == "HME":
            inputs = word_embedder.embed(self.inputs,
                                         self.char_inputs,
                                         self.bpe_inputs,
                                         zero_pad=True)
        else:
            inputs = word_embedder.embed(self.inputs,
                                         zero_pad=True)
        self.word_embeddings = inputs
        if not time_major:
            inputs = transpose_first_two_dims(inputs)  # (seq_len, batch_size, input_size)
        return inputs

    def _build_init_state(self, cell, input_dict):
        initial_state = input_dict['init_state']
        if initial_state is not None:
            return initial_state
        else:
            return cell.zero_state(self.batch_size, tf.float32)

    def _build_init_output(self, cell):
        '''
        Initializer for scan. Should have the same shape as the RNN output.
        '''
        return tf.zeros([self.batch_size, cell.output_size])

    def _get_final_state(self, states):
        '''
        Return the final non-pad state from tf.scan outputs.
        '''
        with tf.name_scope(type(self).__name__+'/get_final_state'):
            flat_states = nest.flatten(states)
            flat_last_states = []
            for state in flat_states:
                state = transpose_first_two_dims(state)  # (batch_size, time_seq, state_size)
                # NOTE: when state has dim=4, it's the context which does not change in a seq; just take the last one.
                if len(state.get_shape()) == 4:
                    last_state = state[:, -1, :, :]
                else:
                    last_state = tf.squeeze(batch_embedding_lookup(state, tf.reshape(self.last_inds, [-1, 1])), [1])
                flat_last_states.append(last_state)
            last_states = nest.pack_sequence_as(states, flat_last_states)
        return last_states

    def _build_output_dict(self, rnn_outputs, rnn_states):
        final_state = self._get_final_state(rnn_states)
        self.output_dict.update({'outputs': rnn_outputs, 'final_state': final_state})

    def build_model(self, word_embedder, input_dict, time_major=True, scope=None):
        '''
        inputs: (batch_size, seq_len, input_size)
        '''
        with tf.variable_scope(scope or type(self).__name__):
            self._build_inputs(input_dict)
            self.batch_size = tf.shape(self.inputs)[0]
            self.seq_len = tf.shape(self.inputs)[1]

            cell = self._build_rnn_cell()
            self.init_state = self._build_init_state(cell, input_dict)
            self.output_size = cell.output_size

            inputs = self._build_rnn_inputs(word_embedder, time_major)
            rnn_outputs, states = tf.scan(lambda a, x: cell(x, a[1]),
                                          inputs,
                                          initializer=(self._build_init_output(cell), self.init_state))
            self._build_output_dict(rnn_outputs, states)

    def build_model_bert(self, word_embedder, input_dict, time_major=True, scope=None):
        '''
        inputs: (batch_size, seq_len, input_size)
        '''
        with tf.variable_scope(scope or type(self).__name__):
            self._build_inputs(input_dict)
            self.batch_size = tf.shape(self.inputs)[0]
            self.seq_len = tf.shape(self.inputs)[1]

            cell = self._build_rnn_cell()
            self.init_state = self._build_init_state(cell, input_dict)
            self.output_size = cell.output_size

            inputs = self._build_rnn_inputs(word_embedder, time_major)
            rnn_outputs, states = tf.scan(lambda a, x: cell(x, a[1]),
                                          inputs,
                                          initializer=(self._build_init_output(cell), self.init_state))
            self._build_output_dict(rnn_outputs, states)

    def get_feed_dict(self, **kwargs):
        feed_dict = kwargs.pop('feed_dict', {})
        feed_dict[self.inputs] = kwargs.pop('inputs')
        if self.mode == "HME":
            feed_dict[self.char_inputs] = kwargs.pop('char_inputs')
            feed_dict[self.bpe_inputs] = kwargs.pop('bpe_inputs')
        feed_dict[self.last_inds] = kwargs.pop('last_inds')
        feed_dict[self.keep_prob] = 1. - self.dropout
        optional_add(feed_dict, self.init_state, kwargs.pop('init_state', None))
        return feed_dict

    def run(self, sess, fetches, feed_dict):
        results = sess.run([self.output_dict[x] for x in fetches], feed_dict=feed_dict)
        return {k: results[i] for i, k in enumerate(fetches)}

    def encode(self, sess, **kwargs):
        feed_dict = self.get_feed_dict(**kwargs)
        return self.run(sess, ('final_state',), feed_dict)


class GraphEncoder(BasicEncoder):
    '''
    RNN encoder that update knowledge graph at the end.
    '''
    def __init__(self,
                 rnn_size,
                 graph_embedder,
                 rnn_type='lstm',
                 num_layers=1,
                 dropout=0,
                 bow_utterance=False,
                 node_embed_in_rnn_inputs=False,
                 update_graph=True,
                 mode="RANDOM"):

        super(GraphEncoder, self).__init__(rnn_size, rnn_type, num_layers, dropout)
        self.graph_embedder = graph_embedder
        self.context_size = self.graph_embedder.config.context_size
        # Id of the utterance matrix to be updated: 0 is encoder utterances, 1 is decoder utterances
        self.utterance_id = 0
        self.bow_utterance = bow_utterance
        self.node_embed_in_rnn_inputs = node_embed_in_rnn_inputs
        self.update_graph = update_graph
        self.mode = mode

    def _build_graph_variables(self, input_dict):
        if 'utterances' in input_dict:
            self.utterances = input_dict['utterances']
        else:
            self.utterances = (tf.placeholder(tf.float32,
                                              shape=[None, None, self.graph_embedder.config.utterance_size],
                                              name='encoder_utterances'),
                               tf.placeholder(tf.float32,
                                              shape=[None, None, self.graph_embedder.config.utterance_size],
                                              name='decoder_utterances'))

        if 'context' in input_dict:
            self.context = input_dict['context']
        else:
            with tf.variable_scope(tf.get_variable_scope(), reuse=self.graph_embedder.context_initialized):
                self.context = self.graph_embedder.get_context(self.utterances)
        self.num_nodes = tf.to_int32(tf.shape(self.context[0])[1])

    def _build_inputs(self, input_dict):
        super(GraphEncoder, self)._build_inputs(input_dict)
        with tf.name_scope(type(self).__name__+'/inputs'):
            # Entities whose embedding are to be updated
            self.update_entities = tf.placeholder(tf.int32, shape=[None, None], name='update_entities')
            # Entities in the current utterance. Non-entity words are -1.
            self.entities = tf.placeholder(tf.int32, shape=[None, None], name='entities')

    def _get_node_embedding(self, context, node_ids):
        '''
        Lookup embeddings of nodes from context.
        node_ids: (batch_size, seq_len)
        context: (batch_size, num_nodes, context_size)
        Return node_embeds (batch_size, seq_len, context_size)
        '''
        node_embeddings = batch_embedding_lookup(context, node_ids, zero_ind=-1)  # (batch_size, seq_len, context_size)
        return node_embeddings

    def _build_rnn_inputs(self, word_embedder, time_major):
        '''
        Concatenate word embedding with entity/node embedding.
        '''
        if self.mode == "HME":
            word_embeddings = word_embedder.embed(self.inputs,
                                                  self.char_inputs,
                                                  self.bpe_inputs,
                                                  zero_pad=True)
        else:
            word_embeddings = word_embedder.embed(self.inputs,
                                                  zero_pad=True)

        self.word_embeddings = word_embeddings
        if self.node_embed_in_rnn_inputs:
            entity_embeddings = self._get_node_embedding(self.context[0], self.entities)
            inputs = tf.concat([word_embeddings, entity_embeddings], axis=2)
        else:
            inputs = word_embeddings
        if not time_major:
            inputs = transpose_first_two_dims(inputs)  # (seq_len, batch_size, input_size)
        return inputs

    def build_model(self, word_embedder, input_dict, time_major=True, scope=None):
        # Variable space is GraphEncoderDecoder
        self._build_graph_variables(input_dict)

        # Variable space is type(self)
        super(GraphEncoder, self).build_model(word_embedder, input_dict, time_major=time_major, scope=scope)

        # Variable space is GraphEncoderDecoder
        # Use the final encoder state as the utterance embedding
        final_output = self._get_final_state(self.output_dict['outputs'])
        self.utterance_embedding = tf.reduce_sum(self.word_embeddings, 1)
        if self.bow_utterance:
            new_utterances = self.graph_embedder.update_utterance(self.update_entities,
                                                                  self.utterance_embedding,
                                                                  self.utterances,
                                                                  self.utterance_id)
        else:
            new_utterances = self.graph_embedder.update_utterance(self.update_entities,
                                                                  final_output,
                                                                  self.utterances,
                                                                  self.utterance_id)
        if not self.update_graph:
            new_utterances = self.utterances
        with tf.variable_scope(tf.get_variable_scope(), reuse=self.graph_embedder.context_initialized):
            context = self.graph_embedder.get_context(new_utterances)

        self.output_dict['utterances'] = new_utterances
        self.output_dict['context'] = context
        self.output_dict['final_output'] = final_output
        self.output_dict['utterance_embedding'] = self.utterance_embedding

    def get_feed_dict(self, **kwargs):
        feed_dict = super(GraphEncoder, self).get_feed_dict(**kwargs)
        feed_dict[self.entities] = kwargs.pop('entities')
        optional_add(feed_dict, self.utterances, kwargs.pop('utterances', None))
        optional_add(feed_dict, self.update_entities, kwargs.pop('update_entities', None))
        return feed_dict

    def encode(self, sess, **kwargs):
        feed_dict = self.get_feed_dict(**kwargs)
        feed_dict = self.graph_embedder.get_feed_dict(feed_dict=feed_dict, **kwargs['graph_data'])
        return self.run(sess, ('final_state', 'final_output', 'utterances', 'context'), feed_dict)


class BasicDecoder(BasicEncoder):
    def __init__(self,
                 rnn_size,
                 num_symbols,
                 rnn_type='lstm',
                 num_layers=1,
                 dropout=0,
                 sample_t=0,
                 sample_select=None,
                 mode="RANDOM"):

        super(BasicDecoder, self).__init__(rnn_size, rnn_type, num_layers, dropout)
        self.num_symbols = num_symbols
        self.sampler = Sampler(sample_t, sample_select)
        self.mode = mode

    def get_feed_dict(self, **kwargs):
        feed_dict = super(BasicDecoder, self).get_feed_dict(**kwargs)
        optional_add(feed_dict, self.matched_items, kwargs.pop('matched_items', None))
        return feed_dict

    def _build_inputs(self, input_dict):
        super(BasicDecoder, self)._build_inputs(input_dict)
        with tf.name_scope(type(self).__name__+'/inputs'):
            self.matched_items = tf.placeholder(tf.int32, shape=[None], name='matched_items')

    def _build_output(self, output_dict):
        '''
        Take RNN outputs and produce logits over the vocab.
        '''
        outputs = output_dict['outputs']
        outputs = transpose_first_two_dims(outputs)  # (batch_size, seq_len, output_size)
        logits = batch_linear(outputs, self.num_symbols, True)
        #logits = self.penalize_repetition(logits)
        return logits

    @classmethod
    def penalize_repetition(cls, logits):
        #return logits
        exp_logits = tf.exp(tf.clip_by_value(logits, -100, 100))
        logits = tf.log(tf.clip_by_value(exp_logits, 1e-10, 1e10)) - tf.log(tf.clip_by_value((tf.cumsum(exp_logits, axis=1) - exp_logits), 1e-10, 1e10))
        return logits

    def compute_loss(self, targets, pad, select):
        logits = self.output_dict['logits']
        loss, seq_loss, total_loss = self._compute_loss(logits, targets, pad)
        # -1 is selection loss
        return loss, seq_loss, total_loss, tf.constant(-1)

    @classmethod
    def _compute_loss(cls, logits, targets, pad):
        '''
        logits: (batch_size, seq_len, vocab_size)
        targets: (batch_size, seq_len)
        '''
        batch_size = tf.shape(logits)[0]
        num_symbols = tf.shape(logits)[2]
        # sparse_softmax_cross_entropy_with_logits only takes 2D tensors
        logits = tf.reshape(logits, [-1, num_symbols])
        targets = tf.reshape(targets, [-1])

        # Mask padded tokens
        token_weights = tf.cast(tf.not_equal(targets, tf.constant(pad)), tf.float32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets) * token_weights
        total_loss = tf.reduce_sum(loss)
        token_weights_sum = tf.reduce_sum(tf.reshape(token_weights, [batch_size, -1]), 1) + EPS
        # Average over words in each sequence
        seq_loss = tf.reduce_sum(tf.reshape(loss, [batch_size, -1]), 1) / token_weights_sum

        # Average over sequences
        loss = tf.reduce_sum(seq_loss) / tf.to_float(batch_size)
        # total_loss is used to compute perplexity
        return loss, seq_loss, (total_loss, tf.reduce_sum(token_weights))

    def build_model(self, word_embedder, input_dict, time_major=True, scope=None):
        super(BasicDecoder, self).build_model(word_embedder, input_dict, time_major=time_major, scope=scope)  # outputs: (seq_len, batch_size, output_size)
        with tf.variable_scope(scope or type(self).__name__):
            logits = self._build_output(self.output_dict)
        self.output_dict['logits'] = logits

    def pred_to_input(self, preds, **kwargs):
        '''
        Convert predictions to input of the next decoding step.
        '''
        textint_map = kwargs.pop('textint_map')
        inputs = textint_map.pred_to_input(preds)
        return inputs

    def decode(self, sess, max_len, batch_size=1, stop_symbol=None, **kwargs):
        if stop_symbol is not None:
            assert batch_size == 1, 'Early stop only works for single instance'
        feed_dict = self.get_feed_dict(**kwargs)
        preds = np.zeros([batch_size, max_len], dtype=np.int32)
        # last_inds=0 because input length is one from here on
        last_inds = np.zeros([batch_size], dtype=np.int32)
        for i in range(max_len):
            logits, final_state = sess.run((self.output_dict['logits'], self.output_dict['final_state']), feed_dict=feed_dict)
            step_preds = self.sampler.sample(logits)
            preds[:, [i]] = step_preds
            if step_preds[0][0] == stop_symbol:
                break
            feed_dict = self.get_feed_dict(inputs=self.pred_to_input(step_preds, **kwargs),
                                           last_inds=last_inds,
                                           init_state=final_state)
        return {'preds': preds, 'final_state': final_state}


class GraphDecoder(GraphEncoder):
    '''
    Decoder with attention mechanism over the graph.
    '''
    def __init__(self,
                 rnn_size,
                 num_symbols,
                 graph_embedder,
                 rnn_type='lstm',
                 num_layers=1,
                 dropout=0,
                 bow_utterance=False,
                 scoring='linear',
                 output='project',
                 checklist=True,
                 sample_t=0,
                 sample_select=None,
                 node_embed_in_rnn_inputs=False,
                 update_graph=True,
                 mode="RANDOM"):

        super(GraphDecoder, self).__init__(rnn_size, graph_embedder, rnn_type, num_layers, dropout, bow_utterance, node_embed_in_rnn_inputs, update_graph)
        self.sampler = Sampler(sample_t, sample_select)
        self.num_symbols = num_symbols
        self.utterance_id = 1
        self.scorer = scoring
        self.output_combiner = output
        self.checklist = checklist
        self.mode = mode

    def compute_loss(self, targets, pad, select):
        logits = self.output_dict['logits']
        loss, seq_loss, total_loss = BasicDecoder._compute_loss(logits, targets, pad)
        # -1 is selection loss
        return loss, seq_loss, total_loss, tf.constant(-1)

    def _build_rnn_cell(self):
        return AttnRNNCell(self.rnn_size,
                           self.context_size,
                           self.rnn_type,
                           self.keep_prob,
                           self.scorer,
                           self.output_combiner,
                           self.num_layers,
                           self.checklist)

    def _build_init_output(self, cell):
        '''
        Output includes both RNN output and attention scores.
        '''
        output = super(GraphDecoder, self)._build_init_output(cell)
        return (output, tf.zeros_like(self.graph_embedder.node_ids, dtype=tf.float32))

    def _build_output(self, output_dict):
        '''
        Take RNN outputs and produce logits over the vocab.
        '''
        outputs = output_dict['outputs']
        outputs = transpose_first_two_dims(outputs)  # (batch_size, seq_len, output_size)
        logits = batch_linear(outputs, self.num_symbols, True)
        #logits = BasicDecoder.penalize_repetition(logits)
        return logits

    def _build_init_state(self, cell, input_dict):
        self.init_output = input_dict['init_output']
        self.init_rnn_state = input_dict['init_state']
        if self.init_rnn_state is not None:
            # NOTE: we assume that the initial state comes from the encoder and is just
            # the rnn state. We need to compute attention and get context for the attention
            # cell's initial state.
            return cell.init_state(self.init_rnn_state, self.init_output, self.context, tf.cast(self.init_checklists[:, 0, :], tf.float32))
        else:
            return cell.zero_state(self.batch_size, self.context)

    # TODO: hacky interface
    def compute_init_state(self, sess, init_rnn_state, init_output, context, init_checklists):
        init_state = sess.run(self.init_state,
                              feed_dict={self.init_output: init_output,
                                         self.init_rnn_state: init_rnn_state,
                                         self.context: context,
                                         self.init_checklists: init_checklists,}
                              )
        return init_state

    def _build_inputs(self, input_dict):
        super(GraphDecoder, self)._build_inputs(input_dict)
        with tf.name_scope(type(self).__name__+'/inputs'):
            self.matched_items = tf.placeholder(tf.int32, shape=[None], name='matched_items')
            self.init_checklists = tf.placeholder(tf.int32, shape=[None, None, None], name='init_checklists')

    def _build_rnn_inputs(self, word_embedder, time_major):
        inputs = super(GraphDecoder, self)._build_rnn_inputs(word_embedder, time_major)

        checklists = tf.cumsum(tf.one_hot(self.entities, self.num_nodes, on_value=1, off_value=0), axis=1) + self.init_checklists
        # cumsum can cause >1 indicator
        checklists = tf.cast(tf.greater(checklists, 0), tf.float32)
        self.output_dict['checklists'] = checklists

        checklists = transpose_first_two_dims(checklists)  # (seq_len, batch_size, num_nodes)
        return inputs, checklists

    def build_model(self, word_embedder, input_dict, time_major=True, scope=None):
        super(GraphDecoder, self).build_model(word_embedder, input_dict, time_major=time_major, scope=scope)  # outputs: (seq_len, batch_size, output_size)
        with tf.variable_scope(scope or type(self).__name__):
            logits = self._build_output(self.output_dict)
        self.output_dict['logits'] = logits
        self.output_dict['probs'] = tf.nn.softmax(logits)

    def _build_output_dict(self, rnn_outputs, rnn_states):
        final_state = self._get_final_state(rnn_states)
        outputs, attn_scores = rnn_outputs
        self.output_dict.update({'outputs': outputs, 'attn_scores': attn_scores, 'final_state': final_state})

    def get_feed_dict(self, **kwargs):
        feed_dict = super(GraphDecoder, self).get_feed_dict(**kwargs)
        feed_dict[self.init_checklists] = kwargs.pop('init_checklists')
        optional_add(feed_dict, self.matched_items, kwargs.pop('matched_items', None))
        return feed_dict

    def pred_to_input(self, preds, **kwargs):
        '''
        Convert predictions to input of the next decoding step.
        '''
        textint_map = kwargs.pop('textint_map')
        inputs = textint_map.pred_to_input(preds)
        return inputs

    def pred_to_entity(self, pred, graphs, vocab):
        return graphs.pred_to_entity(pred, vocab.size_word)

    def decode(self, sess, max_len, batch_size=1, stop_symbol=None, **kwargs):
        if stop_symbol is not None:
            assert batch_size == 1, 'Early stop only works for single instance'
        feed_dict = self.get_feed_dict(**kwargs)
        cl = kwargs['init_checklists']
        preds = np.zeros([batch_size, max_len], dtype=np.int32)
        generated_word_types = None
        # last_inds=0 because input length is one from here on
        last_inds = np.zeros([batch_size], dtype=np.int32)
        attn_scores = []
        probs = []
        graphs = kwargs['graphs']
        vocab = kwargs['vocab']
        select = vocab.to_ind_word(markers.SELECT)
        word_embeddings = 0

        for i in range(max_len):
            # NOTE: since we're running for one step, utterance_embedding is essentially word_embedding
            output_nodes = [self.output_dict['logits'], self.output_dict['final_state'], self.output_dict['final_output'], self.output_dict['utterance_embedding'], self.output_dict['attn_scores'], self.output_dict['probs'], self.output_dict['checklists']]
            if 'selection_scores' in self.output_dict:
                output_nodes.append(self.output_dict['selection_scores'])

            if 'selection_scores' in self.output_dict:
                logits, final_state, final_output, utterance_embedding, attn_score, prob, cl, selection_scores = sess.run(output_nodes, feed_dict=feed_dict)
            else:
                logits, final_state, final_output, utterance_embedding, attn_score, prob, cl = sess.run(output_nodes, feed_dict=feed_dict)

            word_embeddings += utterance_embedding
            # attn_score: seq_len x batch_size x num_nodes, seq_len=1, so we take attn_score[0]
            attn_scores.append(attn_score[0])
            # probs: batch_size x seq_len x num_symbols
            probs.append(prob[:, 0, :])
            step_preds = self.sampler.sample(logits, prev_words=None)

            if generated_word_types is None:
                generated_word_types = np.zeros([batch_size, logits.shape[2]])
            generated_word_types[np.arange(batch_size), step_preds[:, 0]] = 1

            preds[:, [i]] = step_preds
            if step_preds[0][0] == stop_symbol:
                break
            entities = self.pred_to_entity(step_preds, graphs, vocab)

            feed_dict = self.get_feed_dict(inputs=self.pred_to_input(step_preds, **kwargs),
                                           last_inds=last_inds,
                                           init_state=final_state,
                                           init_checklists=cl,
                                           entities=entities,
                                           )
        # NOTE: the final_output may not be at the stop symbol when the function is running
        # in batch mode -- it will be the state at max_len. This is fine since during test
        # we either run with batch_size=1 (real-time chat) or use the ground truth to update
        # the state (see generate()).
        output_dict = {'preds': preds, 'final_state': final_state, 'final_output': final_output, 'attn_scores': attn_scores, 'probs': probs, 'utterance_embedding': word_embeddings, 'checklists': cl}
        if 'selection_scores' in self.output_dict:
            output_dict['selection_scores'] = selection_scores
        return output_dict

    def _print_cl(self, cl):
        print ('checklists:')
        for i in range(cl.shape[0]):
            nodes = []
            for j, c in enumerate(cl[i][0]):
                if c != 0:
                    nodes.append(j)
            if len(nodes) > 0:
                print (i, nodes)

    def _print_copied_nodes(self, cn):
        print ('copied_nodes:')
        cn, mask = cn
        for i, (c, m) in enumerate(zip(cn, mask)):
            if m:
                print (i, c)

    def update_context(self, sess, entities, final_output, utterance_embedding, utterances, graph_data):
        feed_dict = {self.update_entities: entities,
                self.output_dict['final_output']: final_output,
                self.output_dict['utterance_embedding']: utterance_embedding,
                self.utterances: utterances}
        feed_dict = self.graph_embedder.get_feed_dict(feed_dict=feed_dict, **graph_data)
        new_utterances, new_context = sess.run([self.output_dict['utterances'], self.output_dict['context']], feed_dict=feed_dict)
        return new_utterances, new_context


class CopyGraphDecoder(GraphDecoder):
    '''
    Decoder with copy mechanism over the attention context.
    '''
    def _build_output(self, output_dict):
        '''
        Take RNN outputs and produce logits over the vocab and the attentions.
        '''
        logits = super(CopyGraphDecoder, self)._build_output(output_dict)  # (batch_size, seq_len, num_symbols)
        attn_scores = transpose_first_two_dims(output_dict['attn_scores'])  # (batch_size, seq_len, num_nodes)
        return tf.concat([logits, attn_scores], axis=2)

    def pred_to_entity(self, pred, graphs, vocab):
        '''
        Return copied nodes for a single time step.
        '''
        offset = vocab.size_word
        pred = graphs.copy_preds(pred, offset)
        node_ids = graphs._pred_to_node_id(pred, offset)
        return node_ids

    def pred_to_input(self, preds, **kwargs):
        graphs = kwargs.pop('graphs')
        vocab = kwargs.pop('vocab')
        textint_map = kwargs.pop('textint_map')
        preds = graphs.copy_preds(preds, vocab.size_word)
        preds = textint_map.pred_to_input(preds)
        return preds


class BasicEncoderDecoder(object):
    '''
    Basic seq2seq model.
    '''
    def __init__(self, encoder_word_embedder, decoder_word_embedder, encoder, decoder, pad, select, scope=None):
        self.PAD = pad  # Id of PAD in the vocab
        self.SELECT = select
        self.encoder = encoder
        self.decoder = decoder
        self.build_model(encoder_word_embedder, decoder_word_embedder, encoder, decoder, scope)

    def compute_loss(self, output_dict, targets):
        return self.decoder.compute_loss(targets, self.PAD, self.SELECT)

    def _encoder_input_dict(self):
        return {
                'init_state': None,
               }

    def _decoder_input_dict(self, encoder_output_dict):
        return {
                'init_state': encoder_output_dict['final_state'],
               }

    def build_model(self, encoder_word_embedder, decoder_word_embedder, encoder, decoder, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # Encoding
            with tf.name_scope('Encoder'):
                encoder_input_dict = self._encoder_input_dict()
                encoder.build_model(encoder_word_embedder, encoder_input_dict, time_major=False)

            # Decoding
            with tf.name_scope('Decoder'):
                decoder_input_dict = self._decoder_input_dict(encoder.output_dict)
                decoder.build_model(decoder_word_embedder, decoder_input_dict, time_major=False)

            self.targets = tf.placeholder(tf.int32, shape=[None, None], name='targets')

            # Loss
            self.loss, self.seq_loss, self.total_loss, self.select_loss = self.compute_loss(decoder.output_dict, self.targets)

    def get_feed_dict(self, **kwargs):
        feed_dict = kwargs.pop('feed_dict', {})
        feed_dict = self.encoder.get_feed_dict(**kwargs.pop('encoder'))
        feed_dict = self.decoder.get_feed_dict(feed_dict=feed_dict, **kwargs.pop('decoder'))
        optional_add(feed_dict, self.targets, kwargs.pop('targets', None))
        return feed_dict

    def generate(self, sess, batch, encoder_init_state, max_len, copy=False, vocab=None, graphs=None, utterances=None, textint_map=None):
        encoder_inputs = batch['encoder_inputs']
        encoder_char_inputs = batch['encoder_char_inputs']
        encoder_bpe_inputs = batch['encoder_bpe_inputs']

        decoder_inputs = batch['decoder_inputs']
        decoder_char_inputs = batch['decoder_char_inputs']
        decoder_bpe_inputs = batch['decoder_bpe_inputs']
        batch_size = encoder_inputs.shape[0]

        # Encode true prefix
        encoder_args = {'inputs': encoder_inputs,
                        'char_inputs': encoder_char_inputs,
                        'bpe_inputs': encoder_bpe_inputs,
                        'last_inds': batch['encoder_inputs_last_inds'],
                        'init_state': encoder_init_state
                        }
        if graphs:
            graph_data = graphs.get_batch_data(batch['encoder_tokens'], None, batch['encoder_entities'], None, utterances, vocab)
            encoder_args['update_entities'] = graph_data['encoder_entities']
            encoder_args['entities'] = graph_data['encoder_nodes']
            encoder_args['utterances'] = graph_data['utterances']
            encoder_args['graph_data'] = graph_data
        encoder_output_dict = self.encoder.encode(sess, **encoder_args)

        # Decode max_len steps
        decoder_args = {'inputs': decoder_inputs[:, [0]],
                        'char_inputs': decoder_char_inputs[:, :, [0]],
                        'bpe_inputs': decoder_bpe_inputs[:, :, [0]],
                        'last_inds': np.zeros([batch_size], dtype=np.int32),
                        'init_state': encoder_output_dict['final_state'],
                        'textint_map': textint_map
                        }
        if graphs:
            init_checklists = graphs.get_zero_checklists(1)
            entities = graphs.get_zero_entities(1)
            decoder_args['init_state'] = self.decoder.compute_init_state(sess,
                                                                         encoder_output_dict['final_state'],
                                                                         encoder_output_dict['final_output'],
                                                                         encoder_output_dict['context'],
                                                                         init_checklists,
                                                                         )
            decoder_args['init_checklists'] = init_checklists
            decoder_args['entities'] = entities
            decoder_args['graphs'] = graphs
            decoder_args['vocab'] = vocab
        decoder_output_dict = self.decoder.decode(sess, max_len, batch_size, **decoder_args)

        # Decode true utterances (so that we always condition on true prefix)
        decoder_args['inputs'] = decoder_inputs
        decoder_args['last_inds'] = batch['decoder_inputs_last_inds']
        if graphs is not None:
            # TODO: why do we need to do encoding again
            # Read decoder tokens and update graph
            new_graph_data = graphs.get_batch_data(None, batch['decoder_tokens'], batch['encoder_entities'], batch['decoder_entities'], utterances, vocab)
            decoder_args['encoder_entities'] = new_graph_data['encoder_nodes']
            # Add checklists
            decoder_args['init_checklists'] = graphs.get_zero_checklists(1)
            # Add copied nodes
            decoder_args['entities'] = new_graph_data['decoder_nodes']
            # Update utterance matrix size and decoder entities given the true decoding sequence
            encoder_args['utterances'] = new_graph_data['utterances']
            decoder_args['update_entities'] = new_graph_data['decoder_entities']
            # Continue from encoder state, don't need init_state
            decoder_args.pop('init_state')
            kwargs = {'encoder': encoder_args, 'decoder': decoder_args, 'graph_embedder': new_graph_data}
            feed_dict = self.get_feed_dict(**kwargs)
            true_final_state, utterances, true_checklists = sess.run((self.decoder.output_dict['final_state'], self.decoder.output_dict['utterances'], self.decoder.output_dict['checklists']), feed_dict=feed_dict)

            result = {'preds': decoder_output_dict['preds'],
                      'final_state': decoder_output_dict['final_state'],
                      'true_final_state': true_final_state,
                      'utterances': utterances,
                      'attn_scores': decoder_output_dict['attn_scores'],
                      'probs': decoder_output_dict['probs'],
                      }
            if 'selection_scores' in decoder_output_dict:
                result['selection_scores'] = decoder_output_dict['selection_scores']
                result['true_checklists'] = true_checklists
            return result
        else:
            feed_dict = self.decoder.get_feed_dict(**decoder_args)
            true_final_state = sess.run((self.decoder.output_dict['final_state']), feed_dict=feed_dict)
            return {'preds': decoder_output_dict['preds'],
                    'final_state': decoder_output_dict['final_state'],
                    'true_final_state': true_final_state,
                    }


class GraphEncoderDecoder(BasicEncoderDecoder):
    def __init__(self,
                 encoder_word_embedder,
                 decoder_word_embedder,
                 graph_embedder,
                 encoder,
                 decoder,
                 pad,
                 select,
                 scope=None):

        self.graph_embedder = graph_embedder
        super(GraphEncoderDecoder, self).__init__(encoder_word_embedder,
                                                  decoder_word_embedder,
                                                  encoder,
                                                  decoder,
                                                  pad,
                                                  select,
                                                  scope)

    def _decoder_input_dict(self, encoder_output_dict):
        input_dict = super(GraphEncoderDecoder, self)._decoder_input_dict(encoder_output_dict)
        # This is used to compute the initial attention
        input_dict['init_output'] = self.encoder._get_final_state(encoder_output_dict['outputs'])
        input_dict['utterances'] = encoder_output_dict['utterances']
        input_dict['context'] = encoder_output_dict['context']
        return input_dict

    def get_feed_dict(self, **kwargs):
        feed_dict = super(GraphEncoderDecoder, self).get_feed_dict(**kwargs)
        feed_dict = self.graph_embedder.get_feed_dict(feed_dict=feed_dict, **kwargs['graph_embedder'])
        return feed_dict
