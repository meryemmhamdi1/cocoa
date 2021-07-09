'''
Preprocess examples in a dataset and generate data for models.
Things I have changed here

def item_to_entity(id_):
    item_str = item_to_str(id_)
    return (item_str, (item_str, 'item'))

def item_to_entities(self, item, attrs):
    entities = [(value, (value, type_)) for value, type_ in ((item[attr.name].lower(), self.attribute_types[attr.name]) for attr in attrs)]
    return entities

def process_event(self, e, kb, mentioned_entities=None, tokenizer=tokenize_wp, known_kb=True):
    if e.action == 'message':
    # Lower, tokenize, link entity
    entity_tokens = self.lexicon.link_entity(tokenizer(e.data), kb=kb, mentioned_entities=mentioned_entities, known_kb=known_kb)
    entity_tokens = [normalize_number(x) if not is_entity(x) else x for x in entity_tokens]
    if entity_tokens:
        # NOTE: have two copies because we might change it given decoding/encoding
        return (entity_tokens, copy.copy(entity_tokens))
    else:
        return None
    elif e.action == 'select':
        # Convert an item to item-id (wrt to the speaker)
        item_id = self.get_item_id(kb, e.data)
        # We use the entities to represent the item during encoding and item-id during decoding
        print("e.data:", e.data, "  self.item_to_entities(e.data, kb.attributes):", self.item_to_entities(e.data, kb.attributes))
        return ([markers.SELECT] + self.item_to_entities(e.data, kb.attributes), [markers.SELECT, item_to_entity(item_id)])
    else:
        raise ValueError('Unknown event action.')

'''

import random
import re
import numpy as np
from src.model.vocab import Vocabulary, is_entity
from src.model.graph import Graph, GraphBatch, inv_rel, item_to_str
from itertools import chain
from collections import namedtuple, defaultdict
import copy
from bert import tokenization
import collections
import six
import tensorflow as tf
import os
from bpemb import BPEmb

if "/nas/clear/users/meryem/" in os.getcwd():
    vocab_file = "/nas/clear/users/meryem/Embeddings/BERT/multi_cased_L-12_H-768_A-12/vocab.txt"
elif "/home1/mmhamdi" in os.getcwd():
    vocab_file = "/home1/mmhamdi/Models/bert-base-multilingual-cased/vocab.txt"
else:
    vocab_file = "/Users/d22admin/USCGDrive/ISI/EventExtraction/4Embeddings/Multilingual/multi_cased_L-12_H-768_A-12-2/vocab.txt"
do_lower_case = False


def add_preprocess_arguments(parser):
    parser.add_argument('--entity-encoding-form', choices=['type', 'canonical'], default='canonical', help='Input entity form to the encoder')
    parser.add_argument('--entity-decoding-form', choices=['canonical', 'type'], default='canonical', help='Input entity form to the decoder')
    parser.add_argument('--entity-target-form', choices=['canonical', 'type', 'graph'], default='canonical', help='Output entity form to the decoder')


SpecialSymbols = namedtuple('SpecialSymbols', ['EOS', 'GO', 'SELECT', 'PAD', 'EOE'])
markers = SpecialSymbols(EOS='</s>', GO='<go>', SELECT='<select>', PAD='<pad>', EOE='</e>')


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


vocab = load_vocab(vocab_file)
tokenizer = tokenization.WordpieceTokenizer(vocab=vocab)


def tokenize_wp(utterance):
    print("Word Piece Tokenization")
    utterance = utterance.lower()
    for s in (' - ', '-'):
        utterance = utterance.replace(s, ' ')

    tokens = ["[CLS]"] + tokenizer.tokenize(utterance) + ["[SEP]"]
    print("tokens:", tokens)
    return tokens


def tokenize_type_based(utterance):
    '''
    'hi there!' => ['hi', 'there', '!']
    '''
    utterance = utterance.lower()
    # Remove '-' to match lexicon preprocess
    for s in (' - ', '-'):
        utterance = utterance.replace(s, ' ')
    # Split on punctuation
    tokens = re.findall(r"[\w']+|[.,!?;&-]", utterance)
    return tokens


word_to_num = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}


def normalize_number(token):
    if token in word_to_num:
        return word_to_num[token]
    return token


def item_to_entity(id_):
    item_str = item_to_str(id_)
    return item_str
    #return (item_str, (item_str, 'item'))


def build_schema_mappings(schema, num_items):
    entity_map = Vocabulary(unk=True)
    for type_, values in schema.values.items():
        entity_map.add_words(((value.lower(), type_) for value in values))
    # Add item nodes
    for i in range(num_items):
        entity_map.add_word(item_to_entity(i)[1])
    # Add attr nodes
    #for attr in schema.attributes:
    #    entity_map.add_word((attr.name.lower(), 'attr'))

    relation_map = Vocabulary(unk=False)
    attribute_types = schema.get_attributes()  # {attribute_name: value_type}
    relation_map.add_words((a.lower() for a in attribute_types.keys()))
    relation_map.add_word('has')
    # Inverse relation
    relation_map.add_words([inv_rel(r) for r in relation_map.word_to_ind])

    return entity_map, relation_map


def build_bpe(args, bpe_cache=""):
    # BPE-LEVEL
    bpe_embs = []
    if args.bpe_lang_list is not None:
        print("Loading BPE:", args.bpe_lang_list)
        for i in range(len(args.bpe_lang_list)):
            bpemb = BPEmb(lang=args.bpe_lang_list[i], dim=args.bpe_emb_size, vs=args.bpe_vocab)#, cache_dir=bpe_cache
            bpe_embs.append(bpemb)
    return bpe_embs


def build_bpe_constant(bpe_lang_list=["en", "es", "hi"], bpe_emb_size=300, bpe_vocab=5000, bpe_cache=""):
    # BPE-LEVEL
    bpe_embs = []
    if bpe_lang_list is not None:
        print("Loading BPE:", bpe_lang_list)
        for i in range(len(bpe_lang_list)):
            bpemb = BPEmb(lang=bpe_lang_list[i], dim=bpe_emb_size, vs=bpe_vocab)#, cache_dir=bpe_cache
            bpe_embs.append(bpemb)
    return bpe_embs


def build_vocab(dialogues, special_symbols=[], entity_forms=[]):
    vocab = Vocabulary(offset_word=0, offset_char=0, unk=True)

    def _add_entity(entity):
        for entity_form in entity_forms:
            # If copy entity embedding from the graph embedding, don't need entity in vocab
            if entity_form != 'graph':
                word = Preprocessor.get_entity_form(entity, entity_form)
                vocab.add_word(word)

    # Add words
    for dialogue in dialogues:
        assert dialogue.is_int is False
        for turns in dialogue.token_turns:
            for turn in turns:
                for token in chain.from_iterable(turn):
                    if is_entity(token):
                        _add_entity(token)
                    else:
                        vocab.add_word(token)
                    # Add Chars to the vocabulary
                    for char in token:
                        vocab.add_char(char)

    # Add special symbols
    vocab.add_words(special_symbols)
    print('Vocabulary word size:', vocab.size_word, ' char size:', vocab.size_char)
    return vocab


def create_mappings(args, dialogues, schema, num_items, entity_forms):
    #print("dialogues[0].token_turns:", dialogues[0].token_turns)
    #exit(0)
    vocab = build_vocab(dialogues, markers, entity_forms)
    bpe_embs = build_bpe(args)
    entity_map, relation_map = build_schema_mappings(schema, num_items)
    return {'vocab': vocab,
            'entity': entity_map,
            'relation': relation_map,
            'bpe_embs': bpe_embs
            }


class TextIntMap(object):
    '''
    Map between text and int for visualizing results.
    '''
    def __init__(self, vocab, entity_map, preprocessor):
        self.vocab = vocab
        self.entity_map = entity_map
        self.entity_forms = preprocessor.entity_forms
        self.preprocessor = preprocessor
        self.setting = {k: self.use_entity_map(v) for k, v in self.entity_forms.items()}

    def pred_to_input(self, preds):
        '''
        Convert decoder outputs to decoder inputs.
        '''
        if self.entity_forms['target'] == self.entity_forms['decoding']:
            return preds
        preds_utterances = [self.int_to_text(pred, 'target') for pred in preds]
        input_utterances = [self.preprocessor.process_utterance(utterance, 'decoding') for utterance in preds_utterances]
        inputs = np.array([self.text_to_int(utterance, 'decoding') for utterance in input_utterances])
        return inputs

    def process_entity(self, token_array, stage):
        '''
        token_array: 2D int array of tokens, assuming entities are mapped by entity_map offset by vocab_size.
        Transform entities to specified form, e.g. type, canonical form.
        Return the new token_array and an entity_array which has the same shape as token_array and use -1
        for non-entity words and map entities by entity_map.
        '''
        offset = self.vocab.size_word
        entity_array = np.full(token_array.shape, -1, dtype=np.int32)
        entity_inds = token_array >= offset
        entity_array[entity_inds] = token_array[entity_inds] - offset

        use_entity_map = self.setting[stage]
        # If use_entity_map, nothing needs to be done as entities are already mapped by the entity_map
        if not use_entity_map:
            # Entities needs to be transformed and mapped by vocab
            for tok_int, is_ent in zip(np.nditer(token_array, op_flags=['readwrite']), np.nditer(entity_inds)):
                if is_ent:
                    entity = self.entity_map.to_word(tok_int - offset)
                    # NOTE: at this point we have lost the surface form of the entity: using an empty string
                    processed_entity = self.preprocessor.get_entity_form(('', entity), self.entity_forms[stage])
                    tok_int[...] = self.vocab.to_ind_word(processed_entity)
        return token_array, entity_array

    def use_entity_map(self, entity_form):
        if entity_form == 'graph':
            return True
        return False

    def get_entities(self, utterance):
        return [-1 if not is_entity(token) else self.entity_map.to_ind_word(token) for token in tokens]

    def text_to_int(self, utterance, stage=None):
        '''
        Process entities in the utterance based on whether it is used for encoding, decoding
        or ground truth.
        '''
        if stage is not None:
            use_entity_map = self.setting[stage]
            tokens = self.preprocessor.process_utterance(utterance, stage)
            if not use_entity_map:
                return [self.vocab.to_ind_word(token) for token in tokens]
            else:
                offset = self.vocab.size_word
                return [self.vocab.to_ind_word(token) if not is_entity(token) else self.entity_map.to_ind_word(token) + offset for token in tokens]
        else:
            tokens = self.preprocessor.process_utterance(utterance)
            offset = self.vocab.size_word
            return [self.vocab.to_ind_word(token) if not is_entity(token) else self.entity_map.to_ind_word(token) + offset for token in tokens]

    def int_to_text(self, inds, stage):
        '''
        Inverse of text_to_int.
        '''
        use_entity_map = self.setting[stage]
        if not use_entity_map:
            return [self.vocab.to_word(ind) for ind in inds]
        else:
            return [self.vocab.to_word(ind) if ind < self.vocab.size_word else self.entity_map.to_word(ind - self.vocab.size_word) for ind in inds]


class Dialogue(object):
    textint_map = None
    ENC = 0
    DEC = 1
    TARGET = 2
    num_stages = 3  # encoding, decoding, target

    def __init__(self, args, kbs, uuid):
        '''
        Dialogue data that is needed by the model.
        '''
        self.uuid = uuid
        self.args = args
        self.kbs = kbs
        self.matched_items = self.get_correct_item(kbs)
        # token_turns: tokens and entitys (output of entitylink)
        self.token_turns = ([], [])

        # char_turns: chars
        self.char_turns = ([], [])
        # bpe_turns: bpe
        self.bpe_turns = ([], [])
        # entities: -1 for non-entity words, entities are mapped based on entity_map
        self.entities = ([], [])
        # turns: input tokens of encoder, decoder input and target, later converted to integers
        self.turns = ([], [], [])
        self.agents = []
        self.is_int = False  # Whether we've converted it to integers
        self.flattened = False
        self.bpe_embs = build_bpe(self.args)

    @classmethod
    def get_correct_item(cls, kbs):
        for id1, item1 in enumerate(kbs[0].items):
            for id2, item2 in enumerate(kbs[1].items):
                if item1 == item2:
                    return (id1, id2)
        raise Exception('No matched item.')

    def create_graph(self):
        assert not hasattr(self, 'graphs')
        self.graphs = [Graph(kb) for kb in self.kbs]

    def get_chars(self, utterances, i):
        char_input_ids = []
        for word in utterances[i]:
            char_arr_id = []
            for char in word:
                char_arr_id.append(char)
            char_input_ids.append(char_arr_id)
        return char_input_ids

    def get_bpe(self, utterances, i):
        bpe_input_ids = []
        for j in range(len(self.bpe_embs)):
            bpe_input_id = []
            for word in utterances[i]:
                subwords = self.bpe_embs[j].encode_ids(word)
                bpe_input_id.append(subwords)
            bpe_input_ids.append(bpe_input_id)
        return bpe_input_ids

    def add_utterance(self, agent, agent_dict, styles, utterances):
        # Same agent talking
        if len(self.agents) > 0 and agent == self.agents[-1]:
            for i in range(2):
                #self.token_turns[i][-1].append(utterances[i]+["[SEP]"])
                if agent_dict[str(agent)] == "rule_bot":
                    self.token_turns[i][-1].append([styles]+utterances[i])
                elif agent_dict["0"] + "_" + agent_dict["1"] == "human_human":
                    self.token_turns[i][-1].append(["en_mono"]+utterances[i])
                else:
                    self.token_turns[i][-1].append(["MISC"]+utterances[i])

                self.char_turns[i][-1].append(self.get_chars(utterances, i)) # chars
                self.bpe_turns[i][-1].append(self.get_bpe(utterances, i)) # bpe ids
        else:
            self.agents.append(agent)
            for i in range(2):
                if agent_dict[str(agent)] == "rule_bot":
                    #print("utterances[i]:", utterances[i])
                    self.token_turns[i].append([[styles]+utterances[i]])
                #self.token_turns[i].append([["[CLS]"]+utterances[i]+["[SEP]"]])
                elif agent_dict["0"] + "_" + agent_dict["1"] == "human_human":
                    self.token_turns[i].append([["en_mono"]+utterances[i]])
                else:
                    self.token_turns[i].append([["MISC"]+utterances[i]])

                self.char_turns[i].append(self.get_chars(utterances, i)) # chars
                self.bpe_turns[i].append(self.get_bpe(utterances, i)) # bpe ids

    def convert_to_int(self):
        if self.is_int:
            return
        for i, turns in enumerate(self.token_turns):
            for turn in turns:
                self.turns[i].append([self.textint_map.text_to_int(utterance)
                                      for utterance in turn])
        # Target tokens
        stage = 'target'
        for turn in self.token_turns[self.DEC]:
            self.turns[self.TARGET].append([self.textint_map.text_to_int(utterance)
                                            for utterance in turn])
        self.is_int = True

    def flatten_turns(self):
        '''
        Flatten turns to a list of tokens with </s> between utterances and </t> at the end.
        '''
        if self.flattened:
            return

        for i, turns in enumerate(self.turns):
            for j, turn in enumerate(turns):
                for utterance in turn:
                    utterance.append(int_markers.EOS)
                self.turns[i][j] = [x for x in chain.from_iterable(turn)]

        if hasattr(self, 'token_turns'):
            for i, turns in enumerate(self.token_turns):
                for j, turn in enumerate(turns):
                    for utterance in turn:
                        utterance.append(markers.EOS)
                    self.token_turns[i][j] = [x for x in chain.from_iterable(turn)]

        self.flattened = True

    def pad_turns(self, num_turns):
        '''
        Pad turns to length num_turns.
        '''
        turns, agents = self.turns, self.agents
        assert len(turns[0]) == len(turns[1]) and len(turns[0]) == len(turns[2])
        for i in range(len(turns[0]), num_turns):
            agents.append(1 - agents[-1])
            for j in range(len(turns)):
                turns[j].append([])


class DialogueBatch(object):
    use_kb = False
    copy = False

    def __init__(self, dialogues):
        self.dialogues = dialogues

    def _normalize_dialogue(self):
        '''
        All dialogues in a batch should have the same number of turns.
        '''
        max_num_turns = max([len(d.turns[0]) for d in self.dialogues])
        for dialogue in self.dialogues:
            dialogue.flatten_turns()
            dialogue.pad_turns(max_num_turns)
        self.num_turns = len(self.dialogues[0].turns[0])

    def _normalize_turn(self, turn_batch):
        '''
        All turns at the same time step should have the same number of tokens.
        '''
        max_num_tokens = max([len(t) for t in turn_batch])
        batch_size = len(turn_batch)
        T = np.full([batch_size, max_num_tokens+1], int_markers.PAD, dtype=np.int32)
        for i, turn in enumerate(turn_batch):
            T[i, 1:len(turn)+1] = turn
            # Insert <go> at the beginning at each turn because for decoding we want to
            # start from <go> to generate, except for padded turns
            if T[i][1] != int_markers.PAD:
                T[i][0] = int_markers.GO
        return T

    def _create_turn_batches(self):
        turn_batches = []
        for i in range(Dialogue.num_stages):
            turn_batches.append([self._normalize_turn(
                [dialogue.turns[i][j] for dialogue in self.dialogues])
                for j in range(self.num_turns)])
        return turn_batches

    def _get_agent_batch(self, i):
        return [dialogue.agents[i] for dialogue in self.dialogues]

    def _get_kb_batch(self, agents):
        return [dialogue.kbs[agent] for dialogue, agent in zip(self.dialogues, agents)]

    def _get_matched_item_batch(self, agents):
        item_entities = [item_to_entity(dialogue.matched_items[agent]) for dialogue, agent in zip(self.dialogues, agents)]
        return Dialogue.textint_map.text_to_int(item_entities, 'target')

    def _get_graph_batch(self, agents):
        return GraphBatch([dialogue.graphs[agent] for dialogue, agent in zip(self.dialogues, agents)])

    def _remove_last(self, array, value):
        '''
        For each row, replace (in place) the last occurence of value to <pad>.
        The last token input to decoder should not be </s> otherwise the model will learn
        </s> <pad> (deterministically).
        '''
        nrows, ncols = array.shape
        for i in range(nrows):
            for j in range(ncols-1, -1, -1):
                if array[i][j] == value:
                    array[i][j] = int_markers.PAD
                    break
        return array

    def _create_one_batch(self,
                          encode_turn,
                          decode_turn,
                          target_turn,
                          encode_chars,
                          decode_chars,
                          encode_bpes,
                          decode_bpes,
                          encode_tokens,
                          decode_tokens):

        if encode_turn is not None:
            # Remove <go> at the beginning of utterance
            encoder_inputs = encode_turn[:, 1:]
        else:
            batch_size = decode_turn.shape[0]
            # If there's no input to encode, use </s> as the encoder input.
            encoder_inputs = np.full((batch_size, 1), int_markers.EOS, dtype=np.int32)
            # encode_tokens are empty lists
            encode_tokens = [[''] for _ in range(batch_size)]

        # Decoder inputs: start from <go> to generate, i.e. <go> <token>
        assert decode_turn.shape == target_turn.shape
        decoder_inputs = np.copy(decode_turn)
        decoder_inputs = self._remove_last(decoder_inputs, int_markers.EOS)[:, :-1]
        decoder_targets = target_turn[:, 1:]

        # Process entities. NOTE: change turns in place!
        encoder_inputs, encoder_entities = Dialogue.textint_map.process_entity(encoder_inputs, 'encoding')
        decoder_inputs, decoder_entities = Dialogue.textint_map.process_entity(decoder_inputs, 'decoding')
        decoder_targets, _ = Dialogue.textint_map.process_entity(decoder_targets, 'target')
        #encoder_char_ids = Dialogue.textint_map.process_char(encode_chars, 'encoding')
        #decoder_char_ids = Dialogue.textint_map.process_char(decode_chars, 'decoding')

        # TODO: use textint_map to process encoder/decoder_inputs here
        print("encoder_inputs:", encoder_inputs, " encoder_tokens:", encode_tokens)
        print("encoder_char_inputs:", encode_chars)
        print("decoder_inputs:", decoder_inputs, " decoder_tokens:", decode_tokens)
        print("decoder_targets:", decoder_targets)
        batch = {
                 'encoder_inputs': encoder_inputs,
                 'encoder_inputs_last_inds': self._get_last_inds(encoder_inputs, int_markers.PAD),
                 'decoder_inputs': decoder_inputs,
                 'decoder_inputs_last_inds': self._get_last_inds(decoder_inputs, int_markers.PAD),
                 'targets': decoder_targets,
                 'encoder_char_inputs': encode_chars,
                 'decoder_char_inputs': decode_chars,
                 'encoder_bpe_inputs': encode_bpes,
                 'decoder_bpe_inputs': decode_bpes,
                 'encoder_tokens': encode_tokens,
                 'decoder_tokens': decode_tokens,
                 'encoder_entities': encoder_entities,
                 'decoder_entities': decoder_entities,
                }
        return batch

    def _get_last_inds(self, inputs, stop_symbol):
        '''
        Return the last index which is not stop_symbol.
        inputs: (batch_size, input_size)
        '''
        assert type(stop_symbol) is int
        inds = np.argmax(inputs == stop_symbol, axis=1)
        inds[inds == 0] = inputs.shape[1]
        inds = inds - 1
        return inds

    def _get_char_turns(self, i, stage):
        if not hasattr(self.dialogues[0], 'char_turns'):
            return None
        # Return None for padded turns
        return [dialogue.char_turns[stage][i] if i < len(dialogue.char_turns[stage]) else ''
                for dialogue in self.dialogues]

    def _get_bpe_turns(self, i, stage):
        if not hasattr(self.dialogues[0], 'bpe_turns'):
            return None
        # Return None for padded turns
        return [dialogue.bpe_turns[stage][i] if i < len(dialogue.bpe_turns[stage]) else ''
                for dialogue in self.dialogues]

    def _get_token_turns(self, i, stage):
        if not hasattr(self.dialogues[0], 'token_turns'):
            return None
        # Return None for padded turns
        return [dialogue.token_turns[stage][i] if i < len(dialogue.token_turns[stage]) else ''
                for dialogue in self.dialogues]

    def create_batches(self):
        self._normalize_dialogue()
        turn_batches = self._create_turn_batches()  # (batch_size, num_turns)
        # A sequence of batches should be processed in turn as the state of each batch is
        # passed on to the next batch
        batches = []
        enc, dec, tgt = Dialogue.ENC, Dialogue.DEC, Dialogue.TARGET
        for start_encode in (0, 1):
            encode_turn_ids = range(start_encode, self.num_turns-1, 2)
            batch_seq = [self._create_one_batch(turn_batches[enc][i], # encoder inputs ids
                                                turn_batches[dec][i+1], # decoder input ids
                                                turn_batches[tgt][i+1], # target input ids (gold labels)
                                                self._get_char_turns(i, enc),  # encoder char ids
                                                self._get_char_turns(i+1, dec), # decoder char ids
                                                self._get_bpe_turns(i, enc), # encoder bpe ids
                                                self._get_bpe_turns(i+1, dec), # decoder bpe ids
                                                self._get_token_turns(i, enc), # encoder token ids
                                                self._get_token_turns(i+1, dec) # decoder token ids
                                                ) for i in encode_turn_ids]
            if start_encode == 1:
                # We still want to generate the first turn
                batch_seq.insert(0, self._create_one_batch(None, #
                                                           turn_batches[dec][0], # decoder input ids
                                                           turn_batches[tgt][0], # ?
                                                           None, #
                                                           self._get_char_turns(0, dec), # decoder char ids
                                                           None, #
                                                           self._get_bpe_turns(0, dec), # decoder bpe ids
                                                           None,  # encoder token ids
                                                           self._get_token_turns(0, dec)))
            # Add agents and kbs
            agents = self._get_agent_batch(1 - start_encode)  # Decoding agent
            kbs = self._get_kb_batch(agents)
            matched_items = self._get_matched_item_batch(agents)
            batch = {
                     'agent': agents,
                     'kb': kbs,
                     'matched_items': matched_items,
                     'batch_seq': batch_seq,
                    }
            if self.use_kb:
                batch['graph'] = self._get_graph_batch(agents)
            batches.append(batch)
        return batches


class Preprocessor(object):
    '''
    Preprocess raw utterances: tokenize, entity linking.
    Convert an Example into a Dialogue data structure used by DataGenerator.
    '''
    def __init__(self, args, schema, lexicon, entity_encoding_form, entity_decoding_form, entity_target_form):
        self.args = args
        self.attributes = schema.attributes
        self.attribute_types = schema.get_attributes()
        self.lexicon = lexicon
        self.entity_forms = {'encoding': entity_encoding_form,
                             'decoding': entity_decoding_form,
                             'target': entity_target_form}

    @classmethod
    def get_entity_form(cls, entity, form):
        '''
        An entity is represented as (surface_form, (canonical_form, type)).
        '''
        assert len(entity) == 2
        if form == 'surface':
            return entity[0]
        elif form == 'type':
            return '<%s>' % entity[1][1]
        elif form == 'canonical':
            return entity[1]
        elif form == 'graph':
            return entity[1]
        else:
            raise ValueError('Unknown entity form %s' % form)

    def process_utterance(self, utterance, stage=None):
        if stage is None:
            return [self.get_entity_form(x, 'canonical') if is_entity(x) else x for x in utterance]
        else:
            return [self.get_entity_form(x, self.entity_forms[stage]) if is_entity(x) else x for x in utterance]

    def _process_example(self, ex, tokenizer):
        '''
        Convert example to turn-based dialogue.
        '''
        kbs = ex.scenario.kbs
        dialogue = Dialogue(self.args, kbs, ex.uuid)

        mentioned_entities = set()
        for e in ex.events:
            utterances = self.process_event(e, kbs[e.agent], mentioned_entities, tokenizer)
            print("utterances:", utterances, " len(utterances):", len(utterances))
            if utterances:
                for token in utterances[0]:
                    if is_entity(token):
                        mentioned_entities.add(token[1][0])

                dialogue.add_utterance(e.agent,
                                       ex.agents,
                                       ex.scenario.styles,
                                       utterances)

        return dialogue

    def item_to_entities(self, item, attrs):
        '''
        Convert an item to a list of entities representing that item.
        '''
        #entities = [(value, (value, type_)) for value, type_ in ((item[attr.name].lower(), self.attribute_types[attr.name]) for attr in attrs)]
        entities = []
        for attr in attrs:
            entities.extend([self.attribute_types[attr.name], item[attr.name].lower()])
        #entities = [[value, type_] for value, type_ in ((item[attr.name].lower(), self.attribute_types[attr.name]) for attr in attrs)]
        return entities

    @classmethod
    def get_item_id(cls, kb, item):
        '''
        Return id of the item in kb.
        '''
        item_id = None
        for i, it in enumerate(kb.items):
            print("it:", it, " item:", item)
            if it == item:
                print("Equal")
                item_id = i
                break
        if item_id is None:
            kb.dump()
            print (item)
        assert item_id is not None
        return item_id

    def process_event(self, e, kb, mentioned_entities=None, tokenizer=tokenize_wp, known_kb=True):
        '''
        Convert event to two lists of tokens and entities for encoding and decoding.
        '''
        if e.action == 'message':
            # Lower, tokenize, link entity
            entity_tokens = self.lexicon.link_entity(tokenizer(e.data), kb=kb,
                                                     mentioned_entities=mentioned_entities,
                                                     known_kb=known_kb)
            #entity_tokens = tokenizer(e.data)
            entity_tokens_before = [normalize_number(x) if not is_entity(x) else x for x in tokenizer(e.data)]
            print("BEFORE: ", tokenizer(e.data))
            print(e.data)
            print(entity_tokens)
            print("AFTER: ", entity_tokens)

            entity_tokens = [normalize_number(x) if not is_entity(x) else x for x in entity_tokens]
            print("AFTER NORMALIZATION: ", entity_tokens)

            if entity_tokens:
                # NOTE: have two copies because we might change it given decoding/encoding
                return (entity_tokens,
                        copy.copy(entity_tokens),
                        entity_tokens_before,
                        copy.copy(entity_tokens_before))
            else:
                return None
        elif e.action == 'select':
            # Convert an item to item-id (wrt to the speaker)
            item_id = self.get_item_id(kb, e.data)
            # We use the entities to represent the item during encoding and item-id during decoding
            #print("e.data:", e.data, "  self.item_to_entities(e.data, kb.attributes):", self.item_to_entities(e.data, kb.attributes))
            #return ([markers.SELECT] + tokenizer(e.data), [markers.SELECT, item_to_entity(item_id)])
            print("SELECT utterance: ", [markers.SELECT, item_to_entity(item_id)])
            return ([markers.SELECT] + self.item_to_entities(e.data, kb.attributes),
                    [markers.SELECT, item_to_entity(item_id)],
                    [markers.SELECT, item_to_entity(item_id)],
                    [markers.SELECT, item_to_entity(item_id)])
        else:
            raise ValueError('Unknown event action.')

    @classmethod
    def count_words(cls, examples, tokenizer):
        counts = defaultdict(int)
        for ex in examples:
            for event in ex.events:
                if event.action == 'message':
                    tokens = tokenizer(event.data)
                    for token in tokens:
                        counts[token] += 1
        return counts

    def preprocess(self, examples, tokenizer):
        dialogues = []
        for ex in examples:
            d = self._process_example(ex, tokenizer)
            # Skip incomplete chats
            if len(d.agents) < 2 or ex.outcome['reward'] == 0:
                continue
                print ('Removing dialogue %s' % d.uuid)
                for event in ex.events:
                    print (event.to_dict())
            else:
                dialogues.append(d)
        return dialogues


class DataGenerator(object):
    def __init__(self, args, train_examples, dev_examples, test_examples, preprocessor, schema, num_items, mappings=None, use_kb=False, copy=False, tokenizer=tokenize_wp):
        examples = {'train': train_examples or [], 'dev': dev_examples or [], 'test': test_examples or []}
        self.num_examples = {k: len(v) if v else 0 for k, v in examples.items()}
        self.use_kb = use_kb  # Whether to generate graph
        self.copy = copy

        DialogueBatch.use_kb = use_kb
        DialogueBatch.copy = copy

        self.dialogues = {k: preprocessor.preprocess(v, tokenizer) for k, v in examples.items()}

        for fold, dialogues in self.dialogues.items():
            print('%s: %d dialogues out of %d examples' % (fold, len(dialogues), self.num_examples[fold]))

        if not mappings:
            mappings = create_mappings(args, self.dialogues['train'], schema, num_items, preprocessor.entity_forms.values())
        self.mappings = mappings

        self.textint_map = TextIntMap(mappings['vocab'], mappings['entity'], preprocessor)
        Dialogue.textint_map = self.textint_map
        Dialogue.preprocessor = preprocessor

        global int_markers
        int_markers = SpecialSymbols(*[mappings['vocab'].to_ind_word(m) for m in markers])

    def convert_to_int(self):
        '''
        Convert tokens to integers.
        '''
        for fold, dialogues in self.dialogues.items():
            for dialogue in dialogues:
                dialogue.convert_to_int()

    def create_dialogue_batches(self, dialogues, batch_size):
        dialogues.sort(key=lambda d: len(d.turns[0]))
        N = len(dialogues)
        dialogue_batches = []
        start = 0
        while start < N:
            # NOTE: last batch may have a smaller size if we don't have enough examples
            end = min(start + batch_size, N)
            dialogue_batch = dialogues[start:end]
            dialogue_batches.extend(DialogueBatch(dialogue_batch).create_batches())
            start = end
        return dialogue_batches

    def create_graph(self, dialogues):
        if not self.use_kb:
            return
        for dialogue in dialogues:
            dialogue.create_graph()

    def reset_graph(self, dialogue_batches):
        if not self.use_kb:
            return
        for dialogue_batch in dialogue_batches:
            for graph in dialogue_batch['graph'].graphs:
                graph.reset()

    def generator(self, name, batch_size, shuffle=True):
        dialogues = self.dialogues[name]
        for dialogue in dialogues:
            dialogue.convert_to_int()
        # NOTE: we assume that GraphMetadata has been constructed before DataGenerator is called
        self.create_graph(dialogues)
        dialogue_batches = self.create_dialogue_batches(dialogues, batch_size)
        yield len(dialogue_batches)
        inds = list(range(len(dialogue_batches)))
        while True:
            if shuffle:
                random.shuffle(inds)
            for ind in inds:
                yield dialogue_batches[ind]
            # We want graphs clean of dialgue history for the new epoch
            self.reset_graph(dialogue_batches)
