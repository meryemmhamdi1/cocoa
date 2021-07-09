import tensorflow as tf
import numpy as np
import io
from src.model import transformer
from src.model.bpemetaembedding import BPEMetaEmbedding


class MUSEWordEmbedder(object):
    def __init__(self, num_symbols, embed_size, W, pad=None, scope=None):
        self.num_symbols = num_symbols
        self.embed_size = embed_size
        self.pad = pad

        self.embedding_np = W

        self.build_model(scope)

    def build_model(self, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            #W = tf.Variable(tf.constant(0.0, shape=[self.num_symbols, self.embed_size]), trainable=True, name="W")
            #embedding_placeholder = tf.placeholder(tf.float32, [self.num_symbols, self.embed_size])
            #embedding_init = W.assign(embedding_placeholder)
            print("self.num_symbols,:", self.num_symbols)
            self.embedding = tf.compat.v1.get_variable(name='embedding', shape=[self.num_symbols, self.embed_size],
                                             initializer=tf.constant_initializer(self.embedding_np), trainable=True)

    def embed(self, inputs, zero_pad=False):
        embeddings = tf.nn.embedding_lookup(self.embedding, inputs)
        if self.pad is not None and zero_pad:
            embeddings = tf.where(inputs == self.pad, tf.zeros_like(embeddings), embeddings)
        return embeddings


class WordEmbedder(object):
    def __init__(self, num_symbols, embed_size, pad=None, scope=None):
        self.num_symbols = num_symbols
        self.embed_size = embed_size
        self.pad = pad

        self.build_model(scope)

    def build_model(self, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            #W = tf.Variable(tf.constant(0.0, shape=[self.num_symbols, self.embed_size]), trainable=True, name="W")
            #embedding_placeholder = tf.placeholder(tf.float32, [self.num_symbols, self.embed_size])
            #embedding_init = W.assign(embedding_placeholder)
            print("self.num_symbols,:", self.num_symbols)
            self.embedding = tf.compat.v1.get_variable(name='embedding', shape=[self.num_symbols, self.embed_size])

    def embed(self, inputs, zero_pad=False):
        embeddings = tf.nn.embedding_lookup(self.embedding, inputs)
        if self.pad is not None and zero_pad:
            embeddings = tf.where(inputs == self.pad, tf.zeros_like(embeddings), embeddings)
        return embeddings


class BERTEmbedder(object):
    def __init__(self, num_symbols, embed_size, pad=None, scope=None):
        self.num_symbols = num_symbols
        self.embed_size = embed_size
        self.pad = pad
        self.build_model(scope)

    def build_model(self, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            self.embedding = tf.compat.v1.get_variable('embedding', [self.num_symbols, self.embed_size])

    def embed(self, inputs, zero_pad=False):
        embeddings = tf.nn.embedding_lookup(self.embedding, inputs)
        if self.pad is not None and zero_pad:
            embeddings = tf.where(inputs == self.pad, tf.zeros_like(embeddings), embeddings)
        return embeddings


class HMEEmbedder(object):
    def __init__(self,
                 word2id, id2word,
                 char2id, id2char,
                 embed_size,
                 embedding_size_char,
                 char_hidden_size,
                 dim_key,
                 dim_value,
                 filter_size,
                 max_length,
                 input_dropout,
                 dropout,
                 attn_dropout,
                 relu_dropout,
                 bpe_lang_list,
                 bpe_dim,
                 bpe_vocab,
                 bpe_hidden_size,
                 bpe_emb_size,
                 bpe_embs,
                 no_word_emb,
                 mode,
                 add_char_emb,
                 W,
                 no_projection,
                 pad=None,
                 scope=None):

        # WORDS
        self.word2id = word2id
        self.id2word = id2word
        self.embed_size = embed_size
        self.uni_emb_size = 300
        self.W = W

        # CHARS
        self.char2id = char2id
        self.id2char = id2char
        self.char_emb_size = embedding_size_char
        self.char_hidden_size = char_hidden_size
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.filter_size = filter_size

        # Dropout
        self.max_length = max_length
        self.input_dropout = input_dropout
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.relu_dropout = relu_dropout

        # BPE
        self.bpe_lang_list = bpe_lang_list
        self.bpe_dim = bpe_dim
        self.bpe_vocab = bpe_vocab
        self.bpe_hidden_size = bpe_hidden_size
        self.bpe_emb_size = bpe_emb_size
        self.bpe_embs = bpe_embs

        self.no_word_emb = no_word_emb
        self.add_bpe_emb = len(self.bpe_lang_list) > 0
        self.add_char_emb = add_char_emb
        self.mode = mode
        self.no_projection = no_projection

        self.pad = pad

        self.pretrained_emb = []
        with tf.variable_scope(scope or type(self).__name__):
            for j, W_j in enumerate(W): # encompassing different monolingual embeddings
                self.pretrained_emb.append(tf.compat.v1.get_variable(name='embedding_'+str(j),
                                                                     shape=[len(self.word2id), self.embed_size],
                                                                     initializer=tf.constant_initializer(W_j),
                                                                     trainable=False))


        self.build_model(scope)

    def build_model(self, scope=None):
        if self.mode == "attn_sum":

            if self.add_char_emb:
                with tf.variable_scope(scope or type(self).__name__):
                    self.char_embedding = tf.compat.v1.get_variable('char_embedding',
                                                                    [len(self.char2id), self.char_emb_size]) #char lookup table

                    self.char_encoder = transformer.Encoder(input_vocab_size=self.char_emb_size,
                                                            d_model=self.char_hidden_size,
                                                            num_layers=1,
                                                            num_heads=4,
                                                            dff=self.filter_size,
                                                            maximum_position_encoding=self.max_length)

            if self.add_bpe_emb:
                with tf.variable_scope(scope or type(self).__name__):
                    self.bpe_embedding = BPEMetaEmbedding(self.bpe_embs,
                                                          self.bpe_hidden_size,
                                                          num_layers=1,
                                                          num_heads=4,
                                                          dim_key=self.dim_key,
                                                          dim_value=self.dim_value,
                                                          filter_size=self.filter_size,
                                                          max_length=self.max_length,
                                                          input_dropout=self.input_dropout,
                                                          layer_dropout=self.dropout,
                                                          attn_dropout=self.attn_dropout,
                                                          relu_dropout=self.relu_dropout,
                                                          mode="attn_sum",
                                                          no_projection=False,
                                                          cuda=True)

    def concat_embeddings(self, word_emb_vec):
        # word_emb_vec => num_emb x batch_size x seq_len x uni_emb_size
        if len(self.pretrained_emb) > 1:
            for i in range(len(word_emb_vec)):
                word_emb_vec[i] = tf.squeeze(word_emb_vec[i], axis=-1)

            if len(word_emb_vec[0]) == 1:
                embeddings = tf.expand_dims(tf.squeeze(tf.concat(word_emb_vec, axis=-1)), axis=0) # 1 x seq_len x (uni_emb_size x num_emb)
            else:
                embeddings = tf.squeeze(tf.concat(word_emb_vec, axis=-1), axis=0) # batch_size x seq_len x (uni_emb_size x num_emb)

        else:
            embeddings = tf.squeeze(word_emb_vec[0], axis=-1)

        return embeddings

    def sum_embeddings(self, word_emb_vec):
        embeddings = None
        if len(self.pretrained_emb) > 1:
            for i in range(len(self.pretrained_emb)):
                if embeddings is None:
                    embeddings = word_emb_vec[i]
                else:
                    embeddings = tf.add(embeddings, word_emb_vec[i])
            embeddings = tf.squeeze(embeddings, axis=-1)
        else:
            embeddings = tf.squeeze(word_emb_vec[0], axis=-1)
        return embeddings

    def word_mme(self, word_emb_vec):
        if len(self.pretrained_emb) > 1:
            print("tf.shape(word_emb_vec)[0]:", tf.shape(word_emb_vec)[0])
            if tf.shape(word_emb_vec)[0] == 1:
                embeddings = tf.expand_dims(tf.squeeze(tf.stack(word_emb_vec, axis=-1)), axis=0) # 1 x seq_len x uni_emb_size x num_emb
            else:
                embeddings = tf.squeeze(tf.stack(word_emb_vec, axis=-1)) # batch_size x seq_len x uni_emb_size x num_emb
            attn = tf.tanh(embeddings)
            attn_scores = tf.nn.softmax(attn, axis=-1)

            embeddings = tf.reduce_sum(tf.multiply(attn_scores, embeddings), -1) # embeddings[:,:,:,i] * attn_scores[:,:,:,i]
        else:
            embeddings = tf.squeeze(word_emb_vec[0], axis=-1)

        return embeddings

    def char_mme(self, char_inputs):
        print("VAR char_inputs:", char_inputs)
        char_outputs = tf.nn.embedding_lookup(self.char_embedding, char_inputs)
        batch_size, max_seq_len, max_word_len, uni_emb = tf.shape(char_outputs)
        char_outputs = tf.reshape(char_outputs, [batch_size * max_seq_len, max_word_len, uni_emb])
        char_outputs = self.char_transformer_enc(char_outputs) # max_seq_len, max_word_len, uni_emb
        char_outputs = tf.reshape(char_outputs, [batch_size, max_seq_len, max_word_len, self.char_hidden_size])
        return tf.reduce_sum(char_outputs, 2)

    def embed(self, word_inputs, char_inputs, bpe_inputs, zero_pad=False):
        print("VAR word_inputs: ", word_inputs)
        word_emb_vec = []
        for i in range(len(self.pretrained_emb)):
            looked_up_emb = tf.nn.embedding_lookup(self.pretrained_emb[i], word_inputs)
            if self.no_projection:
                new_emb = looked_up_emb
            else:
                with tf.variable_scope('projection_{}'.format(i)):
                    new_emb = tf.contrib.layers.fully_connected(looked_up_emb, self.uni_emb_size)
            new_emb = tf.expand_dims(new_emb, axis=-1) # batch_size x seq_len x uni_emb_size x 1
            word_emb_vec.append(new_emb)

        if self.mode == "concat":
            embeddings = self.concat_embeddings(word_emb_vec)
            if self.pad is not None and zero_pad:
                embeddings = tf.where(word_inputs == self.pad, tf.zeros_like(embeddings), embeddings)

        elif self.mode == "linear":
            embeddings = self.sum_embeddings(word_emb_vec)
            if self.pad is not None and zero_pad:
                embeddings = tf.where(word_inputs == self.pad, tf.zeros_like(embeddings), embeddings)

        else: ## HME or some version of MME
            all_embs = []
            word_mme = self.word_mme(word_emb_vec)
            if self.pad is not None and zero_pad:
                word_mme = tf.where(word_inputs == self.pad, tf.zeros_like(word_mme), word_mme)
            all_embs.append(word_mme)
            if self.add_char_emb:
                char_mme = self.char_mme(char_inputs)
                all_embs.append(char_mme)
            if self.add_bpe_emb:
                bpe_mme = self.bpe_embedding.encoder(bpe_inputs)
                all_embs.append(bpe_mme)

            if len(all_embs) == 1 and len(self.pretrained_emb) == 1:
                embeddings = word_mme
            else:
                embeddings = tf.concat(all_embs, axis=-1)

        return embeddings


