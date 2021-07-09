from src.model import transformer
import tensorflow as tf


def gen_new_bpe_embedding(emb_vectors, num_vocab, emb_size, j):
    """
        Generate bpe embeddings
    """
    with tf.variable_scope("bpe_embeddings_"+str(j)):
        emb = tf.compat.v1.get_variable(name='bpe_embedding_'+str(j),
                                        shape=[num_vocab, emb_size],
                                        initializer=tf.constant_initializer(emb_vectors),
                                        trainable=False)
    return emb


class BPEMetaEmbedding(object):
    def __init__(self,
                 embs,
                 bpe_hidden_size,
                 num_layers=1,
                 num_heads=4,
                 dim_key=32,
                 dim_value=32,
                 filter_size=32,
                 max_length=100,
                 input_dropout=0.1,
                 layer_dropout=0.1,
                 attn_dropout=0.1,
                 relu_dropout=0.1,
                 mode="attn_sum",
                 no_projection=False,
                 cuda=False):

        super(BPEMetaEmbedding, self).__init__()

        self.embs = embs
        self.bpe_embs = [gen_new_bpe_embedding(bpe_emb.vectors, bpe_emb.vectors.shape[0], bpe_emb.vectors.shape[1], j)
                         for j, bpe_emb in enumerate(self.embs)]

        self.bpe_emb_sizes = [self.embs[i].vectors.shape[1] for i in range(len(self.embs))]
        self.mode = mode
        self.no_projection = no_projection
        self.bpe_hidden_size = bpe_hidden_size

        self.bpe_encoders = [
            transformer.Encoder(
                input_vocab_size=self.bpe_emb_sizes[i],
                d_model=bpe_hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                dff=filter_size,
                maximum_position_encoding=max_length) for i in range(len(self.bpe_embs))]

        ### num_layers -> num_layers
        ### d_model -> (bpe_)hidden_size
        ### num_heads -> num_heads
        ### dff -> filter_size
        ### input_vocab_size -> self.bpe_emb_sizes[i]
        ### maximum_position_encoding -> max_length

    def encoder(self, bpe_inputs):
        """
            input_bpes: batch_size, num_of_emb, max_seq_len, uni_emb
        """
        attn_scores = None
        bpe_emb_vec = []

        input_bpes = bpe_inputs.transpose(0, 1) # num_of_emb, batch_size, max_word_len, max_bpe_len
        for i in range(len(input_bpes)):
            emb = tf.nn.embedding_lookup(self.bpe_embs[i], input_bpes[i])
            batch_size, max_seq_len, max_bpe_len, emb_size = tf.shape(emb)

            emb = tf.reshape(emb, [batch_size * max_seq_len, max_bpe_len, emb_size])
            bpe_emb = self.bpe_encoders[i](emb) # batch_size * max_word_len, max_bpe_len, uni_emb
            bpe_emb = tf.reshape(bpe_emb, [batch_size, max_seq_len, max_bpe_len, self.bpe_hidden_size]) # batch_size, max_word_len, max_bpe_len, uni_emb
            trained_bpe_emb = tf.expand_dims(tf.reduce_sum(bpe_emb, axis=2), axis=-1) # batch_size, max_word_len, uni_emb, 1
            bpe_emb_vec.append(trained_bpe_emb)

        if len(self.bpe_embs) > 1:
            if len(bpe_emb_vec[0]) == 1:
                embedding = tf.expand_dims(tf.squeeze(tf.stack(bpe_emb_vec, axis=-1)), axis=0) # 1 x word_len x bpe_seq_len x uni_emb_size x num_emb
            else:
                embedding = tf.squeeze(tf.stack(bpe_emb_vec, axis=-1)) # batch_size x word_len x bpe_seq_len x uni_emb_size x num_emb

            if self.mode == "concat":
                sum_embedding = tf.squeeze(tf.concat(bpe_emb_vec, axis=-1)) # batch_size x seq_len x (uni_emb_size x num_emb)
            elif self.mode == "linear":
                sum_embedding = tf.reduce_sum(tf.multiply(attn_scores, embedding), -1)
            else: #"attn_sum"
                attn = tf.tanh(embedding)
                attn_scores = tf.nn.softmax(attn, axis=-1)
                sum_embedding = tf.reduce_sum(tf.multiply(attn_scores, embedding), -1)

            embedding = sum_embedding
        else:
            embedding = tf.squeeze(bpe_emb_vec[0], axis=-1)

        return embedding
