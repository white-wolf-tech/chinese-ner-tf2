import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import LayerNormalization,Conv1D,Dropout,Embedding
from tf_utils import gelu, shape_list, merge_heads, split_heads

class attention(tf.keras.layers.Layer):
    def __init__(self, config,training):
        super(attention, self).__init__()
        self.conv1 = Conv1D(filters = 3 * config.embed,kernel_size=1)
        self.conv2 = Conv1D(filters = config.embed,kernel_size=1)
        self.dropout = Dropout(config.a_dropout)
        self.head = config.head
        self.n_state = config.embed
        self.training = training
    def call(self,x, scale_att=False):
        x = self.conv1(x)
        q, k, v = tf.split(x, 3, axis=2)

        assert self.n_state % self.head == 0
        q = split_heads(q, self.head)
        k = split_heads(k, self.head)
        v = split_heads(v, self.head)

        w = tf.matmul(q, k, transpose_b=True)
        if scale_att:
            dk = tf.cast(shape_list(k)[-1], tf.float32)
            w = w / tf.math.sqrt(dk)
        w = tf.nn.softmax(w, axis=-1)
        w = self.dropout(w, training=self.training)
        a = tf.matmul(w, v)

        a = merge_heads(a)
        a = self.conv2(a)
        a = self.dropout(a, training=self.training)

        return a

class FFN(tf.keras.layers.Layer):
    """docstring for FFN"""
    def __init__(self, config,training):
        super(FFN, self).__init__()
        self.training = training
        self.conv1 = Conv1D(filters=config.ffw_rate*config.embed,kernel_size=1)
        self.conv2 = Conv1D(filters=config.embed,kernel_size=1)
        self.dropout = Dropout(config.a_dropout)
    def call(self,x):
        ffn0 = self.conv1(x)
        act_ffn0 = gelu(ffn0)
        ffn1 = self.conv2(act_ffn0)
        ffn1 = self.dropout(ffn1, training=self.training)
        return ffn1

class attention_block(tf.keras.layers.Layer):
    def __init__(self, config,training):
        super(attention_block, self).__init__()
        self.attention = attention(config,training)
        self.ln = LayerNormalization(epsilon=config.layer_norm_epsilon)
        self.ffn = FFN(config,training)
    def call(self,x, scale_att=False):
        a = self.attention(x,scale_att=scale_att)
        x = x + a
        x = self.ln(x)

        m = self.ffn(x)
        x = x + m
        x = self.ln(x)

        return x

class transformer(tf.keras.layers.Layer):
    """docstring for transformer"""
    def __init__(self, config,training):
        super(transformer, self).__init__()
        self.config = config
        self.wde = Embedding(config.vocab_size, config.embed)
        self.pte = Embedding(config.max_length, config.embed)
        self.attention_blocks = [attention_block(config,training) for i in range(self.config.n_layer)]
        self.ln = LayerNormalization(epsilon=self.config.layer_norm_epsilon)
    def call(self,input_ids):
        seq_embedding = self.wde(input_ids)
        batch,seq_len,emb = shape_list(seq_embedding)
        position_ids = tf.range(0, seq_len, dtype=tf.int32)[tf.newaxis, :]
        position_embedding = self.pte(position_ids)
        hidden_state = self.ln(seq_embedding + position_embedding)
        for i in range(self.config.n_layer):
            hidden_state = self.attention_blocks[i](hidden_state,scale_att=True)
        return hidden_state
