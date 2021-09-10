import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization,Conv1D,Dropout,Embedding, Lambda
from tf_utils import shape_list, merge_heads, split_heads, gelu

def get_embedding(num_embeddings,
                  embedding_dim):

    half_dim = embedding_dim // 2
    emb = tf.math.log(10000.0) / (half_dim - 1)
    emb = tf.exp(tf.range(0, half_dim, dtype=tf.float32) * -emb)
    emb_tmp0 = tf.expand_dims(emb,0)
    emb_tmp1 = tf.range(-num_embeddings//2, num_embeddings//2, dtype=tf.float32)
    emb_tmp1 = tf.expand_dims(emb_tmp1,1)
    emb = emb_tmp1 * emb_tmp0
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)
    emb = tf.reshape(emb,[num_embeddings,-1])
    return emb

def RelativeEmbedding(input_tensor, weights, max_len):
    bsz, seq_len = shape_list(input_tensor)
    positions = tf.range(-seq_len , seq_len) + max_len // 2 + 1
    embed = tf.nn.embedding_lookup(weights, positions)
    return embed

class RelativeMultiHeadAttn(tf.keras.layers.Layer):
    def __init__(self, d_model, n_head, dropout,max_len,
                trainning=True, scale=False, layer_name=""):

        super().__init__()
        self.qkv_linear = Conv1D(filters = 3 * d_model, kernel_size=1)
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.trainning = trainning
        if trainning:
            self.dropout_layer = Dropout(dropout)
        if scale:
            self.scale = tf.math.sqrt(float(self.head_dim))
        else:
            self.scale = 1.0
        #变量u和v
        w_init = tf.initializers.GlorotUniform()
        self.r_r_bias = tf.Variable(
            initial_value = w_init(shape=[n_head, self.head_dim],dtype=tf.float32),
            trainable=True, name=layer_name + "rr_bias")
        self.r_w_bias = tf.Variable(
            initial_value=w_init(shape=[n_head, self.head_dim],dtype=tf.float32),
            trainable=True, name=layer_name + "rw_bias")

    def call(self, x, pos_embed):
        batch_size, max_len, d_model = shape_list(x)
        qkv = self.qkv_linear(x)  # batch_size x max_len x d_model3
        q, k, v = tf.split(qkv, 3, axis=2)
        q = split_heads(q, self.n_head)
        k = split_heads(k, self.n_head)
        v = split_heads(v, self.n_head) # b x n x l x d
        rw_head_q = q + self.r_r_bias[:, None]
        AC = tf.einsum('bnqd,bnkd->bnqk', rw_head_q, k)  # b x n x l x d, n是head
        D_ = tf.einsum('nd,ld->nl', self.r_w_bias, pos_embed)[None, :, None]  # head x 2max_len, 每个head对位置的bias
        B_ = tf.einsum('bnqd,ld->bnql', q, pos_embed)  # bsz x head  x max_len x 2max_len，每个query对每个shift的偏移
        E_ = tf.einsum('bnqd,ld->bnql', k, pos_embed)  # bsz x head x max_len x 2max_len, key对relative的bias
        BD = B_ + D_  # bsz x head x max_len x 2max_len, 要转换为bsz x head x max_len x max_len
        BDE = self._shift(BD) + self._transpose_shift(E_)
        attn = AC + BDE
        attn = attn / self.scale

        attn = tf.nn.softmax(attn, axis=-1)
        if self.trainning:
            attn = self.dropout_layer(attn)
        v = tf.matmul(attn, v)
        v = merge_heads(v)
        return v

    def _shift(self, BD):
        """
        类似
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        -3 -2 -1 0 1 2
        转换为
        0   1  2
        -1  0  1
        -2 -1  0
        :param BD: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = shape_list(BD)
        zero_pad = tf.zeros([bsz, n_head, max_len, 1])
        BD = tf.concat([BD, zero_pad], axis=-1)
        BD = tf.reshape(BD,[bsz, n_head, -1, max_len])# bsz x n_head x (2max_len+1) x max_len
        BD = tf.reshape(BD[:, :, :-1],[bsz, n_head, max_len, -1]) # bsz x n_head x 2max_len x max_len
        BD = BD[:, :, :, max_len:]
        return BD
    
    def _transpose_shift(self, E):
        """
        类似
          -3   -2   -1   0   1   2
         -30  -20  -10  00  10  20
        -300 -200 -100 000 100 200
        转换为
          0  -10   -200
          1   00   -100
          2   10    000
        :param E: batch_size x n_head x max_len x 2max_len
        :return: batch_size x n_head x max_len x max_len
        """
        bsz, n_head, max_len, _ = shape_list(E)
        zero_pad = tf.zeros([bsz, n_head, max_len, 1])
        # bsz x n_head x -1 x (max_len+1)
        E = tf.concat([E, zero_pad], axis=-1)
        E = tf.reshape(E,[bsz, n_head, -1, max_len])
        indice = tf.range(0,max_len)*2+1
        E = tf.gather(E, indice, axis=-2)
        return E


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
    def __init__(self, config,training,scale,layer_name):
        super(attention_block, self).__init__()
        self.attention = RelativeMultiHeadAttn(
                                config.embed,
                                config.head,
                                config.a_dropout,
                                training,
                                scale=scale,
                                layer_name=layer_name)
        self.ln0 = LayerNormalization(epsilon=config.layer_norm_epsilon)
        self.ln1 = LayerNormalization(epsilon=config.layer_norm_epsilon)
        self.ffn = FFN(config, training)
    def call(self,x, pos_embed):
        a = self.attention(x, pos_embed)
        x = x + a
        x = self.ln0(x)

        m = self.ffn(x)
        x = x + m
        x = self.ln1(x)

        return x


class RelTranformer(tf.keras.layers.Layer):
    """docstring for RelTranformer"""
    def __init__(self, config, trainning):
        super(RelTranformer, self).__init__()
        self.scale = config.scale
        self.config = config
        self.tf_layers = [attention_block(config, trainning, self.scale, layer_name="L"+str(i)) for i in range(config.n_layer)]
        self.wde = Embedding(config.vocab_size, config.embed)
        #位置编码
        self.weights_embeding = get_embedding(config.max_length, config.embed // config.head)
    def call(self,input_ids):
        mask = tf.cast(tf.not_equal(input_ids,0),dtype=tf.int32)
        hidden_state = self.wde(input_ids)
        position_enc = RelativeEmbedding(mask,
                                        self.weights_embeding,
                                        self.config.max_length)
        for tf_layer in self.tf_layers:
            hidden_state = tf_layer(hidden_state, position_enc)
        return hidden_state