import tensorflow as tf
from transformer.transformer import transformer
from transformer.tf_utils import gelu
from transformer.relative_transformer import RelTranformer
from .rnn import rnn_layer
from .crf import crf_log_likelihood, crf_decode
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dropout, LayerNormalization

class Forward(tf.keras.layers.Layer):
    def __init__(self, config, training):
        super(Forward, self).__init__()
        self.config = config
        tgt_size = config.tgt_size
        self.training = training
        if config.decode_type == "crf": #build crf var
            init = tf.keras.initializers.GlorotUniform()
            self.transition_params = tf.Variable(lambda : init([tgt_size,tgt_size]),trainable=True)
            self.dense_layer = tf.keras.layers.Dense(config.tgt_size)
        elif config.decode_type == "span": #build span network
            self.span_ln = LayerNormalization(epsilon=config.layer_norm_epsilon)
            self.head_network = tf.keras.layers.Dense(config.span_tgt, activation='sigmoid')
            self.tail_network = tf.keras.layers.Dense(config.span_tgt, activation='sigmoid')
        else:
            raise ValueError

    def crf(self, output, label_input=None):
        output = self.dense_layer(output)
        if self.training:
            ins = [output,label_input,self.transition_params]
            loss, self.transition_params = crf_log_likelihood(ins)
            return loss
        else:
            decode_tags, _ = crf_decode([output,self.transition_params])
            return decode_tags

    def span(self,
             output,
             label_input=None,
             label_input0=None,
             mask=None):
        head_logits = self.head_network(output)
        tail_input = self.span_ln(tf.concat([output, head_logits], -1))
        tail_logits = self.tail_network(tail_input)
        if self.training:
            def loss_compute(ground_trueth, pred, mask):
                ground_trueth = tf.cast(ground_trueth,tf.float32)
                bce = K.binary_crossentropy(ground_trueth,pred)
                #消除最后一维度
                bce = K.sum(bce,axis=-1,keepdims=True)
                #去除pad部分
                bce = bce * mask
                loss = K.sum(bce)/K.sum(mask)
                return loss
            head_loss = loss_compute(label_input, head_logits, mask)
            tail_loss = loss_compute(label_input0, tail_logits, mask)
            return head_loss + tail_loss
        else:
            return (head_logits, tail_logits)
    
    def call(self, output, label_input=None, label_input0=None, mask=None):
        if self.config.decode_type == 'crf':
            return self.crf(output, label_input)
        elif self.config.decode_type == 'span':
            return self.span(output, label_input, label_input0, mask)
        else:
            raise ValueError
 
class ner_model(tf.keras.Model):
    """docstring for ner_model"""
    def __init__(self, config,training):
        super(ner_model, self).__init__()
        self.config = config
        self.training = training
        if config.enable_reltf:
            self.tf_layer = RelTranformer(config,training)
        else:
            self.tf_layer = transformer(config,training)
        if config.rnn:
            self.rnn_layer = rnn_layer(config,training)
        self.ffw = Forward(config,training)

    def mask(self, input_tensor):
        mask = tf.expand_dims(input_tensor,2)
        mask = tf.greater(mask,0)
        mask = tf.cast(mask,dtype=tf.float32)
        return mask

    def call(self,
            seq_input,
            label_input=None,
            label_input0=None):
        mask = self.mask(seq_input)
        output = self.tf_layer(seq_input)
        if self.config.rnn:
            output = self.rnn_layer.bi_rnn(output)
        if self.training:
            return self.ffw(output, label_input, label_input0, mask)
        else:
            return self.ffw(output)