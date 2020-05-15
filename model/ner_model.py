import tensorflow as tf
from .transformer import transformer
from .rnn import rnn_layer
from .crf import crf_log_likelihood,crf_decode

class Forward(tf.keras.layers.Layer):
    def __init__(self, config,training):
        super(Forward, self).__init__()
        init = tf.keras.initializers.GlorotUniform()
        tgt_size = config.tgt_size
        self.training = training
        self.transition_params = tf.Variable(lambda : init([tgt_size,tgt_size]))
    def call(self,output,label_input=None):
        if self.training:
            ins = [output,label_input,self.transition_params]
            loss, self.transition_params = crf_log_likelihood(ins)
            return loss
        else:
            decode_tags, _ = crf_decode([output,self.transition_params])
            return decode_tags
        
class ner_model(tf.keras.Model):
    """docstring for ner_model"""
    def __init__(self, config,training):
        super(ner_model, self).__init__()
        self.config = config
        self.training = training
        self.tf_layer = transformer(config,training)
        self.rnn_layer = rnn_layer(config,training)
        self.dense_layer = tf.keras.layers.Dense(config.tgt_size)
        self.ffw = Forward(config,training)
    def call(self,seq_input,label_input=None):
        output = self.tf_layer(seq_input)
        if self.config.rnn:
            output = self.rnn_layer.bi_rnn(output)
        output = self.dense_layer(output)
        if self.training:
            return self.ffw(output, label_input)
        else:
            return self.ffw(output)