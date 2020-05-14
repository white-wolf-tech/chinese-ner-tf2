from tensorflow.keras.layers import LSTM, Dropout, Bidirectional
import tensorflow as tf

class rnn_layer(tf.keras.layers.Layer):
    """docstring for rnn_layer"""
    def __init__(self,config, training):
        super(rnn_layer, self).__init__()
        self.bi_lstm = Bidirectional(LSTM(config.rnn_unit, return_sequences=True,return_state=False))
        self.lstm = LSTM(config.rnn_unit,return_sequences=True,return_state=False)
        self.dropout = Dropout(config.rnn_dropout)
        self.training = training
    def bi_rnn(self,feature):
        rnn_layer = self.bi_lstm(feature)
        return self.dropout(rnn_layer, training =self.training)
    def rnn(self,feature,label_input):
        rnn_layer = self.lstm(feature)
        return self.dropout(rnn_layer, training=self.training)