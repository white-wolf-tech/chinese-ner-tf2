import tensorflow as tf
import numpy as np

def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) *
        (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def shape_list(x):
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def merge_heads(x):
    x = tf.transpose(x, [0, 2, 1, 3])
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2] + [x_shape[-2] * x_shape[-1]]
    return tf.reshape(x, new_x_shape)

def split_heads(x,n_head):
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-1] + [n_head, x_shape[-1] // n_head]
    x = tf.reshape(x, new_x_shape)
    return tf.transpose(x, (0, 2, 1, 3))#(batch, head, seq_length, head_features)