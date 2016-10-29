# -*- coding: utf-8 -*-

"""
Straightforward adaptation of the TensorFlow-based autoencoder available at
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py

The only changes are the shape of the input and output layers as well as the data parsing.
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf


def autoencoder_model():
    # Network Parameters
    n_hidden_1 = 400  # 1st layer num features
    n_input = 520  # MNIST data input (img shape: 28*28)
    n_output = 494  # MNIST data input (img shape: 28*28)

    misspelling = tf.placeholder("float", [None, n_input], name="mistake")
    correct = tf.placeholder("float", [None, n_output], name="correct")

    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_output])),
    }
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b1': tf.Variable(tf.random_normal([n_output])),
    }

    # Building the encoder
    def encoder(x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                       biases['encoder_b1']))
        return layer_1

    # Building the decoder
    def decoder(x):
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                       biases['decoder_b1']))
        return layer_1

    # Construct model
    encoder_op = encoder(misspelling)
    decoder_op = decoder(encoder_op)

    # Input
    input = misspelling
    # Prediction
    y_pred = decoder_op
    # Targets (Labels) are the input data.
    y_true = correct

    return input, y_pred, y_true
