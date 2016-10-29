# -*- coding: utf-8 -*-

"""
Straightforward adaptation of the TensorFlow-based autoencoder available at
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py

The only changes are the shape of the input and output layers as well as the data parsing.
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf


def transpose_autoencoder():
    # Network Parameters
    n_hidden_1 = 260  # 1st layer num features
    n_hidden_2 = 130  # 2nd layer num features
    n_input = 520  # MNIST data input (img shape: 28*28)
    n_output = 494  # MNIST data input (img shape: 28*28)

    misspelling = tf.placeholder("float", [None, n_input], name="mistake")
    correct = tf.placeholder("float", [None, n_output], name="correct")

    encoder_m2 = tf.random_normal([n_hidden_1, n_hidden_2])
    weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'encoder_h2': tf.Variable(encoder_m2),
        'decoder_h1': tf.Variable(tf.transpose(encoder_m2)),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_output])),
    }
    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b2': tf.Variable(tf.random_normal([n_output])),
    }

    # Building the encoder
    def encoder(x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                       biases['encoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                       biases['encoder_b2']))
        return layer_2

    # Building the decoder
    def decoder(x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                       biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                       biases['decoder_b2']))
        return layer_2

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
