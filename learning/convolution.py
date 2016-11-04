# -*- coding: utf-8 -*-

"""
This implements a network with:
One convolutional layer: stride 2, width 2, channels 26, features 600
One fully connected layer
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf


def build_model():
    # Network Parameters
    n_characters = 8
    n_alphabet = 26
    n_input = n_alphabet * n_characters
    n_output = n_alphabet * n_characters
    conv1_features = 600
    conv1_stride = 2
    conv1_columns = int(n_characters / conv1_stride)

    misspelled = tf.placeholder("float", [None, n_input], name="mistake")
    correct = tf.placeholder("float", [None, n_output], name="correct")

    weights = tf.Variable(tf.random_normal([conv1_columns * conv1_features, n_output]))
    biases = tf.Variable(tf.random_normal([n_output]))

    conv_in = tf.reshape(misspelled, [-1, n_characters, n_alphabet])
    # Input shape: batch = 1000, in_width = n_characters, in_channels = n_alphabet
    # Kernel shape: filter_with = n_alphabet, in_channels = n_alphabet, out_channels = conv1_features
    kernel_shape = [conv1_stride, n_alphabet, conv1_features]
    kernel = tf.Variable(tf.truncated_normal(kernel_shape, stddev=5e-2))
    conv = tf.nn.conv1d(conv_in, kernel, conv1_stride, "SAME")
    conv_reshaped = tf.reshape(conv, [-1, 1, conv1_columns, conv1_features])
    norm = tf.nn.local_response_normalization(conv_reshaped)
    norm_reshaped = tf.reshape(norm, [-1, conv1_columns * conv1_features])

    output = tf.nn.sigmoid(tf.add(tf.matmul(norm_reshaped, weights), biases))

    return misspelled, output, correct
