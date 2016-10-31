# -*- coding: utf-8 -*-

"""
This implements a simple perceptron
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf


def build_model():
    # Network Parameters
    n_characters = 9
    n_alphabet = 26
    n_input = n_alphabet * n_characters
    n_hidden1 = n_alphabet * n_characters
    n_output = n_alphabet * n_characters
    conv1_features = 200

    misspelled = tf.placeholder("float", [None, n_input], name="mistake")
    correct = tf.placeholder("float", [None, n_output], name="correct")

    weights = tf.Variable(tf.random_normal([n_characters * conv1_features, n_output]))
    biases = tf.Variable(tf.random_normal([n_output]))

    conv_in = tf.reshape(misspelled, [-1, n_characters, n_alphabet])
    conv_in = tf.reshape(misspelled, [-1, 1, n_characters, n_alphabet])

    # Input shape: batch = 1000, in_width = n_characters, in_channels = n_alphabet
    # Kernel shape: filter_with = n_alphabet, in_channels = n_alphabet, out_channels = conv1_features
    kernel_shape = [1, 3, n_alphabet, conv1_features]
    kernel = tf.Variable(tf.truncated_normal(kernel_shape, stddev=5e-2))
    conv = tf.nn.conv2d(conv_in, kernel, [1, 1, 1, 1], "SAME")
    conv1_biases = tf.Variable(tf.constant(0.0, shape=[conv1_features]))
    conv1 = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

    norm = tf.nn.local_response_normalization(conv1)
    norm_reshaped = tf.reshape(norm, [-1, n_characters * conv1_features])

    normalized = tf.nn.sigmoid(tf.add(tf.matmul(norm_reshaped, weights), biases))

    hidden1_weights = tf.Variable(tf.random_normal([n_input, n_hidden1]))
    hidden1_biases = tf.Variable(tf.random_normal([n_hidden1]))
    hidden1 = tf.nn.sigmoid(tf.add(tf.matmul(normalized, hidden1_weights), hidden1_biases))

    output_weights = tf.Variable(tf.random_normal([n_input, n_output]))
    output_biases = tf.Variable(tf.random_normal([n_output]))
    output = tf.nn.sigmoid(tf.add(tf.matmul(hidden1, output_weights), output_biases))

    return misspelled, output, correct
