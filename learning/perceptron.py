# -*- coding: utf-8 -*-

"""
This implements a simple perceptron
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf


def build_model():
    # Network Parameters
    n_input = 520
    n_output = 494

    misspelled = tf.placeholder("float", [None, n_input], name="mistake")
    correct = tf.placeholder("float", [None, n_output], name="correct")

    weights = tf.Variable(tf.random_normal([n_input, n_output]))
    biases = tf.Variable(tf.random_normal([n_output]))
    output = tf.nn.sigmoid(tf.add(tf.matmul(misspelled, weights), biases))
    return misspelled, output, correct
