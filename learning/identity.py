# -*- coding: utf-8 -*-

"""
Straightforward adaptation of the TensorFlow-based autoencoder available at
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py

The only changes are the shape of the input and output layers as well as the data parsing.
"""

import tensorflow as tf


def build_model():
    # Network Parameters
    n_input = 520
    n_output = 494

    misspelled = tf.placeholder("float", [None, n_input], name="input")
    correct = tf.placeholder("float", [None, n_output], name="true-output")

    weights = tf.Variable(tf.random_normal([n_input, n_output]))
    biases = tf.Variable(tf.random_normal([n_output]))

    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(misspelled, weights), biases))
    sliced = [tf.slice(layer1, [0, i], [-1, 26]) for i in xrange(0, 19)]
    softmaxed = [tf.nn.softmax(s) for s in sliced]
    output = tf.concat(1, softmaxed)

    return misspelled, output, correct
