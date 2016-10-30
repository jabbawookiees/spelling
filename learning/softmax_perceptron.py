"""
Same as the model in perceptron.py, but the output is softmaxed every 26 letters.
"""

import tensorflow as tf


def build_model():
    # Network Parameters
    n_input = 234
    n_output = 234

    misspelled = tf.placeholder("float", [None, n_input], name="input")
    correct = tf.placeholder("float", [None, n_output], name="true-output")

    weights = tf.Variable(tf.random_normal([n_input, n_output]))
    biases = tf.Variable(tf.random_normal([n_output]))

    layer1 = tf.add(tf.matmul(misspelled, weights), biases)
    sliced = [tf.slice(layer1, [0, i], [-1, 26]) for i in xrange(0, 9)]
    softmaxed = [tf.nn.softmax(s) for s in sliced]
    output = tf.concat(1, softmaxed)

    return misspelled, output, correct
