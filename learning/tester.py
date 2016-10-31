import os

import click
import numpy as np
import tensorflow as tf
import IPython

from learners import learners


def vectorize(string, size):
    array = np.zeros((1, 26 * size))
    for i, c in enumerate(string):
        array[0][26 * i + ord(c) - ord('a')] = 1
    return array


def unvectorize(arr):
    arr.shape = (arr.size / 26, 26)
    result = []
    for row in arr:
        pairs = []
        for i, c in enumerate(row):
            pairs.append((c, i))
        pairs.sort()
        pairs.reverse()
        result.append(pairs[0][1])
    return ''.join([chr(c + 97) for c in result])


@click.command()
@click.option('--model', help='The model used. Options are: perceptron, mnist, softmax_perceptron')
@click.option('--checkpoint', default=None, show_default=True,
              help='Checkpoint file to save the model. Default is checkpoints/`model_name`.ckpt')
def main(model, checkpoint):
    if checkpoint is None:
        checkpoint = os.path.join("checkpoints", "{}.ckpt".format(model))

    if model in learners:
        input, prediction, output = learners[model].build_model()
    else:
        raise Exception("Model name must be provided")

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    saver = tf.train.Saver()
    if os.path.exists(checkpoint):
        saver.restore(sess, checkpoint)

    def check(mistake):
        v_mistake = vectorize(mistake, 9)
        result = sess.run(prediction, feed_dict={input: v_mistake})
        print unvectorize(result)

    IPython.embed()

if __name__ == '__main__':
    main()
