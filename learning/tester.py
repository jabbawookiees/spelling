import os
import csv
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


def interactive_mode(sess, input, prediction, length):
    def check(mistake):
        v_mistake = vectorize(mistake, length)
        result = sess.run(prediction, feed_dict={input: v_mistake})
        print unvectorize(result)
    print "You are in interactive mode."
    print "Call `check(word)` to see what the predicted correction is."
    IPython.embed()


def batch_mode(sess, input, prediction, length, data):
    inp = open(data)
    reader = csv.reader(inp)
    correct_answers = 0
    total_answers = 0
    for correct, mistake, count, edits in reader:
        v_mistake = vectorize(mistake, length)
        result = sess.run(prediction, feed_dict={input: v_mistake})
        answer = unvectorize(result)
        total_answers += 1
        if correct in answer:
            correct_answers += 1
        if total_answers % 10000 == 0:
            print "Got {}/{} correct".format(correct_answers, total_answers)
    print "Got {}/{} correct".format(correct_answers, total_answers)


@click.command()
@click.option('--model', help='The model used. Options are: perceptron, mnist, softmax_perceptron')
@click.option('--length', default=9, show_default=True,
              help='Maximum length of each string')
@click.option('--checkpoint', default=None, show_default=True,
              help='Checkpoint file to save the model. Default is checkpoints/`model_name`.ckpt')
@click.option('--interactive', default="True", show_default=True,
              help='Whether we should test interactively or not')
@click.option('--data', default="data/deduplicated.csv", show_default=True,
              help='Whether we should test interactively or not')
def main(model, length, checkpoint, interactive, data):
    if checkpoint is None:
        checkpoint = os.path.join("checkpoints", "{}.ckpt".format(model))
    interactive = interactive == "True"

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

    if interactive:
        interactive_mode(sess, input, prediction, length)
    else:
        batch_mode(sess, input, prediction, length, data)

if __name__ == '__main__':
    main()
