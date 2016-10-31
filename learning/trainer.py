import os
import sys
import time

import click
import tensorflow as tf

from reader import read_data_set
from learners import learners

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)


@click.command()
@click.option('--data', default='data/serialized.hdf5', show_default=True,
              help='File containing trimmed csv')
@click.option('--model', help='The model used. Options are: perceptron, mnist, softmax_perceptron')
@click.option('--checkpoint', default=None, show_default=True,
              help='Checkpoint file to save the model. Default is checkpoints/`model_name`.ckpt')
@click.option('--batch_size', default=-1, show_default=True,
              help='Batch size to load and feed to the network')
@click.option('--epochs', default=50, show_default=True,
              help='Model epoch count')
@click.option('--learning_rate', default=0.30, show_default=True,
              help='Model learning rate')
@click.option('--display_step', default=1, show_default=True,
              help='How often to print epoch update')
@click.option('--save_delay', default=60, show_default=True,
              help='Seconds between each save')
def main(data, model, checkpoint, batch_size, epochs, learning_rate, display_step, save_delay):
    batch_size = int(batch_size)
    epochs = int(epochs)
    learning_rate = float(learning_rate)
    display_step = int(display_step)
    save_delay = int(save_delay)

    # Construct the model
    if model in learners:
        input, prediction, output = learners[model].build_model()
    else:
        raise Exception("Model name must be provided")

    if checkpoint is None:
        checkpoint = os.path.join("checkpoints", "{}.ckpt".format(model))

    # Define loss and optimizer, minimize the squared error
    # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(prediction, tf.slice(input, [0, 0], [-1, 234]))
    # cost = tf.reduce_mean(cross_entropy)
    cost = tf.nn.l2_loss(prediction - output)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Launch the graph
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    dataset = read_data_set(data)

    saver = tf.train.Saver()
    start_time = time.time()
    last_saved = start_time

    if os.path.exists(checkpoint):
        saver.restore(sess, checkpoint)

    # Training cycle
    display_counter = 0
    for epoch in range(epochs):
        # Loop over all batches
        c_sum, c_count = 0, 0
        for batch_mistake, batch_correct in dataset.batched_training(batch_size):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={output: batch_correct, input: batch_mistake})
            c_sum += c
            c_count += 1
            display_counter += 1
            if os.environ.get("DEBUG_TRAINER") == "True" and display_counter % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1),
                      "cost=", "{}".format(c),
                      "Time: {} seconds".format(time.time() - start_time))

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "cost:", "{:.9f}".format(c_sum / c_count),
                  "Time: {} seconds".format(time.time() - start_time))

        if time.time() - last_saved > save_delay:
            last_saved = time.time()
            saver.save(sess, checkpoint)
            print("Saved at {}".format(checkpoint))

    saver.save(sess, checkpoint)
    print("Optimization Finished!")


if __name__ == '__main__':
    main()
