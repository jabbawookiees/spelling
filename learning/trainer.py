import os
import time

import click
import tensorflow as tf

from reader import read_data_set
from autoencoder import autoencoder_model


@click.command()
@click.option('--source', default='data/serialized.hdf5', show_default=True,
              help='File containing trimmed csv')
@click.option('--checkpoint', default='checkpoints/default.ckpt', show_default=True,
              help='Checkpoint file to load and save at')
@click.option('--batch_size', default=1000, show_default=True,
              help='Batch size to load and feed to the network')
@click.option('--epochs', default=20, show_default=True,
              help='Model epoch count')
@click.option('--learning_rate', default=0.01, show_default=True,
              help='Model learning rate')
@click.option('--display_step', default=1, show_default=True,
              help='How often to print epoch update')
@click.option('--save_delay', default=60, show_default=True,
              help='Seconds between each save')
def main(source, checkpoint, batch_size, epochs, learning_rate, display_step, save_delay):
    batch_size = int(batch_size)
    epochs = int(epochs)
    learning_rate = float(learning_rate)
    display_step = int(display_step)
    save_delay = int(save_delay)

    # Construct the model
    input, prediction, output = autoencoder_model()

    # Define loss and optimizer, minimize the squared error
    cost = tf.reduce_mean(tf.pow(output - prediction, 2))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

    # Launch the graph
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    dataset = read_data_set('data/serialized.hdf5')

    saver = tf.train.Saver()
    start_time = time.time()
    last_saved = start_time

    if os.path.exists(checkpoint):
        saver.restore(sess, checkpoint)

    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        for batch_mistake, batch_correct in dataset.batched_training(batch_size):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={output: batch_correct, input: batch_mistake})

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "cost=", "{:.9f}".format(c),
                  "Time: {} seconds".format(time.time() - start_time))

        if time.time() - last_saved > save_delay:
            last_saved = time.time()
            saver.save(sess, checkpoint)
            print("Saved at {}".format(checkpoint))

    print("Optimization Finished!")


if __name__ == '__main__':
    main()