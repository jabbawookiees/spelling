import csv
from multiprocessing.pool import Pool

import click
import h5py
import numpy as np


def group_by_k(iterable, k):
    group = []
    for i in iterable:
        group.append(i)
        if len(group) == k:
            yield group
            group = []
    if len(group) > 0:
        yield group


def vectorize(string, size):
    array = np.zeros((1, 26 * size))
    for i, c in enumerate(string):
        array[0][26 * i + ord(c) - ord('a')] = 1
    return array


def process(args):
    correct, mistake = args
    correct_arr = vectorize(correct, 19)
    mistake_arr = vectorize(mistake, 20)
    return correct_arr, mistake_arr


@click.command()
@click.option('--source', default='data/trimmed.csv', show_default=True,
              help='File containing trimmed csv')
@click.option('--destination', default='data/serialized.hdf5', show_default=True,
              help='Destination of serialized items')
@click.option('--processors', default=4, show_default=True,
              help='Number of processors.')
def main(source, destination, processors):
    """Pre-compute the matrices for the strings, then serialize them into a numpy-ready format.
    This pre-computation provides a speed up on the order of 100x because we have to re-compute every
    pass otherwise since the data set is so huge"""
    pool = Pool(processors)

    inp = open(source)
    dest_file = h5py.File(destination, "w")
    correct_dataset = dest_file.create_dataset("correct", (0, 494), maxshape=(None, 494), dtype='int8')
    mistake_dataset = dest_file.create_dataset("mistake", (0, 520), maxshape=(None, 520), dtype='int8')
    # dest_file = h5py.File(destination, "a")
    # correct_dataset = dest_file["correct"]
    # mistake_dataset = dest_file["mistake"]
    reader = csv.reader(inp)

    counter = 0
    for group in group_by_k(reader, 1000):
        args = []
        for correct, mistake, count, edits in group:
            args.append((correct, mistake))
            # if counter > 2614000:
            #     print correct, mistake
        counter += len(args)
        print counter
        # if counter <= 2614000:
        #     continue
        mapped = pool.map(process, args)

        for correct_arr, mistake_arr in mapped:
            correct_dataset.resize(correct_dataset.shape[0] + 1, 0)
            mistake_dataset.resize(mistake_dataset.shape[0] + 1, 0)
            correct_dataset[correct_dataset.shape[0] - 1] = correct_arr
            mistake_dataset[mistake_dataset.shape[0] - 1] = mistake_arr


if __name__ == '__main__':
    main()
