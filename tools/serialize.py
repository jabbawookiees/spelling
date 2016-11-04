"""
Takes the data from deduplicate.py and computes the one-hot
vectors for each character. It is then stored in an hdf5 file
which allows the data to be loaded directly as numpy objects.
This provides a huge speedup in training time since the data
does not fit in memory.


Usage is:
python tools/serialize.py --source data/deduplicated.csv

Use `python tools/serialize.py --help` for a list of options
"""

import os
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
    correct, mistake, length = args
    correct_arr = vectorize(correct, length)
    mistake_arr = vectorize(mistake, length)
    return correct_arr, mistake_arr


@click.command()
@click.option('--source', default='data/deduplicated.csv', show_default=True,
              help='File containing deduplicated data')
@click.option('--destination', default=None, show_default=True,
              help='Destination of serialized items')
@click.option('--length', default=9, show_default=True,
              help='Maximum length of each string')
@click.option('--processors', default=4, show_default=True,
              help='Number of processors.')
def main(source, destination, length, processors):
    """Pre-compute the matrices for the strings, then serialize them into a numpy-ready format.
    This pre-computation provides a speed up on the order of 100x because we have to re-compute every
    pass otherwise since the data set is so huge"""
    if destination is None:
        destination = os.path.join(os.path.dirname(source), "serialized.hdf5")
    length = int(length)
    pool = Pool(int(processors))

    inp = open(source)
    dest_file = h5py.File(destination, "w")
    correct_dataset = dest_file.create_dataset("correct", (0, 26 * length), maxshape=(None, 26 * length), dtype='int8')
    mistake_dataset = dest_file.create_dataset("mistake", (0, 26 * length), maxshape=(None, 26 * length), dtype='int8')

    reader = csv.reader(inp)

    counter = 0
    for group in group_by_k(reader, 1000):
        args = []
        for correct, mistake, count, edits in group:
            args.append((correct, mistake, length))

        counter += len(args)
        print counter

        mapped = pool.map(process, args)

        for correct_arr, mistake_arr in mapped:
            correct_dataset.resize(correct_dataset.shape[0] + 1, 0)
            mistake_dataset.resize(mistake_dataset.shape[0] + 1, 0)
            correct_dataset[correct_dataset.shape[0] - 1] = correct_arr
            mistake_dataset[mistake_dataset.shape[0] - 1] = mistake_arr


if __name__ == '__main__':
    main()
