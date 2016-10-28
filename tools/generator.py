import csv
import os
import re
from multiprocessing.pool import Pool

import click

from edits import edits1, edits2


def read_input(filename):
    words_with_counts = []
    with open(filename) as data:
        for line in data:
            word, count = re.split("\\s+", line.strip())
            words_with_counts.append((word, int(count)))
    return words_with_counts


def chunks(list, n_chunks):
    batchsize = (len(list) + n_chunks - 1) // n_chunks
    for i in xrange(0, len(list), batchsize):
        yield list[i: i + batchsize]


def process(args):
    words_with_counts, func, filename = args
    with open(filename, "w") as output:
        writer = csv.writer(output)
        for word, count in words_with_counts:
            for mistake in func(word):
                writer.writerow([word, mistake, count])


@click.command()
@click.option('--source', default='data/raw-data.txt', show_default=True,
              help='Raw data file.')
@click.option('--destination', default='data/output', show_default=True,
              help='Where to save garbled data.')
@click.option('--processors', default=4, show_default=True,
              help='Number of processors.')
def main(source, destination, processors):
    try:
        os.makedirs(destination)
    except OSError:
        pass

    words_with_counts = read_input(source)

    pool = Pool(processors)
    args = []
    args.append([words_with_counts, edits1, os.path.join(destination, "data-edit-1.txt")])
    for i, chunk in enumerate(chunks(words_with_counts, processors)):
        fname = os.path.join(destination, "data-edit-2-part-{}.txt".format(i))
        args.append([chunk, edits2, fname])
    pool.map(process, args)


if __name__ == '__main__':
    main()
