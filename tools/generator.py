import csv
import os
import re
from multiprocessing.pool import Pool

import click

from edits import edits0, edits1, edits2


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
    words_with_counts, edit_distance, filename = args
    func = edits0
    if edit_distance == 1:
        func = edits1
    elif edit_distance == 2:
        func = edits2
    with open(filename, "w") as output:
        writer = csv.writer(output)
        for word, count in words_with_counts:
            if not (set(word) <= set("abcdefghijklmnopqrstuvwxyz")):
                continue
            for mistake in func(word):
                writer.writerow([word, mistake, "{:010}".format(count), edit_distance])


@click.command()
@click.option('--source', default='data/raw-data.txt', show_default=True,
              help='Raw data file.')
@click.option('--destination', default='data/generated', show_default=True,
              help='Where to save garbled data.')
@click.option('--edits', default=2, show_default=True,
              help='Maximum edit distance allowed.')
@click.option('--fragments', default=None, show_default=True,
              help='Number of fragments to chunk the edit distance 2 set.')
@click.option('--processors', default=4, show_default=True,
              help='Number of processors.')
def main(source, destination, edits, fragments, processors):
    try:
        os.makedirs(destination)
    except OSError:
        pass
    edits = int(edits)
    processors = int(processors)
    if fragments is None:
        fragments = processors
    else:
        fragments = int(fragments)

    words_with_counts = read_input(source)

    pool = Pool(processors)
    args = []
    args.append([words_with_counts, 0, os.path.join(destination, "data-edit-0.csv")])
    if 1 <= edits:
        args.append([words_with_counts, 1, os.path.join(destination, "data-edit-1.csv")])
    if 2 <= edits:
        for i, chunk in enumerate(chunks(words_with_counts, fragments)):
            fname = os.path.join(destination, "data-edit-2-part-{:02}.csv".format(i))
            args.append([chunk, 2, fname])
    pool.map(process, args)


if __name__ == '__main__':
    main()
