"""
Takes the generated data from generator.py and combines them into one
set.

It's possible for two words to get the same typo. For example "hellx" is
one edit distance from "hello" and "hell" and two edit distance from "help".
In this case, we want to map "hellx" to "hello" because it has a smaller
edit distance than "help" and it is more common than "hell".


Usage is:
python tools/deduplicate.py --source data/generated

Use `python tools/deduplicate.py --help` for a list of options
"""
import csv
import os
from multiprocessing.pool import Pool
import heapq

import click


def sort_csvs(args):
    input_fname, output_fname = args
    print input_fname, output_fname
    inp = open(input_fname, "r")
    out = open(output_fname, "w")
    reader = csv.reader(inp)
    data = [(word, mistake, int(count), int(edits)) for word, mistake, count, edits in reader]
    data.sort(key=lambda (word, mistake, count, edits): (mistake, edits, -count, word))
    writer = csv.writer(out)
    for word, mistake, count, edits in data:
        writer.writerow([word, mistake, "{:010}".format(count), edits])
    inp.close()
    out.close()
    return output_fname


def heappush(heap, reader):
    try:
        word, mistake, count, edits = reader.next()
        heapq.heappush(heap, (mistake, int(edits), -int(count), word, reader))
    except StopIteration:
        return False


def group_words(fnames):
    files = [open(f) for f in fnames]
    readers = [csv.reader(f) for f in files]
    heap = []
    for reader in readers:
        heappush(heap, reader)

    group = []
    while len(heap) > 0:
        mistake, edits, count, word, reader = heapq.heappop(heap)
        if len(group) != 0 and group[0][1] != mistake:
            yield group
            group = []
        group.append((word, mistake, -count, edits))
        heappush(heap, reader)


def merge(fnames, destination_writer):
    """Takes in a list of filenames, each a csv of sorted items.
    For the merge step, we since we are guaranteed that it's sorted
    primarily by the misspelled word, we can find all misspellings that
    end up the same, then we prioritize the least edit distance, followed
    by the popularity of that word"""
    for group in group_words(fnames):
        word, mistake, count, edits = group[0]
        destination_writer.writerow([word, mistake, "{:010}".format(count), edits])


@click.command()
@click.option('--source', default='data/generated', show_default=True,
              help='Directory containing the output of generate.py')
@click.option('--destination', default=None, show_default=True,
              help='File containing deduplicated csv')
@click.option('--sorted_files', default=None, show_default=True,
              help='Where to save sorted data before the merge step.')
@click.option('--processors', default=4, show_default=True,
              help='Number of processors.')
def main(source, destination, sorted_files, processors):
    if destination is None:
        destination = os.path.join(os.path.dirname(source.rstrip("/")), "deduplicated.csv")
    if sorted_files is None:
        sorted_files = os.path.join(os.path.dirname(destination), "sorted")
    try:
        os.makedirs(sorted_files)
    except OSError:
        pass

    pool = Pool(processors)

    args = []
    filenames = [fname for folder, _, fnames in os.walk(source) for fname in fnames]
    for fname in filenames:
        args.append([os.path.join(source, fname), os.path.join(sorted_files, fname)])

    # This part is sort of like mergesort. Sort the different files by misspelling, edit distance, and -count
    sorted_fnames = pool.map(sort_csvs, args)

    # Then apply the merge algorithm over all the files to create one super file
    out = open(destination, "w")
    output = csv.writer(out)
    merge(sorted_fnames, output)
    out.close()


if __name__ == '__main__':
    main()
