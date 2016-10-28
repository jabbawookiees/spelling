import csv

import click


@click.command()
@click.option('--source', default='data/output', show_default=True,
              help='File containing deduplicated csv')
@click.option('--destination', default='data/result.csv', show_default=True,
              help='Destination of removed items')
def main(source, destination):
    "Make the longest correct and mistaken word to be length 19 and 20 respectively"
    inp = open(source)
    out = open(destination, 'w')
    reader = csv.reader(inp)
    writer = csv.writer(out)
    for word, mistake, count, edits in reader:
        if len(word) > 19 or len(mistake) > 20:
            continue
        else:
            writer.writerow([word, mistake, count, edits])


if __name__ == '__main__':
    main()
