import h5py


class SpellingDataBatcher(object):
    "Generator that yields things by batch"
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __iter__(self):
        file = self.data.file
        for idx in xrange(0, file["correct"].shape[0], self.batch_size):
            yield file["mistake"][idx: idx + self.batch_size], file["correct"][idx:idx + self.batch_size]


class SpellingData(object):
    "Generator that yields items one by one"
    def __init__(self, filename):
        self.file = h5py.File(filename, 'r')
        self.index = 0

    def batched_training(self, batch_size):
        return SpellingDataBatcher(self, batch_size)

    def __iter__(self):
        index = 0
        for correct, mistake, count, edits in self.reader:
            yield self.file["mistake"][index], self.file["correct"][index]
            index += 1


def read_data_set(filename):
    return SpellingData(filename)
