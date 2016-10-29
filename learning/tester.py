
def vectorize(string, size):
    array = np.zeros((1, 26 * size))
    for i, c in enumerate(string):
        array[0][26 * i + ord(c) - ord('a')] = 1
    return array


def unvectorize(arr, size=20):
    arr.shape = (size, 26)
    result = []
    for row in arr:
        for i, c in enumerate(row):
            if c == 1:
                result.append(i)
                break
    return ''.join([chr(c + 97) for c in result])
