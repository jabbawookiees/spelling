from collections import defaultdict


def replaceables(letter, replacements=defaultdict(list)):
    "All characters we can replace `letter` with"
    if len(replacements) == 0:
        keyboard = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
        dx = [-1, -1, -1,  0, 0,  1, 1, 1]
        dy = [-1,  0,  1, -1, 1, -1, 0, 1]
        position = {}
        character = {}
        for x, row in enumerate(keyboard):
            for y, char in enumerate(row):
                position[char] = x, y
                character[x, y] = char

        for char in ''.join(keyboard):
            x, y = position[char]
            for tx, ty in zip(dx, dy):
                nx = x + tx
                ny = y + ty
                if (nx, ny) in character:
                    replacements[char].append(character[nx, ny])
            replacements[char].sort()
    return replacements[letter]


def insertables(before, after):
    "All characters we can insert in between before and after"
    if len(before) > 0:
        if len(after) > 0:
            if before[-1] == after[0]:
                return [after[0]]
            else:
                return [before[-1], after[0]]
        else:
            return [before[-1]]
    elif len(after) > 0:
        return [after[0]]
    else:
        return [None]


def edits0(word):
    "All edits that are zero edits away from `word`."
    return [word]


def edits1(word):
    "All edits that are one edit away from `word`."
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in replaceables(R[0])]
    inserts    = [L + c + R               for L, R in splits for c in insertables(L, R)]
    return [s for s in set(deletes + transposes + replaces + inserts) if len(s) > 0]


def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
