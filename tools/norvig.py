# Norvig's spell checker from http://norvig.com/spell-correct.html

import re
from collections import Counter
from edits import edits1, edits2

def read_input(filename):
    with open(filename) as data:
        for line in data:
            split = re.split("\\s+", line)
            word, count = re.split("\\s+", line)
            WORDS[word] = int(count)

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

WORDS = read_input("raw-data.txt")
