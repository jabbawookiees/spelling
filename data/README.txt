# Data
The data that we use is the 50,000 most-commonly occurring words in English.

It was taken from https://github.com/hermitdave/FrequencyWords with the author, Hermit Dave, implementing the data cleaning.
He took the original data from http://opus.lingfil.uu.se/OpenSubtitles2016.php, the Open Subtitles 2016 corpus.

I took the following steps to preprocess the data:

1. raw-data.txt - This comes from https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2016/en/en_50k.txt
2. output/filename.csv - Break up raw-data.txt into different groups and files (tools/generator.py)
3. deduplicated.csv - Combined the output from the previous one into a de-duplicated one using a mergesort-based algorithm (tools/deduplicate.py)
4. trimmed.csv - Remove all words of length 20 or higher and all misspellings of length 21 or higher (tools/trim_long.py)
5. serialized.h5py - Preprocess the data and serialize it into a numpy-ready format to vastly improve training time (tools/serialize.py)
