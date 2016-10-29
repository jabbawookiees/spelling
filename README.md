# Data
The data that we use is the 50,000 most-commonly occurring words in English.

It was taken from https://github.com/hermitdave/FrequencyWords with the author, Hermit Dave, implementing the data cleaning.
He took the original data from http://opus.lingfil.uu.se/OpenSubtitles2016.php, the Open Subtitles 2016 corpus.

The raw-data is available at data/raw-data.txt
This comes from https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2016/en/en_50k.txt

For pre-processing, I did the following steps:
I took the following steps to preprocess the data:
1. Break up raw-data.txt into different groups and compute all words up to a fixed edit distance.
   After that I store them in separate files because the data set becomes too big to load in memory. (tools/generator.py)
   During this step I also remove all words that contain non-alphanumeric characters and all words betwen 4 and 9 in length.
2. I sort each individual file, then apply a mergesort-based algorithm to delete duplicates. This gives me a one-to-one map for
   misspelling to correct spelling. I prioritize items by smaller edit distance and more popular correct word. (tools/deduplicate.py)
3. After this, I serialize the data into the format used in my networks. Each character is converted into a 26-length vector,
   so 'a' becomes [1, 0, 0, ...., 0], 'b' becomes [0, 1, 0, ..., 0], and so on. This is stored in an hdf5 file so that data can
   be readily loaded as a numpy array. (tools/serialize.py)
