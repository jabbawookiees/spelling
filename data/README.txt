# Data
The data that we use is the 50,000 most-commonly occurring words in English.

It was taken from https://github.com/hermitdave/FrequencyWords with the author, Hermit Dave, implementing the data cleaning.
He took the original data from http://opus.lingfil.uu.se/OpenSubtitles2016.php, the Open Subtitles 2016 corpus.

The following are descriptions of the files as I have keep them:

raw-data.txt
This comes from https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2016/en/en_50k.txt

training.txt
The training set is a list of pairs (wrong, correct) generated from the raw data.

validation.txt
The validation set is another list of pairs (wrong, correct) also generated from the raw data.
