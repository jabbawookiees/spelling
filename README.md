# Requirements
This project requires the following python packages:
* `click`
* `tensorflow`
* `ipython`
* `h5py`
* `numpy`

I think that covers everything, but `requirements.txt` contains a pip freeze of my virtual environment in case anything is missing.

# Data Set
The data that we use is the 50,000 most-commonly occurring words in English.

It was taken from https://github.com/hermitdave/FrequencyWords with the author, Hermit Dave, implementing the data cleaning.
He took the original data from http://opus.lingfil.uu.se/OpenSubtitles2016.php, the Open Subtitles 2016 corpus.

The direct link to the raw data file is: https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2016/en/en_50k.txt

# Pre-processing

**The quick way to generate all words of edit distance 1 for training is running the three commands:**

`python tools/generate.py --source data/raw-data --edits 1`

`python tools/deduplicate.py --source data/generated`

`python tools/serialize.py --source data/deduplicated.csv`

To understand what the above commands do, we will explain our three steps to generate data.
1. Compute all the words up to a fixed edit distance.  The tool for this step is at `tools/generate.py`. Possible arguments are as follows:
    * `source` - The raw data file. (default: `data/raw-data.txt`)
    * `destination` - Directory to save generated files (default: directory named `generated` beside the source file)
    * `edits` - Maximum edit distance allowed. Possible values are 0, 1, and 2 (default: 1)
    * `min_length` - The minimum word length allowed (default: 4)
    * `max_length` - The maximum word length allowed (default: 9)
    * `fragments` - How many fragments to break up the words of edit distance 2. Each fragment will be sorted in-memory, so the total should fit in the memory of your system. The fragments will be sorted through an N-way mergesort later. (default: 128)
    * `processors` - How many processors are available to distribute this work. (default: 4)

2. Merge and de-duplicate all the words generated from the previous step.  The tool for this step is at `tools/deduplicate.py`. Possible arguments are as follows:
    * `source` - The directory containing the output of `generate.py`. (default: `data/generated/`)
    * `destination` - Location of the final deduplicated file (default: `deduplicated.csv` beside the source directory)
    * `sorted_files` - Directory to temporarily store sorted fragments (default: directory named `sorted` beside the source directory)
    * `processors` - How many processors are available to distribute this work. (default: 4)

2. Apply one-hot encoding to all the strings, then store them in the HDF5 data format so that we can directly read the data from numpy:
    * `source` - The directory containing the deduplicated file from `deduplicate.py`. (default: `data/deduplicated.csv`)
    * `destination` - Directory to save the encoded file (default: `serialized.hdf5` beside the source directory)
    * `length` - Maximum length of each string. If this is set to 9 but the longest string is shorter, all strings will be padded. If this is set to shorter than the longest string in the source file, an error will be thrown. (default: 9)
    * `processors` - How many processors are available to distribute this work. (default: 4)



# Training
To train a network, we have a script at `learning/trainer.py`. The possible arguments are as follows:
* `data` - The serialized data file created by `serialize.py`. (default: `data/serialized.hdf5`)
* `model` - The model that is trained. Options will be mentioned later.
* `checkpoint` - The checkpoint file used so we don't start over. (default: `checkpoints/{model}.ckpt`)
* `batch_size` - Batch size while training. Set this to `-1` to use the whole data set per batch. (default: -1)
* `epochs` - How many epochs to train for (default: 50)
* `learning_rate` - Learning rate (default: 0.30)
* `display_step` - How many epochs before we print out diagnostic information. (default: 1)
* `save_delay` - How many seconds must pass before we update the checkpoint file (default: 60)

### Models
The models we have implemented are the following:
* `autoencoder` - This is the single hidden layer autoencoder that accepts strings of length 9 and outputs strings of length 9. There are 234 neurons in the input layer, 200 in the hidden layer, and 234 in the output layer.
* `autoencoder2` - This is the three-hidden-layer autoencoder that accepts strings of length 9 and outputs strings of length 9. There are 234 neurons in the input layer, 217 in the first hidden layer, 200 in the second, 217 in the third, and 234 in the output layer. Warning: This model just generates noise.
* `convolution` - This network treats the string as an array of 9 with 26 channels (similar to how an image has 3 color channels). A convolution matrix is applied with stride 2, retrieving 600 features, then a dense hidden layer with 200 neurons is applied. Warning: This model just generates noise no matter how much I played with the features and the dense layer.
* `perceptron` - This is just a fully-connected input and output layer. The associated checkpoint file was trained with the identity data set so I could debug the trainer. This is just the identity function. :)
* There are a few more, but some of those were simply experiments on TensorFlow and none of them converged anywhere.

# Testing
To test a network, we have a script at `learning/tester.py`. The possible arguments are as follows:
* `model` - The model that will be used for testing. Same options as in `trainer.py`.
* `length` - Length of the longest word accepted by the model. (default: 9)
* `checkpoint` - The checkpoint file where we load the weights. (default: `checkpoints/{model}.ckpt`)
* `interactive` - Whether testing will be interactive or not (default: True)
* `data` - If interactive mode is false, this is the data we use (default: `data/deduplicated.csv`)

For the non-interactive testing mode, we read in data from the csv file and predict each answer. It will periodically print how much it's gotten correct and at the end it will give a final result.

For the interactive mode, it provides a function, `check` which we call to get the predicted answer. For example, `check("hello")` should print out "hello" or some noise if we used a bad model.
