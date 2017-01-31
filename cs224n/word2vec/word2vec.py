# word2vec
# Author: Nishith Khandwala (nishith@stanford.edu)
# Adapted from https://www.tensorflow.org/tutorials/word2vec/

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import tensorflow as tf

from utils import *

'''
Consider the following sentence:
"the first cs224n homework was a lot of fun"

With a window size of 1, we have the dataset:
([the, cs224n], first), ([lot, fun], of) ...

Remember that Skipgram tries to predict each context word from 
its target word, and so the task becomes to predict 'the' and
'cs224n' from first, 'lot' and 'fun' from 'of' and so on.

Our dataset now becomes:
(first, the), (first, cs224n), (of, lot), (of, fun) ...
'''

# Let's define some constants first
batch_size = 128
vocabulary_size = 50000
embedding_size = 128  # Dimension of the embedding vector.
num_sampled = 64    # Number of negative examples to sample.

'''
load_data loads the already preprocessed training and val data.

train data is a list of (batch_input, batch_labels) pairs.
val data is a list of all validation inputs.
reverse_dictionary is a python dict from word index to word
'''
train_data, val_data, reverse_dictionary = load_data()
print("Number of training examples:", len(train_data)*batch_size)
print("Number of validation examples:", len(val_data))

def skipgram():
    raise NotImplementedError

def run():
    raise NotImplementedError

# Let's start training
final_embeddings = run()

# Visualize the embeddings.
visualize_embeddings(final_embeddings, reverse_dictionary)

