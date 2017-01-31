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
    batch_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    batch_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    val_dataset = tf.constant(val_data, dtype=tf.int32)

    with tf.variable_scope('word2vec') as scope:
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, 
                                                    embedding_size], 
                                                    -1.0, 1.0))
        batch_embeddings = tf.nn.embedding_lookup(embeddings, batch_inputs)

        weights = tf.Variable(tf.truncated_normal([vocabulary_size, 
                                                   embedding_size],
                                                   stddev=1.0/math.sqrt(embedding_size)))
        biases = tf.Variable(tf.zeros([vocabulary_size]))

        # This objective is maximized when the model assigns high probabilities
        # to the real words, and low probabilities to noise words.
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=weights, 
                                             biases=biases,
                                             labels=batch_labels,
                                             inputs=batch_embeddings,
                                             num_sampled=num_sampled,
                                             num_classes=vocabulary_size))


        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings/norm
        
        val_embeddings = tf.nn.embedding_lookup(normalized_embeddings, 
                                                val_dataset)
        similarity = tf.matmul(val_embeddings, 
                               normalized_embeddings, transpose_b=True)

    return batch_inputs, batch_labels, normalized_embeddings, similarity, loss

def run():
    # load model
    batch_inputs, batch_labels, normalized_embeddings, similarity, loss = skipgram()
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        average_loss = 0
        for step, batch_data in enumerate(train_data):
            inputs, labels = batch_data
            feed_dict = {batch_inputs: inputs, batch_labels: labels}

            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val
            
            if step % 1000 == 0:
                if step > 0:
                    average_loss /= 1000
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0
                    
            if step % 5000 == 0:
                sim = similarity.eval()
                for i in xrange(len(val_data)):
                    top_k = 8  # number of nearest neighbors                       
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1] 
                    print_closest_words(val_data[i], nearest, reverse_dictionary)
  
        final_embeddings = normalized_embeddings.eval()
        return final_embeddings

# Let's start training
final_embeddings = run()

# Visualize the embeddings.
visualize_embeddings(final_embeddings, reverse_dictionary)

