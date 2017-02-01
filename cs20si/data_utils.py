import sys
import tensorflow
from tensorflow.examples.tutorials.mnist import input_data

def load_dataset(dataset='mnist'):
    if dataset == 'mnist':
        return input_data.read_data_sets('MNIST_data')
    else:
        print "Sorry! This tutorial only supports the MNIST dataset right now."
        sys.exit(1)
