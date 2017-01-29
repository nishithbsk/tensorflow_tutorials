import numpy as np
import tensorflow as tf

def conv(input, name, weight_dims, stride_dims, padding='SAME'):
    assert(len(weight_dims) == 4) # height, width, num_channels in and out
    assert(len(stride_dims) == 2) # stride height and width

    filter_h, filter_w, num_channels_in, num_channels_out = weight_dims
    stride_h, stride_w = stride_dims

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', weight_dims)
        biases = tf.get_variable('biases', [num_channels_out])
        out = tf.nn.conv2d(input,
                           weight_dims,
                           [1, stride_h, stride_w, 1],
                           padding=padding)
        out = tf.nn.bias_add(out, biases)
        return tf.nn.relu(out, name=scope.name)

def deconv():
    raise NotImplementedError

def fc(input, name, out_dim):
    with tf.variable_scope(name) as scope:
        input_dims = input.get_shape()
        # the input to the fc layer should be flattened
        if input_dims.ndims == 4:
            # for eg. the output of a conv layer
            in_dim = 1
            # ignore the batch dimension
            for dim in input_dims.as_list()[1:]:
                in_dim *= dim
            flat_input = tf.reshape(input, [int(input_dims[0]), in_dim])
        else:
            in_dim = int(input_shape[-1])
            flat_input = input

        weights = tf.get_variable('weights', [in_dim, out_dim])
        biases = tf.get_variable('biases', [out_dim])
        return tf.nn.relu_layer(flat_input, weights, biases, name=scope.name)

def encoder():
    raise NotImplementedError

def decoder():
    raise NotImplementedError

def autoencoder():
    raise NotImplementedError
