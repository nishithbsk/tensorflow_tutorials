import tensorflow as tf

from layer_utils import get_deconv2d_output_dims

def conv(input, name, filter_dims, stride_dims, padding='SAME',
         non_linear_fn=tf.nn.relu):
    input_dims = input.get_shape().as_list()
    assert(len(input_dims) == 4) # batch_size, height, width, num_channels_in
    assert(len(filter_dims) == 3) # height, width and num_channels out
    assert(len(stride_dims) == 2) # stride height and width

    num_channels_in = input_dims[-1]
    filter_h, filter_w, num_channels_out = filter_dims
    stride_h, stride_w = stride_dims

    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', [filter_h,
                                              filter_w,
                                              num_channels_in,
                                              num_channels_out])
        biases = tf.get_variable('biases', [num_channels_out])
        out = tf.nn.conv2d(input,
                           weights,
                           [1, stride_h, stride_w, 1],
                           padding=padding)
        out = tf.nn.bias_add(out, biases)
        if non_linear_fn:
            return non_linear_fn(out, name=scope.name)
        else:
            return out

def deconv(input, name, filter_dims, stride_dims, padding='SAME',
           non_linear_fn=tf.nn.relu):
    input_dims = input.get_shape().as_list()
    assert(len(input_dims) == 4) # batch_size, height, width, num_channels_in
    assert(len(filter_dims) == 3) # height, width and num_channels out
    assert(len(stride_dims) == 2) # stride height and width

    num_channels_in = input_dims[-1]
    filter_h, filter_w, num_channels_out = filter_dims
    stride_h, stride_w = stride_dims
    output_dims = get_deconv2d_output_dims(input_dims,
                                           filter_dims,
                                           stride_dims,
                                           padding)

    with tf.variable_scope(name) as scope:
        # note that num_channels_out and in positions are flipped for deconv.
        weights = tf.get_variable('weights', [filter_h,
                                              filter_w,
                                              num_channels_out,
                                              num_channels_in])
        biases = tf.get_variable('biases', [num_channels_out])
        out = tf.nn.conv2d_transpose(input,
                                     weights,
                                     output_dims,
                                     [1, stride_h, stride_w, 1],
                                     padding=padding)
        out = tf.nn.bias_add(out, biases)
        if non_linear_fn:
            return non_linear_fn(out, name=scope.name)
        else:
            return out

def max_pool(input, name, filter_dims, stride_dims, padding='SAME'):
    assert(len(filter_dims) == 2) # filter height and width
    assert(len(stride_dims) == 2) # stride height and width

    filter_h, filter_w = filter_dims
    stride_h, stride_w = stride_dims
    return tf.nn.max_pool(input, 
                          [1, filter_h, filter_w, 1],
                          [1, stride_h, stride_w, 1],
                          padding=padding)

def fc(input, name, out_dim, non_linear_fn=tf.nn.relu):
    assert(type(out_dim) == int)

    with tf.variable_scope(name) as scope:
        input_dims = input.get_shape().as_list()
        # the input to the fc layer should be flattened
        if len(input_dims) == 4:
            # for eg. the output of a conv layer
            batch_size, input_h, input_w, num_channels = input_dims
            # ignore the batch dimension
            in_dim = input_h * input_w * num_channels
            flat_input = tf.reshape(input, [batch_size, in_dim])
        else:
            in_dim = input_dims[-1]
            flat_input = input

        weights = tf.get_variable('weights', [in_dim, out_dim])
        biases = tf.get_variable('biases', [out_dim])
        out = tf.nn.xw_plus_b(flat_input, weights, biases)
        if non_linear_fn:
            return non_linear_fn(out, name=scope.name)
        else:
            return out

