import tensorflow as tf

from layers import *

def encoder(input):
    input_reshaped = tf.reshape(input, [batch_size,
                                        image_dims[0],
                                        image_dims[1],
                                        image_dims[2]])
    conv1 = conv(input_reshaped, 'conv1', [5, 5, 32], [2, 2])
    conv2 = conv(conv1, 'conv2', [5, 5, 64], [2, 2])
    conv3 = conv(conv2, 'conv3', [3, 3, 128], [2, 2])
    conv4 = conv(conv3, 'conv4', [3, 3, 256], [2, 2])
    conv5 = conv(conv4, 'conv5', [3, 3, 512], [2, 2])
    fc_enc = fc(conv5, 'fc_enc', hidden_size, non_linear_fn=None)
    return fc_enc

def decoder(input):
    fc_dec = fc(input, 'fc_dec', 7*7*512)
    fc_dec_reshaped = tf.reshape(fc_dec, [batch_size,
                                          7, 7, 512])
    deconv1 = deconv(fc1, 'deconv1', [3, 3, 256], [2, 2])
    deconv2 = deconv(deconv1, 'deconv2', [3, 3, 128], [2, 2])
    deconv3 = deconv(deconv2, 'deconv3', [3, 3, 64], [2, 2])
    deconv4 = deconv(deconv3, 'deconv4', [5, 5, 32], [2, 2])
    deconv5 = deconv(deconv4, 'deconv5', [5, 5, 3], [2, 2], non_linear_fn=tf.sigmoid)
    return deconv5

def autoencoder():
    input_image = tf.placeholder(tf.float32,
                                 [batch_size,
                                  image_dims[0],
                                  image_dims[1],
                                  image_dims[2]], name='input_image')

    with tf.variable_scope('autoencoder') as scope:
        encoding = encoder(input_image)
        reconstructed_image = decoder(encoding)
    return input_image, reconstructed_image

