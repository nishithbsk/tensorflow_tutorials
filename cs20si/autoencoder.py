import tensorflow as tf

from layers import *

def encoder(input):
    conv1 = conv(input, 'conv1', [3, 3, 1], [2, 2])
    conv2 = conv(conv1, 'conv2', [3, 3, 8], [2, 2])
    conv3 = conv(conv2, 'conv3', [3, 3, 8], [2, 2])
    fc_enc = fc(conv3, 'fc_enc', 100, non_linear_fn=None)
    return fc_enc

def decoder(input):
    fc_dec = fc(input, 'fc_dec', 128)
    fc_dec_reshaped = tf.reshape(fc_dec, [-1, 4, 4, 8])
    deconv1 = deconv(fc_dec_reshaped, 'deconv1', [3, 3, 8], [2, 2])
    deconv2 = deconv(deconv1, 'deconv2', [8, 8, 1], [2, 2], padding='VALID')
    deconv3 = deconv(deconv2, 'deconv3', [7, 7, 1], [1, 1], padding='VALID',
                     non_linear_fn=tf.sigmoid)
    return deconv3

def encoder_fc(input):
    input_flat = tf.reshape(input, (-1, 784))
    fc1 = fc(input_flat, 'fc1', 128)
    fc2 = fc(fc1, 'fc2', 64)
    fc3 = fc(fc2, 'fc3', 32)
    return fc3

def decoder_fc(input):
    fc1_dec = fc(input, 'fc1_dec', 64)
    fc2_dec = fc(fc1_dec, 'fc2_dec', 128)
    fc3_dec = fc(fc2_dec, 'fc3_dec', 784, non_linear_fn=tf.sigmoid)
    fc3_dec_reshaped = tf.reshape(fc3_dec, (-1, 28, 28, 1))
    return fc3_dec_reshaped

def autoencoder(input_shape):
    input_image = tf.placeholder(tf.float32,
                                 input_shape,
                                 name='input_image')

    with tf.variable_scope('autoencoder') as scope:
        encoding = encoder(input_image)
        reconstructed_image = decoder(encoding)
        return input_image, reconstructed_image

