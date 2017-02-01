import tensorflow as tf

from data_utils import *
from autoencoder import *

batch_size = ...
lr = ...
num_iters = ...

def calculate_loss(original, reconstructed):
    return tf.div(tf.reduce_sum(tf.square(tf.sub(reconstructed,
                                                 original))), 
                  tf.constant(float(batch_size)))

def train(dataset):
    input_image, reconstructed_image = autoencoder()
    loss = calculate_loss(input_image, reconstructed_image)
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)

        for step in xrange(num_iters):
            input_batch, _ = dataset.train.next_batch(batch_size)
            loss_val,  _ = session.run([loss, optimizer], 
                                       feed_dict={input_image: input_batch})
            print "Loss at step", step, ":", loss_val

        reconstruction = session.run(reconstructed_image,
                                     feed_dict={input_image: dataset.test.images})



if __name__ == '__main__':
    dataset = load_dataset('mnist')
    train(dataset)
    
