import tensorflow as tf
import numpy as np
import math
import random

from scipy.io.wavfile import read, write
from scipy.fftpack import rfft, irfft

DIMS=[256, 128]
SIZE=1024

def create(x, layer_sizes):

        # Build the encoding layers
        next_layer_input = x

        encoding_matrices = []
        for dim in layer_sizes:
                input_dim = int(next_layer_input.get_shape()[1])

                # Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
                W = tf.Variable(tf.random_uniform([input_dim, dim], -1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim)))

                # Initialize b to zero
                b = tf.Variable(tf.zeros([dim]))

                # We are going to use tied-weights so store the W matrix for later reference.
                encoding_matrices.append(W)

                output = tf.nn.tanh(tf.matmul(next_layer_input,W) + b)

                # the input into the next layer is the output of this layer
                next_layer_input = output

        # The fully encoded x value is now stored in the next_layer_input
        encoded_x = next_layer_input

        # build the reconstruction layers by reversing the reductions
        layer_sizes.reverse()
        encoding_matrices.reverse()


        for i, dim in enumerate(layer_sizes[1:] + [ int(x.get_shape()[1])]) :
                # we are using tied weights, so just lookup the encoding matrix for this step and transpose it
                W = tf.transpose(encoding_matrices[i])
                b = tf.Variable(tf.zeros([dim]))
                output = tf.nn.tanh(tf.matmul(next_layer_input,W) + b)
                next_layer_input = output

        # the fully encoded and reconstructed value of x is here:
        reconstructed_x = next_layer_input

        return {
                'encoded': encoded_x,
                'decoded': reconstructed_x,
                'cost' : tf.sqrt(tf.reduce_mean(tf.square(x-reconstructed_x)))
        }

def deep_test():
        sess = tf.Session()
        rate, input = read('input.wav')

        transformed_raw = np.array(rfft(input))
        transformed = transformed_raw / transformed_raw.max(axis=0)


        x = tf.placeholder("float", [None, SIZE])
        autoencoder = create(x, DIMS)
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver()
        saver.save(sess, 'model.ckpt')
        train_step = tf.train.GradientDescentOptimizer(3.0).minimize(autoencoder['cost'])


        #output = irfft(filtered)

        #write('output.wav', rate, output)



        for k in range(100):
            # do 1000 training steps
            for i in range(len(transformed)/SIZE):
                    # Our dataset consists of two centers with gaussian noise w/ sigma = 0.1
                    c1 = transformed[i*SIZE:i*SIZE+SIZE]
                    batch = [c1]
                    sess.run(train_step, feed_dict={x: np.array(batch)})
                    if i % 1000 == 1:
                            saver.save(sess, 'model.ckpt')
                            print(i, " cost", sess.run(autoencoder['cost'], feed_dict={x: batch}))
                            print(i, " original", batch[0])
                            print( i, " decoded", sess.run(autoencoder['decoded'], feed_dict={x: batch}))
            print(k)

def deep_gen():
        sess = tf.Session()
        rate, input = read('input.wav')

        print('rate',rate)
        transformed_raw = np.array(rfft(input), dtype=np.float32)
        transformed = transformed_raw / transformed_raw.max(axis=0)

        output = irfft(transformed)
        print(input)
        print(output)

        #write('sanity.wav', rate, np.array(output, dtype=np.float32))

        x = tf.placeholder("float", [None, SIZE])
        autoencoder = create(x, DIMS)
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, 'model.ckpt')





        filtered = np.array([])
        # do 1000 training steps
        for i in range(len(transformed)/SIZE):
		if(i>1000):
		    break;
                # Our dataset consists of two centers with gaussian noise w/ sigma = 0.1
                c1 = transformed[i*SIZE:i*SIZE+SIZE]
                batch = [c1]
                decoded = sess.run(autoencoder['decoded'], feed_dict={x: np.array(batch)})
                ded = decoded.ravel()
                #filtered = np.append(filtered, batch)
                filtered = np.append(filtered,ded)
                if i % 100 == 1:
			print(filtered)
                        print(i, " cost", sess.run(autoencoder['cost'], feed_dict={x: batch}))
                        print(i, " original", batch[0])
                        print( i, " decoded", sess.run(autoencoder['decoded'], feed_dict={x: batch}))

	output = irfft(filtered * transformed_raw.max(axis=0))
	write('output.wav', rate, np.array(output, dtype=np.float32))
                       
if __name__ == '__main__':
        deep_test()
	#deep_gen()


