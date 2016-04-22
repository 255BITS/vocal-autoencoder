import tensorflow as tf
import numpy as np
import math
import random

import sys

import os
import glob

from wav import loadfft2, savefft2, get_wav, save_wav


from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq


import matplotlib.pyplot as plt



PREDICT_ROLL=8
TRAIN_REPEAT=100000
SIZE=8192//16
LEARNING_RATE = tf.Variable(2e-3, trainable=False)
BATCH_SIZE=2048
WAVELETS=SIZE//2
CHANNELS=1

NUM_MIXTURES=24

SAVE_DIR='save'
#BATCH_SIZE=349
layers = [
    {
        'type': 'autoencoder',
        'wavelets': WAVELETS
    },
    #{
    #    'type': 'feedforward',
    #    'wavelets': WAVELETS
    #}

]
def feed_forward_nn(input, layer_def, nextMethod):
    input_dim = int(input.get_shape()[1])
    print("-- Begin feed forward nn", input_dim, input.get_shape())
    W = tf.Variable(tf.random_normal([input_dim, input_dim]))

    # Initialize b to zero
    b = tf.Variable(tf.zeros([input_dim]))

    output = tf.nn.tanh(tf.matmul(tf.reshape(input, [-1,input_dim]),W) + b)


    return nextMethod(output)


def wnn_decode(output, output_dim):
    wavelets = output.get_shape()[1]
    with tf.variable_scope('wnn_decode'):
        summer = tf.get_variable('summer', [1, output_dim], initializer= tf.random_uniform_initializer())
        w = tf.get_variable('w', [wavelets, output_dim])
        output = tf.matmul(output, w) + summer
        return output

def wnn_encode(input, wavelets):
    # this is the depth of the tree
    dim_in = input.get_shape()[1]
    def initial_dt_tree(a, b, n):
        if(n <=1):
            return []
        t = (a+b)/2
        lam = (b-a)/2
        tree_left = initial_dt_tree(a, t, n-1)
        tree_right = initial_dt_tree(t, b, n-1)
        concat = [[t, lam]] 
        if(len(tree_left) > 0):
            concat += tree_left
        if(len(tree_right) > 0):
            concat+=tree_right
        return concat
    def mother(input):
        #return (-input)*(-tf.exp(tf.square(input)))
        #mexican hat
        square = tf.square(input)
        return (1-square)*tf.exp(-square/2)

        #mortlet
        #return 0.75112554446494 * tf.cos(input * 5.336446256636997) * tf.exp((-tf.square(input)) / 2)
    with tf.variable_scope('wnn_encode'):
        full_resolutions = math.log(wavelets)/math.log(2)
        tree = initial_dt_tree(-1,1, full_resolutions)
        print(tree)
        d_c = [leaf[1] for leaf in tree]
        t_c = [leaf[0] for leaf in tree]
        print('yer vals', np.shape(d_c), np.shape(t_c))
        translation = tf.get_variable('translation', [1, wavelets], initializer = tf.constant_initializer(t_c))
        dilation = tf.get_variable('dilation', [1, wavelets], initializer = tf.constant_initializer(d_c))
        w = tf.get_variable('w', [dim_in,wavelets])
        input_proj = (tf.matmul(input, w) - translation)/dilation
        return mother(input_proj)
def autoencoder(input, layer_def, nextMethod):
    input_dim = int(input.get_shape()[1])
    output_dim = input_dim
    wavelets = layer_def['wavelets']
    print("-- Begin autoencoder", input_dim, input.get_shape())
    output = wnn_encode(input, wavelets)

    output = nextMethod(output)
    output = wnn_decode(output, output_dim)

    # Such as output = build_wavelon(resolution)
    # where build_wavelon is recursively building the input array, one per resolution(1 main, 2*N smaller, 2**M smaller, etc)
    # then, make sure to introoduce the right vars and abstraction into wnn_encode
    #  multiply by translations, which is one translation per wavelon(per batch)

    # add bias term at the end
    return output



layer_index=0
def create(x):
    ops = {
        'autoencoder':autoencoder,
        'feedforward':feed_forward_nn,
    }


    results = {}
    def nextMethod(current_layer):
        global layer_index
        if(len(layers) == layer_index+1):
            return current_layer
        layer_index += 1
        layer_def = layers[layer_index]
        return ops[layer_def['type']](current_layer, layer_def, nextMethod)

    flat_x = tf.reshape(x, [BATCH_SIZE, -1])
    decoded = ops[layers[0]['type']](flat_x, layers[0], nextMethod)


    # below is where we need to do MDN splitting of distribution params
    def get_mixture_coef(output):
      # returns the tf slices containing mdn dist params
      # ie, eq 18 -> 23 of http://arxiv.org/abs/1308.0850
      z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(1, 6, z)

      # process output z's into MDN paramters

      # softmax all the pi's:
      max_pi = tf.reduce_max(z_pi, 1, keep_dims=True)
      z_pi = tf.sub(z_pi, max_pi)
      z_pi = tf.exp(z_pi)
      normalize_pi = tf.inv(tf.reduce_sum(z_pi, 1, keep_dims=True))
      z_pi = tf.mul(normalize_pi, z_pi)

      # exponentiate the sigmas and also make corr between -1 and 1.
      z_sigma1 = tf.exp(z_sigma1)
      z_sigma2 = tf.exp(z_sigma2)
      z_corr = tf.tanh(z_corr)
      return [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen]


    def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
      # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
      norm1 = tf.sub(x1, mu1)
      norm2 = tf.sub(x2, mu2)
      s1s2 = tf.mul(s1, s2)
      z = tf.square(tf.div(norm1, s1))+tf.square(tf.div(norm2, s2))-2*tf.div(tf.mul(rho, tf.mul(norm1, norm2)), s1s2)
      negRho = 1-tf.square(rho)
      result = tf.exp(tf.div(-z,2*negRho))
      denom = 2*np.pi*tf.mul(s1s2, tf.sqrt(negRho))
      result = tf.div(result, denom)
      return result

    def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, x1_data, x2_data, pen_data):
      result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
      # implementing eq # 26 of http://arxiv.org/abs/1308.0850
      epsilon = 1e-20
      result1 = tf.mul(result0, z_pi)
      result1 = tf.reduce_sum(result1, 1, keep_dims=True)
      result1 = -tf.log(tf.maximum(result1, 1e-20)) # at the beginning, some errors are exactly zero.
      result_shape = tf.reduce_mean(result1)

    reconstructed_x = tf.reshape(decoded, [BATCH_SIZE, CHANNELS,SIZE])
    results['decoded']=tf.reshape(decoded, [BATCH_SIZE, CHANNELS,SIZE])
    results["cost"]= tf.sqrt(tf.reduce_mean(tf.square(reconstructed_x-x)))

    with tf.variable_scope('wnn_encode'):
        tf.get_variable_scope().reuse_variables()
        translation = tf.get_variable('translation', [1, WAVELETS])
        dilation = tf.get_variable('dilation', [1, WAVELETS])
        results['translation' ] = translation
        results['dilation'] = dilation
        return results

def get_input():
    return tf.placeholder("float", [BATCH_SIZE, CHANNELS, SIZE], name='x')
def deep_test():
        sess = tf.Session()

        x = get_input()
        autoencoder = create(x)
        grad_clip = 5
        #train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(autoencoder['cost'])
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(autoencoder['cost'])
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.d_loss, self.d_vars),
                                grad_clip)
        train_step = train_step.apply_gradients(zip(grads, tf.trainable_variables()))
        #train_step = None
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver(tf.all_variables())
        saver.save(sess, SAVE_DIR+'/modellstm.ckpt', global_step=0)

        tf.train.write_graph(sess.graph_def, 'log', 'modellstm.pbtxt', False)
        

        #output = irfft(filtered)
        i=0
        #write('output.wav', rate, output)
        for trains in range(TRAIN_REPEAT):
            print("Starting epoch", trains)
            for file in glob.glob('training/*.wav'):
                i+=1
                learn(file, sess, train_step, x,i, autoencoder, saver)
                if(i%1000==1):
                    i=i
                    print("Saving")
                    saver.save(sess, SAVE_DIR+"/modellstm.ckpt", global_step=i+1)
        

def collect_input(data, dims):
    length = len(data)
    # discard extra info
    arr= np.array(data[0:int(length/dims[0]/BATCH_SIZE)*dims[0]*BATCH_SIZE])

    reshaped =  arr.reshape((-1, BATCH_SIZE, dims[0]))
    return reshaped

def learn(filename, sess, train_step, x, k, autoencoder, saver):
        #print("Loading "+filename)
        # not really loading fft
        wavobj = get_wav(filename)
        transformed_raw = wavobj['data']
        rate = wavobj['rate']

        input_squares = collect_input(transformed_raw, [SIZE*CHANNELS])

        #print(input_squares)
        #print("Running " + filename + str(np.shape(input_squares)[0]))
        for square in input_squares:
            square = np.reshape(square, [BATCH_SIZE, SIZE,CHANNELS])
            square = np.swapaxes(square, 1, 2)
            to_plot = np.reshape(square[0,0,:], [-1])
            #print(to_plot)
            plt.clf()
            plt.plot(to_plot)

            plt.xlim([0, SIZE])
            plt.ylabel("Amplitude")
            plt.xlabel("Time")
            ## set the title  
            plt.title("batch")
            #plt.show()
            plt.savefig('visualize/input-'+str(k)+'.png')
            _, cost, translation, dilation = sess.run([train_step,autoencoder['cost'], autoencoder['translation'], autoencoder['dilation']], feed_dict={x: square})

            print(" cost", cost,np.mean(translation), np.mean(dilation), k, filename )
        #print("Finished " + filename)
        #print(i, " original", batch[0])
        #print( " decoded", sess.run(autoencoder['conv2'], feed_dict={x: input_squares}))

def deep_gen():
    with tf.Session() as sess:
        wavobj = get_wav('input.wav')
        transformed = wavobj['data']
        save_wav(wavobj, 'sanity.wav')
        batches = collect_input(transformed, [SIZE*CHANNELS])

        x = get_input()
        autoencoder = create(x)
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(SAVE_DIR)
        if(checkpoint and checkpoint.model_checkpoint_path):
            saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            print("ERROR: No checkpoint found")
            exit(-1)




        all_out=[]
        for batch in batches:
            print(np.shape(batch), BATCH_SIZE, SIZE, CHANNELS)
            batch = np.reshape(batch, [BATCH_SIZE, SIZE, CHANNELS])
            batch =np.swapaxes(batch, 1, 2)
            decoded = sess.run(autoencoder['decoded'], feed_dict={x: np.array(batch)})
            all_out.append(np.swapaxes(decoded, 1, CHANNELS))
        all_out = np.array(all_out)
        wavobj['data']=np.reshape(all_out, [-1, CHANNELS])
        print('saving to output2.wav', np.min(all_out), np.max(all_out))
        save_wav(wavobj, 'output2.wav')
                       
if __name__ == '__main__':
    if(sys.argv[1] == 'train'):
        print("Train")
        deep_test()
    else:
        print("Generate")
        deep_gen()


