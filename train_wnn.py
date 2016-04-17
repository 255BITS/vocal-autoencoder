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


PREDICT_ROLL=8
TRAIN_REPEAT=100000
SIZE=8192//4
LEARNING_RATE = tf.Variable(2e-4, trainable=False)
BATCH_SIZE=512

SAVE_DIR='save'
#BATCH_SIZE=349
layers = [
    {
        'type': 'autoencoder',
        'wavelets': SIZE//2
    },

]

def wnn_decode(output, output_dim):
    wavelets = output.get_shape()[1]
    with tf.variable_scope('wnn_decode'):
        summer = tf.get_variable('summer', [1, output_dim], initializer= tf.random_uniform_initializer())
        w = tf.get_variable('w', [wavelets, output_dim])
        output = tf.matmul(output, w) + summer
        return output

def wnn_encode(input, wavelets):
    # this is the depth of the tree
    #full_resolutions = math.log(wavelets)/math.log(2)
    dim_in = input.get_shape()[1]
    def gaus(input, translation, dilation):
        input = (input - translation)/dilation
        #return (-input)*(-tf.exp(tf.square(input)))
        #mexican hat
        square = tf.square(input)
        return (1-square)*tf.exp(-square/2)
    with tf.variable_scope('wnn_encode'):
        translation = tf.get_variable('translation', [1, wavelets], initializer = tf.random_uniform_initializer())
        dilation = tf.get_variable('dilation', [1, wavelets], initializer = tf.random_uniform_initializer())
        w = tf.get_variable('w', [dim_in,wavelets])
        input_proj = tf.matmul(input, w)
        return gaus(input_proj, translation, dilation)
def autoencoder(input, layer_def, nextMethod):
    input_dim = int(input.get_shape()[1])
    output_dim = input_dim
    wavelets = layer_def['wavelets']
    print("-- Begin autoencoder", input_dim, input.get_shape())
    output = wnn_encode(input, wavelets)

    output = nextMethod(output)
    output = wnn_decode(input, output_dim)

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
    reconstructed_x = tf.reshape(decoded, [BATCH_SIZE, SIZE,2])
    results['decoded']=tf.reshape(decoded, [BATCH_SIZE, SIZE,2])
    results["cost"]= tf.sqrt(tf.reduce_mean(tf.square(reconstructed_x-x)))

    return results

def get_input():
    return tf.placeholder("float", [BATCH_SIZE, SIZE,2], name='x')
def deep_test():
        sess = tf.Session()

        x = get_input()
        autoencoder = create(x)
        #train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(autoencoder['cost'])
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(autoencoder['cost'])
        #train_step = None
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver(tf.all_variables())
        saver.save(sess, SAVE_DIR+'/modellstm3.ckpt', global_step=0)

        tf.train.write_graph(sess.graph_def, 'log', 'modellstm3.pbtxt', False)

        #output = irfft(filtered)
        i=0
        #write('output.wav', rate, output)
        for trains in range(TRAIN_REPEAT):
            print("Starting epoch", trains)
            for file in glob.glob('training/*.wav'):
                i+=1
                learn(file, sess, train_step, x,i, autoencoder, saver)
                if(i%100==1):
                    i=i
                    print("Saving")
                    saver.save(sess, SAVE_DIR+"/modellstm3.ckpt", global_step=i+1)
        

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

        input_squares = collect_input(transformed_raw, [SIZE*2])

        #print(input_squares)
        #print("Running " + filename + str(np.shape(input_squares)[0]))
        for square in input_squares:
            square = np.reshape(square, [BATCH_SIZE, SIZE,2])
            _, cost = sess.run([train_step,autoencoder['cost']], feed_dict={x: square})

            print(" cost", cost,k,filename )
        #print("Finished " + filename)
        #print(i, " original", batch[0])
        #print( " decoded", sess.run(autoencoder['conv2'], feed_dict={x: input_squares}))

def deep_gen():
    with tf.Session() as sess:
        wavobj = get_wav('input.wav')
        transformed = wavobj['data']
        batches = collect_input(transformed, [SIZE*2])

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
            batch = np.reshape(batch, [BATCH_SIZE, SIZE, 2])
            decoded = sess.run(autoencoder['decoded'], feed_dict={x: np.array(batch)})
            all_out.append(decoded)
        wavobj['data']=np.reshape(all_out, [-1, SIZE, 2])
        print('saving to output2.wav', decoded)
        save_wav(wavobj, 'output2.wav')
                       
if __name__ == '__main__':
    if(sys.argv[1] == 'train'):
        print("Train")
        deep_test()
    else:
        print("Generate")
        deep_gen()


