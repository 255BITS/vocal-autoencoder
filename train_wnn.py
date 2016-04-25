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
SIZE=441*4
LEARNING_RATE = tf.Variable(2e-3, trainable=False)
BATCH_SIZE=4096
WAVELETS=64*4
CHANNELS = 1

PLOT_EVERY = 50
SAVE_EVERY = 200

SAVE_DIR='save'
#BATCH_SIZE=349
layers = [
        {
            'type': 'autoencoder',
            'wavelets': WAVELETS,
            'name': 'l1'
            },
        #{
        #    'type': 'autoencoder',
        #    'wavelets': WAVELETS//2,
        #    'name': 'l2'
        #},
        #{
        #    'type': 'autoencoder',
        #    'wavelets': WAVELETS//4,
        #    'name': 'l3'
        #},
        #{
        #    'type': 'autoencoder',
        #    'wavelets': WAVELETS//2,
        #    'name': 'l2'
        #},
        #{
        #    'type': 'feedforward',
        #    'wavelets': WAVELETS
        #}

        ]

def wnn_decode(output, output_dim, name):
    wavelets = output.get_shape()[1]
    with tf.variable_scope('wnn_encode_'+name):
        tf.get_variable_scope().reuse_variables()
        w = tf.get_variable('w', [output_dim, wavelets])
        w = tf.transpose(w)
    with tf.variable_scope('wnn_decode_'+name):
        summer = tf.get_variable('summer', [output_dim], initializer= tf.constant_initializer(0))
        output = tf.nn.xw_plus_b(output, w, summer)
        return output



def wnn_encode(input, wavelets, name):
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
        #square = tf.square(input)
        #start = tf.get_variable("mothersub", 1, initializer=tf.constant_initializer(1.))
        #div = tf.get_variable("motherdiv", 1, initializer=tf.constant_initializer(2.))
        #return (start-square)*tf.exp(-square/div)

        #mortlet
        return 0.75112554446494 * tf.cos(input * 5.336446256636997) * tf.exp((-tf.square(input)) / 2)
    with tf.variable_scope('wnn_encode_'+name):
        full_resolutions = math.log(wavelets*2)/math.log(2)
        tree = initial_dt_tree(-1,1, full_resolutions)
        print(tree)
        d_c = [(leaf[1]) for leaf in tree]
        d_c.append(0.01)
        t_c = [leaf[0] for leaf in tree]
        t_c.append(0.01)
        print('yer vals', len(d_c), len(t_c))
        t_c = np.tile(t_c,BATCH_SIZE)
        d_c = np.tile(d_c,BATCH_SIZE)
        print('-tc',np.shape(t_c))
        print('-dc',d_c)
        translation = tf.reshape(tf.constant(t_c, dtype=tf.float32), [BATCH_SIZE, WAVELETS])
        dilation = tf.reshape(tf.constant(d_c, dtype=tf.float32), [BATCH_SIZE, WAVELETS])
        translation = tf.get_variable('translation', [BATCH_SIZE, wavelets], initializer = tf.constant_initializer(t_c))
        dilation = tf.get_variable('dilation', [BATCH_SIZE, wavelets], initializer = tf.constant_initializer(d_c))
        #w = tf.get_variable('w', [dim_in,wavelets], initializer=tf.constant_initializer(0.001), trainable=False)
        w = tf.get_variable('w', [dim_in,wavelets], initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
        #w = tf.ones([dim_in, wavelets])
        input_proj = tf.mul(tf.sub(tf.matmul(input, w), translation),dilation)
        return mother(input_proj)


def autoencoder(input, layer_def, nextMethod):
    input_dim = int(input.get_shape()[1])
    output_dim = input_dim
    wavelets = layer_def['wavelets']
    name = layer_def['name']
    print("-- Begin autoencoder", input_dim, input.get_shape())
    output = wnn_encode(input, wavelets, name)

    output = nextMethod(output)
    output = wnn_decode(output, output_dim, name)

    # Such as output = build_wavelon(resolution)
    # where build_wavelon is recursively building the input array, one per resolution(1 main, 2*N smaller, 2**M smaller, etc)
    # then, make sure to introoduce the right vars and abstraction into wnn_encode
    #  multiply by translations, which is one translation per wavelon(per batch)

    # add bias term at the end
    return output



layer_index=0
def create(x):
    ops = {
        'autoencoder':autoencoder
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
    reconstructed_x = tf.reshape(decoded, [BATCH_SIZE, CHANNELS,SIZE])
    results['decoded']=tf.reshape(decoded, [BATCH_SIZE, CHANNELS,SIZE])
    results["cost"]= tf.sqrt(tf.reduce_mean(tf.square(reconstructed_x-x)))
    return results


def get_input():
    return tf.placeholder("float", [BATCH_SIZE, CHANNELS, SIZE], name='x')
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
        j=0
        #write('output.wav', rate, output)
        for trains in range(TRAIN_REPEAT):
            print("Starting epoch", trains)
            for file in glob.glob('training/*.wav'):
                i+=1
                k = learn(file, sess, train_step, x,j, autoencoder, saver)
                j+= k
                if(i%SAVE_EVERY==1):
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

        input_squares = collect_input(transformed_raw, [SIZE*CHANNELS])

        #print(input_squares)
        #print("Running " + filename + str(np.shape(input_squares)[0]))
        
        i=0
        for square in input_squares:
            square = np.reshape(square, [BATCH_SIZE, SIZE,CHANNELS])
            square = np.swapaxes(square, 1, 2)
            _, cost, decoded = sess.run([train_step,autoencoder['cost'], autoencoder['decoded']], feed_dict={x: square})
            i+=1
            if((i+k) % PLOT_EVERY == 3):
                to_plot = np.reshape(square[0,0,:], [-1])
                #print(to_plot)
                plt.clf()
                plt.plot(to_plot)

                plt.xlim([0, SIZE])
                plt.ylim([-2, 2])
                plt.ylabel("Amplitude")
                plt.xlabel("Time")
                ## set the title  
                plt.title("batch")
                #plt.show()
                plt.plot(np.reshape(decoded[0,0,:], [-1]))
                plt.savefig('visualize/input-'+str(k+i)+'.png')

            print(" cost", cost, k+i, filename )
        #print("Finished " + filename)
        #print(i, " original", batch[0])
        #print( " decoded", sess.run(autoencoder['conv2'], feed_dict={x: input_squares}))
        return i

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
            batch = np.reshape(batch, [BATCH_SIZE, SIZE, CHANNELS])
            batch =np.swapaxes(batch, 1, 2)
            decoded = sess.run(autoencoder['decoded'], feed_dict={x: np.array(batch)})
            all_out.append(np.swapaxes(decoded, 1, 2))
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


