import tensorflow as tf
import numpy as np
import math
import random

import sys

import os
import glob

from wav import loadfft2, savefft2, sanity

TRAIN_REPEAT=1
SIZE=32
DEPTH=1
#DIMS=[[4096,4096,1],[None,None,2],[None,None,4]]#, [256,256,8], [64,64,16]]
#DIMS=[[1024,1024,1], [512,512,2], [256,256,4]]
DIMS=[[1024,1024,1], [512,512,2], [256,256,4]]
#DIMS=[[256,256,1],[128, 128, 2],[64, 64, 4]]
#DIMS=[[128,128,1],[64, 64, 2], [32,32,4]]
FILTER_SIZE=[]
#SIZE=256

def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')

def create(x):
    prev_layer = x
    prev_height = DIMS[0][0]
    prev_depth=DIMS[0][2]
    hidden = []
    results = {}
    for i, dim in enumerate(DIMS):
        #print(i, dim)
        depth = dim[2]
        height = dim[0]
        fw = fh = 2
        sv = sd = 2
        filterr = tf.Variable(tf.random_normal([fw, fh, prev_depth, depth]))
        print(filterr)
        #print(tf.shape(filterr))
        #print(tf.shape(prev_layer))
        conv = tf.nn.conv2d(prev_layer, filterr, [1, sv, sd, 1], padding='VALID')
        biases = tf.Variable(tf.zeros([depth]))
        #conv = max_pool(conv, 1)
        #conv = tf.nn.dropout(conv, 0.75)
        hidden = tf.nn.relu(conv + biases)
        prev_layer=hidden
        prev_depth = depth
        prev_height =height
        #print(prev_depth)
        results['conv'+str(i)]=conv
    filterr = tf.truncated_normal([8, 8, DEPTH, DEPTH], stddev=0.1)

    deconv_shape = tf.pack([tf.shape(prev_layer)[0], SIZE, SIZE, DEPTH])
    #print('prev_depth', prev_depth)
    #print('prev_layer', tf.shape(prev_layer))
    arranged_prev_layer = tf.depth_to_space(prev_layer, 2)
    #print('shape',tf.shape(arranged_prev_layer)[0])
    conv_transposed = tf.nn.conv2d_transpose(arranged_prev_layer, 
            filterr,
            output_shape=deconv_shape,
            strides=[1,4,4,1],
            padding='SAME'
            )
    prev_layer =conv_transposed
    W = tf.Variable(tf.random_normal([SIZE, SIZE]))
    b = tf.Variable(tf.zeros([SIZE]))
    reshaped = tf.reshape(prev_layer, [-1, W.get_shape().as_list()[0]])
    mat = tf.matmul(reshaped,W)
    output = tf.nn.tanh(mat + b)

    decoded = output
    reconstructed_x = tf.reshape(decoded, [-1, SIZE,SIZE,DEPTH])
    results["decoded"]=reconstructed_x
    results["cost"]= tf.sqrt(tf.reduce_mean(tf.square(x-reconstructed_x)))
    #results['arranged']= arranged_prev_layer
    #results['transposed']= conv_transposed
    return results

def get_input():
    return tf.placeholder("float", [None, SIZE, SIZE, DEPTH], name='x')
def deep_test():
        sess = tf.Session()

        x = get_input()
        autoencoder = create(x)
        #train_step = tf.train.GradientDescentOptimizer(3.0).minimize(autoencoder['cost'])
        train_step = tf.train.AdamOptimizer(1e-5).minimize(autoencoder['cost'])
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver()
        saver.save(sess, 'modelconv.ckpt')

        tf.train.write_graph(sess.graph_def, 'log', 'modelcon.pbtxt', False)

        #output = irfft(filtered)
        i=0
        #write('output.wav', rate, output)
        for trains in range(TRAIN_REPEAT):
            for file in glob.glob('training/*.wav'):
                i+=1
                learn(file, sess, train_step, x,i, autoencoder, saver)
        

# given fft, return back a stack 3-dimensional SIZExSIZE squares
def collect_input(data, dims):
    slice_size = dims[0]*dims[1]*dims[2]
    length = len(data)
    # discard extra info
    relevant = int(length/slice_size)*slice_size
    arr= np.array(data[0:relevant])

    reshaped =  arr.reshape((-1, SIZE, SIZE, DEPTH))
    return reshaped
def learn(filename, sess, train_step, x, k, autoencoder, saver):
        print("Loading "+filename)
        wavobj = loadfft2(filename)
        transformed = wavobj['transformed']
        transformed_raw = wavobj['raw']
        rate = wavobj['rate']

        input_squares = collect_input(transformed, [SIZE, SIZE, DEPTH])
        #print(input_squares)
        print("Running " + filename + str(np.shape(input_squares)[0]))
        sess.run(train_step, feed_dict={x: input_squares})
        print(k,filename, " cost", sess.run(autoencoder['cost'], feed_dict={x: input_squares}))
        print("Finished " + filename)
        #print(i, " original", batch[0])
        #print( " decoded", sess.run(autoencoder['conv2'], feed_dict={x: input_squares}))
        saver.save(sess, 'modelconv.ckpt')

def deep_gen():
        sess = tf.Session()
        wavobj = loadfft2('input.wav')
        sanity(wavobj)
        transformed = wavobj['transformed']

        x = get_input()
        autoencoder = create(x)
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, 'modelconv.ckpt')



        batch = collect_input(transformed, [SIZE, SIZE, DEPTH])
        filtered = np.array([])

        decoded = sess.run(autoencoder['decoded'], feed_dict={x: np.array(batch)})
        #decoded = sess.run(autoencoder['decoded'], feed_dict={x: np.array(np.random.normal(0,1,[len(batch), 8192]))})
        #filtered = np.append(filtered, batch)
        filtered = np.append(filtered,decoded.reshape([-1]))
        #print(i, " cost", sess.run(autoencoder['cost'], feed_dict={x: batch}))
        #print(i, " original", batch[0])
        #print( i, " decoded", sess.run(autoencoder['decoded'], feed_dict={x: batch}))
        savefft2('output.wav', wavobj, filtered)
                       
if __name__ == '__main__':
    if(sys.argv[1] == 'train'):
        print("Train")
        deep_test()
    else:
        print("Generate")
        deep_gen()


