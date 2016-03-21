import tensorflow as tf
import numpy as np
import math
import random

import sys

import os
import glob

from wav import loadfft2, savefft2, sanity


TRAIN_REPEAT=1
SIZE=64
DEPTH=1
LEARNING_RATE = 7e-5
#BATCH_SIZE=349
layers = [
    {
        'type':'conv1d',
        'filter':[2, 1, DEPTH*2],
        'stride':[1,2,2,1],
        'padding':"SAME"
    },
    {
        'type':'conv1d',
        'filter':[2, 2, DEPTH*4],
        'stride':[1,2,2,1],
        'padding':"SAME"
    },
    {
        'type':'conv1d',
        'filter':[2, 4, DEPTH*8],
        'stride':[1,2,2,1],
        'padding':"SAME"
    },
    #{
    #    'type':'conv1d',
    #    'filter':[2, 32, DEPTH*64],
    #    'stride':[1,2,2,1],
    #    'padding':"SAME"
    #},

>>>>>>> [fix] actually emitting sound now
    #{
    #    'type': 'autoencoder',
    #    'output_dim': 32
    #},
    #{
    #    'type': 'autoencoder',
    #    'output_dim': 24
    #},
    {
        'type': 'feed_forward_nn',
    },
    #{
    #    'type': 'conv1d',
    #    'filter':[2, 1, DEPTH*2],
    #    'stride':[1,2,2,1],
    #    'padding':"SAME"
    #},



]

def feed_forward_nn(input, layer_def, nextMethod):
    input_dim = int(input.get_shape()[1])
    print("-- Begin feed forward nn", input_dim, input.get_shape())
    # Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
    W = tf.Variable(tf.random_uniform([input_dim, input_dim], -1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim)))

    # Initialize b to zero
    b = tf.Variable(tf.zeros([input_dim]))

    output = tf.nn.tanh(tf.matmul(tf.reshape(input, [-1,input_dim]),W) + b)

 
    return nextMethod(output)

def conv1d(input, layer_def, nextMethod):
    print('---')
    filters = layer_def['filter']
    filter_normal = tf.Variable(tf.random_normal([2, filters[0], filters[1], filters[2]]))
    padding = layer_def['padding']
    stride =layer_def['stride']
    print('in shape', input.get_shape())

    input_t = tf.transpose(tf.reshape(input, [-1, int(input.get_shape()[1]), 1]), [0,1,2])
    print('in_t shape', input_t.get_shape())
    #filter_t = tf.transpose(filter_normal, [1,0,2])
    expand_input = tf.expand_dims(input_t, 1)
    expand_filter = filter_normal
    #expand_filter = tf.expand_dims(filter_t, 0)
    #add_layer=tf.tile(expand_input, [1,2,1,1])
    add_layer=tf.zeros_like(expand_input)
    print('shapes:')
    print('add layer', add_layer.get_shape())
    print('expand input', expand_input.get_shape())
    print('expand filter', expand_filter.get_shape())
    #expand_input_add = tf.concat(1, (expand_input, add_layer))
    expand_input_add = add_layer

    print('expand input add', expand_input_add.get_shape())
    #print('expand filter', expand_filter.get_shape())
    print("stride", stride)
    conv = tf.nn.conv2d(expand_input_add, expand_filter, stride, padding=padding)
    print("conv:", conv.get_shape())
    #conv = max_pool(conv, 2)
    #conv = tf.nn.dropout(conv, 0.75)
    #slice= tf.slice(conv, [0,0,0,0], [-1, 1, -1, -1])
    #squeeze = tf.squeeze(slice, squeeze_dims=[1])
    squeeze = tf.squeeze(conv, squeeze_dims=[1])
    print("Squeeze:", squeeze.get_shape())

    biases = tf.Variable(tf.zeros([squeeze.get_shape()[-1]]))
    relu = tf.nn.relu(squeeze + biases)
    hidden = tf.maximum(0.2*relu, relu)

    return nextMethod(hidden)

def conv1d_transpose(input, layer_def):
    print("--- Begin conv1d_transpose")
    print("input ", input.get_shape()) 
    padding = layer_def['padding']
    stride =layer_def['stride']
    filters = layer_def['filter']
    output_shape = layer_def['output_shape']

    input_t = tf.transpose(input, [0,1,2])

    expand_input = input_t

    #add_layer=tf.zeros_like(expand_input)
    print('shapes:')
    #print('add layer', add_layer.get_shape())
    print('expand input', expand_input.get_shape())
    #expand_input_add = tf.concat(1, (expand_input, add_layer))
    expand_input_add = expand_input
    expand_input_add = tf.expand_dims(input_t, 1)
    add_layer=tf.zeros_like(expand_input_add)
    expand_input_add = tf.concat(1, (expand_input_add, add_layer))
    #expand_input_add = tf.depth_to_space(expand_input_add, 2)
    print('expand input add', expand_input_add.get_shape())
    #expand_input_add = tf.concat(1, (expand_input, expand_input_add))
    filter_normal = tf.Variable(tf.random_normal([filters[0], filters[1],output_shape[-1],int(expand_input_add.get_shape()[3])]))
    expand_filter = filter_normal
    print('expand filter', expand_filter.get_shape())


    output_shape_pack = [int(input.get_shape()[0]), filters[1], output_shape[1], output_shape[2]]
    print("output shape", output_shape, output_shape_pack)

    #print('expand input add', expand_input_add.get_shape())

    conv_transposed = tf.nn.conv2d_transpose(expand_input_add, 
            expand_filter,
            output_shape=output_shape_pack,
            strides=stride,
            padding=padding
            )
    #print("conv_transposed", conv_transposed)
    #squeeze = tf.squeeze(conv_transposed, squeeze_dims=[1])
    #squeeze = conv_transposed
    #slice= tf.slice(conv_transposed, [0,0,0,0], [BATCH_SIZE, 1, output_shape[1], output_shape[2]])
    slice = conv_transposed
    #print("Sslice:", slice.get_shape())
    biases = tf.Variable(tf.zeros([slice.get_shape()[-1]]))
    hidden = tf.nn.relu(conv_transposed + biases)
    return hidden

def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')


def autoencoder(input, layer_def, nextMethod):
    input_dim = int(input.get_shape()[1])
    output_dim = layer_def['output_dim']
    print("-- Begin autoencoder", input_dim, input.get_shape())
    W = tf.Variable(tf.random_uniform([input_dim, output_dim], -1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim)))
    # Initialize b to zero
    b = tf.Variable(tf.zeros([output_dim]))
    output = tf.nn.tanh(tf.matmul(tf.reshape(input, [-1,input_dim]),W) + b)

    print("autoencoder", output.get_shape())
    inner_layer = nextMethod(output)
    inner_layer = tf.reshape(inner_layer, [-1, output_dim])

    W2 = tf.transpose(W)
    b2 = tf.Variable(tf.zeros([input_dim]))
    print("autoencoder 2", inner_layer.get_shape(), W2.get_shape(), b2)
    return tf.nn.tanh(tf.matmul(inner_layer,W2) + b2)



layer_index=0
def create(x):
    ops = {
        'conv1d':conv1d,
        'conv1d_transpose':conv1d_transpose,
        'feed_forward_nn':feed_forward_nn,
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

    decoded = ops[layers[0]['type']](x, layers[0], nextMethod)
    #print("Reshaping ", decoded.get_shape(), " to ", [BATCH_SIZE, SIZE, DEPTH]) 
    #reconstructed_x = tf.reshape(decoded, [BATCH_SIZE, SIZE,DEPTH])
    reconstructed_x = tf.reshape(decoded, [-1, SIZE,DEPTH])
    print("Completed reshaping")
    #results["decoded"]=reconstructed_x
    results['decoded']=tf.reshape(decoded, [-1])
    results["cost"]= tf.sqrt(tf.reduce_mean(tf.square(x-reconstructed_x)))
    #results['arranged']= arranged_prev_layer
    #results['transposed']= conv_transposed
    return results

def get_input():
    return tf.placeholder("float", [None, SIZE, DEPTH], name='x')
def deep_test():
        sess = tf.Session()

        x = get_input()
        autoencoder = create(x)
        #train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(autoencoder['cost'])
        train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(autoencoder['cost'])
        #train_step = None
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver()
        saver.save(sess, 'model1d.ckpt')

        tf.train.write_graph(sess.graph_def, 'log', 'modelcon.pbtxt', False)

        #output = irfft(filtered)
        i=0
        #write('output.wav', rate, output)
        for trains in range(TRAIN_REPEAT):
            for file in glob.glob('training/*.wav'):
                i+=1
                learn(file, sess, train_step, x,i, autoencoder, saver)
        saver.save(sess, 'model1.ckpt')
        

def collect_input(data, dims):
    slice_size = dims[0]*dims[1]
    length = len(data)
    # discard extra info
    arr= np.array(data[0:int(length/SIZE)*SIZE])

    reshaped =  arr.reshape((-1, dims[0], dims[1]))
    return reshaped
def learn(filename, sess, train_step, x, k, autoencoder, saver):
        #print("Loading "+filename)
        wavobj = loadfft2(filename)
        transformed = wavobj['transformed']
        transformed_raw = wavobj['raw']
        rate = wavobj['rate']

        input_squares = collect_input(transformed, [SIZE, DEPTH])
        print("wav size", len(input_squares))
        #print(input_squares)
        #print("Running " + filename + str(np.shape(input_squares)[0]))
        sess.run(train_step, feed_dict={x: input_squares})
        print(k,filename, " cost", sess.run(autoencoder['cost'], feed_dict={x: input_squares}))
        #print("Finished " + filename)
        #print(i, " original", batch[0])
        #print( " decoded", sess.run(autoencoder['conv2'], feed_dict={x: input_squares}))

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
        saver.restore(sess, 'model1d.ckpt')



        batch = collect_input(transformed, [SIZE, DEPTH])
        #print('np.shape(bat', np.shape(batch))
        batch_copy = collect_input(transformed, [SIZE, DEPTH])

        decoded = sess.run(autoencoder['decoded'], feed_dict={x: np.array(batch)})
        #decoded = sess.run(autoencoder['decoded'], feed_dict={x: np.array(np.random.normal(0,1,[len(batch), 8192]))})
        print(decoded)
        #filtered = np.append(filtered, batch)
        #res = np.transpose(batch_copy, [0,1,2]).reshape([-1])
        #sanity({"transformed":res, "rate": wavobj["rate"], "raw": wavobj['raw']})
        #print(i, " cost", sess.run(autoencoder['cost'], feed_dict={x: batch}))
        #print(i, " original", batch[0])
        #print( i, " decoded", sess.run(autoencoder['decoded'], feed_dict={x: batch}))
        savefft2('output2.wav', wavobj, filtered)
                       
if __name__ == '__main__':
    if(sys.argv[1] == 'train'):
        print("Train")
        deep_test()
    else:
        print("Generate")
        deep_gen()


