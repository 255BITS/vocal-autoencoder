import tensorflow as tf
import numpy as np
import math
import random

import sys

import os
import glob

from wav import loadfft2, savefft2, sanity


from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq


PREDICT_ROLL=8
TRAIN_REPEAT=5
SIZE=441
DEPTH=1
LEARNING_RATE = tf.Variable(1e-3, trainable=False)

SAVE_DIR='save'
#BATCH_SIZE=349
layers = [
    #{
    #    'type': 'autoencoder',
    #    'output_dim': 220
    #},
    {
        'type': 'autoencoder',
        'output_dim': 80
    },

    #{ 'type': 'lstm',
    #    'size': 128
    #    }
    #{
    #    'type':'conv1d',
    #    'filter':[8, 1, DEPTH*2],
    #    'stride':[1,2,2,1],
    #    'padding':"SAME"
    #   },
    #{
    #    'type':'conv1d',
    #    'filter':[16, 2, DEPTH*4],
    #    'stride':[1,2,2,1],
    #    'padding':"SAME"
    #   },




#
#
]

def lstm(input, layer_def, next_method):
    return None#next_method(logits)
    

def reshape(input, layer_def, nextMethod):
    reshape = tf.reshape(input, layer_def['output_dims'])
    return nextMethod(reshape)

def feed_forward_nn(input, layer_def, nextMethod):
    input_dim = int(input.get_shape()[1])
    print("-- Begin feed forward nn", input_dim, input.get_shape())
    W = tf.Variable(tf.random_normal([input_dim, input_dim]))

    # Initialize b to zero
    b = tf.Variable(tf.zeros([input_dim]))

    output = tf.nn.tanh(tf.matmul(tf.reshape(input, [-1,input_dim]),W) + b)


    return nextMethod(output)

def conv1d(input, layer_def, nextMethod):
    if(len(input.get_shape())==2):
        input = tf.expand_dims(input, 2)

    print('---')
    filters = layer_def['filter']
    filter_normal = tf.Variable(tf.random_normal([2, filters[0], filters[1], filters[2]]))
    padding = layer_def['padding']
    stride =layer_def['stride']
    print('in shape', input.get_shape())

    #input_t = tf.transpose(tf.reshape(input, [-1, int(input.get_shape()[1]), 1]), [0,1,2])
    #print('in_t shape', input_t.get_shape())
    #filter_t = tf.transpose(filter_normal, [1,0,2])
    expand_input = tf.expand_dims(input, 1)
    expand_filter = filter_normal
    #expand_filter = tf.expand_dims(filter_t, 0)
    #add_layer=tf.tile(expand_input, [1,2,1,1])
    add_layer=tf.zeros_like(expand_input)
    print('shapes:')
    print('add layer', add_layer.get_shape())
    print('expand input', expand_input.get_shape())
    print('expand filter', expand_filter.get_shape())
    expand_input_add = tf.concat(1, (expand_input, add_layer))
    #expand_input_add = add_layer

    print('expand input add', expand_input_add.get_shape())
    #print('expand filter', expand_filter.get_shape())
    print("stride", stride)
    conv = tf.nn.conv2d(expand_input_add, expand_filter, stride, padding=padding)
    print("conv:", conv.get_shape())
    #conv = max_pool(conv, 1)
    conv = tf.nn.dropout(conv, 0.9)
    #slice= tf.slice(conv, [0,0,0,0], [-1, 1, -1, -1])
    #squeeze = tf.squeeze(slice, squeeze_dims=[1])
    squeeze = tf.squeeze(conv, squeeze_dims=[1])
    print("Squeeze:", squeeze.get_shape())

    biases = tf.Variable(tf.zeros([squeeze.get_shape()[-1]]))
    relu = tf.nn.relu(squeeze + biases)
    hidden = tf.maximum(0.2*relu, relu)
    #hidden = squeeze

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
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
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
def create(x, targets,batch_size=-1):
    ops = {
        'conv1d':conv1d,
        'conv1d_transpose':conv1d_transpose,
        'feed_forward_nn':feed_forward_nn,
        'autoencoder':autoencoder,
        'reshape':reshape,
        'lstm':lstm
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
    #decoded=input
    reconstructed_x = tf.reshape(decoded, [-1, SIZE,DEPTH])
    print("Completed reshaping")


    ## hack build lstm
    size = SIZE#layer_def['size']
    cell = rnn_cell.BasicLSTMCell(size)

    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, last_state = seq2seq.rnn_decoder([decoded], initial_state, cell)
    extra_outputs = tf.concat(1, outputs)
    print("shape of extra", extra_outputs)
    output = tf.reshape(extra_outputs, [-1, size])
    print("shape of output", output.get_shape())

#    softmax_w = tf.get_variable("softmax_w", [size, tf.shape(input)[0]]) #wrong
#    softmax_b = tf.get_variable("softmax_b", [tf.shape(input)[0]]) #wrong

#    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
    #print("shape of logits", logits.get_shape())
    #probs = tf.nn.softmax(logits)
    #print("shape of probs", probs.get_shape())

    #`weights = tf.ones_like(logits)
    #print("shape of targets", targets.get_shape())
    num_decoder_symbols = 10
    #loss = seq2seq.sequence_loss_by_example([logits], [targets], [weights], num_decoder_symbols)
    #output=loss
    #results["cost"]= tf.reduce_sum(loss) / SIZE / 1000
    ## end hack
    predict = output
    results["cost"]= tf.sqrt(tf.reduce_mean(tf.square(targets-reconstructed_x)))*0.1+tf.sqrt(tf.reduce_mean(tf.square(x-reconstructed_x)))*0.9
    results['predict']=predict

    results['decoded']=tf.reshape(decoded, [-1])

    #results['arranged']= arranged_prev_layer
    #results['transposed']= conv_transposed
    return results

def get_input():
    return tf.placeholder("float", [None, SIZE, DEPTH], name='x')
def deep_test():
        sess = tf.Session()

        x = get_input()
        targets = tf.placeholder(tf.float32, [None, SIZE, DEPTH], name='targets')
        autoencoder = create(x, targets)
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
            for file in glob.glob('training/*.wav'):
                i+=1
                learn(file, sess, train_step, x,i, autoencoder, saver, targets)
                if(i%100==1):
                    i=i
                    print("Saving")
                    saver.save(sess, SAVE_DIR+"/modellstm3.ckpt", global_step=i+1)
        

def collect_input(data, dims):
    slice_size = dims[0]*dims[1]
    length = len(data)
    # discard extra info
    arr= np.array(data[0:int(length/SIZE)*SIZE])

    reshaped =  arr.reshape((-1, dims[0], dims[1]))
    return reshaped

def learn(filename, sess, train_step, x, k, autoencoder, saver, targets):
        #print("Loading "+filename)
        wavobj = loadfft2(filename)
        transformed = wavobj['transformed']
        transformed_raw = wavobj['raw']
        rate = wavobj['rate']

        input_squares = collect_input(transformed, [SIZE, DEPTH])
        predict_squares = np.roll(input_squares, SIZE*DEPTH*PREDICT_ROLL)

        #print(input_squares)
        #print("Running " + filename + str(np.shape(input_squares)[0]))
        sess.run(train_step, feed_dict={x: input_squares, targets: predict_squares})
        print(k,filename, " cost", sess.run(autoencoder['cost'], feed_dict={x: input_squares, targets:predict_squares}))
        #print("Finished " + filename)
        #print(i, " original", batch[0])
        #print( " decoded", sess.run(autoencoder['conv2'], feed_dict={x: input_squares}))

def deep_gen():
    with tf.Session() as sess:
        wavobj = loadfft2('input.wav')
        sanity(wavobj)
        transformed = wavobj['transformed']
        batch = collect_input(transformed, [SIZE, DEPTH])

        x = get_input()
        targets = tf.placeholder(tf.float32, [None, SIZE, DEPTH], name='targets')
        autoencoder = create(x, targets, batch_size=np.array(batch).shape[0])
        init = tf.initialize_all_variables()
        sess.run(init)
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(SAVE_DIR)
        if(checkpoint and checkpoint.model_checkpoint_path):
            saver.restore(sess, checkpoint.model_checkpoint_path)
        else:
            print("ERROR: No checkpoint found")
            exit(-1)




        #decoded = sess.run(autoencoder['predict'], feed_dict={x: np.array(batch), targets: np.array(batch)})
        #decoded = sess.run(autoencoder['decoded'], feed_dict={x: np.array(np.random.normal(0,1,[len(batch), SIZE, DEPTH]))})
        decoded = sess.run(autoencoder['decoded'], feed_dict={x: np.array(batch), targets: np.array(batch)})
        print(decoded)
        #filtered = np.append(filtered, batch)
        #res = np.transpose(batch_copy, [0,1,2]).reshape([-1])
        #sanity({"transformed":res, "rate": wavobj["rate"], "raw": wavobj['raw']})
        #print(i, " cost", sess.run(autoencoder['cost'], feed_dict={x: batch}))
        #print(i, " original", batch[0])
        #print( i, " decoded", sess.run(autoencoder['decoded'], feed_dict={x: batch}))
        savefft2('output2.wav', wavobj, decoded)
                       
if __name__ == '__main__':
    if(sys.argv[1] == 'train'):
        print("Train")
        deep_test()
    else:
        print("Generate")
        deep_gen()


