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

from ops import lstm, autoencoder


PREDICT_ROLL=-1
EPOCH=5
LEARNING_RATE = tf.Variable(1e-3, trainable=False)

SAVE_DIR='save'




BATCH_SIZE=441*100
layers = [
    {
        'type': 'autoencoder_discrete',
        'output_dim': 1
    },
    #{
    #    'type': 'autoencoder_int',
    #    'output_dim': 80
    #},

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


def autoencoder_discrete(input, layer_def, nextMethod):
    if(len(input.get_shape())==1):
        input = tf.reshape(input, [-1, 1])
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
def create(x, targets,batch_size=BATCH_SIZE):
    ops = {
        'autoencoder_discrete':autoencoder_discrete,
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
    reconstructed_x = tf.reshape(decoded, [-1])
    print("Completed reshaping")


    ## hack build lstm
    size = 128#layer_def['size']
    cell = rnn_cell.BasicLSTMCell(size)

    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, last_state = seq2seq.rnn_decoder([reconstructed_x], initial_state, cell)
    extra_outputs = tf.concat(1, outputs)
    print("shape of extra", extra_outputs)
    output = tf.reshape(extra_outputs, [-1, size])
    print("shape of output", output.get_shape())
    results = {}

    softmax_w = tf.get_variable("softmax_w", [size, x.get_shape()[0]]) #wrong
    softmax_b = tf.get_variable("softmax_b", [x.get_shape()[0]]) #wrong

    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
    #print("shape of logits", logits.get_shape())
    probs = tf.nn.softmax(logits)
    #print("shape of probs", probs.get_shape())

    weights = tf.ones_like(logits)
    #print("shape of targets", targets.get_shape())
    num_decoder_symbols = 10
    loss = seq2seq.sequence_loss_by_example([logits], [targets], [weights], num_decoder_symbols)
    #output=loss
    #results["cost"]= tf.reduce_sum(loss) / SIZE / 1000
    ## end hack
    predict = output
    #results["cost"]= tf.sqrt(tf.reduce_mean(tf.square(targets-reconstructed_x)))*0.1+tf.sqrt(tf.reduce_mean(tf.square(x-reconstructed_x)))*0.9
    results['cost'] = loss
    results['predict']=predict

    #results['arranged']= arranged_prev_layer
    #results['transposed']= conv_transposed
    return results

def get_input():
    return tf.placeholder("float", BATCH_SIZE, name='x')
def deep_test():
        sess = tf.Session()

        x = get_input()
        targets = tf.placeholder(tf.float32, BATCH_SIZE, name='targets')
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
        for trains in range(EPOCH):
            for file in glob.glob('training/*.wav'):
                i+=1
                learn(file, sess, train_step, x,i, autoencoder, saver, targets)
                if(i%100==1):
                    i=i
                    print("Saving")
                    saver.save(sess, SAVE_DIR+"/modellstm3.ckpt", global_step=i+1)
        

def learn(filename, sess, train_step, x, k, autoencoder, saver, targets):
        #print("Loading "+filename)
        wavobj = loadfft2(filename)
        transformed = wavobj['transformed']
        transformed_raw = wavobj['raw']
        rate = wavobj['rate']

        x_in = np.array(transformed[:BATCH_SIZE])
        y_in = np.roll(x_in, PREDICT_ROLL)

        print(x_in,y_in)
        #print(input_squares)
        #print("Running " + filename + str(np.shape(input_squares)[0]))
        sess.run(train_step, feed_dict={x: x_in, targets: y_in})
        print(k,filename, " cost", sess.run(autoencoder['cost'], feed_dict={x: x_in, targets:y_in}))
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
        targets = tf.placeholder(tf.float32, [BATCH_SIZE], name='targets')
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


