import tensorflow as tf
import numpy as np
import math
import random

import sys

import os
import glob

from wav import loadfft2, savefft2, get_wav, save_wav


from tensorflow.models.rnn import rnn_cell, rnn
from tensorflow.models.rnn import seq2seq


import matplotlib.pyplot as plt



PREDICT_ROLL=8
TRAIN_REPEAT=100000
SIZE=256
LEARNING_RATE = tf.Variable(2e-3, trainable=False)
BATCH_SIZE=64
WAVELETS=256
Z_SIZE=64#WAVELETS//4
CHANNELS = 1
SEQ_LENGTH = 32

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
        {
            'type': 'softmax',
            'name': 's1',
            'size': WAVELETS,
            },
        #   {
        #    'type': 'autoencoder',
        #    'wavelets': WAVELETS,
        #    'name': 'l2'
        #    },
        #{
        #    'type': 'softmax',
        #    'name':'s2'
        #    },
            
        
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

def wnn_decode(output, output_dim, name, reuse=False):
    wavelets = output.get_shape()[1]
    with tf.variable_scope('wnn_encode_'+name):
        tf.get_variable_scope().reuse_variables()
        w = tf.get_variable('w', [output_dim, wavelets])
        w = tf.transpose(w)
    with tf.variable_scope('wnn_decode_'+name):
        if(reuse):
            tf.get_variable_scope().reuse_variables()
        summer = tf.get_variable('summer', [output_dim], initializer= tf.constant_initializer(0))
        output = tf.nn.xw_plus_b(output, w, summer)
        return output



def wnn_encode(input, wavelets, name, reuse=False):
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
        if(reuse):
            tf.get_variable_scope().reuse_variables()
        full_resolutions = math.log(wavelets*2)/math.log(2)
        tree = initial_dt_tree(-1,1, full_resolutions)
        d_c = [(leaf[1]) for leaf in tree]
        d_c.append(0.01)
        t_c = [leaf[0] for leaf in tree]
        t_c.append(0.01)
        t_c = np.tile(t_c,BATCH_SIZE)
        d_c = np.tile(d_c,BATCH_SIZE)
        translation = tf.reshape(tf.constant(t_c, dtype=tf.float32), [BATCH_SIZE, wavelets])
        dilation = tf.reshape(tf.constant(d_c, dtype=tf.float32), [BATCH_SIZE, wavelets])
        translation = tf.get_variable('translation', [BATCH_SIZE, wavelets], initializer = tf.constant_initializer(t_c))
        dilation = tf.get_variable('dilation', [BATCH_SIZE, wavelets], initializer = tf.constant_initializer(d_c))
        #w = tf.get_variable('w', [dim_in,wavelets], initializer=tf.constant_initializer(0.001), trainable=False)
        w = tf.get_variable('w', [dim_in,wavelets], initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
        #w = tf.ones([dim_in, wavelets])
        input_proj = tf.mul(tf.sub(tf.matmul(input, w), translation),dilation)
        #killer = tf.greater(dilation, 0.01225)
        #killer = tf.cast(killer, tf.float32)
        return mother(input_proj)#*killer


def linear(input_, output_size, scope=None, stddev=0.2, bias_start=0.0, with_w=False, reuse=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        if(reuse):
            tf.get_variable_scope().reuse_variables()
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


lstm_state = None
lstm_dec_state = None
def lstm_softmax(output):
    global lstm_state,lstm_dec_state
    out_shape = output[0].get_shape()[1]
    memory = Z_SIZE
    cell = rnn_cell.BasicLSTMCell(memory)
    cell = rnn_cell.MultiRNNCell([cell]*2)
    if(lstm_state == None):
        lstm_state = cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)
    #enc_inp = output
    #dec_inp = [tf.zeros_like(enc_inp[0], name="GO")]+ enc_inp[:-1]
    #dec_outputs, dec_state = seq2seq.basic_rnn_seq2seq(enc_inp, dec_inp, cell)
    dec_outputs, lstm_dec_state = rnn.rnn(cell, output,initial_state=lstm_state, dtype=tf.float32)
    print("dec_outputs  is", dec_outputs)
    #dec_outputs = [linear(o, out_shape, 'dec_out'+str(i)) for i,o in enumerate(dec_outputs)]
    #dec_outputs = [tf.nn.sigmoid(o) for o in dec_outputs]
    return dec_outputs



def softmax_layer(output, layer_def, nextMethod):
    name = layer_def['name']
    size = layer_def['size']

    [output, soft] = output#tf.split(1, 2, output)
    input_dim = int(output[0].get_shape()[1])//2

    soft = lstm_softmax(soft)

    sizes_down = [WAVELETS//2]
    sizes_up = reversed(sizes_down)
    for size in sizes_down:
        output = [linear(output[i], size, name+'down'+str(size), reuse=i > 0) for i in range(SEQ_LENGTH)]
        output = [tf.nn.tanh(o) for o in output]
    #output = soft
    #output = tf.nn.softmax(output)
    output = [linear(output[i], Z_SIZE, name+'downlast', reuse=i > 0) for i in range(SEQ_LENGTH)]
    noise = tf.random_normal(output[0].get_shape(), stddev=0.001)
    output = [s for o,s in zip(output, soft)]
    for size in sizes_up:
        output = [linear(output[i], size, name+'up'+str(size), reuse=i > 0) for i in range(SEQ_LENGTH)]
        output = [tf.nn.tanh(o) for o in output]
    output = [linear(output[i], WAVELETS, name+'uplast', reuse=i > 0)  for i in range(SEQ_LENGTH)]
    return output
def autoencoder(input, layer_def, nextMethod):
    input_dim = int(input.get_shape()[3])
    print("output dim is", input_dim)
    output_dim = input_dim
    wavelets = layer_def['wavelets']
    name = layer_def['name']
    print("-- Begin autoencoder", input_dim, input.get_shape())
    #output = input
    output = tf.split(1, SEQ_LENGTH, input)
    soft = [wnn_encode(tf.squeeze(output[i]), Z_SIZE, name+'lin', reuse = i > 0) for i in range(SEQ_LENGTH)]
    output = [wnn_encode(tf.squeeze(output[i]), wavelets, name, reuse = i > 0) for i in range(SEQ_LENGTH)]

    output = nextMethod([output,soft])
    output = [wnn_decode(output[i], output_dim, name, reuse = i>0) for i in range(SEQ_LENGTH)]
    output = [tf.reshape(o, [BATCH_SIZE, 1, CHANNELS*SIZE]) for o in output]
    output = tf.concat(1, output)

    # Such as output = build_wavelon(resolution)
    # where build_wavelon is recursively building the input array, one per resolution(1 main, 2*N smaller, 2**M smaller, etc)
    # then, make sure to introoduce the right vars and abstraction into wnn_encode
    #  multiply by translations, which is one translation per wavelon(per batch)

    # add bias term at the end
    return output



layer_index=0
def create(x,y=None):
    if y is None:
        y = tf.ones_like(x)
    ops = {
        'autoencoder':autoencoder,
        'softmax':softmax_layer
    }


    results = {}
    def nextMethod(current_layer):
        global layer_index
        if(len(layers) == layer_index+1):
            return current_layer
        layer_index += 1
        layer_def = layers[layer_index]
        return ops[layer_def['type']](current_layer, layer_def, nextMethod)

    #flat_x = tf.reshape(x, [BATCH_SIZE, -1])
    decoded = ops[layers[0]['type']](x, layers[0], nextMethod)

    #reconstructed_x = tf.reshape(output, [BATCH_SIZE, SEQ_LENGTH, CHANNELS,SIZE])
    results['decoded']=tf.reshape(decoded, [BATCH_SIZE, SEQ_LENGTH, CHANNELS,SIZE])
    y = tf.reshape(y, [BATCH_SIZE, SEQ_LENGTH, CHANNELS, SIZE])
    def get_entropy(o):
        return tf.reshape(o,[BATCH_SIZE,-1])
        os = tf.split(1, SEQ_LENGTH, o)
        return [tf.reshape(o, [BATCH_SIZE, -1]) for o in os]
    d = get_entropy(results['decoded'])
    ys = get_entropy(y)
    #xent = 1/2*tf.abs(ys-d)+((-1)/2*tf.sign(d)*ys)+(tf.cast(-1/2*(tf.sign(d)*tf.sign(ys)) > 0, dtype=tf.float32))
    xent = tf.square(ys-d)
    results['cost']=tf.sqrt(tf.reduce_mean(xent))#+((-1)/2*tf.reduce_mean(tf.sign(d)*ys))
    #results['cost']=results['cost']
    #results['cost']=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(xent, tf.zeros_like(ys)))
    #results["cost"]= tf.sqrt(tf.reduce_mean(tf.square(reconstructed_x-x)))
    #results["cost"]= tf.sqrt(tf.reduce_mean(tf.square(y-results['decoded'])))
    return results


def get_input():
    return tf.placeholder("float", [BATCH_SIZE, SEQ_LENGTH, CHANNELS, SIZE], name='x')
def get_y():
    return tf.placeholder("float", [BATCH_SIZE, SEQ_LENGTH, CHANNELS, SIZE], name='y')
def deep_test():
        global learn_state,lstm_state
        sess = tf.Session()

        x = get_input()
        y = get_y()
        autoencoder = create(x, y)
        learn_state = sess.run(lstm_state)
        #train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(autoencoder['cost'])
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        grad_clip = 5.
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(autoencoder['cost'], tvars), grad_clip)
        train_step = optimizer.apply_gradients(zip(grads, tvars))
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
                k = learn(file, sess, train_step, x,y,j, autoencoder, saver)
                j+= k
       

def collect_input(data, dims, pad=False):
    # discard extra info

    if(pad and len(data) < dims[0]*BATCH_SIZE*SEQ_LENGTH):
        zeros=np.zeros(dims[0]*BATCH_SIZE*SEQ_LENGTH)
        zeros[:len(data)]= data
        print(np.array(zeros))
        data = zeros
    length = len(data)
    arr= np.array(data[0:int(length/dims[0]/BATCH_SIZE/SEQ_LENGTH)*dims[0]*BATCH_SIZE*SEQ_LENGTH])
    print(np.shape(arr), "SHAPE")

    
    reshaped =  arr.reshape((-1, BATCH_SIZE, dims[0]))
    return reshaped

learn_state = None
def learn(filename, sess, train_step, x, y,k, autoencoder, saver):
        global lstm_state,lstm_dec_state, learn_state
        #print("Loading "+filename)
        # not really loading fft
        wavobj = get_wav(filename)
        transformed_raw = wavobj['data']
        rate = wavobj['rate']

        input_squares = collect_input(transformed_raw, [SIZE*CHANNELS*SEQ_LENGTH])
        y_squares = np.roll(input_squares, -SIZE)

        #print(input_squares)
        #print("Running " + filename + str(np.shape(input_squares)[0]))
        
        i=0
        for square,y_square in zip(input_squares, y_squares):
            square = np.reshape(square, [BATCH_SIZE, SEQ_LENGTH, SIZE,CHANNELS])
            square = np.swapaxes(square, 2, 3)
            y_square = np.reshape(y_square, [BATCH_SIZE, SEQ_LENGTH, SIZE,CHANNELS])
            y_square = np.swapaxes(y_square, 2, 3)
            _, cost, decoded,learn_state = sess.run([train_step,autoencoder['cost'], autoencoder['decoded'], lstm_dec_state], feed_dict={x: square, y:y_square, lstm_state: learn_state})
            i+=1
            if((i+k) % PLOT_EVERY == 3):
                to_plot = np.reshape(y_square[0,0,0,:], [-1])
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
                plt.plot(np.reshape(decoded[0,0,0,:], [-1]))
                plt.savefig('visualize/input-'+str(k+i)+'.png')
            if((i+k)%SAVE_EVERY==1):
                print("Saving")
                saver.save(sess, SAVE_DIR+"/modellstm3.ckpt", global_step=i+1)
 
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
        batches = collect_input(transformed, [SIZE*CHANNELS*SEQ_LENGTH], pad=True)

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
        decoded = None
        for batch in batches:
            batch = np.reshape(batch, [BATCH_SIZE, SEQ_LENGTH, SIZE, CHANNELS])
            batch =np.swapaxes(batch, 2, 3)
            if(decoded is None):
                decoded = batch# * 0.9 + np.random.uniform(-0.1, 0.1, batch.shape)
            else:
                decoded = decoded * 0.1 + batch*0.8 + np.random.uniform(-0.1, 0.1, batch.shape)

            #batch += np.random.uniform(-0.1,0.1,batch.shape)
            decoded = sess.run(autoencoder['decoded'], feed_dict={x: decoded})
            all_out.append(np.swapaxes(decoded, 2, 3))
        all_out = np.array(all_out)
        wavobj['data']=np.reshape(all_out, [-1, CHANNELS])
        print(all_out)
        print('saving to output2.wav', np.min(all_out), np.max(all_out))
        save_wav(wavobj, 'output2.wav')
                       
if __name__ == '__main__':
    if(sys.argv[1] == 'train'):
        print("Train")
        deep_test()
    else:
        print("Generate")
        deep_gen()


