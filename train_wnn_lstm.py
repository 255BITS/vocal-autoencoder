import tensorflow as tf
import numpy as np
import math
import random

import sys

import os
import glob

from tensorflow.models.rnn import rnn_cell, rnn
from tensorflow.models.rnn import seq2seq


import matplotlib.pyplot as plt

import trainer
from wav import get_wav, save_wav



TRAIN_REPEAT=100000
SIZE=1024
LEARNING_RATE = tf.Variable(2.55 * 1e-3, trainable=False)
LEARNING_RATE_G_LOSS = LEARNING_RATE/10
BATCH_SIZE=512
WAVELETS=512
Z_SIZE=128#WAVELETS//4
CHANNELS = 1
SEQ_LENGTH = 2

PLOT_EVERY = 50
SAVE_EVERY = 1000

PREDICT=SEQ_LENGTH
SAVE_DIR='save'
#BATCH_SIZE=349
layers = [
        {
            'type': 'autoencoder',
            'wavelets': WAVELETS,
            'name': 'l1'
            },
        {
            'type': 'rnn',
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
        d_c = [(leaf[1])*100 for leaf in tree]
        d_c.append(0.01)
        t_c = [leaf[0] for leaf in tree]
        t_c.append(0.01)
        t_c = np.tile(t_c,BATCH_SIZE)
        d_c = np.tile(d_c,BATCH_SIZE)
        translation = tf.reshape(tf.constant(t_c, dtype=tf.float32), [BATCH_SIZE, wavelets])
        dilation = tf.reshape(tf.constant(d_c, dtype=tf.float32), [BATCH_SIZE, wavelets])
        #translation = tf.get_variable('translation', [BATCH_SIZE, wavelets], initializer = tf.constant_initializer(t_c))
        #dilation = tf.get_variable('dilation', [BATCH_SIZE, wavelets], initializer = tf.constant_initializer(d_c))
        #w = tf.get_variable('w', [dim_in,wavelets], initializer=tf.constant_initializer(0.001), trainable=False)
        w = tf.get_variable('w', [dim_in,wavelets], initializer=tf.random_normal_initializer(mean=0, stddev=0.05))
        #w = tf.ones([dim_in, wavelets])
        input_proj = tf.mul(tf.sub(tf.matmul(input, w), translation),dilation)

        #scales = tf.get_variable('scale_w',[wavelets], initializer=tf.random_normal_initializer(mean=1, stddev=0.01))
        #input_proj = tf.mul(input_proj, scales)
        #killer = tf.greater(dilation, 0.02551225)
        #killer = tf.cast(killer, tf.float32)
        return mother(input_proj)#*killer


def linear(input_, output_size, name=None, stddev=0.02, bias_start=0.0, with_w=False, reuse=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name or "Linear"):
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

def deconv2d(input_, output_shape,
        k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, biasstart=0.0, padding='SAME',
        name="deconv2d", with_w=False, no_bias=False,reuse=False):
    with tf.variable_scope(name):
        if(reuse):
            tf.get_variable_scope().reuse_variables()
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_h, output_shape[-1], input_.get_shape()[-1]],
                initializer=tf.truncated_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, padding=padding,
                    strides=[1, d_h, d_w, 1])

            # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

            if(no_bias):
                print("Skipping bias")

        else:
            biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(biasstart))
            deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def conv2d(input_, output_dim, 
        k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.2, padding='SAME',
        name="conv2d", reuse=False):
    with tf.variable_scope(name):
        if(reuse):
            tf.get_variable_scope().reuse_variables()
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


def lstm(output):
    global x_hat, y_hat
    out_shape = output[0].get_shape()[1]
    memory = Z_SIZE
    cell = rnn_cell.BasicLSTMCell(memory)
    cell = rnn_cell.MultiRNNCell([cell]*4)
    lstm_state = cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)
    enc_inp = output
    dec_inp = [tf.zeros_like(enc_inp[0], name="GO")]+ y_hat
    dec_outputs, dec_state = seq2seq.basic_rnn_seq2seq(enc_inp, dec_inp, cell)
    #dec_outputs, lstm_dec_state = rnn.rnn(cell, output,initial_state=lstm_state, dtype=tf.float32)
    print("dec_outputs  is", dec_outputs)
    #dec_outputs = [linear(o, out_shape, 'RNN_dec_out'+str(i)) for i,o in enumerate(dec_outputs)]
    #dec_outputs = [tf.nn.sigmoid(o) for o in dec_outputs]
    x_hat = dec_outputs
    return dec_outputs



def rnn_layer(output,layer_def, nextMethod):
    #output = [((o)) for o,o2 in zip(output[0], output[1])]
    output = [(tf.nn.tanh(o)*tf.nn.sigmoid(o2)) for o,o2 in zip(output[0], output[1])]
    output = lstm(output)

    return output
def autoencoder(input, layer_def, nextMethod):
    output_dim = int(input.get_shape()[3])
    wavelets = layer_def['wavelets']
    name = layer_def['name']
    return build_autoencoder(input, wavelets, name, output_dim, nextMethod, reuse=True)

loss_term = 0
def build_autoencoder(input, wavelets, name, output_dim, nextMethod, reuse=False):
    output = tf.split(1, SEQ_LENGTH, input)
    orig_output = output

    output = [wnn_encode(tf.squeeze(output[i]), WAVELETS, name, reuse = reuse or (i>0)) for i in range(SEQ_LENGTH)]
    output = [tf.reshape(o, [BATCH_SIZE, 32,16,1]) for o in output]
    output = [conv2d(output[i], 64,name=name+'conv1', reuse=reuse or i > 0) for i in range(SEQ_LENGTH)]
    output = [tf.nn.relu(o) for o in output]
    #output = [tf.nn.dropout(o, 0.9) for o in output]
    output = [conv2d(output[i], 32,name=name+'conv2', reuse=reuse or i > 0) for i in range(SEQ_LENGTH)]
    output = [tf.nn.relu(o) for o in output]
    output = [conv2d(output[i], 16,k_w=4, k_h=4, name=name+'conv3', reuse=reuse or i > 0) for i in range(SEQ_LENGTH)]


    #output = [tf.reshape(o, [BATCH_SIZE, -1]) for o in output]
    output = [tf.reshape(o, [BATCH_SIZE, Z_SIZE]) for o in output]
    #output = [linear(output[i], Z_SIZE, name=name+'convlin', reuse=reuse or i > 0) for i in range(SEQ_LENGTH)]
    print("OUTPUT DIM IS", output_dim)


    extra = [wnn_encode(tf.squeeze(orig_output[i]), Z_SIZE, name+'z', reuse = reuse or (i>0)) for i in range(SEQ_LENGTH)]
    if nextMethod is not None:
        output = nextMethod([output, extra])


    output = build_generator(output, reuse = reuse)

    wnn_encode_output = output

    # Such as output = build_wavelon(resolution)
    # where build_wavelon is recursively building the input array, one per resolution(1 main, 2*N smaller, 2**M smaller, etc)
    # then, make sure to introoduce the right vars and abstraction into wnn_encode
    #  multiply by translations, which is one translation per wavelon(per batch)

    # add bias term at the end
    return output


def deep_autoencoder(output, reuse=False):
    name = 'l1'
    wavelets = WAVELETS
    z={}
    def nonlinear(output):
        output = [tf.nn.tanh(o)*tf.nn.sigmoid(o2) for o,o2 in zip(output[0], output[1])]
        #output = [tf.nn.dropout(o, 0.8) for o in output]
        z['value'] = output
        return output
    output_dim = int(output.get_shape()[3])
    output = build_autoencoder(output, wavelets, name, output_dim, nonlinear, reuse=reuse)

    return [output,z['value']]


def build_generator(output, reuse):
    name='l1'
    #print('output', output[0].get_shape())
    #output = [linear(o,4*2*16,name+'linear_prj', reuse=reuse or (i>0),stddev=0.02) for i, o in enumerate(output)]
    #output = [tf.nn.dropout(o, 0.5) for o in output]
    #output = [tf.nn.relu(o) for o in output]
    output = [tf.reshape(o, [BATCH_SIZE, 4,2,16]) for o in output]
    output = [deconv2d(output[i], [BATCH_SIZE, 8, 4, 8], name=name+'deconv1', reuse = reuse or (i>0)) for i in range(SEQ_LENGTH)]
    output = [tf.nn.tanh(o) for o in output]
    output = [deconv2d(output[i], [BATCH_SIZE, 16, 8, 4], name=name+'deconv2', reuse = reuse or (i>0)) for i in range(SEQ_LENGTH)]
    output = [tf.nn.tanh(o) for o in output]
    output = [deconv2d(output[i], [BATCH_SIZE, 32, 16, 1], name=name+'deconv3', reuse = reuse or (i>0)) for i in range(SEQ_LENGTH)]

    output = [tf.reshape(o, [BATCH_SIZE, WAVELETS]) for o in output]
    output = [wnn_decode(output[i], SIZE*CHANNELS, name, reuse = reuse or (i>0)) for i in range(SEQ_LENGTH)]
    output = [tf.reshape(o, [BATCH_SIZE, 1, CHANNELS, SIZE]) for o in output]
    output = tf.concat(1, output)
    return output


def discriminator(output, reuse=True):
    name='discriminator'
    output = tf.split(1, SEQ_LENGTH, output)
    output = [wnn_encode(tf.squeeze(output[i]), WAVELETS, 'l1', reuse = True) for i in range(SEQ_LENGTH)]
    output = [tf.reshape(o, [BATCH_SIZE, 32,16,1]) for o in output]
    output = [conv2d(output[i], 32,name=name+'conv1', reuse=reuse or i > 0) for i in range(SEQ_LENGTH)]
    output = [tf.nn.relu(o) for o in output]
    output = [conv2d(output[i], 16,name=name+'conv3', reuse=reuse or i > 0) for i in range(SEQ_LENGTH)]
    output = [tf.nn.relu(o) for o in output]
    output = [tf.reshape(o, [BATCH_SIZE, -1]) for o in output]
    output = [linear(o,output[0].get_shape()[1],name+'disc_lin', reuse=reuse or (i>0),stddev=0.02) for i, o in enumerate(output)]
    output = [tf.nn.relu(o) for o in output]
    output = tf.concat(1, output)
    output = linear(output, 1,name=name+'linear1', reuse=reuse, stddev=0.01)
    print('SHAPE is', output.get_shape())
    return tf.sigmoid(output)
 

layer_index=0
def create(x,y=None):
    global x_hat, y_hat
    if y is None:
        y = tf.ones_like(x)
    ops = {
        'autoencoder':autoencoder,
        'rnn':rnn_layer
    }
    autoencoded_x, _ = deep_autoencoder(x)
    _, y_hat = deep_autoencoder(y, reuse=True)



    results = {}
    def nextMethod(current_layer):
        global layer_index
        if(len(layers) == layer_index+1):
            return current_layer
        layer_index += 1
        layer_def = layers[layer_index]
        return ops[layer_def['type']](current_layer,layer_def, nextMethod)

    #flat_x = tf.reshape(x, [BATCH_SIZE, -1])
    decoded = ops[layers[0]['type']](x,layers[0], nextMethod)

    #reconstructed_x = tf.reshape(output, [BATCH_SIZE, SEQ_LENGTH, CHANNELS,SIZE])
    results['decoded']=tf.reshape(decoded, [BATCH_SIZE, SEQ_LENGTH, CHANNELS,SIZE])
    y = tf.reshape(y, [BATCH_SIZE, SEQ_LENGTH, CHANNELS, SIZE])
    def get_entropy(o):
        return tf.reshape(o,[BATCH_SIZE,-1])
    d = get_entropy(results['decoded'])
    ys = get_entropy(y)
    sqs = tf.square(ys-d)

    hats = [tf.square(y_h-x_h) for y_h,x_h in zip(y_hat, x_hat)]
    hats = tf.add_n(hats)

    results['cost']=tf.sqrt(tf.reduce_sum(hats))

    results['pretrain_cost']=tf.sqrt(tf.reduce_sum(tf.square(x-autoencoded_x)))#+tf.reduce_mean(loss_term)

    gen = build_generator([tf.nn.tanh(tf.random_normal([BATCH_SIZE, Z_SIZE], mean=0, stddev=0.5)* \
        tf.nn.sigmoid(tf.random_uniform([BATCH_SIZE,Z_SIZE],-6, 6))) \
            for i in range(SEQ_LENGTH)], reuse=True)
    real_d = discriminator(x, reuse=False)
    fake_d = discriminator(gen, reuse=True)
    # from https://arxiv.org/pdf/1604.07379v1.pdf
    # eq 2
    eps=1e-10
    results['d_loss']=tf.reduce_mean(-tf.log(real_d+eps))
    results['fake_loss']=tf.reduce_mean(-tf.log(1-fake_d+eps))
    results['adversarial_cost']=results['d_loss']+results['fake_loss']
    results['g_loss']=tf.reduce_mean(-tf.log(fake_d+eps))
    results['g_sample']=gen
    results['joint_loss']=0.001*results['adversarial_cost']+0.999*results['pretrain_cost']
    results['autoencoded_x']=autoencoded_x
    return results


def get_input():
    return tf.placeholder("float", [BATCH_SIZE, SEQ_LENGTH, CHANNELS, SIZE], name='x')
def get_y():
    return tf.placeholder("float", [BATCH_SIZE, SEQ_LENGTH, CHANNELS, SIZE], name='y')
def deep_train(clobber=False):
    sess = tf.Session()

    x = get_input()
    y = get_y()
    autoencoder = create(x, y)
    #train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(autoencoder['cost'])
    train_step = create_cost_optimizer(autoencoder)
    create_pretrain_cost_optimizer(autoencoder)
    g_train_step = create_optim(autoencoder, 'g_loss', LEARNING_RATE_G_LOSS)
    ac_train_step = create_optim(autoencoder, 'joint_loss', LEARNING_RATE)
    init = tf.initialize_all_variables()
    sess.run(init)
    saver = save_or_load(sess, clobber)

    tf.train.write_graph(sess.graph_def, 'log', 'modellstm3.pbtxt', False)

    i=0
    for epoch, batch, predict in trainer.each_batch('training/*.wav', \
            size=SIZE*SEQ_LENGTH*CHANNELS, \
            batch_size=BATCH_SIZE,
            predict=PREDICT*SIZE,
            epochs=TRAIN_REPEAT):
        learn(batch, predict, sess, train_step, x,y,i, autoencoder, saver)
        i=i+1

def learn(batch, predict, sess, train_step, x, y,k, autoencoder, saver):
    batch = np.reshape(batch, [BATCH_SIZE, SEQ_LENGTH, SIZE,CHANNELS])
    batch = np.swapaxes(batch, 2, 3)
    predict = np.reshape(predict, [BATCH_SIZE, SEQ_LENGTH, SIZE,CHANNELS])
    predict = np.swapaxes(predict, 2, 3)
    _, cost, decoded = sess.run([train_step,autoencoder['cost'], autoencoder['decoded']], feed_dict={x: batch, y:predict})
    if((k) % PLOT_EVERY == 3):
        to_plot = np.reshape(predict[0,0,0,:], [-1])
        plt.clf()
        plt.plot(to_plot)

        plt.xlim([0, SIZE])
        plt.ylim([-2, 2])
        plt.ylabel("Amplitude")
        plt.xlabel("Time")
        ## set the title  
        plt.title("batch")
        plt.plot(np.reshape(decoded[0,0,0,:], [-1]))
        plt.savefig('visualize/input-'+str(k)+'.png')
    if((k)%SAVE_EVERY==99):
        print("Saving")
        saver.save(sess, SAVE_DIR+"/modellstm3.ckpt", global_step=k+1)

    print(" cost", cost, k)

def pretrain_learn(batch, predict, sess, train_step, g_train_step, ac_train_step, x, y,k, autoencoder, saver):
    batch = np.reshape(batch, [BATCH_SIZE, SEQ_LENGTH, SIZE,CHANNELS])
    batch = np.swapaxes(batch, 2, 3)
 
    #_, cost, decoded = sess.run([train_step,autoencoder['pretrain_cost'], autoencoder['autoencoded_x']], feed_dict={x: batch})
    #_ = sess.run([ac_train_step], feed_dict={x: batch})
    #g_loss, g_sample, d_loss, d_fake_loss = [0, np.zeros_like(batch), 0, 0]
    #_, g_loss, g_sample = sess.run([g_train_step,autoencoder['g_loss']], feed_dict={x: batch})
    #_ = sess.run([train_step], feed_dict={x: batch})
    _, = sess.run([ac_train_step ], feed_dict={x: batch})
    cost, decoded, d_loss, d_fake_loss = sess.run([autoencoder['pretrain_cost'], autoencoder['autoencoded_x'],
        autoencoder['d_loss'], autoencoder['fake_loss']], feed_dict={x: batch})
    g_loss, g_sample = sess.run([ autoencoder['g_loss'], autoencoder['g_sample']], feed_dict={x: batch})


    if((k) % PLOT_EVERY == 3):
        to_plot = np.reshape(batch[0,0,0,:], [-1])
        plt.clf()
        plt.plot(to_plot)

        plt.xlim([0, SIZE])
        plt.ylim([-2, 2])
        plt.ylabel("Amplitude")
        plt.xlabel("Time")
        plt.title("batch")
        g_plot = np.reshape(g_sample[0,0,0,:], [-1])
        plt.plot(g_plot+np.ones_like(g_plot))
        plt.plot(np.reshape(decoded[0,0,0,:], [-1]))
        print("SHAPE IS", g_sample.shape, decoded.shape)
        plt.savefig('visualize/input-'+str(k)+'.png')
    if((k)%SAVE_EVERY==99):
        print("Saving")
        saver.save(sess, SAVE_DIR+"/modellstm3.ckpt", global_step=k+1)

    print(str(k)+" cost", cost, "d_loss", d_loss, 'g_loss', g_loss, 'd_fake_loss', d_fake_loss )


def deep_gen():
    with tf.Session() as sess:
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



        i=0
        g_all_out=[]
        all_out=[]
        decoded_all_out=[]
        for epoch, batch, predict in trainer.each_batch('input.wav', \
                    size=SIZE*SEQ_LENGTH*CHANNELS, \
                    batch_size=BATCH_SIZE,
                    predict=PREDICT*SIZE,
                    epochs=1):

            if(i>0):
                print("Not doing anything")
            else:
                decoded = None
                batch = np.reshape(batch, [BATCH_SIZE, SEQ_LENGTH, SIZE, CHANNELS])
                batch =np.swapaxes(batch, 2, 3)
                if(decoded is None):
                    decoded = batch# * 0.9 + np.random.uniform(-0.1, 0.1, batch.shape)
                else:
                    decoded = batch# * 0.1 + batch*0.8# + np.random.uniform(-0.1, 0.1, batch.shape)

                #batch += np.random.uniform(-0.1,0.1,batch.shape)
                decoded = sess.run(autoencoder['autoencoded_x'], feed_dict={x: decoded})
                g_sample = sess.run(autoencoder['g_sample'], feed_dict={x: decoded})
                all_out.append(np.swapaxes(decoded, 2, 3))
                g_all_out.append(np.swapaxes(g_sample, 2, 3))
                decoded = sess.run(autoencoder['decoded'], feed_dict={x: decoded})
                decoded_all_out.append(np.swapaxes(decoded, 2, 3))
        all_out = np.array(all_out)
        g_all_out = np.array(g_all_out)
        decoded_all_out = np.array(decoded_all_out)
        wavobj = get_wav('input.wav')
        wavobj['data']=np.reshape(all_out, [-1, CHANNELS])
        print(all_out)
        print('saving to output2.wav', np.min(all_out), np.max(all_out))
        save_wav(wavobj, 'output2.wav')
        wavobj['data']=np.reshape(g_all_out, [-1, CHANNELS])
        save_wav(wavobj,'g_output2.wav')
        wavobj['data']=np.reshape(decoded_all_out, [-1, CHANNELS])
        save_wav(wavobj,'decoded_output.wav')

def create_cost_optimizer(autoencoder):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        grad_clip = 5.
        tvars = tf.trainable_variables()
        tvars = [var for var in tvars if 'RNN' in var.name]
        print("train trainables", [ v.name for v in tvars])
        grads, _ = tf.clip_by_global_norm(tf.gradients(autoencoder['cost'], tvars), grad_clip)
        train_step = optimizer.apply_gradients(zip(grads, tvars))
        return train_step
def create_pretrain_cost_optimizer(autoencoder):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        grad_clip = 5.
        tvars = tf.trainable_variables()
        tvars = [var for var in tvars if 'l1' in var.name]
        print("Pretrain trainables", [ v.name for v in tvars])
        grads, _ = tf.clip_by_global_norm(tf.gradients(autoencoder['pretrain_cost'], tvars), grad_clip)
        train_step = optimizer.apply_gradients(zip(grads, tvars))
        return train_step

def create_optim(autoencoder, name, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grad_clip = 5.
    tvars = tf.trainable_variables()
    #tvars = [var for var in tvars if 'l1' in var.name]
    print("Pretrain trainables", [ v.name for v in tvars])
    grads, _ = tf.clip_by_global_norm(tf.gradients(autoencoder[name], tvars), grad_clip)
    train_step = optimizer.apply_gradients(zip(grads, tvars))
    return train_step



def deep_pretrain(clobber=False):
        sess = tf.Session()

        x = get_input()
        autoencoder = create(x)
        create_cost_optimizer(autoencoder)
        train_step = create_pretrain_cost_optimizer(autoencoder)
        g_train_step = create_optim(autoencoder, 'g_loss', LEARNING_RATE_G_LOSS)
        ac_train_step = create_optim(autoencoder, 'joint_loss', LEARNING_RATE)



        init = tf.initialize_all_variables()
        sess.run(init)
        saver = save_or_load(sess,clobber)

        tf.train.write_graph(sess.graph_def, 'log', 'modellstm3.pbtxt', False)

        i=0

        for epoch, batch, predict in trainer.each_batch('training/*.wav', \
                    size=SIZE*SEQ_LENGTH*CHANNELS, \
                    batch_size=BATCH_SIZE,
                    predict=PREDICT*SIZE,
                    epochs=TRAIN_REPEAT):

            pretrain_learn(batch, predict, sess, train_step, g_train_step, ac_train_step, x,None, i, autoencoder, saver)
            i=i+1

def save_or_load(sess, clobber):
    if(clobber):
        print("Clobbering ...")
        saver = tf.train.Saver(tf.all_variables())
        #saver.save(sess, SAVE_DIR+'/modellstm3.ckpt', global_step=0)
    else:
        print("Loading ...")
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(SAVE_DIR)
        saver.restore(sess, checkpoint.model_checkpoint_path)

    return saver
                       
if __name__ == '__main__':
    clobber = False
    if(len(sys.argv) > 2 and sys.argv[2] == '--clobber'):
        clobber = True

    if(sys.argv[1] == 'train'):
        print("Train")
        deep_train(clobber = clobber)
    elif(sys.argv[1] == 'pretrain'):
        print('pretrain')
        deep_pretrain(clobber =clobber)
    else:
        print("Generate")
        deep_gen()


