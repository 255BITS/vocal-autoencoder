import tensorflow as tf
import numpy as np
import math
import random

import sys

from scipy.io.wavfile import read, write
from scipy.fftpack import rfft, irfft
from numpy.fft import hfft, ihfft, fft, ifft
import os
from restore_graph import restore_graph
from tensorflow.python.platform import gfile

SIZE=128
def deep_test():

    with tf.Session() as sess:
        
        with gfile.FastGFile("log/model.pbtxt",'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            whatever, saver = restore_graph(
                graph_def,
                "model.ckpt",
            )
        x =  sess.graph.get_tensor_by_name('x:0')#tf.placeholder("float", [None, SIZE], name='x')
        node = sess.graph.get_tensor_by_name('encoder-0:0')
        out = sess.run(node, feed_dict={x:[np.random.normal(0,1, [SIZE])]}) 
        print(out)
                      
if __name__ == '__main__':
    if(sys.argv[1] == 'train'):
        print("Train")
        deep_test()
    else:
        print("Generate todot")


