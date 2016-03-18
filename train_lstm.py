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
def deep_test():

    sess = tf.InteractiveSession()
    
    with gfile.FastGFile("log/model.pbtxt",'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        whatever, saver = restore_graph(
            graph_def,
            "model.ckpt",
        )
        print(whatever)
 
                      
if __name__ == '__main__':
    if(sys.argv[1] == 'train'):
        print("Train")
        deep_test()
    else:
        print("Generate todot")


