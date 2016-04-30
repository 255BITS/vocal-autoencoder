import glob
from multiprocessing import Queue, Process
import numpy as np
import time

from wav import get_wav

queue = Queue()

def each_batch(globPattern, batch_size, size, predict, epochs):
    files = glob.glob(globPattern)
    for epoch in range(epochs):
        load(files, batch_size, size, predict)
        batch = True
        while(batch):
            batch = next_batch()
            if(batch is not None):
                yield epoch, batch[0], batch[1]


def load(files, batch_size, size, predict):
    p = Process(target=add_to_queue, args=([files, batch_size, size, predict]))
    p.start()

def get_batch(file, batch_size, size):

    print("Loading", file)

    out = get_wav(file)
    out = np.reshape(out['data'], [-1])
    end = (len(out)//batch_size//size)*size*batch_size
    extra = out[end:]
    out = out[:end]
    if(len(extra) > 0):
        pad_extra = np.zeros([batch_size*size])
        pad_extra[:len(extra)]=extra
        out = np.concatenate([out, pad_extra])
    out = np.reshape(out, [-1, batch_size, size])
    return out

def get_predict(batches, i, batch_size, size, predict):
    x = batches.reshape([-1])
    begin = batch_size*(i)*size+predict
    end = batch_size*(i+1)*size+predict
    if(end > len(x)):
        return np.zeros([(batch_size*size)])
    return x[begin:end]

def add_to_queue(files,batch_size, size, predict_x):
    for filea in files:
       batches = get_batch(filea, batch_size, size)#, get_batch(fileb)]
       for i, batch in enumerate(batches):
           while(queue.qsize() > 100):
               time.sleep(0.1)
           predict = get_predict(batches, i, batch_size, size, predict_x)
           queue.put([batch, predict, i/len(batches[0]), 1.0/batch_size])
       time.sleep(0.1)
    queue.put("DONE")

def next_batch():
    pop = queue.get()
    if(pop == "DONE"):
        return None
    else:
        return pop
