from tensorflow.models.rnn import rnn_cell, seq2seq

import tempfile
import tensorflow as tf
import numpy as np
import os

from wav import loadfft, savefft, sanity
sess = tf.InteractiveSession()

seq_length = 5
SIZE=10
batch_size = seq_length * SIZE

vocab_size = 7
embedding_dim = 50

memory_dim = 100


enc_inp = [tf.placeholder(tf.float32, shape=(None,SIZE),
                          name="inp%i" % t)
           for t in range(seq_length)]

labels = [tf.placeholder(tf.float32, shape=(None,SIZE),
                        name="labels%i" % t)
          for t in range(seq_length)]

weights = [tf.ones_like(labels_t, dtype=tf.float32)
           for labels_t in labels]

# Decoder input: prepend some "GO" token and drop the final
# token of the encoder input
dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.float32, name="GO")]
           + enc_inp[:-1])

# Initial memory value for recurrence.
#prev_mem = tf.zeros((batch_size, memory_dim))

print("shapes", np.array(enc_inp).shape, np.array(dec_inp).shape, np.array(labels).shape)
cell = rnn_cell.GRUCell(memory_dim)

dec_outputs, dec_memory = seq2seq.basic_rnn_seq2seq(
    enc_inp, dec_inp, cell)

labels_t = tf.reshape(labels, [5,100])
print(labels_t)
print(dec_outputs)
loss = seq2seq.sequence_loss(dec_outputs, labels_t, weights, vocab_size)
tf.scalar_summary("loss", loss)
#magnitude = tf.sqrt(tf.reduce_sum(tf.square(dec_memory[1])))
#tf.scalar_summary("magnitude at t=1", magnitude)
summary_op = tf.merge_all_summaries()


learning_rate = 0.05
momentum = 0.9
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train_op = optimizer.minimize(loss)
logdir = tempfile.mkdtemp()
print(logdir)
summary_writer = tf.train.SummaryWriter(logdir, sess.graph_def)

sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
saver.save(sess, 'seq2seq_model.ckpt')
def train_batch(X, Y):
    for t in range(seq_length):
        print(np.array(Y[t]).shape) 
    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})

    _, loss_t, summary = sess.run([train_op], feed_dict)#, loss, summary_op], feed_dict)
    return loss_t, summary


i=0
for file in glob.glob('training/*.wav'):
    i+=1
    filename = 'training/'+file
    wavobj = loadfft(filename)
    batch = []
    transformed=wavobj['transformed']
    for i in range(int(len(transformed)/batch_size)): # Our dataset consists of two centers with gaussian noise w/ sigma = 0.1
        c1 = transformed[i*batch_size:i*batch_size+batch_size]
        batch += [np.array(c1).reshape([seq_length, SIZE])]
    X = np.array(batch)
    Y = X[:]

    print(X.shape)

    loss_t, summary = train_batch(X, Y)
    print(loss_t, summary)
    summary_writer.add_summary(summary, loss_t)
summary_writer.flush()


feed_dict = {enc_inp[t]: X_batch[t] for t in range(seq_length)}
dec_outputs_batch = sess.run(dec_outputs, feed_dict)

print(X_batch)


print([logits_t.argmax(axis=1) for logits_t in dec_outputs_batch])
