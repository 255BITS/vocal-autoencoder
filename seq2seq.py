from tensorflow.models.rnn import rnn_cell, seq2seq

import tempfile
import tensorflow as tf
import numpy as np
sess = tf.InteractiveSession()

seq_length = 5
batch_size = 64

vocab_size = 7
embedding_dim = 50

memory_dim = 100


enc_inp = [tf.placeholder(tf.int32, shape=(None,),
                          name="inp%i" % t)
           for t in range(seq_length)]

labels = [tf.placeholder(tf.int32, shape=(None,),
                        name="labels%i" % t)
          for t in range(seq_length)]

weights = [tf.ones_like(labels_t, dtype=tf.float32)
           for labels_t in labels]

# Decoder input: prepend some "GO" token and drop the final
# token of the encoder input
dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.int32, name="GO")]
           + enc_inp[:-1])

# Initial memory value for recurrence.
prev_mem = tf.zeros((batch_size, memory_dim))

cell = rnn_cell.GRUCell(memory_dim)

dec_outputs, dec_memory = seq2seq.embedding_rnn_seq2seq(
    enc_inp, dec_inp, cell, vocab_size, vocab_size)

loss = seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)
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

def train_batch(batch_size):
    X = [np.random.choice(vocab_size, size=(seq_length,), replace=False)
         for _ in range(batch_size)]
    Y = [np.random.choice(vocab_size, size=(seq_length,), replace=False)
         for _ in range(batch_size)]
    
    # Dimshuffle to seq_len * batch_size
    X = np.array(X).T
    Y = np.array(Y).T

    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})

    _, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)
    return loss_t, summary


for t in range(500):
    loss_t, summary = train_batch(batch_size)
    summary_writer.add_summary(summary, t)
summary_writer.flush()


X_batch = [np.random.choice(vocab_size, size=(seq_length,), replace=False)
           for _ in range(10)]
X_batch = np.array(X_batch).T

feed_dict = {enc_inp[t]: X_batch[t] for t in range(seq_length)}
dec_outputs_batch = sess.run(dec_outputs, feed_dict)

print(X_batch)


print([logits_t.argmax(axis=1) for logits_t in dec_outputs_batch])
