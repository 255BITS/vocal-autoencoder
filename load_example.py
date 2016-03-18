import tensorflow as tf
import os
import numpy as np
from tensorflow.python.platform import gfile

data = np.arange(10,dtype=np.int32)
with tf.Session() as sess:
  print("# build graph and run")
  input1= tf.placeholder(tf.int32, [10], name="input")
  output1= tf.add(input1, tf.constant(100,dtype=tf.int32), name="output") #  data depends on the input data
  saved_result= tf.Variable(data, name="saved_result")
  do_save=tf.assign(saved_result,output1)
  tf.initialize_all_variables()
  os.system("rm -rf /tmp/load")
  tf.train.write_graph(sess.graph_def, "/tmp/load", "test.pb", False) #proto
  # now set the data:
  result,_=sess.run([output1,do_save], {input1: data}) # calculate output1 and assign to 'saved_result'
  saver = tf.train.Saver(tf.all_variables())
  saver.save(sess,"checkpoint.data")

with tf.Session() as persisted_sess:
    from restore_graph import restore_graph
    from tensorflow.python.summary.event_accumulator import EventAccumulator
    events = EventAccumulator('/tmp/load/test.pb')
    events.Reload()
    graph = events.Graph()
    
    #with gfile.FastGFile("model.pbtxt",'rb') as f:
    #    graph_def = tf.GraphDef()
    #    graph_def.ParseFromString(f.read())

    #    print(graph_def)
    #    whatever, saver = restore_graph(
    #        graph_def,
    #        "model.ckpt"
    #    )
    #    print(whatever)
 
    (x,y), saver = restore_graph(
        graph,
        "model.ckpt",
        return_elements=['x', 'y']
    )
    print(x,y)
     
    print("load graph")
    with gfile.FastGFile("/tmp/load/test.pb",'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      persisted_sess.graph.as_default()
      tf.import_graph_def(graph_def, name='')
    print("map variables")
    persisted_result = persisted_sess.graph.get_tensor_by_name("saved_result:0")
    tf.add_to_collection(tf.GraphKeys.VARIABLES,persisted_result)
    try:
      saver = tf.train.Saver(tf.all_variables()) # 'Saver' misnomer! Better: Persister!
    except:pass
    print("load data")
    saver.restore(persisted_sess, "checkpoint.data")  # now OK
    print(persisted_result.eval())
    print("DONE")
