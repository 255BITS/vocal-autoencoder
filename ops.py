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



