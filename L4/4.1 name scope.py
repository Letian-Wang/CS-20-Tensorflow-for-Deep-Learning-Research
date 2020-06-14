# Phase 1 : Assemble graph
# 1. Define placeholders for input and output
# 2. Define inference and model
# 3. Define loss function
# 4. Define optimizer

# Phase 2 : Compute
# 1. Initialize model parameters
# 2. Input training data
# 3. Execute inference on training data
# 4. Compute loss
# 5. Adjust model paramter (return to 2)

import tensorflow as tf
''' -------------------------- Name Scope -------------------------------------'''
# group ops together, better for Tensorboard
with tf.name_scope(name_of_that_scope):
    # declare op_1
    # declare op_2
    # ...

''' example ''' 
with tf.name_scope('data'):
    center_words = tf.placeholder(tf.int32, shape=[batch_size], name='center_words')
    target_words = tf.placeholder(tf.int32, shape=[batch_size], name='target_words')
with tf.name_scope('embed'):
    embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0), name='embed_matrix')
with tf.name_scope('loss'):
    embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')
    nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE], stddev=1.0/math.sqrt(EMBED_SIZE)), name='nce_weight')
    nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, biases=nce_bias, 
            lables=target_words, inputs=embed, num_sampled=NUM_SAMPLED, num_classes=VOCAB_SIZE), name = 'loss')
            