import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
learning_rate = 0.01
batch_size = 128
n_epochs = 30
''' phase 1: build our model '''
# Step1: read in data
mnist = input_data.read_data_sets('/data/mnist', one_hot=True)

# Step2: create placeholders
X = tf.placeholder(tf.float32, [batch_size, 784], name='image')
Y = tf.placeholder(tf.float32, [batch_size, 10], name='label')

# Step3: create variables
W = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), dtype=tf.float32, name='weights')
b = tf.Variable(tf.zeros([1, 10]), dtype=tf.float32, name='bias')

# Step4: build model
# this logits will be later passed through softmax layer
logits = tf.matmul(X, W) + b

# Step5: define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits, Y, name='loss')       # [batch_size, 1]
loss = tf.reduce_mean(entropy)          # compute the mean over all the examples in the batch

# Step6: define optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

''' phase 2 : train our model '''
with tf.Session() as sess:
    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    n_batches = int(mnist.train.num_examples/batch_size)
    for i in range(n_epochs):
        total_loss = 0
        for j in range(n_batches):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            _, batch_loss = sess.run([optimizer, loss], feed_dict={X:x_batch, Y: y_batch})
            total_loss += batch_loss
        print('Epoch {0} Average loss: {1}'.format(i, total_loss / n_batches))
    print('Total time: {0} seconds'.format(time.time()-start_time))
    print('Optimization finnished!')
            
    
# test the model
n_batches = int(mnist.test.num_examples / batch_size)
total_correct_preds = 0
for i in range(n_batches):
    x_batch, y_batch = mnist.test.next_batch(batch_size)
    logits_batch = sess.run([logits], feed_dict={X: x_batch, Y: y_batch})
    probs = tf.nn.softmax(logits_batch)
    correct_preds = tf.equal(tf.argmax(probs, 1), tf.argmax(y_batch, 1))            # 1 means max in row
    correct_num = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
    total_correct_preds += sess.run(correct_num)

print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))
writer.close()
