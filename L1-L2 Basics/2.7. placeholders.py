import tensorflow as tf

tf.placeholder(dtype, shape=None, name=None)

# create a placeholder of type float 32-bit, shape is a vector of 3 elements
a = tf.placeholder(tf.float32, shape=[3])

# create a constant of type float 32-bit, shape is a vector of 3 elements
b = tf.constant([5, 5, 5], tf.float32)

# use placeholder as you would with a constant or a variable
c = a + b           # Short for tf.add(a, b)

with tf.Session() as sess:
    print(sess.run(c))                              # error because a doesn't have any value
    print(sess.run(c, {a: [1, 2, 3]}))              # feed [1,2,3] to placeholder a 

# feed multiple data points in, one at a time (for training)
with tf.Session() as sess:
    for a_value in list_of_values_for_a:
        print(sess.run(c, {a: a_value}))

''' Quirk '''
# shape=None means that tensors of any shape will be accepted as value for placeholder
# shape=None is easy to construct graphs, but nightmarish for debugging
# shape=None also breaks all following shape inference, which makes many ops not work because they expect certain rank

''' feed: extremly helpful for testing '''
# You can feed_dict any feedable tensor. Placeholder is just a way to indicate that something must be fed.
tf.Graph.is_feedable(tensor)        # True if and only if tensor is feedable

a = tf.add(2, 5)
b = tf.mul(a, 3)
with tf.Session() as sess:
    replace_dict = {a:15}                   # define a dictionary that replaces the value of 'a' with 15
    sess.run(b, feed_dict=replace_dict)     # run session, passing in 'replace_dict' as the value to 'feed_dict'
    # return 45

