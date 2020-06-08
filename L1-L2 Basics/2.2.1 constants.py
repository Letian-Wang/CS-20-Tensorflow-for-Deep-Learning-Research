import tensorflow as tf
''' constants '''
tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)
# verify_shape=True: make sure the value and shape match, throw error
# verify_shape=False: repeat the last value of value to match shape
# dtype/shape: assigned or infered
a = tf.constant(2, shape = [2,2], verify_shape=True)
b = tf.constant(2, shape = [2,2])
c = tf.constant([2,1],shape=[3,3])

tf.InteractiveSession()     # no need to call session


### name, broadcast
a = tf.constant([2, 2], name="a")
b = tf.constant([[0,1], [2,3]], name="b")
x = tf.add(a, b, name="add")                    # broadcast similar to numpy
y = tf.multiply(a, b, name="mul")
with tf.Session() as sess:
    x, y = sess.run([x, y])
    print(x, y)