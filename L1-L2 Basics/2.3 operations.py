a = tf.constant([3, 6])
b = tf.constant([2, 2])
tf.add(a, b)
tf.add_n([a, b, b])         # a + b + b
tf.mul(a, b)                # element wise
tf.matmul(a, b)             # ValueError
tf.matmul(tf.reshape(a, [1,2]), tf.reshape(b, [2,1]))
tf.div(a, b)                # element wise
tf.mod(a, b)                # element wise
