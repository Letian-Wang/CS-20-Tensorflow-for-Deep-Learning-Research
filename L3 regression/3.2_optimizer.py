import tensorflow as tf
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
_, l = sess.run([optimizer, loss], feed_dict={X:x, Y:y})

Session looks at all trainable variables that loss depends on and update them
tf.Variable(initializer=None, trainable=True, collections=None, validate_shape=True, caching_device=None,
            name=None, variable_def=None, dtype=None, expected_shape=None, import_scope=None)

List of optimizers in TF
1. tf.train.GradientDescentOptimizer
2. tf.train.AdagradOptimizer
3. tf.train.MomentumOptimizer
4. tf.train.AdamOptimizer
5. tf.train.ProximalGradientDescentOptimizer
6. tf.train.ProximalAdagradOptimizer
7. tf.train.RMSPropOptimizer
And more