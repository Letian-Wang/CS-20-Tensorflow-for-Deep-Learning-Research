# number
tf.linspace(start, stop, num, name=None)        # start/stop are supposed to be float (different from numpy)
tf.linspace(10.0, 13.0, 4)

# step
tf.range(start, limit=None, delta=1, dtype=None, name='range')  # limit excluded
tf.range(3, 18, 3)

tf.range(5)
