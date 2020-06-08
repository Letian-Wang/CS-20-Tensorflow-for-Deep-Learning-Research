''' zeros '''
tf.zeros(shape, dtype=tf.float32, name=None)
tf.zeros([2, 3], tf.int32)
# input_tensor is [0,1],[2,3],[4,5] (tensor or numpy)
tf.zeros_like(input_tensor, dtype=None, name=None, optimizer=True)
tf.zeros_like(input_tensor)

''' ones '''
tf.ones(shape, dtype=tf.float32, name=None)
tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)


''' fill '''
tf.fill(dims, value, name=None)
tf.fill([2,3], 8)       #[[8,8,8], [8,8,8]]
# numpy:
# 1. create numpy array a
# 2. a.fill(value)

''' muscle memory '''