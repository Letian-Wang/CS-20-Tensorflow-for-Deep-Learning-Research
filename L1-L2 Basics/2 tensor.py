import tensorflow as tf
''' constants '''
tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)
# verify_shape=True: make sure the value and shape match
# verify_shape=False: repeat the last value of value to match shape

a = tf.constant(2, shape = [2,2], verify_shape=True)
b = tf.constant(2, shape = [2,2])

tf.InteractiveSession()


###3
a = tf.constant([2, 2], name="a")
b = tf.constant([[0,1], [2,3]], name="b")
x = tf.add(a, b, name="add")
y = tf.mul(a, b, name="mul")
with tf.Session() as sess:
    x, y = sess.run([x, y])
    print(x, y)

tf.zeros(shape, dtype=tf.float32, name=None)
tf.zeros([2, 3], tf.int32)
# input_tensor is [0,1],[2,3],[4,5]
tf.zeros_like(input_tensor)

tf.ones(shape, dtype=tf.float32, name=None)
tf.ones_like(input_tensor, dtype=None, name=None, optimize=True)

tf.fill(dims, value, nmae=None)
tf.fill([2,3], 8)       #[[8,8,8], [8,8,8]]
# numpy:
# 1. create numpy array a
# 2. a.fill(value)


tf.linspace(start, stop, num, name=None)
tf.linspace(10.0, 13.0, 4)

tf.range(start, limit=None, delta=1, dtype=None, name='range')
tf.range(3, 18, 3)

tf.range(5)



# randomly generated constants
tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)    
# if sample is 2 stddevs away from mean, truncate it and resample
tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
tf.random_shuffle(value, seed=None, name=None)  # shuffle on the first dimension
a = tf.constant([[2,1], [2,2], [3,3]])
tf.random_shuffle(a)

tf.random_crop(value, size, seed=None, name=None)               # crop a contiguous shape from the value


tf.multinomial(logits, num_samples, seed=None, name=None)       # 
b = tf.constant(np.random.normal(size=(3,4)))
tf.multinomial(b, 5)

tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)
tf.random_gamma([10], 5)
tf.random_gamma([10], [5, 15])
