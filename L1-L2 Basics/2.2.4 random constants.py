# randomly generated constants

''' normal distribution '''
tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)    
# if sample is 2 stddevs away from mean, truncate it and resample

''' uniform distribution '''
tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)

''' shuffle on the first dimension '''
tf.random_shuffle(value, seed=None, name=None)  
    a = tf.constant([[2,1], [2,2], [3,3]])
    tf.random_shuffle(a)

''' crop a contiguous shape from the value '''
tf.random_crop(value, size, seed=None, name=None)              

''' categorical sample '''
tf.multinomial(logits, num_samples, seed=None, name=None)       # 
    b = tf.constant(np.random.normal(size=(3,4)))
    tf.multinomial(b, 5).eval()
    b.eval()

''' gamma distribution '''
tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)
    tf.random_gamma([10], 5)
    tf.random_gamma([10], [5, 15])

''' random seed '''
tf.set_random_seed(seed)        # set deteministic randomness
                                # after graph, before session