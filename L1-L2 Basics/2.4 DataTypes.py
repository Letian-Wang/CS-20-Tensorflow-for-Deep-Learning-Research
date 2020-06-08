''' TF takes python natives types: boolean, numeric(int, float), strings '''
# numeric 
t_0 = 19
tf.zeros_like(t_0)      # ==> 0
tf.ones_like(t_0)       # ==> 1

# string
t_1 = ['apple', 'peach', 'banana']
tf.zeros_like(t_1)      # ==> ['', '', '']
tf.ones_like(t_1)       # ==> TypeError

# boolean
t_2 = [[True, False, False],
       [False, False, True],
       [False, True, False]]
tf.zeros_like(t_2)      # ==> all elements are False
tf.ones_like(t_2)       # ==> all elements are True

''' TF vs NP '''
tf.int32 == np.int32    # True
tf.ones([2, 2], np.float32)
tf.Session.run(fetches)        
    # If the requested fetch is a tensor, then the output will be a Numpy ndarray

# not use python native data for tensor
# numpy and tensorflow may not be compatible in future