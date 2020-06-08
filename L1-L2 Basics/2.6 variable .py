# Only use constants for primitive types.
# Use variables or readers for more data that requires more memory
import tensorflow as tf

tf.Variable(initial_value=None, name=None, dtype=None, shape=None)

''' Create '''
# Create variable a with scalar value
a = tf.Variable(2, name='scalar')
# Create variable b as a vector
b = tf.Variable([2, 3], name='vector')
# Create variable c as a matrix
c = tf.Variable([[0,1], [2,3]], name='matrix')
# Create variable W as 784*10 tensor, filled with zeros
W = tf.Variable(tf.zeros([784, 10]))

# tf.Variable() is a class, tf.constant() is an operator

''' operation '''
x = tf.Variable(5)
x.initializer       # init op
x.value()           # read op
x.assign()          # write op
x.assign_add()      # add and assign op


''' Have to initialize your variables '''
# Easiest way to initialize all variables at once
init = tf.global_variables_initializer()        
with tf.Session() as sess:
    sess.run(init)
# Initialize subset of variables
init_ab = tf.variables_initializer([a, b], name='init_ab')      
with tf.Session() as sess:
    sess.run(init_ab)
# Initialize a single variable
W = tf.Variable(tf.zeros((10,10)))
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W.eval())
# Use a variable to initialize another variable
W = tf.Variable(tf.truncated_normal([700, 10]))
U = tf.Variable(2 * W)                              # not that safe
U = tf.Variable(2 * W.initialized_value())          # saft: initialize W then U

''' assign '''
# need to be run
W = tf.Variable(10)
W.assign(100)
assign_op = W.assign(100)
with tf.Session() as sess:
    # sess.run(W.initializer)           # not neccesary
    sess.run(assign_op)                 # need to be run, initialize and assign
    print(W.eval())

# succesive calling operation
my_var = tf.Variable(2, name="my_var")
my_var_times_two = my_var.assign(2 * my_var)
with tf.Session() as sess:
    sess.run(my_var.initializer)
    sess.run(my_var_times_two)          # >> 4
    sess.run(my_var_times_two)          # >> 8
    sess.run(my_var_times_two)          # >> 16

# assign_add() and assign_sub()
my_var = tf.Variable(10)
with tf.Session() as sess:
    sess.run(my_var.initializer)        # assign_add() and assign_sub() can't initialize the variable automatically
    sess.run(my_var.assign_add(10))     # >> 20
    sess.run(my_var.assign_sub(2))      # >> 18

# succesive calling operation
W = tf.Variable(10)
sess = tf.Session()
sess.run(W.initializer)
print(sess.run(W.assign_add(10)))      # >> 20
print(sess.run(W.assign_add(100)))      # >> 120
sess.close()

# make sure some ops have been executed to avoid unneccesary errors
# your graph g has 5 ops: a, b, c, d, e
with g.control_dependencies([a, b, c]):
    # 'd', 'e' will only run after 'a', 'b', and 'c' have been executed
    d = ...
    e = ...