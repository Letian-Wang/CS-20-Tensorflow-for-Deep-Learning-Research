''' Lazy loading:   Defer creating/initializing an object until it is needed '''
import tensorflow as tf

''' Normal loading example: single add node '''
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y)                    # create node before executing the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./my_graoh/l2_normal', sess.graph)
    for _ in range(10):
        sess.run(z)                 # only add one node in the graph
    writer.close()
print(tf.get_default_graph().as_graph_def())

''' Lazy loading example: '''
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./my_graph/l2_lazy', sess.graph)
    for _ in range(10):
        sess.run(tf.add(x, y))      # someone decides to be clever to save one line of code
    writer.close()
print(tf.get_default_graph().as_graph_def())
# multiple add node:
#   Graph gets bloated, slow to load, expensive to pass around
        


''' Solution '''
# 1. Separate definition of ops from computing/running ops
# 2. Use python property to ensure function is also loaded once the first time it is called
#       Especailly for prediction


''' Using property '''
Just splitting the code into fnctiosn doesn't work, since every time the functiosn are called, the graph would be extended by new code.
Therefore, we have to ensure that the operations are added to the graph only when the functions is called for the first time.
This is bascically lazy-loading
class Model:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self._prediction = None
        self._optimize = None
        self._error = None

    @property
    def prediction(self):
        if not self._prediction:
            data_size = int(self.data.get_shape()[1])
            target_size = int(self.target.get_shape()[1])
            weight = tf.Variable(tf.truncated_normal([data_size, target_size]))
            bias = tf.Variable(tf.constant(0.1, shape=[target_size]))
            incoming = tf.matmul(self.data, weight) + bias
            self._prediction = tf.nn.softmax(incoming)
        return self._prediction
            
    @property
    def optimize(self):
        if not self._optimize:
            cross_entropy = -tf.reduce_sum(self.target, tf.log(self.prediction))
            optimizer = tf.train.RMSPropOptimizer(0.03)
            self._optimize = optimizer.minimize(cross_entropy)
        return self._optimize
        
    @property
    def error(self):
        if not self._error:
            mistakes = tf.not_equal(
                tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
            self._error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        return self._error