import tensorflow as tf
# Create a graph on a specified GPU
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b')
    c = tf.matmul(a, b)

# Creates a session with log_device_placement set to True
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Run the op
print(sess.run(c))


''' Create graph '''
g = tf.Graph()                  # Create graph
with g.as_default():            # Set graph g as default graph
    a = 3
    b = 5
    x = tf.add(a+b)
sess = tf.Session(graph=g)      # session is run on the graph g

''' '''
g = tf.get_default_graph()      # to handle the default graph



''' not mix graph ''' 
g1 = tf.get_default_graph()     # get default graph provided by tf
g2 = tf.Graph()                 # create a new graph
with g1.as_default():           # operations on graph g1
    a = tf.constant(3)
with g2.as_default():           # operations on graph g2
    b = tf.constant(5)


''' print out the graph def '''
import tensorflow as tf
my_const = tf.constant([1.0, 2.0], name='my_const')
my_const2 = tf.constant([1.0, 2.0], name='my_const2')
with tf.Session() as sess:
    print(sess.graph.as_graph_def())


Why graphs:
1. Save computation (only run subgraphs that lead to the values you want to fetch)
2. Break computation into small, differential pieces to facilitates auto-differentiation
3. Facilitate distributed computation, spread the work across multiple CPUs, GPUs, or devices
4. Many common machine learning models are commonly taught and visualized as directed graphs already