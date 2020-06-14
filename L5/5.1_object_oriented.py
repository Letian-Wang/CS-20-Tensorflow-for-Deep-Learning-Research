import os
import tensorflow as tf
import numpy as np

''' tf.gradients(y, [xs]) '''
# Take derivative of y with respect to each tensor in the list [xs]
x = tf.Variable(2.0)
y = 2.0 * (x**3)
z = 3.0 + y ** 2
grad_z = tf.gradients(z, [x,y])
with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    sess.run(x.initializer)
    print(sess.run(grad_z))

# CS 244
class model:
    def __init__(self, params):
        pass
    def _create_placeholders(self):
        ''' Step 1: Define the placeholders for input and output ''' 
        pass
    def _create_embedding(self):
        ''' Step 2: build variables ''' 
    def _create_loss(self):
        ''' Step 3: define inference and loss function '''
    def _create_optimizers(self):
        ''' Setp 4: define optimizer '''
        
class Model(self):
    def load_data(self):

    def add_placeholders(self):

    def create_feed_dict(self):

    def add_model(self, input_data):

    def add_loss_op(self, pred):

    def run_epoch(self, sess, input_data, input_labels):

    def fit(self, sess, input_data, input_labels):

    def predict(self, sess, input_data, input_labels=None):

    def __init__(self, config):
        self.config = config
        self.load_data()
        self.add_placeholders()
        window = self.add_embedding
        y = self.add_model()
        self.loss = self.add_loss_op(y)
        self.predictions=tf.nn.softmax(y)



''' another model '''
class NERModel(LanguageModel):
    def load_data(self, debug=False):

    def add_placeholders(self):
        self.input_placeholder = tf.placehodler(tf.init32, [None, self.config.window_size])
        self.labels_placeholder = tf.placeholder(tf.float32, [None, self.config.label_size])
        