import tensorflow as tf
import xlrd
import numpy as np
import os
import matplotlib.pyplot as plt

def huber_loss(predicted, label, delta=1.0):            # less sensitive to outlier
    residual = tf.abs(predicted - label)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)
    
# Step1: read in data
DATA_FILE = '../stanford-tensorflow-tutorials/2017/data/fire_theft.xls'
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1
print(data.shape)

# Step2: create placeholders for input X (number of fire) and label Y(number of theft)
X = tf.placeholder(tf.float32, shape=[], name="fire")
Y = tf.placeholder(tf.float32, shape=[], name='theft')

# Step3: Create weight and bias, initialized to 0
W = tf.Variable(0, shape=[], name="weights", dtype=tf.float32)
b = tf.Variable(0, shape=[], name="bias", dtype=tf.float32)

# Step4: build model to predict Y
Y_hat = X * W + b

# Step5: use the square error as the loss function
# loss = tf.square(Y - Y_hat, name='loss')
loss = huber_loss(Y, Y_hat)

# Step6: using gradient descent with learningrate of 0.01 to minimize loss
opt = tf.train.GradientDescentOptimizer(0.001)
train_op = opt.minimize(loss)

with tf.Session() as sess:
    # Step7: initialize the necessary variables
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./my_graph/0./linear_reg', sess.graph)

    # Step8: train the model
    for i in range(100):
        total_loss = 0
        for x, y in data:
            _, l = sess.run([train_op, loss], feed_dict={X:x, Y:y})
            total_loss += l
        print('Epoch {0} Loss: {1}'.format(i, total_loss/n_samples))
    writer.close()
    # 9. output the values of w and b
    w_value, b_value = sess.run([W, b])

X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data' )
plt.plot(X, X*w_value + b_value, 'r', label='Predicted data')
plt.legend()
plt.grid('on')
plt.show()

