import tensorflow as tf
a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
x = tf.add(a, b, name="add")
# Create the summary writer after graph definition and before running your session
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(x))


# 1. python -m tensorboard.main --logdir="./graphs"
# 2. open it from browser


solid lines: data flows
dotted lines: control dependencies