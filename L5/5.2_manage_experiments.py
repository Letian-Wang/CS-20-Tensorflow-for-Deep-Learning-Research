import tensorflow as tf
# Save graph's variables in binary files
tf.train.Saver()

# Save sessions, not graphs
tf.train.Saver.save(sess, save_path, global_step=None)


''' example '''
# 1. define model
# 2. create a saver object
saver = tf.train.Saver()
# 3. launch a session to compute the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    for step in range(training_steps):
        sess.run([optimizer])
        if (step + 1) % 1000 == 0:
            saver.save(sess, 'checkpoing_directory/model_name', global_step=model.global_step)
            

self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)   # automatically increment

saver.save(sess, 'my-model', global_step=0)     # ==> filename: my-model-0
saver.save(sess, 'my-model', global_step=1000)     # ==> filename: my-model-1000


''' example '''
saver = tf.train.Saver(...variables...)
sess = tf.Session()
for step in range(1000000):
    sess.run(..training_op..)
    if step%1000==0:
        saver.save(sess, 'my-model', global_step=step)


v1 = tf.Variable(..., name='v1')
v2 = tf.Variable(..., name='v2')

# Pass variables as a dict
saver = tf.train.Saver({'v1':v1, 'v2':v2})
# Or pass them as a list
saver = tf.train.Saver([v1, v2])
# Passing a list is equivalent to passing a dict with the variable op names as keys
saver = tf.train.Saver({v.op.name: v for v in [v1, v2]})


''' Restore variables '''
saver.restore(sess, save_path)
saver.restore(sess, 'checkpoints/name_of_the_checkpoint')
saver.restore(sess, 'checkpoints/skip-gram-99999')

# Restore the latest checkpoint
# 1. checkpoint keeps track of the latest checkpoint
ckpt = tf.train.get_checkpoint_state(os.path.dirnam('checkpoints/checkpoint'))
# 2. Safeguard to restore checkpoints only when there are checkpoints
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)

tf.train.latest_checkpoint(checkpoint_dir, latest_filename=None)


''' Restore graph and variables'''
# Save model:
w1 = tf.Variable(tf.truncated_normal(shape=[10], name='w1'))
w2 = tf.Variable(tf.truncated_normal(shape=[20], name='w2'))
tf.add_to_collection('vars', w1)
tf.add_to_collection('vars', w2)
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, 'my-model')
    # 'save' method will call 'export_meta_graph' implicitly
    # you will get saved graph files:my-model.meta

# Restore model:
sess = tf.Session()
new_saver = tf.train.import_meta_graph('my-model.meta')             # restore graph
new_saver.restore(sess, tf.train.latesst_checkpoint('./'))          # restore variables
all_vars = tf.get_collection('vars')
for v in all_vars:
    v_ = sess.run(v)
    print(v_)