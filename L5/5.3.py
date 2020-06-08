# Visualize our summary statistics furing our training
tf.summary.scalar()
tf.summary.histogram()
tf.summary.image()

# Step1: create summaries
with tf.name_scope('summaries'):
    tf.summary.scalar('loss', self.loss)
    tf.summary.scalar('accuracy', self.accuracy)
    tf.summary.histogram('histogram loss', self.loss)
    # merge then all, not need to run them one by one
    self.summary_op = tf.summary.merge_all()
    
# Step2: run them (summaries are ops, too)
loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op], feed_dict=feed_dict)

# Step3； write summaries to file
writer.add_summary(summary, global_step=step)