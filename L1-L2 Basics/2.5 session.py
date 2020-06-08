# Each session maintains its own copy of variable
W = tf.Variable(10)

sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(W.initializer)
sess2.run(W.initializer)

print(sess1.run(W.assign_add(10)))      # >> 20
print(sess2.run(W.assign_add(2)))       # >> 8

print(sess1.run(W.assign_add(100)))      # >> 120
print(sess2.run(W.assign_add(50)))       # >> -42

sess1.close()
sess2.close()


# Session vs InteractiveSession
# InteractiveSession makes itself the default
sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
print(c.eval())
sess.run
