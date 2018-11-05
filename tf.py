import tensorflow as tf

i1 = tf.placeholder(tf.int32)
i2 = tf.placeholder(tf.int32)
out = tf.multiply(i1, i2)

sess = tf.Session()

out = print(sess.run(out, feed_dict={i1: [100], i2: [432]}))
