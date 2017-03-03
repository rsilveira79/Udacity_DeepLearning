import tensorflow as tf

a = tf.constant(1234)
b = tf.constant([111,222,222])
x = tf.placeholder(tf.int32)

init = tf.global_variables_initializer()

# defining hyperparameters
n_features = 150
n_labels = 5
W = tf.Variable(tf.truncated_normal((n_features,n_labels)))
b = tf.Variable(tf.zeros(n_labels))

with tf.Session() as sess:
	output = sess.run(x, feed_dict = {x:123})
	y = tf.add(tf.multiply(2,3),3)
	print(y)
	print(output)
	output2=sess.run(y)
	print(output2)
	sess.run(init)