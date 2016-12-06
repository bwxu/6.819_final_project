import tensorflow as tf

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def atrous_conv(x, W, stride, dilation):
	return tf.nn.atrous_conv2d(x, W, strides=[1, stride, stride, 1],
		dilation, 'SAME')

def conv(x, W, stride):
	return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1],
		'SAME')

def deconv(x, W, stride, dopad):
	return tf.nn.deconv2d(x, W, strides=[1, stride, stride, 1],
		                'SAME' if dopad else 'VALID')

def batch_norm(x, n_out, phase_train):
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

# training_data = matrix with dimension [num_images, 256, 256, 1]


# https://github.com/richzhang/colorization/blob/master/models/colorization_train_val_v2.prototxt

phase_train = tf.placeholder(tf.bool, name='phase_train')

Wconv1_1 = weight_variable([3, 3, 1, 64])
bconv1_1 = bias_variable([64])

Rconv1_1 = tf.nn.relu(conv(training_data, Wconv1_1, 1) + bconv1_1)

Wconv1_2 = weight_variable([3, 3, 64, 64])
bconv1_2 = bias_variable([64])

Rconv1_2 = tf.nn.relu(conv(Rconv1_1, Wconv1_2, 2) + bconv1_2)

Rnorm1 = batch_norm(Rconv1_2, 64, phase_train)




# training and evaluating, but not for our case
# https://www.tensorflow.org/versions/r0.12/tutorials/mnist/pros/index.html#convolution-and-pooling
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))







