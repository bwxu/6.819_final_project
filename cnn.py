import tensorflow as tf
import getdata
'''
     \      ,    I'M RICHARD STALLMAN AND I DON'T WEAR SHOES IN PUBLIC
     l\   ,/     BECAUSE NO ONE TAUGHT ME HOW TO BEHAVE               
._   `|] /j                                                           
 `\\, \|f7 _,/'                                                       
   "`=,k/,x-'                                                         
    ,z/fY-=-                                                          
  -'" .y \                                                            
      '   \itz   OR WASH FOR THAT MATTER                              

'''

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def atrous_conv(x, W, stride, dilation):
    return tf.nn.atrous_conv2d(x, W,
        rate=dilation, padding='SAME')

def conv(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1],
        padding='SAME')

def deconv(x, W, out_shape, stride, dopad):
    return tf.nn.conv2d_transpose(x, W, out_shape, strides=[1, stride, stride, 1],
                        padding='SAME' if dopad else 'VALID')

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


# https://github.com/richzhang/colorization/blob/master/models/colorization_train_val_v2.prototxt
# github.com/richzhang/colorization/blob/master/models/colorization_deploy_v2.prototxt
def cnn_model(x):
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    BATCH_SIZE = int(list(x.get_shape())[0])
    print 'Batch Size is ' + str(BATCH_SIZE)

    ################## Conv 1 ##################
    
    Wconv1_1 = weight_variable([3, 3, 1, 64])
    bconv1_1 = bias_variable([64])
    
    Rconv1_1 = tf.nn.relu(conv(x, Wconv1_1, 1) + bconv1_1)

    Wconv1_2 = weight_variable([3, 3, 64, 64])
    bconv1_2 = bias_variable([64])

    Rconv1_2 = tf.nn.relu(conv(Rconv1_1, Wconv1_2, 2) + bconv1_2)

    Rnorm1 = batch_norm(Rconv1_2, 64, phase_train)
    print Rnorm1.get_shape()
    
    print 'END LAYER 1'
    
    ################## Conv 2 ##################
    
    Wconv2_1 = weight_variable([3, 3, 64, 128])
    bconv2_1 = bias_variable([128])
    
    Rconv2_1 = tf.nn.relu(conv(Rnorm1, Wconv2_1, 1) + bconv2_1)
    
    Wconv2_2 = weight_variable([3, 3, 128, 128])
    bconv2_2 = bias_variable([128])
    
    Rconv2_2 = tf.nn.relu(conv(Rconv2_1, Wconv2_2, 2) + bconv2_2)
        
    Rnorm2 = batch_norm(Rconv2_2, 128, phase_train)
    print Rnorm2.get_shape()
    
    print 'END LAYER 2'

    ################## Conv 3 ################## 

    Wconv3_1 = weight_variable([3, 3, 128, 256])
    bconv3_1 = bias_variable([256])
    
    Rconv3_1 = tf.nn.relu(conv(Rnorm2, Wconv3_1, 1) + bconv3_1)
    
    Wconv3_2 = weight_variable([3, 3, 256, 256])
    bconv3_2 = bias_variable([256])
    
    Rconv3_2 = tf.nn.relu(conv(Rconv3_1, Wconv3_2, 1) + bconv3_2)
    
    Wconv3_3 = weight_variable([3, 3, 256, 256])
    bconv3_3 = bias_variable([256])
    
    Rconv3_3 = tf.nn.relu(conv(Rconv3_2, Wconv3_3, 2) + bconv3_3)
    
    Rnorm3 = batch_norm(Rconv3_3, 256, phase_train)
    print Rnorm3.get_shape()
    print 'END LAYER 3'

    ################## Conv 4 ################## 
    
    Wconv4_1 = weight_variable([3, 3, 256, 512])
    bconv4_1 = bias_variable([512])
    
    Rconv4_1 = tf.nn.relu(conv(Rnorm3, Wconv4_1, 1) + bconv4_1)
    
    Wconv4_2 = weight_variable([3, 3, 512, 512])
    bconv4_2 = bias_variable([512])
    
    Rconv4_2 = tf.nn.relu(conv(Rconv4_1, Wconv4_2, 1) + bconv4_2)
    
    Wconv4_3 = weight_variable([3, 3, 512, 512])
    bconv4_3 = bias_variable([512])
    
    Rconv4_3 = tf.nn.relu(conv(Rconv4_2, Wconv4_3, 1) + bconv4_3)
    
    Rnorm4 = batch_norm(Rconv4_3, 512, phase_train)
    print Rnorm4.get_shape()
    print 'END LAYER 4'

    ################## Conv 5 ################## 
    
    Wconv5_1 = weight_variable([3, 3, 512, 512])
    bconv5_1 = bias_variable([512])
    
    Rconv5_1 = tf.nn.relu(atrous_conv(Rnorm4, Wconv5_1, 1, 2) + bconv5_1)
    
    Wconv5_2 = weight_variable([3, 3, 512, 512])
    bconv5_2 = bias_variable([512])
    
    Rconv5_2 = tf.nn.relu(atrous_conv(Rconv5_1, Wconv5_2, 1, 2) + bconv5_2)
    
    Wconv5_3 = weight_variable([3, 3, 512, 512])
    bconv5_3 = bias_variable([512])
    
    Rconv5_3 = tf.nn.relu(atrous_conv(Rconv5_2, Wconv5_3, 1, 2) + bconv5_3)
    
    Rnorm5 = batch_norm(Rconv5_3, 512, phase_train)
    print Rnorm5.get_shape()
    print 'END LAYER 5'

    ################## Conv 6 ################## 
    
    Wconv6_1 = weight_variable([3, 3, 512, 512])
    bconv6_1 = bias_variable([512])
    
    Rconv6_1 = tf.nn.relu(atrous_conv(Rnorm5, Wconv6_1, 1, 2) + bconv6_1)
    
    Wconv6_2 = weight_variable([3, 3, 512, 512])
    bconv6_2 = bias_variable([512])
    
    Rconv6_2 = tf.nn.relu(atrous_conv(Rconv6_1, Wconv6_2, 1, 2) + bconv6_2)
    
    Wconv6_3 = weight_variable([3, 3, 512, 512])
    bconv6_3 = bias_variable([512])
    
    Rconv6_3 = tf.nn.relu(atrous_conv(Rconv6_2, Wconv6_3, 1, 2) + bconv6_3)
    
    Rnorm6 = batch_norm(Rconv6_3, 512, phase_train)
    print Rnorm6.get_shape()
    print 'END LAYER 6'


    ################## Conv 7 ################## 
    
    Wconv7_1 = weight_variable([3, 3, 512, 512])
    bconv7_1 = bias_variable([512])
    
    Rconv7_1 = tf.nn.relu(conv(Rnorm6, Wconv7_1, 1) + bconv7_1)
    
    Wconv7_2 = weight_variable([3, 3, 512, 512])
    bconv7_2 = bias_variable([512])
    
    Rconv7_2 = tf.nn.relu(conv(Rconv7_1, Wconv7_2, 1) + bconv7_2)
    
    Wconv7_3 = weight_variable([3, 3, 512, 512])
    bconv7_3 = bias_variable([512])
    
    Rconv7_3 = tf.nn.relu(conv(Rconv7_2, Wconv7_3, 1) + bconv7_3)
    
    Rnorm7 = batch_norm(Rconv7_3, 512, phase_train)
    print Rnorm7.get_shape()
    print 'END LAYER 7'

    ################## Conv 8 ################## 
    
    Wconv8_1 = weight_variable([4, 4, 256, 512])
    bconv8_1 = bias_variable([256])
    
    Rconv8_1 = tf.nn.relu(deconv(Rnorm7, Wconv8_1, [BATCH_SIZE, 64, 64, 256], 2, True) + bconv8_1)
    
    Wconv8_2 = weight_variable([3, 3, 256, 256])
    bconv8_2 = bias_variable([256])
    
    Rconv8_2 = tf.nn.relu(conv(Rconv8_1, Wconv8_2, 1) + bconv8_2)
    
    Wconv8_3 = weight_variable([3, 3, 256, 256])
    bconv8_3 = bias_variable([256])
    
    Rconv8_3 = tf.nn.relu(conv(Rconv8_2, Wconv8_3, 1) + bconv8_3)  
    print Rconv8_3.get_shape()
    print 'END LAYER 8'

    ################## Softmax Layer ##################
    
    WconvS_313 = weight_variable([1, 1, 256, 313])
    bconvS_313 = weight_variable([313])
    
    convS_313 = conv(Rconv8_3, WconvS_313, 1) + bconvS_313
    convS_shape = [int(i) for i in convS_313.get_shape()] 
    WconvS_scale = weight_variable(convS_shape)
    
    softmax_out = tf.nn.softmax(tf.mul(convS_313, WconvS_scale))
    print softmax_out.get_shape()
    print 'END SOFTMAX'

    ################## Decoding ##################
    
    WconvD = weight_variable([1, 1, 2, 313])
    bconvD = weight_variable([2])
    
    convD = deconv(softmax_out, WconvD, [BATCH_SIZE, 256, 256, 2], 1, 1) + bconvD
    print convD.get_shape()

    return convD


# training and evaluating, but not for our case
# https://www.tensorflow.org/versions/r0.12/tutorials/mnist/pros/index.html#convolution-and-pooling

def train_cnn(x, y_):
    prediction = cnn_model(x)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    with tf.Session() as sess:
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
        
if __name__ == '__main__':

    # training_data = matrix with dimension [num_images, 256, 256, 1]
    # x = tf.placeholder('float', [10, 256, 256, 1])
    # y = tf.placeholder('float', [10, 256, 256, 3])
    x, y = getdata.read_data()
    train_cnn(x, y)

    

