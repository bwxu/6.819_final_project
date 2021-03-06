import tensorflow as tf
import numpy as np
import getdata
from skimage import io, color, exposure
from PIL import Image
from scipy import misc

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

# This method contains our CNN model.
# This method also contains the CNN architecture based off of Richard Zhang's paper
# https://github.com/richzhang/colorization/blob/master/models/colorization_train_val_v2.prototxt
# github.com/richzhang/colorization/blob/master/models/colorization_deploy_v2.prototxt
def cnn_model(x):
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    BATCH_SIZE = int(list(x.get_shape())[0])


    W = weight_variable([3, 3, 1, 2])
    b = weight_variable([2])

    conv1 = tf.nn.relu(conv(x, W, 1) + b)

    W2 = weight_variable([1, 1, 2, 4])
    b2 = weight_variable([4])
    
    conv2 = tf.nn.relu(conv(conv1, W2, 1) + b2)

    W3 = weight_variable([1, 1, 4, 2])
    b3 = weight_variable([2])

    conv3 = tf.nn.relu(conv(conv2, W3, 1) + b3)

    W4 = weight_variable([1, 1, 2, 2])
    b4 = weight_variable([2])

    conv4 = conv(conv3, W4, 1) + b4

    return conv4

    # Here is the CNN implementation inspired from the paper
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
    print convS_313.get_shape()
    convS_shape = [int(i) for i in convS_313.get_shape()] 
    WconvS_scale = weight_variable(convS_shape)
    print WconvS_scale.get_shape()
    softmax_out = tf.nn.softmax(tf.mul(convS_313, WconvS_scale))
    print softmax_out.get_shape()
    print 'END SOFTMAX'

    ################## Decoding ##################
    
    WconvD = weight_variable([1, 1, 2, 313])
    bconvD = weight_variable([2])
    
    convD = deconv(softmax_out, WconvD, [BATCH_SIZE, 256, 256, 2], 1, True) + bconvD
    print convD.get_shape()

    return convD


def save_image(lab, image_name):
    scaled_lab = exposure.rescale_intensity(lab, in_range=(np.amin(lab), np.amax(lab)))
    rgb = color.lab2rgb(scaled_lab)
    misc.imsave(image_name, rgb)


# training and evaluating, but not for our case
# https://www.tensorflow.org/versions/r0.12/tutorials/mnist/pros/index.html#convolution-and-pooling

def train_cnn():
    batch_size = 1
    x = tf.placeholder('float', [batch_size, 256, 256, 1])
    y = tf.placeholder('float', [batch_size, 256, 256, 2])
    prediction = cnn_model(x)
    square_loss = tf.reduce_sum(tf.square(prediction-y))
    train_step = tf.train.AdamOptimizer().minimize(square_loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Get the Lab values of each of the images
        print 'READING IMAGES...'
        a, b = getdata.read_data_directly()
        num_examples = a.shape[0]
        
        image_num = 0

        print 'TRAINING NEURAL NET...'
        num_epochs = 10
        for epoch in range(num_epochs):
            epoch_loss = 0
            for i in range(int(num_examples/batch_size)):
                # don't train with test image
                if image_num >= i*batch_size and image_num < (i+1)*batch_size:
                    continue

                x_val = a[i*batch_size:(i+1)*batch_size,:,:,:]
                if x_val.shape != (batch_size, 256, 256, 1):
                    continue

                y_val = b[i*batch_size:(i+1)*batch_size,:,:,:]
                if y_val.shape != (batch_size, 256, 256, 2):
                    continue

                _, loss_val = sess.run([train_step, square_loss], feed_dict={x: x_val , y: y_val})
                epoch_loss += loss_val
            print 'EPOCH: ' + str(epoch+1), 'LOSS VALUE: ' + str(epoch_loss)

        test_x = a[image_num:image_num+batch_size,:,:,:]
        test_y = sess.run(prediction, feed_dict={x: test_x})
        test_lab = np.concatenate((test_x, test_y), axis=3)
        test_lab = test_lab[0,:,:,:]
        save_image(test_lab, 'prediction.jpg')
        #actual_lab = np.concatenate((a[image_num,:,:,:], b[image_num,:,:,:]), axis = 2)
        #save_image(actual_lab, 'actual.jpg')
        
if __name__ == '__main__':
    train_cnn()

    

