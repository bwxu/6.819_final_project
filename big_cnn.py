import tensorflow as tf
import numpy as np
import getdata_224
from skimage import io, color, exposure
from PIL import Image
from scipy import misc
import lol_model

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


def cnn_model(x):
    # phase_train = tf.placeholder(tf.bool, name='phase_train')
    BATCH_SIZE = int(list(x.get_shape())[0])
    convS_313 = lol_model.most_of_cnn_model(x)
    convS_shape = [int(i) for i in convS_313.get_shape()] 
    WconvS_scale = weight_variable(convS_shape)
    print WconvS_scale.get_shape()
    softmax_out = tf.nn.softmax(tf.mul(convS_313, WconvS_scale))
    print softmax_out.get_shape()
    print 'END SOFTMAX'

    ################## Decoding ##################
    
    # # WconvD = weight_variable([1, 1, 2, 313])
    # # bconvD = weight_variable([2])
    
    # # convD = deconv(softmax_out, WconvD, [BATCH_SIZE, 224, 224, 2], 1, True) + bconvD
    # # print convD.get_shape()
    # convD = lol_model.conv(softmax_out, 1, 1, 2, 1, 1, name='convD')
    # print convD.get_shape()
    kernel = lol_model.make_var('weights2', shape=[1, 1, 2, softmax_out.get_shape()[-1]])
    softmax_out = tf.nn.conv2d_transpose(softmax_out, kernel, [BATCH_SIZE, 224, 224, 2], strides = [1, 4, 4, 1], padding='SAME')
    softmax_out = tf.nn.relu(softmax_out)
    print softmax_out.get_shape()
    return softmax_out

# saves lab value as image in current folder
def save_image(lab, image_name):
    # scaled_lab = exposure.rescale_intensity(lab, in_range=(np.amin(lab), np.amax(lab)))
    for i in range(len(lab)):
    	for j in range(len(lab[0])):
            lab[i][j][0] = (lab[i][j][0]/50.0) - 1
            lab[i][j][1] = lab[i][j][1]/80.0 * 0
            lab[i][j][2] = lab[i][j][2]/80.0 * 0

    rgb = color.lab2rgb(lab)
    misc.imsave(image_name, rgb)

# method to train and test cnn
def train_cnn():
    batch_size = 1
    x = tf.placeholder('float', [batch_size, 224, 224, 1])
    y = tf.placeholder('float', [batch_size, 224, 224, 2])
    # prediction = lol_model.most_of_cnn_model(x)
    prediction = cnn_model(x)
    square_loss = tf.reduce_sum(tf.abs(prediction-y))
    train_step = tf.train.AdamOptimizer().minimize(square_loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        # Get the Lab values of each of the images
        # Go to getdata.py to change the MAX_NUMBER of images to read
        # and the IMAGE_DIR to the location of images on your computer
        print 'READING IMAGES...'
        a, b = getdata_224.read_data_directly()
        num_examples = a.shape[0]
        
        print 'TRAINING NEURAL NET...'
        num_epochs = 20
        for epoch in range(num_epochs):
            loss = 0
            image_count = 0
            for i in range(int(num_examples/batch_size)):
                # first image (test image) not used for training
                x_val = a[i*batch_size+1:(i+1)*batch_size+1,:,:,:]
                if x_val.shape != (batch_size, 224, 224, 1):
                    continue
                y_val = b[i*batch_size+1:(i+1)*batch_size+1,:,:,:]
                if y_val.shape != (batch_size, 224, 224, 2):
                    continue
                print "hi"
                image_count += 1
                _, loss_val = sess.run([train_step, square_loss], feed_dict={x: x_val , y: y_val})
                loss += loss_val
            print 'EPOCH: ' + str(epoch+1), 'LOSS VALUE: ' + str(loss*1./image_count)
        
        # test image is first image read
        test_x = a[0:batch_size,:,:,:]
        test_y = sess.run(prediction, feed_dict={x: test_x})
        test_lab = np.concatenate((test_x, test_y), axis=3)
        test_lab = test_lab[0,:,:,:]
        print test_x[0][0][0], test_y[0][0][0], test_lab[0][0]
        save_image(test_lab, 'prediction2.jpg')
        actual_lab = np.concatenate((a[0,:,:,:], b[0,:,:,:]), axis = 2)
        save_image(actual_lab, 'actual2.jpg')
        
if __name__ == '__main__':
    train_cnn()

    
