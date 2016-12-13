# from kaffe.tensorflow import Network
import tensorflow as tf


def make_var(name, shape):
        '''Creates a new TensorFlow variable.'''
        return tf.get_variable(name, shape)

def conv(input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             dilation = 1,
             relu=True,
             padding='SAME',
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid
        # Convolution for a given input and kernel
        # kernel height/width, number challens, stride height/width, relu = true/false
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        # if dilation is not None:
        # convolve = lambda i, k: tf.nn.convolution(i, k, padding, strides=[1, s_h, s_w, 1], dilation_rate=dilation)

        with tf.variable_scope(name) as scope:
            kernel = make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
                # This is the common-case. Convolve the input without any further complications.
            output = convolve(input, kernel)
            # Add the biases
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

def batch_normalization(input, name, scale_offset=True, relu=False):
        # NOTE: Currently, only inference is supported
        with tf.variable_scope(name) as scope:
            shape = [input.get_shape()[-1]]
            if scale_offset:
                scale = make_var('scale', shape=shape)
                offset = make_var('offset', shape=shape)
            else:
                scale, offset = (None, None)
            output = tf.nn.batch_normalization(
                input,
                mean=make_var('mean', shape=shape),
                variance=make_var('variance', shape=shape),
                offset=offset,
                scale=scale,
                # TODO: This is the default Caffe batch norm eps
                # Get the actual eps from parameters
                variance_epsilon=1e-5,
                name=name)
            if relu:
                output = tf.nn.relu(output)
            return output

def unpool(value, name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        dim = len(sh[1:-1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat(i, [out, tf.zeros_like(out)])
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size, name=scope)
    return out

def most_of_cnn_model(x):
    def layer(lol):
        print lol, x.get_shape()
    batch_size = 1
    x = conv(x, 3, 3, 64, 1, 1, name='bw_conv1_1')
    layer('bw_conv1_1')
    x = conv(x, 3, 3, 64, 2, 2, name='conv1_2')
    layer('conv1_2')
    x = batch_normalization(x, 'conv1_2norm', False)

    x = conv(x, 3, 3, 128, 1, 1, name='conv2_1')
    layer('conv2_1')
    x = conv(x, 3, 3, 128, 2, 2, name='conv2_2')
    layer('conv2_2')
    x = batch_normalization(x, 'conv2_2norm', False)

    x = conv(x, 3, 3, 256, 1, 1, name='conv3_1')
    layer('conv3_1')
    x = conv(x, 3, 3, 256, 1, 1, name='conv3_2')
    layer('conv3_2')
    x = conv(x, 3, 3, 256, 2, 2, name='conv3_3')
    layer('conv3_3')
    x = batch_normalization(x, 'conv3_3norm', False)

    x = conv(x, 3, 3, 512, 1, 1, name='conv4_1')
    layer('conv4_1')
    x = conv(x, 3, 3, 512, 1, 1, name='conv4_2')
    layer('conv4_2')
    x = conv(x, 3, 3, 512, 1, 1, name='conv4_3')
    layer('conv4_3')
    x = batch_normalization(x, 'conv4_3norm', False)

    x = conv(x, 3, 3, 512, 1, 1, name='conv5_1', dilation=2)
    layer('conv5_1')
    x = conv(x, 3, 3, 512, 1, 1, name='conv5_2', dilation=2)
    layer('conv5_2')
    x = conv(x, 3, 3, 512, 1, 1, name='conv5_3', dilation=2)
    layer('conv5_3')
    x = batch_normalization(x, 'conv5_3norm', False)
    
    x = conv(x, 3, 3, 512, 1, 1, name='conv6_1', dilation=2)
    layer('conv6_1')
    x = conv(x, 3, 3, 512, 1, 1, name='conv6_2', dilation=2)
    layer('conv6_2')
    x = conv(x, 3, 3, 512, 1, 1, name='conv6_3', dilation=2)
    layer('conv6_3')
    x = batch_normalization(x, 'conv6_3norm', False)
    
    x = conv(x, 3, 3, 256, 1, 1, name='conv7_1')
    layer('conv7_1')
    x = conv(x, 3, 3, 256, 1, 1, name='conv7_2')
    layer('conv7_2')
    x = conv(x, 3, 3, 256, 1, 1, name='conv7_3')
    layer('conv7_3')
    x = batch_normalization(x, 'conv7_3norm', False)
    
    # x = conv(x, 4, 4, 128, 1, 1, name='conv8_1')
    # from 28 28 256 to 56 56 128
    kernel = make_var('weights', shape=[4, 4, 128, x.get_shape()[-1]])
    x = tf.nn.conv2d_transpose(x, kernel, [batch_size, 56, 56, 128], strides = [1, 2, 2, 1], padding='SAME')
    x = tf.nn.relu(x)

    layer('conv8_1')
    x = conv(x, 3, 3, 128, 1, 1, name='conv8_2')
    layer('conv8_2')
    x = conv(x, 3, 3, 128, 1, 1, name='conv8_3')
    layer('conv8_3')

    x = conv(x, 1, 1, 313, 1, 1, relu=False, name='conv8_313')
    return x


