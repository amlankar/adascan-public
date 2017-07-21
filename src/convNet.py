import os
import tensorflow as tf
import numpy as np
import time

'''
VGG-16 code adapted from https://github.com/machrisaa/tensorflow-vgg
'''

class Vgg16:
    def __init__(self, vgg16_npy_path='/home1/amlan/data/vgg16.npy'):
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, model_options, input_, input_type, keep_prob = 1):
        """
        load variable from npy to build the VGG
        params:
        input_: Input placeholder of shape [batch_size, height, width, num_channels]
        input_type: 'flow' or 'rgb'
        keep_prob: keep probability for dropout applied on fc layers
        """

        start_time = time.time()
        print("Building VGG16...")
        print("Dropout prob is: %f" %(1-keep_prob))

        if input_type == 'rgb':
            VGG_MEAN = [103.939, 116.779, 123.68]
            # Convert RGB to BGR
            red, green, blue = tf.split(3, 3, input_)
            assert red.get_shape().as_list()[1:] == [224, 224, 1]
            assert green.get_shape().as_list()[1:] == [224, 224, 1]
            assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            input_ = tf.concat(3, [
                  blue - VGG_MEAN[0],
                  green - VGG_MEAN[1],
                  red - VGG_MEAN[2],
            ])

        elif input_type == 'flow':
            FLOW_MEAN = [127.5]*model_options['num_channels']
            input_ = input_ - FLOW_MEAN

        if input_type == 'rgb':
            self.conv1_1 = self.conv_layer(input_, "conv1_1", input_type = 'rgb', train_flag=True)
        elif input_type == 'flow':
            self.conv1_1 = self.conv_layer(input_, "conv1_1", input_type = 'flow', train_flag=True)
        else:
            print "Type %s for VGG not understood" %(input_type)
            raise NotImplementedError

        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2", True)
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1", True)
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2", True)
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1", True)
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2", True)
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3", True)
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1", True)
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2", True)
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3", True)
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1", True)
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2", True)
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3", True)
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = tf.nn.dropout(self.fc_layer(self.pool5, "fc6"), keep_prob)
        assert self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = tf.nn.dropout(self.fc_layer(self.relu6, "fc7"), keep_prob)
        self.relu7 = tf.nn.relu(self.fc7)

        #self.fc8 = self.fc_layer(self.relu7, "fc8")
        #self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
        print("build model finished: %ds" % (time.time() - start_time))

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name, train_flag = False, input_type='rgb'):
        with tf.variable_scope(name) as scope:
            if input_type == 'flow' and name == 'conv1_1':
                #filters = self.data_dict[name][0]
                if ('weights' in self.data_dict[name]):
                    num_channels = bottom.get_shape().as_list()[-1]
                    filters = self.data_dict[name]['weights'][:,:,:num_channels,:]
                else:
                    filters = self.data_dict[name][0]
                
                with tf.device('/cpu:0'):
                    filt = tf.get_variable(name='filter',
                                           initializer=filters,
                                           trainable=train_flag)

            elif input_type == 'rgb':
                with tf.device('/cpu:0'):
                    filt = self.get_conv_filter(name, train_flag)

            else:
                print "Convolutional Filter Type %s not understood"%(type)
                raise NotImplementedError

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            with tf.device('/cpu:0'):
                conv_biases = self.get_conv_bias(name,train_flag=train_flag)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            #tf.histogram_summary('adascan/'+name+'_activations', bias)
            #tf.histogram_summary('adascan/'+name+'_weights', filt)

            scope.reuse_variables()
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name) as scope:
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            with tf.device('/cpu:0'):
                weights = self.get_fc_weight(name)
                biases = self.get_fc_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            #tf.histogram_summary('adascan/'+name+'_activations', fc)
            #tf.histogram_summary('adascan/'+name+'_weights', weights)
            scope.reuse_variables()
            return fc

    def get_conv_filter(self, name, train_flag=False):
        if ('weights' in self.data_dict[name]):
            return tf.get_variable(name='filter',
                                   initializer=self.data_dict[name]['weights'],
                                   trainable=train_flag)
        else:
            return tf.get_variable(name='filter',
                                   initializer=self.data_dict[name][0],
                                   trainable=train_flag)

    def get_conv_bias(self, name, train_flag=False):
        if ('biases' in self.data_dict[name]):
            return tf.get_variable(name='biases',
                                   initializer=self.data_dict[name]['biases'],
                                   trainable=train_flag)
        else:
            return tf.get_variable(name='biases',
                                   initializer=self.data_dict[name][1],
                                   trainable=train_flag)

    def get_fc_bias(self, name):
        if ('biases' in self.data_dict[name]):
            return tf.get_variable(name='biases',
                                   initializer=self.data_dict[name]['biases'])
        else:
            return tf.get_variable(name='biases',
                                   initializer=self.data_dict[name][1])

    def get_fc_weight(self, name):
        if ('weights' in self.data_dict[name]):
            return tf.get_variable(name='filter',
                                   initializer=self.data_dict[name]['weights'])
        else:
            return tf.get_variable(name='filter',
                                   initializer=self.data_dict[name][0])
