# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 16:05:37 2019

"""


import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

cardinality = 8
class_num = 2
dropout_rate = 0.5
def conv_layer(input_data, filter_num, kernel, stride, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs = input_data, use_bias = False, filters = filter_num, kernel_size = kernel, strides=stride, padding=padding)
        return network

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Average_pooling(x, pool_size=[2,2], stride = 2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs = x, pool_size = pool_size, strides = stride, padding = padding)

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope = scope,
                   updates_collections = None,
                   decay = 0.9,
                   center = True,
                   scale = True,
                   zero_debias_moving_mean = False) :
        return tf.cond(training,
                       lambda : batch_norm(inputs = x, is_training = training, reuse = None),
                       lambda : batch_norm(inputs = x, is_training = training, reuse = True))

def Relu(x):
    return tf.nn.relu(x)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x) :
    return tf.layers.dense(inputs = x, use_bias = False, units = class_num, name = 'linear')

def Sigmoid(x) :
    return tf.nn.sigmoid(x)

def Fully_connected(x, units = class_num, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=False, units=units)
    
def _create_fc_layer(x, output_dim, activation, scope, keep_prob):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        w = tf.get_variable('weights', [x.shape[1], output_dim], 
            initializer=tf.truncated_normal_initializer(stddev = 0.1)
        )
    
        b = tf.get_variable('biases', [output_dim], 
            initializer=tf.constant_initializer(0.0)
        )
    
        # add weights and bias to collections
        tf.add_to_collection("var_weight", w)
        tf.add_to_collection("var_bias", b)
    
        layer_out = tf.nn.dropout(tf.matmul(x, w) + b, keep_prob)
    
        if activation == 'relu':
            layer_out = tf.nn.relu(layer_out)
        elif activation == 'sigmoid':
            layer_out = tf.nn.sigmoid(layer_out)
        elif activation == 'tanh':
            layer_out = tf.nn.tanh(layer_out)
        else:
            raise NotImplementedError('activation not recognized')
    
        return layer_out

class SE_ResNeXt():
    def __init__(self, x, training, reduction_ratio = 4):
        self.reduction_ratio = reduction_ratio
        self.training = training
        self.model = self.Build_SE_ResNext(x)

    def first_layer(self, x, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter_num = 32, kernel=[3, 3], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            
            x = conv_layer(x, filter_num = 64, kernel=[3, 3], stride = 2, layer_name=scope+'_conv2')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            
            x = tf.layers.max_pooling2d(inputs = x, pool_size = [3,3], strides = 2, padding = 'SAME')

            return x

    def transform_layer(self, x, stride, scope):
        input_dim = x.get_shape().as_list()[-1]
        nn = input_dim/(2*cardinality)
        with tf.name_scope(scope):
            x = conv_layer(x, filter_num = nn, kernel=[1,1], stride=stride, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            x = conv_layer(x, filter_num = nn, kernel=[3,3], stride=1, layer_name=scope+'_conv2')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            return x

    def transition_layer(self, x, out_dim, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter_num = out_dim, kernel=[1,1], stride = 1, layer_name = scope+'_conv1')
            x = Batch_Normalization(x, training = self.training, scope = scope+'_batch1')
            # x = Relu(x)

            return x

    def split_layer(self, input_x, stride, layer_name):
        with tf.name_scope(layer_name):
            layers_split = list()
            for i in range(cardinality):
                splits = self.transform_layer(input_x, stride = stride, scope = layer_name + '_splitN_' + str(i))
                layers_split.append(splits)

            return Concatenation(layers_split)
        
    def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):

            squeeze = Global_Average_Pooling(input_x)

            excitation = Fully_connected(squeeze, units = out_dim / ratio, layer_name = layer_name+'_fully_connected1')
            excitation = Relu(excitation)
            excitation = Fully_connected(excitation, units = out_dim, layer_name = layer_name+'_fully_connected2')
            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1,1,1,out_dim])
            scale = input_x * excitation

            return scale

    def SE_residual_layer(self, input_x, out_dim, layer_num, res_block = 3):
        # split + transform(bottleneck) + transition + merge

        for i in range(res_block):
            # input_dim = input_x.get_shape().as_list()[-1]
            input_dim = input_x.get_shape().as_list()[-1]

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1
            x = self.split_layer(input_x, stride = stride, layer_name = 'split_layer_'+layer_num+'_'+str(i))
            x = self.transition_layer(x, out_dim = out_dim, scope = 'trans_layer_'+layer_num+'_'+str(i))
            x = self.squeeze_excitation_layer(x, out_dim = out_dim, ratio = self.reduction_ratio, layer_name = 'squeeze_layer_'+layer_num+'_'+str(i))
   
            if flag is True:
                    pad_input_x = Average_pooling(input_x)
                    pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]]) # [?, height, width, channel]
            else:
                pad_input_x = input_x
            
            input_x = Relu(x + pad_input_x)
    
        return input_x 


    def Build_SE_ResNext(self, input_x):
#       SE_resNext-50
        input_x = self.first_layer(input_x, scope = 'first_layer')

        x = self.SE_residual_layer(input_x, out_dim = 64, layer_num = '1', res_block = 3)
        x = self.SE_residual_layer(x, out_dim = 128, layer_num = '2', res_block = 4)
        x = self.SE_residual_layer(x, out_dim = 256, layer_num = '3', res_block = 6)
        x = self.SE_residual_layer(x, out_dim = 512, layer_num = '4', res_block = 3)

        x = Global_Average_Pooling(x)

        return x
