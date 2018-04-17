
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *
from tensorflow.python.framework import ops
import os
from main import batch_size,unsup_rate,embedding_dim,image_size

##### Parameters declear ######
    

##### Model Definition #####
def text_encoder(input_t, is_train=True, reuse=False):

    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    w_init = tf.random_normal_initializer(stddev=0.02)
    # input_t = tf.squeeze(tf.one_hot(input_t,10))

    with tf.variable_scope("text_encoder", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_t_ecd = InputLayer(input_t, name='te_input')

        net_t_ecd = DenseLayer(net_t_ecd, n_units=embedding_dim/2, act=lrelu, 
                        W_init=w_init, b_init=None, name='te_h0/dense')
        net_t_ecd = DenseLayer(net_t_ecd, n_units=embedding_dim, act=tf.nn.sigmoid, 
                        W_init=w_init, b_init=None, name='te_h1/dense')

        return net_t_ecd, net_t_ecd.outputs

def wgangp_loss(real_data, fake_data,batch_size,ebd_text,LAMBDA=10):

    alpha = tf.random_uniform(
        shape=[batch_size], 
        minval=0.,
        maxval=1.
    )
    alpha2 = tf.tile(alpha,[28*28*1])
    alpha2 = tf.reshape(alpha2,[28,28,1,batch_size])
    alpha3 = tf.transpose(alpha2,[3,0,1,2])
    differences = fake_data - real_data
    interpolates = real_data + (alpha3*differences)
    _,yyy = discriminator(interpolates,ebd_text,reuse=True)
    gradients = tf.gradients(yyy, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    
    return LAMBDA*gradient_penalty

def domain_classifer(feature,l=1.0,mode="GRL", is_train=True, reuse=False):
    
    w_init = tf.random_normal_initializer(stddev=0.01)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("domain_classifer", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        if mode=='GRL':
            feat = flip_gradient(feature, l)
            net_domain = InputLayer(feat, name='dc_input')
            net_domain = DenseLayer(net_domain, n_units=100, act=lrelu, 
                                W_init=w_init, b_init=None, name='dc_h0/dense')
            net_domain = DenseLayer(net_domain, n_units=2, act=tf.identity, 
                                W_init=w_init, b_init=None, name='dc_h1/dense')
            domain_pred = tf.nn.softmax(net_domain.outputs)

            return domain_pred
        else:
            net_domain = InputLayer(feature, name='dc_input')
            net_domain = DenseLayer(net_domain, n_units=100, act=lrelu, 
                                W_init=w_init, b_init=None, name='dc_h0/dense')
            net_domain = DenseLayer(net_domain, n_units=1, act=tf.identity, 
                                W_init=w_init, b_init=None, name='dc_h1/dense')
            domain_pred = net_domain.outputs

            return domain_pred

    
def image_encoder(input_i, is_train=True, reuse=False):

    ef_dim = 16  # Initial channel number
    w_init = tf.random_normal_initializer(stddev=0.01)
    gamma_init = tf.random_normal_initializer(1., 0.01)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("image_encoder", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_i_ecd = InputLayer(input_i, name='ie_input')

        net_i_ecd = Conv2d(net_i_ecd, ef_dim, (2, 2), (1, 1), act=None,
                        padding='VALID', W_init=w_init, b_init=None, name='ie_h0/conv2d')
        net_i_ecd = BatchNormLayer(net_i_ecd, act=lrelu, is_train=is_train, gamma_init=gamma_init,
                        name='ie_h0/batch_norm')
        
        net_i_ecd = Conv2d(net_i_ecd, ef_dim*2, (4, 4), (2, 2), act=None,
                        padding='VALID', W_init=w_init, b_init=None, name='ie_h1/conv2d')
        net_i_ecd = BatchNormLayer(net_i_ecd, act=lrelu, is_train=is_train, gamma_init=gamma_init,
                        name='ie_h1/batch_norm')

        net_i_ecd = Conv2d(net_i_ecd, ef_dim*4, (4, 4), (2, 2), act=None,
                        padding='VALID', W_init=w_init, b_init=None, name='ie_h2/conv2d')
        net_i_ecd = BatchNormLayer(net_i_ecd, act=lrelu, is_train=is_train, gamma_init=gamma_init,
                        name='ie_h2/batch_norm')

        net_i_ecd = FlattenLayer(net_i_ecd, name='ie_h2/reshape')

        net_i_ecd = DenseLayer(net_i_ecd, n_units=embedding_dim, act=tf.nn.sigmoid, 
                        W_init=w_init, b_init=None, name='ie_h3/dense')

        return net_i_ecd, net_i_ecd.outputs

def image_encoder2(input_i, is_train=True, reuse=False):

    ef_dim = 16  # Initial channel number
    w_init = tf.random_normal_initializer(stddev=0.01)
    gamma_init = tf.random_normal_initializer(1., 0.01)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)

    with tf.variable_scope("image_encoder2", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_i_ecd = InputLayer(input_i, name='ie_input')

        net_i_ecd = Conv2d(net_i_ecd, ef_dim, (2, 2), (1, 1), act=None,
                        padding='VALID', W_init=w_init, b_init=None, name='ie_h0/conv2d')
        net_i_ecd = BatchNormLayer(net_i_ecd, act=lrelu, is_train=is_train, gamma_init=gamma_init,
                        name='ie_h0/batch_norm')
        
        net_i_ecd = Conv2d(net_i_ecd, ef_dim*2, (4, 4), (2, 2), act=None,
                        padding='VALID', W_init=w_init, b_init=None, name='ie_h1/conv2d')
        net_i_ecd = BatchNormLayer(net_i_ecd, act=lrelu, is_train=is_train, gamma_init=gamma_init,
                        name='ie_h1/batch_norm')

        net_i_ecd = Conv2d(net_i_ecd, ef_dim*4, (4, 4), (2, 2), act=None,
                        padding='VALID', W_init=w_init, b_init=None, name='ie_h2/conv2d')
        net_i_ecd = BatchNormLayer(net_i_ecd, act=lrelu, is_train=is_train, gamma_init=gamma_init,
                        name='ie_h2/batch_norm')

        net_i_ecd = FlattenLayer(net_i_ecd, name='ie_h2/reshape')

        net_i_ecd = DenseLayer(net_i_ecd, n_units=embedding_dim, act=tf.nn.sigmoid, 
                        W_init=w_init, b_init=None, name='ie_h3/dense')

        return net_i_ecd, net_i_ecd.outputs

def image_decoder(input_e, is_train=True, reuse=False):
    
    s = image_size #28 x28
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16) # 14 #7
    w_init = tf.random_normal_initializer(stddev=0.01)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    df_dim = 32

    with tf.variable_scope("image_decoder", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_i_dcd = InputLayer(input_e, name='id_input')

        net_i_dcd = DenseLayer(net_i_dcd, 1024, act=tf.identity,
                          W_init=w_init, name='id_h0/dense')
        net_i_dcd = BatchNormLayer(net_i_dcd, act=tf.nn.relu, is_train=is_train, 
                          gamma_init=gamma_init,name='id_h0/batch_norm')

        net_i_dcd = DenseLayer(net_i_dcd, df_dim*4*s4*s4, act=tf.identity,
                          W_init=w_init, name='id_h02/dense')
        net_i_dcd = BatchNormLayer(net_i_dcd, act=tf.nn.relu, is_train=is_train, 
                          gamma_init=gamma_init,name='id_h02/batch_norm')

        net_i_dcd = ReshapeLayer(net_i_dcd, [-1, s4, s4, df_dim*4], name='id_h0/reshape')

        net_i_dcd = DeConv2d(net_i_dcd, 64, (4, 4), (s2, s2), strides=(2, 2), 
                          padding='SAME', W_init=w_init, act=None, name='id_h1/decon2d')
        net_i_dcd = BatchNormLayer(net_i_dcd, act=tf.nn.relu, is_train=is_train, 
                          gamma_init=gamma_init,name='id_h1/batch_norm')

        net_i_dcd = DeConv2d(net_i_dcd, 1, (4, 4), (s, s), strides=(2, 2), 
                          padding='SAME', W_init=w_init, act=tf.nn.sigmoid, name='id_h3/decon2d')
        # net_i_dcd = BatchNormLayer(net_i_dcd, act=tf.nn.relu, is_train=is_train, 
        #                   gamma_init=gamma_init,name='id_h3/batch_norm')

        # net_i_dcd = DeConv2d(net_i_dcd, 1, (2, 2), (s, s), strides=(2, 2), 
        #                   padding='SAME', W_init=w_init, act=tf.nn.tanh, name='id_h4/decon2d')
        # net_i_dcd = BatchNormLayer(net_i_dcd, act=tf.nn.relu, is_train=is_train, 
        # #                   gamma_init=gamma_init,name='id_h4/batch_norm')
        # net_i_dcd.outputs = tf.nn.tanh(net_i_dcd.outputs)

        return net_i_dcd, net_i_dcd.outputs

def image_decoder2(input_e, is_train=True, reuse=False):
    
    s = image_size #28 x28
    s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16) # 14 #7
    w_init = tf.random_normal_initializer(stddev=0.01)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    df_dim = 32

    with tf.variable_scope("image_decoder2", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_i_dcd = InputLayer(input_e, name='id_input')

        net_i_dcd = DenseLayer(net_i_dcd, 1024, act=tf.identity,
                          W_init=w_init, name='id_h0/dense')
        net_i_dcd = BatchNormLayer(net_i_dcd, act=tf.nn.relu, is_train=is_train, 
                          gamma_init=gamma_init,name='id_h0/batch_norm')

        net_i_dcd = DenseLayer(net_i_dcd, df_dim*4*s4*s4, act=tf.identity,
                          W_init=w_init, name='id_h02/dense')
        net_i_dcd = BatchNormLayer(net_i_dcd, act=tf.nn.relu, is_train=is_train, 
                          gamma_init=gamma_init,name='id_h02/batch_norm')

        net_i_dcd = ReshapeLayer(net_i_dcd, [-1, s4, s4, df_dim*4], name='id_h0/reshape')

        net_i_dcd = DeConv2d(net_i_dcd, 64, (4, 4), (s2, s2), strides=(2, 2), 
                          padding='SAME', W_init=w_init, act=None, name='id_h1/decon2d')
        net_i_dcd = BatchNormLayer(net_i_dcd, act=tf.nn.relu, is_train=is_train, 
                          gamma_init=gamma_init,name='id_h1/batch_norm')

        net_i_dcd = DeConv2d(net_i_dcd, 3, (4, 4), (s, s), strides=(2, 2), 
                          padding='SAME', W_init=w_init, act=tf.nn.sigmoid, name='id_h3/decon2d')
        # net_i_dcd = BatchNormLayer(net_i_dcd, act=tf.nn.relu, is_train=is_train, 
        #                   gamma_init=gamma_init,name='id_h3/batch_norm')

        # net_i_dcd = DeConv2d(net_i_dcd, 1, (2, 2), (s, s), strides=(2, 2), 
        #                   padding='SAME', W_init=w_init, act=tf.nn.tanh, name='id_h4/decon2d')
        # net_i_dcd = BatchNormLayer(net_i_dcd, act=tf.nn.relu, is_train=is_train, 
        # #                   gamma_init=gamma_init,name='id_h4/batch_norm')
        # net_i_dcd.outputs = tf.nn.tanh(net_i_dcd.outputs)

        return net_i_dcd, net_i_dcd.outputs

def discriminator2(input_i, input_ebd=None,is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init=tf.random_normal_initializer(1., 0.01)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    df_dim = 64

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_d = InputLayer(input_i, name='d_input/images')

        net_d = Conv2d(net_d, df_dim, (4, 4), (2, 2), act=lrelu,
                        padding='VALID', W_init=w_init, b_init=None, name='d_h0/conv2d')
        net_d = BatchNormLayer(net_d, act=lrelu,is_train=is_train, 
                        gamma_init=gamma_init, name='dx_h0/batchnorm')
        
        net_d = Conv2d(net_d, df_dim*2, (4, 4), (2, 2), act=lrelu,
                        padding='VALID', W_init=w_init, b_init=None, name='d_h1/conv2d')
        net_d = BatchNormLayer(net_d, act=lrelu,is_train=is_train, 
                        gamma_init=gamma_init, name='dx_h1/batchnorm')
        if input_ebd is not None:
            net_d_ebd = InputLayer(input_ebd, name='d_input_ebd')
            net_d_ebd = DenseLayer(net_d_ebd, n_units=1024, act=lrelu, 
                            W_init=w_init, b_init=None, name='d_t_0/dense')
            net_d_ebd = ExpandDimsLayer(net_d_ebd, 1, name='d_t_1/expand0')
            net_d_ebd = ExpandDimsLayer(net_d_ebd, 1, name='d_t_1/expand1')
            net_d_ebd = TileLayer(net_d_ebd,[1,5,5,1],name='d_t_1/tile')
            net_d = ConcatLayer([net_d, net_d_ebd],concat_dim=3,name='d_cat')

        net_d = Conv2d(net_d, df_dim*4, (4, 4), (2, 2), act=lrelu,
                        padding='VALID', W_init=w_init, b_init=None, name='d_h2/conv2d')
        net_d = BatchNormLayer(net_d, act=lrelu,is_train=is_train, 
                        gamma_init=gamma_init, name='dx_h2/batchnorm')
        
        net_d = FlattenLayer(net_d, name='d_h2/flatten')
        net_d = DenseLayer(net_d, n_units=1, act=tf.identity, 
                        W_init=w_init, b_init=None, name='d_h4/dense')

        return net_d, net_d.outputs
        
def discriminator(input_i, input_ebd=None,is_train=True, reuse=False):

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init=tf.random_normal_initializer(1., 0.01)
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    df_dim = 64

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        net_d = InputLayer(input_i, name='d_input/images')

        net_d = Conv2d(net_d, df_dim, (4, 4), (2, 2), act=lrelu,
                        padding='VALID', W_init=w_init, b_init=None, name='d_h0/conv2d')
        net_d = BatchNormLayer(net_d, act=lrelu,is_train=is_train, 
                        gamma_init=gamma_init, name='dx_h0/batchnorm')
        
        net_d = Conv2d(net_d, df_dim*2, (4, 4), (2, 2), act=lrelu,
                        padding='VALID', W_init=w_init, b_init=None, name='d_h1/conv2d')
        net_d = BatchNormLayer(net_d, act=lrelu,is_train=is_train, 
                        gamma_init=gamma_init, name='dx_h1/batchnorm')
        if input_ebd is not None:
            net_d_ebd = InputLayer(input_ebd, name='d_input_ebd')
            net_d_ebd = DenseLayer(net_d_ebd, n_units=1024, act=lrelu, 
                            W_init=w_init, b_init=None, name='d_t_0/dense')
            net_d_ebd = ExpandDimsLayer(net_d_ebd, 1, name='d_t_1/expand0')
            net_d_ebd = ExpandDimsLayer(net_d_ebd, 1, name='d_t_1/expand1')
            net_d_ebd = TileLayer(net_d_ebd,[1,5,5,1],name='d_t_1/tile')
            net_d = ConcatLayer([net_d, net_d_ebd],concat_dim=3,name='d_cat')

        net_d = Conv2d(net_d, df_dim*4, (4, 4), (2, 2), act=lrelu,
                        padding='VALID', W_init=w_init, b_init=None, name='d_h2/conv2d')
        net_d = BatchNormLayer(net_d, act=lrelu,is_train=is_train, 
                        gamma_init=gamma_init, name='dx_h2/batchnorm')
        
        net_d = FlattenLayer(net_d, name='d_h2/flatten')
        net_d = DenseLayer(net_d, n_units=1, act=tf.identity, 
                        W_init=w_init, b_init=None, name='d_h4/dense')

        return net_d, net_d.outputs

class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls
        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]
        
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)
            
        self.num_calls += 1
        return y

flip_gradient = FlipGradientBuilder()
