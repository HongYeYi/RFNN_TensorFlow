# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2017-12-21

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import RFNN.utils as utils

################################################################################
################################################################################

def rfnn_layer(var_scope, inputs, sigma, filter_extent, num_bases, num_combinations,
               conv_padding, pool_ksize, pool_stride, dropout_kp):

    with tf.variable_scope(var_scope, reuse=tf.AUTO_REUSE):

        np_basis = utils.initialize_hermite_basis(sigma, filter_extent, num_bases) # [num_basis, h, w]
        np_alphas = utils.initialize_alphas(num_combinations, inputs.shape[-1], num_bases)

        # Note: filters are not trainable, alphas are.
        basis  = tf.Variable(np_basis, dtype=tf.float32, name="weights", trainable=False)
        alphas = tf.Variable(np_alphas, dtype=tf.float32, name="alphas")

        # Compute Equation (6)
        weights = tf.multiply(basis[None,None,:,:,:], alphas[:,:,:,None,None])  # [m,1,n_basis,h,w]
        weights = tf.reduce_sum(weights, axis=2)  # [m,1,h,w]
        weights = tf.transpose(weights, perm=[2,3,1,0]) # [h,w,1,m]

        # Compute convolution of filters with inputs (Algorithm 1: Step 2)
        h = tf.pad(inputs, paddings=[[0,0],[conv_padding,conv_padding],[conv_padding,conv_padding],[0,0]], name='padding')
        h = tf.nn.conv2d(h, weights, strides=[1,1,1,1], padding='VALID', name='conv')
        h = tf.nn.relu(h, name='relu')  # [num_batch, h, w, c]
        h = tf.nn.max_pool(h, ksize=[1,pool_ksize,pool_ksize,1], strides=[1,pool_stride,pool_stride,1], padding='SAME')
        h = tf.nn.dropout(h, dropout_kp, name='dropout')
        # add: cross-normalizer

    return h

################################################################################
################################################################################

class RFNN(object):

    def __init__(self, output_dim, sigmas, filter_extents, num_bases, num_combinations,
                 conv_padding, pool_ksize, pool_stride, dropout_kp):

        # Define the output dimensionality
        self._output_dim = output_dim

        # Define the number of conv layers
        self._num_layers = len(sigmas)

        # Check that all parameters have the same number of layers
        if not all(len(lst) == self._num_layers for lst in
                   [filter_extents, num_bases, num_combinations, conv_padding, pool_ksize, pool_stride, dropout_kp]):
            raise ValueError("Input lists defining layers must all have the same size.")

        self._sigmas = sigmas
        self._filter_extents = filter_extents
        self._num_basis = num_bases
        self._num_combinations = num_combinations
        self._conv_padding = conv_padding
        self._pool_ksize = pool_ksize
        self._pool_stride = pool_stride
        self._dropout_kp = dropout_kp

    def logits(self, images):

        layer_outputs = []

        for i in range(self._num_layers):

            # Inputs are images or outputs from previous layer
            inputs = images if i == 0 else layer_outputs[-1]

            # Initialize the RFNN layer
            h = rfnn_layer(
                var_scope="layer_{}".format(i),
                inputs=inputs,
                sigma=self._sigmas[i],
                filter_extent=self._filter_extents[i],
                num_bases=self._num_basis[i],
                num_combinations=self._num_combinations[i],
                conv_padding=self._conv_padding[i],
                pool_ksize=self._pool_ksize[i],
                pool_stride=self._pool_stride[i],
                dropout_kp = self._dropout_kp[i]
            )

            print("Layer {} output shape: {}".format(i+1, h.shape))
            layer_outputs.append(h)

        # Final output layer
        with tf.variable_scope("logits", reuse=tf.AUTO_REUSE):
            conv_flat = tf.layers.flatten(layer_outputs[-1], name='flatten')
            logits = tf.layers.dense(
                conv_flat, self._output_dim,
                kernel_initializer=tf.initializers.variance_scaling, name='logits')
            print("Layer {} output shape: {}".format(self._num_layers+1, logits.shape))

        return logits

    def probabilities(self, images):
        return tf.nn.softmax(self.logits(images), name='probabilities')

    def predictions(self, images):
        return tf.argmax(self.logits(images), dim=1, name='predictions')