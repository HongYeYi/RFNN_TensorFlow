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


class RFNN(object):

    def __init__(self, sigmas, filter_extents, num_bases):

        # Must have the same number of layers
        assert len(sigmas) == len(filter_extents) == len(num_bases)

        self._sigmas = sigmas
        self._filter_extents = filter_extents
        self._num_bases = num_bases


    def logits(self, inputs):

        with tf.variable_scope("layer1", reuse=tf.AUTO_REUSE):

            hermite = utils.initialize_hermite_basis(self._sigmas[0], self._filter_extents[0], self._num_bases[0]) # [num_basis, h, w]
            hermite = np.transpose(hermite, axes=[1,2,0])
            hermite = np.expand_dims(hermite, -1)

            weights = tf.Variable(hermite, dtype=tf.float32, name="weights")
            alphas  = tf.Variable(utils.initialize_alphas(self._num_bases[0], 1, self._num_bases[0]), dtype=tf.float32, name="alphas")

            print("weights", weights.shape)
            print("alphas", alphas.shape)

            h = tf.pad(inputs, paddings=[[0,0],[5,5],[5,5],[0,0]], name='padding')
            print(h.shape)
            h = tf.nn.conv2d(h, weights, strides=[1,1,1,1], padding='SAME', name='conv')
            print(h.shape)
            h = tf.nn.max_pool(h, ksize=3, strides=2)
            h = tf.nn.relu(h, name='relu')

        return h

