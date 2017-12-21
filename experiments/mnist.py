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

from RFNN.model import RFNN

def train_mnist():

    # Define the network parameters
    sigmas         = (1.5, 1.0, 1.0)
    num_bases      = (10, 6, 6)
    filter_extents = (5, 3, 3)

    # Initialize the RFNN model
    model = RFNN(sigmas, filter_extents, num_bases)

    inputs = tf.placeholder(tf.float32, shape=[None,256,256,1])
    logits = model.logits(inputs)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    random_images = np.random.uniform(0, 1, size=(32, 256, 256, 1))
    res = sess.run(logits, feed_dict={inputs: random_images})

    print(res)



if __name__ == "__main__":
    train_mnist()







