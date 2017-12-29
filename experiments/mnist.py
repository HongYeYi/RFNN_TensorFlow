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

import time
import datetime
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from RFNN.model import RFNN


def train_mnist():

    # MNIST
    output_dim = 10

    # RFNN parameters
    sigmas         = (1.5, 1.0, 1.0)
    num_bases      = (10, 6, 6)
    filter_extents = (5, 3, 3)
    num_combine    = (64, 64, 64)

    # Convolutional parameters
    conv_padding = (5, 0, 0)
    pool_stride  = (1, 1, 1)
    pool_ksize   = (2, 2, 2)
    dropout_kp   = (1.0, 1.0, 1.0)

    # Train settings
    num_epochs = 1000
    learning_rate = 0.0005
    batch_size = 64

    # Load MNIST
    dataset = input_data.read_data_sets('./data/MNIST/', one_hot=True)

    # Input placeholders
    images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    labels = tf.placeholder(tf.int32, shape=[None, 10])

    # Initialize the RFNN model
    model = RFNN(
        output_dim=output_dim,
        sigmas=sigmas,
        filter_extents=filter_extents,
        num_bases=num_bases,
        num_combinations=num_combine,
        conv_padding=conv_padding,
        pool_ksize=pool_ksize,
        pool_stride=pool_stride,
        dropout_kp=dropout_kp
    )

    global_step = tf.train.get_or_create_global_step()

    # Model outputs
    logits = model.logits(images)
    accuracy = tf.contrib.metrics.accuracy(tf.argmax(labels, 1), tf.argmax(logits, 1))

    # Cross-entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(loss)

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    num_steps = int((num_epochs*dataset.train.num_examples)/batch_size)

    # Training ops
    ops = {'train_op': train_op, 'loss': loss, 'accuracy': accuracy}

    for step in range(num_steps):

        # Only for time measurement of step through network
        t1 = time.time()

        # Get a train batch
        trn_images, trn_labels = dataset.train.next_batch(batch_size)
        trn_images = trn_images.reshape(-1, 28, 28, 1)

        # Run training operation
        fetches = sess.run(ops, feed_dict={images: trn_images, labels: trn_labels})

        # Only for time measurement of step through network
        t2 = time.time()
        examples_per_second = batch_size / float(t2 - t1)

        print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, Accuracy = {:.2f}, Loss = {:.3f}".format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), step+1, int(num_steps),
            batch_size, examples_per_second, fetches['accuracy'], fetches['loss']
        ))


if __name__ == "__main__":
    train_mnist()







