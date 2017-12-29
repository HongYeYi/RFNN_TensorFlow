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
import argparse
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from RFNN.model import RFNN


def load_mnist():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    train_im     = mnist.train.images.reshape(-1, 28, 28, 1)
    train_labels = mnist.train.labels.astype(np.int32)
    test_im      = mnist.test.images.reshape(-1, 28, 28, 1)
    test_labels  = mnist.test.labels.astype(np.int32)
    return train_im, train_labels, test_im, test_labels


def train_mnist(config):

    # Load MNIST
    train_im, train_labels, test_im, test_labels = load_mnist()

    # Data pipeline
    data_train = tf.data.Dataset.from_tensor_slices((train_im, train_labels))
    data_train = data_train.shuffle(buffer_size=10000)
    data_train = data_train.repeat(config.num_epochs)
    data_train = data_train.batch(config.batch_size)

    data_test = tf.data.Dataset.from_tensor_slices((test_im, test_labels)).batch(config.batch_size)
    iterator  = tf.data.Iterator.from_structure(data_train.output_types, data_train.output_shapes)

    batch_images, batch_labels = iterator.get_next()
    init_train_data_op = iterator.make_initializer(data_train)
    init_test_data_op  = iterator.make_initializer(data_test)

    # Placeholder for dropout keep probability
    dropout_kp = tf.placeholder_with_default(1.0, shape=[], name='dropout_kp')

    # Initialize the RFNN model
    model = RFNN(
        num_classes=10,
        sigmas=[float(x) for x in config.sigmas.split(',')],
        filter_extents=[int(x) for x in config.filter_extents.split(',')],
        num_bases=[int(x) for x in config.num_basis.split(',')],
        num_combinations=[int(x) for x in config.linear_combinations.split(',')],
        conv_padding=[int(x) for x in config.conv_padding.split(',')],
        pool_ksize=[int(x) for x in config.pool_ksize.split(',')],
        pool_stride=[int(x) for x in config.pool_stride.split(',')],
    )

    global_step = tf.train.get_or_create_global_step()

    # Model outputs
    logits = model.logits(batch_images, dropout_kp)
    accuracy = tf.contrib.metrics.accuracy(tf.argmax(batch_labels, 1), tf.argmax(logits, 1))

    # Cross-entropy loss
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=batch_labels, logits=logits)
    loss = tf.reduce_mean(loss)

    # Optimizer
    optimizer = tf.train.AdamOptimizer(config.learning_rate)
    train_op = optimizer.minimize(loss, global_step)

    sess = tf.Session()

    sess.run(init_train_data_op) # Initialize the variables of the data-loader.
    sess.run(tf.global_variables_initializer())  # Initialize the model parameters.

    num_steps = int((config.num_epochs*len(train_im))/config.batch_size)

    # Training ops
    ops = {'train_op': train_op, 'loss': loss, 'accuracy': accuracy}

    for step in range(num_steps):

        # Only for time measurement of step through network
        t1 = time.time()

        # Run training operation
        fetches = sess.run(ops)

        # Only for time measurement of step through network
        t2 = time.time()
        examples_per_second = config.batch_size / float(t2 - t1)

        print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, Accuracy = {:.2f}, Loss = {:.3f}".format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), step+1, int(num_steps),
            config.batch_size, examples_per_second, fetches['accuracy'], fetches['loss']
        ))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # RFNN parameters
    parser.add_argument('--sigmas', type=str, default='1.5,1.0,1.0', help='Gaussian sigmas (comma separated)')
    parser.add_argument('--num_basis', type=str, default='10,6,6', help='Number of basis filters (comma separated)')
    parser.add_argument('--filter_extents', type=str, default='5,3,3', help='Gaussian filter extent (comma separated)')
    parser.add_argument('--linear_combinations', type=str, default='64,64,64', help='Number of linear combinations (comma separated)')

    # CNN parameters
    parser.add_argument('--conv_padding', type=str, default='5,0,0', help='Zero padding before convolution (comma separated)')
    parser.add_argument('--pool_ksize', type=str, default='2,2,2', help='Max-pooling kernel size (comma separated)')
    parser.add_argument('--pool_stride', type=str, default='1,1,1', help='Max-pooling stride (comma separated)')
    parser.add_argument('--dropout_kp', type=float, default='0.75', help='Dropout keep probabilities')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--print_frequency', type=int, default=5, help='Printing frequency')
    parser.add_argument('--eval_frequency', type=int, default=5, help='Evaluation frequency')
    parser.add_argument('--save_frequency', type=int, default=5, help='Model saving frequency')

    config = parser.parse_args()

    train_mnist(config)







