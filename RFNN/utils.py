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
import scipy.ndimage.filters as filters


def initialize_hermite_basis(sigma, filter_extent, num_bases=15):

    # ~meshgrid for filter
    x = np.arange(-filter_extent, filter_extent + 1, dtype=np.float)

    im_size = filter_extent*2 + 1

    # Filter of zeroes with single one in the center
    impulse = np.zeros( (np.int(im_size), np.int(im_size)) )
    impulse[int(np.floor(im_size/2)),int(np.floor(im_size/2))] = 1.0

    # Emptry matrix of size [num_bases, im_size, im_size]
    num_bases_full = 15
    hermite_basis = np.empty((np.int(num_bases_full), np.int(im_size), np.int(im_size)))

    # plain gaussian
    g = 1.0/(np.sqrt(2*np.pi)*sigma)*np.exp(np.square(x)/(-2*np.square(sigma)))
    g = g/g.sum()

    # Gaussian derivatives (written out Equation 3)
    g1 = sigma * -(x/ np.square(sigma)) * g
    g2 = np.power(sigma, 2) * ((np.square(x)-np.square(sigma)) / np.power(sigma,4)) * g
    g3 = np.power(sigma, 3) * -((np.power(x,3) - 3 * x * np.square(sigma)) / np.power(sigma,6)) * g
    g4 = np.power(sigma, 4) * ((np.power(x,4) - 6 *  np.square(x) * np.square(sigma) + 3 * np.power(sigma,4)) / np.power(sigma,8)) * g

    # Create the Gaussian filters by convolving impulse
    gauss0x = filters.convolve1d(impulse, g,  axis=1)
    gauss0y = filters.convolve1d(impulse, g,  axis=0)
    gauss1x = filters.convolve1d(impulse, g1, axis=1)
    gauss1y = filters.convolve1d(impulse, g1, axis=0)
    gauss2x = filters.convolve1d(impulse, g2, axis=1)
    gauss0  = filters.convolve1d(gauss0x, g,  axis=0)

    hermite_basis[0,:,:] = gauss0
    vmax = gauss0.max()
    vmin = -vmax

    hermite_basis[1,:,:]  = filters.convolve1d(gauss0y, g1, axis=1) # g_x
    hermite_basis[2,:,:]  = filters.convolve1d(gauss0x, g1, axis=0) # g_y
    hermite_basis[3,:,:]  = filters.convolve1d(gauss0y, g2, axis=1) # g_xx
    hermite_basis[4,:,:]  = filters.convolve1d(gauss0x, g2, axis=0) # g_yy
    hermite_basis[5,:,:]  = filters.convolve1d(gauss1x, g1, axis=0) # g_yy
    hermite_basis[6,:,:]  = filters.convolve1d(gauss0y, g3, axis=1) # g_xxx
    hermite_basis[7,:,:]  = filters.convolve1d(gauss0x, g3, axis=0) # g_yyy
    hermite_basis[8,:,:]  = filters.convolve1d(gauss1y, g2, axis=1) # g_xxy
    hermite_basis[9,:,:]  = filters.convolve1d(gauss1x, g2, axis=0) # g_yyx
    hermite_basis[10,:,:] = filters.convolve1d(gauss0y, g4, axis=1) # g_xxxx
    hermite_basis[11,:,:] = filters.convolve1d(gauss0x, g4, axis=0) # g_yyyy
    hermite_basis[12,:,:] = filters.convolve1d(gauss1y, g3, axis=1) # g_xxxy
    hermite_basis[13,:,:] = filters.convolve1d(gauss1x, g3, axis=0) # g_yyyx
    hermite_basis[14,:,:] = filters.convolve1d(gauss2x, g2, axis=0) # g_yyxx

    return hermite_basis[0:num_bases]


def initialize_alphas(num_filters, num_channels, num_bases):
    # TODO: experiment with other random initializers
    return np.random.uniform(low=-1.0, high=1.0, size=(num_filters, num_channels, num_bases))



