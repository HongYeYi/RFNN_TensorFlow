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

import numpy as np
import matplotlib.pyplot as plt

from RFNN.utils import *


sigma = 1.5
filter_extent = 5

# Create the basis
bases = initialize_hermite_basis(sigma, filter_extent)

fig, ax = plt.subplots(5,3,figsize=(10,6))
ax = ax.flatten()

for i in range(len(bases)):
    ax[i].imshow(bases[i])
    ax[i].get_xaxis().set_visible(False)
    ax[i].get_yaxis().set_visible(False)

plt.suptitle("Herminte Filter Bases")
plt.show()







