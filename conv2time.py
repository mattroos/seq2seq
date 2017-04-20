# conv2time.py
#
# How to take the output of a convolution layer and convert it into a sequence?
# 4/18/17

from __future__ import print_function

# Set random seed before importing anything from keras
import numpy as np
np.random.seed(0)

from keras.models import Sequential
from keras import layers
from six.moves import range


## Generate dummy data
nSamples = 1
# random set of 0s and 1s as data
nKernSize = 2
nFilters = 1
nRows = nKernSize + 1
nCols = 2 * nRows
nChan = 1

# May need to flip the input image such that space-to-sequence conversion it top-bottom rather than left-right,
# due to way the layers.core.Reshape works.  Or build a transpose layer/operation?
x = np.round(np.random.uniform(size=(nSamples, nRows, nCols, nChan)))

## Build the model
model = Sequential()
layers_conv1 = layers.convolutional.Conv2D(nFilters,
                                           nKernSize,
                                           strides=(1, 1),
                                           padding='valid',
                                           data_format='channels_last',
                                           dilation_rate=(1, 1),
                                           activation=None,
                                           use_bias=False,
                                           kernel_initializer='glorot_uniform',
                                           bias_initializer='zeros',
                                           kernel_regularizer=None,
                                           bias_regularizer=None,
                                           activity_regularizer=None,
                                           kernel_constraint=None,
                                           bias_constraint=None,
                                           input_shape=(nRows, nCols, nChan))
model.add(layers_conv1)



# Reshape, converting columns to timesteps....
# (batch_size, timesteps/cols, rows)
model.add(layers.core.Reshape((nRows-1,(nCols-1)*nChan)))


## Configure the learning process, using the compile method, even if training won't be performed.
# The optimizer and loss must be specified.
model.compile(optimizer='sgd', loss='mean_squared_error')

# ## Set weights 
# w = layer_rnn1.get_weights()
# w[0] = np.ones(w[0].shape)
# w[1] = np.ones(w[1].shape)
# layer_rnn1.set_weights(w)

# Print summary
model.summary()

y = model.predict(x, batch_size=nSamples, verbose=1)

print(x.shape)
print("\n\n")
print(y.shape)
