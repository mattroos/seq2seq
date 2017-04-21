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
nKernSize = 3
nFilters = (8, 16)
nConvLayers = len(nFilters)
nRows = 32
nCols = nRows*10
nChan = 1

RNN = layers.GRU   # Try LSTM, GRU, or SimpleRNN
HIDDEN_SIZE = 256
DECODE_LAYERS = 1

x = np.round(np.random.uniform(size=(nSamples, nRows, nCols, nChan)))
# May need to flip the input images such that space-to-sequence conversion it top-bottom rather than left-right,
# due to way the layers.core.Reshape works.  Or build a transpose layer/operation?
x = x.transpose(0,2,1,3)


## Build the model
model = Sequential()
for i in range(nConvLayers):

      #model.add(layers.normalization.BatchNormalization(input_shape=(nCols, nRows, nChan)))
      model.add(layers.convolutional.Conv2D(nFilters[i],
                                           nKernSize,
                                           strides=(1, 1),
                                           padding='same',
                                           data_format='channels_last',
                                           dilation_rate=(1, 1),
                                           activation=None,
                                           use_bias=True,
                                           kernel_initializer='glorot_uniform',
                                           bias_initializer='zeros',
                                           kernel_regularizer=None,
                                           bias_regularizer=None,
                                           activity_regularizer=None,
                                           kernel_constraint=None,
                                           bias_constraint=None,
                                           input_shape=(nCols, nRows, nChan)))
      model.add(layers.pooling.MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))


# Reshape, converting columns to timesteps....
# (batch_size, timesteps/cols, rows)
inputSeqLen = 80
rnnInputDim = 16*8
model.add(layers.core.Reshape((inputSeqLen, rnnInputDim)))

# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, num_feature).
model.add(RNN(HIDDEN_SIZE,
              input_shape=(inputSeqLen, rnnInputDim), # 7-bit ASCII code, so 7 input nodes
              return_sequences=False,
              activation='relu',     # tanh is the default (documentation wrongly says it's linear)
              unroll=False))         # processing is faster if unroll==True, but requires more memory. Also may need to be True to avoid a theano bug


# As the decoder RNN's input, repeatedly provide with the last hidden state of
# RNN for each time step. Repeat 8 times as that's the maximum
# length of output (longest DOT number possible).
model.add(layers.RepeatVector(8))


# The decoder RNN could be multiple layers stacked or a single layer.
for _ in range(DECODE_LAYERS):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    model.add(RNN(HIDDEN_SIZE,
                  # activation='tanh',
                  return_sequences=True))

# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
model.add(layers.TimeDistributed(layers.Dense(11))) # 11 outputs, for one-hot ' 0123456789'
model.add(layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()


y = model.predict(x, batch_size=nSamples, verbose=1)

print(x.shape)
print("\n\n")
print(y.shape)
