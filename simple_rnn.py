# simple_rnn.py

from __future__ import print_function

# Set random seed before importing anything from keras
import numpy as np
np.random.seed(0)

from keras.models import Sequential
from keras import layers
from six.moves import range


## Generate dummy data
nSamples = 2
nInputNodes = 1
nOutputNodes = 1
nInputLength = 3
x = np.random.random((nSamples, nInputLength, nInputNodes))
x[0,:,0] = np.asarray([1,2,3])
x[1,:,0] = np.asarray([3,2,1])
init_state = [np.asarray([1])]

#labels = np.random.randint(1, size=(nSamples, 1))

## Build the model
model = Sequential()
# Simple RNN unit feeds output back to input.  That is, the hiddens units hold the values of the last output, W_h == W_y
# y(t) = z( W_x*x(t) + W_h*y(t-1))
layer_rnn1 = layers.recurrent.SimpleRNN(nOutputNodes,
                                        input_shape=(nInputLength, nInputNodes),
                                        use_bias=False,
                                        return_sequences=True,
                                        activation = 'linear',
                                        unroll=True)  #  unroll=True to avoid bug in theano when #outputnodes = 1
model.add(layer_rnn1)

# Note that initial_states are not specified.  Thay appear to default to zero.

## Configure the learning process, using the compile method, even if training won't be performed.
# The optimizer and loss must be specified.
model.compile(optimizer='sgd', loss='mean_squared_error')

## Set weights 
w = layer_rnn1.get_weights()
w[0] = np.ones(w[0].shape)
w[1] = np.ones(w[1].shape)
layer_rnn1.set_weights(w)


# Print summary?
model.summary()

y = model.predict(x, batch_size=nSamples, verbose=1)

print(x)
print("\n\n")
print(y)
