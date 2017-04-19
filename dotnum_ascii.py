# -*- coding: utf-8 -*-
'''
TODO:
- Train with noise added
- Train with larger batch sizes
'''

from __future__ import print_function

from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range

import string
import random
import matplotlib.pyplot as plt
import pickle

plt.ion()

def ascii_array(c):
    # Convert a visible character into ASCII, as an np array of 7 binary values.
    # Input can be single character or an integer.
    c = ord(c) if type(c) is str else c     # convert to int if not already an int
    ascii = np.asarray([int(i) for i in bin(c)[2:]])
    return np.concatenate((np.zeros(7-len(ascii)) ,ascii))


def chars_to_onehot(data):
    nSeqLength = data.shape[0]
    ref = np.sort(np.asarray([c for c in ' 0123456789']))
    nClasses = len(ref)
    ix = np.searchsorted(ref,data)
    onehot = np.zeros((nSeqLength, nClasses))
    onehot[np.arange(nSeqLength), ix] = 1
    return onehot


def get_data_pos(nSamples):
    # Build text with DOT numbers
    # Variability includes
    #   1. Use of - or # between "DOT" and the number
    #   2. Addition of extra spaces in between DOT and number
    #   3. Possible extra text before "DOT 12454"
    #   4. Possible extra text after "DOT 12454"
    #   5. Number of digits in dot number
    minNumLength = 5    # maximum number of digits in DOT number, e.g., 12345
    maxNumLength = 8    # maximum number of digits in DOT number, e.g., 12345678
    maxPreText = 5
    maxPostText = 10
    maxTotalLength = maxPreText + 1 + 3 + 3 + maxNumLength + 1 + maxPostText # 1,3,3,1 --> space, 'DOT', space, [space,-/#,space], space
    data = np.zeros((nSamples, maxTotalLength, 7))	# Input data is ASCII, thus 7 bits, 7 input nodes
    labels = np.zeros((nSamples, maxNumLength, 11))	# Output data is one-hot vector with 11 classes: space (' ') and 0-9
    dotnumbers = []
    textlines = []

    # Do some of the stochastic stuff first.
    nDigits = np.random.randint(minNumLength,maxNumLength+1, nSamples)
    nPreText = np.random.randint(0,maxPreText+1, nSamples)
    nPostText = np.random.randint(0,maxPostText+1, nSamples)

    preMidSpace = np.random.choice((' ',''), nSamples).tolist()
    midChar = np.random.choice(('-','#',''), nSamples).tolist()
    postMidSpace = np.random.choice((' ',''), nSamples).tolist()

    # Get list of all printable ASCII printable characters (int 32 to 126)
    printables = []
    for i in range(32,127):
        printables.append(str(unichr(i)))

    digit_array = ('0','1','2','3','4','5','6','7','8','9')
    nClasses = len(digit_array)

    for i in range(nSamples):
        preText = np.random.choice((printables), nPreText[i])
        if len(preText)>0:
            preText = np.append(preText,[' '])
        preText = preText.tostring()

        postText = np.random.choice(printables, nPostText[i])
        if len(postText)>0:
            postText = np.concatenate(([' '],postText))
        postText = postText.tostring()

        # one-hot vectors as labels/output?
        digits = np.random.choice(digit_array, nDigits[i])
        labels[i,:,:] = chars_to_onehot(np.concatenate((np.asarray([' ']*(maxNumLength-nDigits[i])),digits)))
        digits = digits.tostring()
        dotnumbers.append(digits)

        fullstring = preText + 'DOT' + preMidSpace[i] + midChar[i] + postMidSpace[i] + digits + postText
        # fullstring = (' ' * (maxTotalLength-len(fullstring))) + fullstring    # prepad
        fullstring = fullstring + (' ' * (maxTotalLength-len(fullstring)))    # postpad
        textlines.append(fullstring)

        for j in range(maxTotalLength):
            data[i, j, :] = ascii_array(fullstring[j])

    return data, labels, textlines, dotnumbers


def get_data_neg(nSamples, maxInputLength, maxOutputLen):

    data = np.zeros((nSamples, maxInputLength, 7))	# Input data is ASCII, thus 7 bits, 7 input nodes
    textlines = []

    # Get list of all printable ASCII printable characters (int 32 to 126)
    printables = []
    for i in range(32,127):
        printables.append(str(unichr(i)))

    nText = np.random.randint(0,maxInputLength+1, nSamples)

    for i in range(nSamples):
        fullstring = np.random.choice((printables), nText[i])
        fullstring = fullstring.tostring()
        fullstring = ' ' * (maxInputLength-len(fullstring)) + fullstring

        textlines.append(fullstring)

        for j in range(maxInputLength):
            data[i, j, :] = ascii_array(fullstring[j])

    labels = np.zeros((nSamples,maxOutputLen,11))    # one-hot vector has 11 elements (' ', and 0 to 9)
    labels[:,:,0] = 1	# label all outputs as spaces
    return data, labels, textlines



## Build data set...
## Data array dimension ordering is (samples, sequence_length, input_nodes).
print('Building data set...')
nPos = 10000
nNeg = 10000
bInvertSeq = True

dataPos, labelsPos, lineStringsPos, dotStringsPos = get_data_pos(nPos)
maxInputLen = dataPos.shape[1]
maxOutputLen = max([len(i) for i in labelsPos])
dataNeg, labelsNeg, lineStringsNeg = get_data_neg(nNeg, maxInputLen, maxOutputLen)

if bInvertSeq:
    dataPos = np.flip(dataPos,1)
    dataNeg = np.flip(dataNeg,1)

dataAll = np.concatenate((dataPos, dataNeg))
labelsAll = np.concatenate((labelsPos, labelsNeg))
lineStringsAll = lineStringsPos + lineStringsNeg

## Shuffle data
indices = np.arange(len(dataAll))
np.random.shuffle(indices)
dataAll = dataAll[indices]
labelsAll = labelsAll[indices]
lineStringsAll = [lineStringsAll[i] for i in indices]

## Add noise to data
noiseMag = 0.5;
dataAll += np.random.uniform(-noiseMag,noiseMag,dataAll.shape)

## Split into train and validation sets
percentTrain = 80
split_at = int( len(dataAll) * percentTrain/100)
(dataTrain, dataVal) = dataAll[:split_at], dataAll[split_at:]
(labelsTrain, labelsVal) = labelsAll[:split_at], labelsAll[split_at:]
(lineStringsTrain, lineStringsVal) = lineStringsAll[:split_at], lineStringsAll[split_at:]

print('Training Data:')
print(dataTrain.shape)
print(labelsTrain.shape)

print('Validation Data:')
print(dataVal.shape)
print(labelsVal.shape)


## Build the model...
print('Build model...')

#RNN = layers.LSTM	# Try LSTM, GRU, or SimpleRNN
RNN = layers.GRU   # Try LSTM, GRU, or SimpleRNN
HIDDEN_SIZE = 128
BATCH_SIZE = 128    # 128 was initial size during first tries
LAYERS = 1

model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, num_feature).
model.add(RNN(HIDDEN_SIZE,
              input_shape=(maxInputLen, 7), # 7-bit ASCII code, so 7 input nodes
              return_sequences=False,
              activation='relu',     # tanh is the default (documentation wrongly says it's linear)
              unroll=False))         # processing is faster if unroll==True, but requires more memory. Also may need to be True to avoid a theano bug

# As the decoder RNN's input, repeatedly provide with the last hidden state of
# RNN for each time step. Repeat 8 times as that's the maximum
# length of output (longest DOT number possible).
model.add(layers.RepeatVector(8))

# The decoder RNN could be multiple layers stacked or a single layer.
for _ in range(LAYERS):
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


# Train the model each generation and show predictions against the validation
# dataset.
history = model.fit(dataTrain, labelsTrain, batch_size=BATCH_SIZE, epochs=500,
          validation_data=(dataVal, labelsVal))

# for iteration in range(1, 5):
#     print()
#     print('-' * 50)
#     print('Iteration', iteration)
#     history = model.fit(dataTrain, labelsTrain, batch_size=BATCH_SIZE, epochs=1,
#               validation_data=(dataVal, labelsVal))
    # # Select 10 samples from the validation set at random so we can visualize
    # # errors.
    # for i in range(10):
    #     ind = np.random.randint(0, len(x_val))
    #     rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
    #     preds = model.predict_classes(rowx, verbose=0)
    #     q = ctable.decode(rowx[0])
    #     correct = ctable.decode(rowy[0])
    #     guess = ctable.decode(preds[0], calc_argmax=False)
    #     print('Q', q[::-1] if INVERT else q)
    #     print('T', correct)
    #     if correct == guess:
    #         print(colors.ok + '☑' + colors.close, end=" ")
    #     else:
    #         print(colors.fail + '☒' + colors.close, end=" ")
    #     print(guess)
    #     print('---')

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.figure(1)
plt.subplot(1,2,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

## Save loss and accuracy values
a = {'acc': history.history['acc'], 'val_acc': history.history['val_acc'], 'loss': history.history['loss'], 'val_loss': history.history['val_loss']}
with open('results.pkl', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)


## Make predictions on validation set
print('Making predictions on validation set...')
# TODO: Time the predictions.  Run it twice.  Faster the second time?  Is there compilation happening the first time?
labelsValPred = model.predict(dataVal, batch_size=BATCH_SIZE, verbose=1)
# labelsTrainPred = model.predict(dataTrain, batch_size=BATCH_SIZE, verbose=1)

plt.figure(2)
for ix in range(len(dataVal)):

    plt.clf()

    plt.subplot(2,2,1)
    plt.imshow(labelsVal[ix])
    plt.clim(0,1)
    plt.title(lineStringsVal[ix])

    plt.subplot(2,2,2)
    plt.imshow(labelsValPred[ix])
    plt.clim(0,1)
    plt.title(lineStringsVal[ix])

    # plt.subplot(2,2,3)
    # plt.imshow(labelsTrain[ix])
    # plt.clim(0,1)
    # plt.title(lineStringsTrain[ix])
    # plt.colorbar()

    # plt.subplot(2,2,4)
    # plt.imshow(labelsTrainPred[ix])
    # plt.clim(0,1)
    # plt.title(lineStringsTrain[ix])
    # plt.colorbar()

    plt.waitforbuttonpress()
