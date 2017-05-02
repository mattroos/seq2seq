# conv2time.py
#
# How to take the output of a convolution layer and convert it into a sequence?
# 4/18/17

from __future__ import print_function

# Set random seed before importing anything from keras
import numpy as np
# seed = int(np.random.uniform() * 10000)
# print("Seed = " + str(seed))
# np.random.seed(seed)
np.random.seed(2)


from keras.models import Sequential
from keras import layers
from six.moves import range
import time
import cPickle as pickle

import matplotlib.pyplot as plt

import cairocffi as cairo
import editdistance
from scipy import ndimage
import keras.callbacks
from keras.preprocessing import image
from keras.callbacks import History 

plt.ion()

# this creates larger "blotches" of noise which look
# more realistic than just adding gaussian noise
# assumes greyscale with pixels ranging from 0 to 1

def speckle(img):
    severity = np.random.uniform(0, 0.2)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck[img_speck > 1] = 1
    img_speck[img_speck <= 0] = 0
    return img_speck


# paints the string in a random location the bounding box
# also uses a random font, a slight random rotation,
# and a random amount of speckle noise

def paint_text(text, w, h, rotate=False, ud=False, multi_fonts=False):
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, w, h)
    with cairo.Context(surface) as context:
        bTooLong = True
        while bTooLong:
            intensity = np.random.uniform(low=0.0, high=0.3)    # background greyscale intensity
            context.set_source_rgb(intensity, intensity, intensity)   # Background colors in BGR order, values from 0 to 1
            context.paint()
            # this font list works in Ubunut 14.04
            fonts = ['Century Schoolbook', 'Courier', 'STIX', 'URW Chancery L', 'FreeMono', 'serif', 'sans-serif', 'cursive', 'fantasy', 'monospace']
            slants = [cairo.FONT_SLANT_NORMAL, cairo.FONT_SLANT_ITALIC]
            weights = [cairo.FONT_WEIGHT_BOLD, cairo.FONT_WEIGHT_NORMAL]
            if multi_fonts:
                context.select_font_face(np.random.choice(fonts), np.random.choice(slants), np.random.choice(weights))
            else:
                context.select_font_face('Courier', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            context.set_font_size(24)
            box = context.text_extents(text) # (x_bearing, y_bearing, width, height, x_advance, y_advance)
            # border_w_h = (4, 4)
            border_w_h = (2, 2)
            if box[2] > (w - 2 * border_w_h[1]) or box[3] > (h - 2 * border_w_h[0]):
                # raise IOError('Could not fit string into image. Max char count is too large for given image width.')
                continue
            else:
                bTooLong = False

        # teach the RNN translational invariance by
        # fitting text box randomly on canvas, with some room to rotate
        max_shift_x = w - box[2] - border_w_h[0]
        max_shift_y = h - box[3] - border_w_h[1]
        top_left_x = np.random.randint(0, int(max_shift_x))
        if ud:
            top_left_y = np.random.randint(0, int(max_shift_y))
        else:
            top_left_y = h // 2
        context.move_to(top_left_x - int(box[0]), top_left_y - int(box[1]))
        intensity = np.random.uniform(low=0.7, high=1.0)    # background greyscale intensity
        context.set_source_rgb(intensity, intensity, intensity)   # Background colors in BGR order, values from 0 to 1
        context.show_text(text)

    buf = surface.get_data()
    a = np.frombuffer(buf, np.uint8)
    a.shape = (h, w, 4) # red, green, blue, alpha
    a = a[:, :, 0]  # grab single channel

    a = a.astype(np.float32) / 255
    a = np.expand_dims(a, 0)

    if rotate:
        # a = image.random_rotation(a, 3 * (w - top_left_x) / w + 1)
        a = image.random_rotation(a, 1 * (w - top_left_x) / w + 1)
    a = speckle(a)

    ## Randomly choose polarity of image
    if np.random.uniform() > 0.5:
        a = 1. - a

    ## Set areas to left and right of the text to zeros
    a[0,:,:top_left_x] = 0.5
    a[0,:,top_left_x+int(round(box[2])):] = 0.5
    ## Shift text to far right or left of image
    a = np.roll(a, -top_left_x, axis=2)
    # a = np.roll(a, w - (top_left_x+int(round(box[2]))), axis=2)

    return a


def shuffle_mats_or_lists(matrix_list, stop_ind=None):
    ret = []
    assert all([len(i) == len(matrix_list[0]) for i in matrix_list])
    len_val = len(matrix_list[0])
    if stop_ind is None:
        stop_ind = len_val
    assert stop_ind <= len_val

    a = range(stop_ind)
    np.random.shuffle(a)
    a += range(stop_ind, len_val)
    for mat in matrix_list:
        if isinstance(mat, np.ndarray):
            ret.append(mat[a])
        elif isinstance(mat, list):
            ret.append([mat[i] for i in a])
        else:
            raise TypeError('shuffle_mats_or_lists only supports '
                            'numpy.array and list objects')
    return ret


def text_to_labels(text, num_classes):
    ret = []
    for char in text:
        if char >= 'a' and char <= 'z':
            ret.append(ord(char) - ord('a'))
        elif char == ' ':
            ret.append(26)
    return ret


# only a-z and space..probably not to difficult
# to expand to uppercase and symbols

def is_valid_str(in_str):
    search = re.compile(r'[^a-z\ ]').search
    return not bool(search(in_str))


# Uses generator functions to supply train/test with
# data. Image renderings are text are created on the fly
# each time with random perturbations

class TextImageGenerator(keras.callbacks.Callback):

    def __init__(self, monogram_file, bigram_file, minibatch_size,
                 img_w, img_h, downsample_factor, val_split,
                 absolute_max_string_len=16):

        self.minibatch_size = minibatch_size
        self.img_w = img_w
        self.img_h = img_h
        self.monogram_file = monogram_file
        self.bigram_file = bigram_file
        self.downsample_factor = downsample_factor
        self.val_split = val_split
        self.blank_label = self.get_output_size() - 1
        self.absolute_max_string_len = absolute_max_string_len

    def get_output_size(self):
        return 28

    # num_words can be independent of the epoch size due to the use of generators
    # as max_string_len grows, num_words can grow
    def build_word_list(self, num_words, max_string_len=None, mono_fraction=0.5):
        assert max_string_len <= self.absolute_max_string_len
        assert num_words % self.minibatch_size == 0
        assert (self.val_split * num_words) % self.minibatch_size == 0
        self.num_words = num_words
        self.string_list = [''] * self.num_words
        tmp_string_list = []
        self.max_string_len = max_string_len
        self.Y_data = np.ones([self.num_words, self.absolute_max_string_len]) * -1
        self.X_text = []
        self.Y_len = [0] * self.num_words

        # monogram file is sorted by frequency in english speech
        with open(self.monogram_file, 'rt') as f:
            for line in f:
                if len(tmp_string_list) == int(self.num_words * mono_fraction):
                    break
                word = line.rstrip()
                if max_string_len == -1 or max_string_len is None or len(word) <= max_string_len:
                    tmp_string_list.append(word)

        # bigram file contains common word pairings in english speech
        with open(self.bigram_file, 'rt') as f:
            lines = f.readlines()
            for line in lines:
                if len(tmp_string_list) == self.num_words:
                    break
                columns = line.lower().split()
                word = columns[0] + ' ' + columns[1]
                if is_valid_str(word) and \
                        (max_string_len == -1 or max_string_len is None or len(word) <= max_string_len):
                    tmp_string_list.append(word)
        if len(tmp_string_list) != self.num_words:
            raise IOError('Could not pull enough words from supplied monogram and bigram files. ')
        # interlace to mix up the easy and hard words
        self.string_list[::2] = tmp_string_list[:self.num_words // 2]
        self.string_list[1::2] = tmp_string_list[self.num_words // 2:]

        for i, word in enumerate(self.string_list):
            self.Y_len[i] = len(word)
            self.Y_data[i, 0:len(word)] = text_to_labels(word, self.get_output_size())
            self.X_text.append(word)
        self.Y_len = np.expand_dims(np.array(self.Y_len), 1)

        self.cur_val_index = self.val_split
        self.cur_train_index = 0

    # each time an image is requested from train/val/test, a new random
    # painting of the text is performed
    def get_batch(self, index, size, train):
        # width and height are backwards from typical Keras convention
        # because width is the time dimension when it gets fed into the RNN
        if K.image_data_format() == 'channels_first':
            X_data = np.ones([size, 1, self.img_w, self.img_h])
        else:
            X_data = np.ones([size, self.img_w, self.img_h, 1])

        labels = np.ones([size, self.absolute_max_string_len])
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        source_str = []
        for i in range(0, size):
            # Mix in some blank inputs.  This seems to be important for
            # achieving translational invariance
            if train and i > size - 4:
                if K.image_data_format() == 'channels_first':
                    X_data[i, 0, 0:self.img_w, :] = self.paint_func('')[0, :, :].T
                else:
                    X_data[i, 0:self.img_w, :, 0] = self.paint_func('',)[0, :, :].T
                labels[i, 0] = self.blank_label
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = 1
                source_str.append('')
            else:
                if K.image_data_format() == 'channels_first':
                    X_data[i, 0, 0:self.img_w, :] = self.paint_func(self.X_text[index + i])[0, :, :].T
                else:
                    X_data[i, 0:self.img_w, :, 0] = self.paint_func(self.X_text[index + i])[0, :, :].T
                labels[i, :] = self.Y_data[index + i]
                input_length[i] = self.img_w // self.downsample_factor - 2
                label_length[i] = self.Y_len[index + i]
                source_str.append(self.X_text[index + i])
        inputs = {'the_input': X_data,
                  'the_labels': labels,
                  'input_length': input_length,
                  'label_length': label_length,
                  'source_str': source_str  # used for visualization only
                  }
        outputs = {'ctc': np.zeros([size])}  # dummy data for dummy loss function
        return (inputs, outputs)

    def next_train(self):
        while 1:
            ret = self.get_batch(self.cur_train_index, self.minibatch_size, train=True)
            self.cur_train_index += self.minibatch_size
            if self.cur_train_index >= self.val_split:
                self.cur_train_index = self.cur_train_index % 32
                (self.X_text, self.Y_data, self.Y_len) = shuffle_mats_or_lists(
                    [self.X_text, self.Y_data, self.Y_len], self.val_split)
            yield ret

    def next_val(self):
        while 1:
            ret = self.get_batch(self.cur_val_index, self.minibatch_size, train=False)
            self.cur_val_index += self.minibatch_size
            if self.cur_val_index >= self.num_words:
                self.cur_val_index = self.val_split + self.cur_val_index % 32
            yield ret

    def on_train_begin(self, logs={}):
        # self.build_word_list(16000, 4, 1)
        self.build_word_list(1600, 4, 1)
        self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                  rotate=False, ud=False, multi_fonts=False)

    def on_epoch_begin(self, epoch, logs={}):
        # rebind the paint function to implement curriculum learning
        if epoch >= 3 and epoch < 6:
            self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                      rotate=False, ud=True, multi_fonts=False)
        elif epoch >= 6 and epoch < 9:
            self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                      rotate=False, ud=True, multi_fonts=True)
        elif epoch >= 9:
            self.paint_func = lambda text: paint_text(text, self.img_w, self.img_h,
                                                      rotate=True, ud=True, multi_fonts=True)
        if epoch >= 21 and self.max_string_len < 12:
            # self.build_word_list(32000, 12, 0.5)
            self.build_word_list(3200, 12, 0.5)

#-------------------------------------------------------------------------

def chars_to_onehot(data):
    nSeqLength = data.shape[0]
    ref = np.sort(np.asarray([c for c in ' 0123456789']))
    nClasses = len(ref)
    ix = np.searchsorted(ref,data)
    onehot = np.zeros((nSeqLength, nClasses))
    onehot[np.arange(nSeqLength), ix] = 1
    return onehot

#-------------------------------------------------------------------------

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
    print("Max input string length = " + str(maxTotalLength))
    pix_height = 32
    pix_width = 16 * maxTotalLength
    # data = np.zeros((nSamples, maxTotalLength, 7))  # Input data is ASCII, thus 7 bits, 7 input nodes
    data = np.zeros((nSamples, pix_height, pix_width, 1))  # Input data is (samples, height_pixels, width_pixes, channels)
    labels = np.zeros((nSamples, maxNumLength, 11)) # Output data is one-hot vector with 11 classes: space (' ') and 0-9
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

        im = paint_text(fullstring, pix_width, pix_height, rotate=True, ud=True, multi_fonts=True)
        data[i,:,:,0] = im

        # for j in range(maxTotalLength):
        #     data[i, j, :] = ascii_array(fullstring[j])

    return data, labels, textlines, dotnumbers, maxTotalLength

#-------------------------------------------------------------------------

def get_data_neg(nSamples, maxInputLength, maxOutputLen):
    pix_height = 32
    pix_width = 16 * maxInputLength

    # data = np.zeros((nSamples, maxInputLength, 7))  # Input data is ASCII, thus 7 bits, 7 input nodes
    data = np.zeros((nSamples, pix_height, pix_width, 1))  # Input data is (samples, height_pixels, width_pixes, channels)
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

        im = paint_text(fullstring, pix_width, pix_height, rotate=True, ud=True, multi_fonts=True)
        data[i,:,:,0] = im

        # for j in range(maxInputLength):
        #     data[i, j, :] = ascii_array(fullstring[j])

    labels = np.zeros((nSamples,maxOutputLen,11))    # one-hot vector has 11 elements (' ', and 0 to 9)
    labels[:,:,0] = 1   # label all outputs as spaces
    return data, labels, textlines

#-------------------------------------------------------------------------

##=============================================================
## Create the training and testing data
##=============================================================

## Data array dimension ordering is (samples, sequence_length, input_nodes).
BATCH_SIZE = 32
nPos = int(100 / BATCH_SIZE) * BATCH_SIZE
nNeg = int(100 / BATCH_SIZE) * BATCH_SIZE
bInvertSeq = True

print('\nBuilding training data set...')
dataPos, labelsPos, lineStringsPos, dotStringsPos, maxInputStringLength = get_data_pos(nPos)
dataPos = np.transpose(dataPos,(0,2,1,3))
dataPos = np.flip(dataPos, axis=2)

print('Building validation data set...')
dataNeg, labelsNeg, lineStringsNeg = get_data_neg(nNeg, maxInputStringLength, 8)
dataNeg = np.transpose(dataNeg,(0,2,1,3))
dataNeg = np.flip(dataNeg, axis=2)

if bInvertSeq:
    dataPos = np.flip(dataPos, axis=1)
    dataNeg = np.flip(dataNeg, axis=1)

# ## Display some images
# plt.figure(1)
# for i in range(10):
#     # plt.subplot(1,n,i+1)
#     plt.imshow(np.squeeze(np.transpose(dataPos,(0,2,1,3))[i,:,:,0]), cmap='Greys', vmin=0., vmax=1.)
#     plt.imshow(np.squeeze(np.transpose(dataNeg,(0,2,1,3))[i,:,:,0]), cmap='Greys', vmin=0., vmax=1.)
#     # plt.colorbar()
#     plt.show()
#     plt.title(str(i) + " of " + str(nPos))
#     raw_input("Press Enter to continue...")

dataAll = np.concatenate((dataPos, dataNeg))
labelsAll = np.concatenate((labelsPos, labelsNeg))
lineStringsAll = lineStringsPos + lineStringsNeg

## Shuffle data
indices = np.arange(len(dataAll))
np.random.shuffle(indices)
dataAll = dataAll[indices]
labelsAll = labelsAll[indices]
lineStringsAll = [lineStringsAll[i] for i in indices]

# ## Add noise to data
# noiseMag = 0.5;
# dataAll += np.random.uniform(-noiseMag,noiseMag,dataAll.shape)

## Split into train and validation sets
percentTrain = 80
split_at = int( len(dataAll) * percentTrain/100)
(dataTrain, dataVal) = dataAll[:split_at], dataAll[split_at:]
(labelsTrain, labelsVal) = labelsAll[:split_at], labelsAll[split_at:]
(lineStringsTrain, lineStringsVal) = lineStringsAll[:split_at], lineStringsAll[split_at:]
nTrain = len(dataTrain)
nVal = len(dataVal)

print('\nTraining data and label array shapes:')
print(dataTrain.shape)
print(labelsTrain.shape)

print('\nValidation data and label array shapes:')
print(dataVal.shape)
print(labelsVal.shape)


# # Plot the text regions
# plt.figure(1)
# for i in range(len(dataTrain)):
#   plt.clf()
#   plt.imshow(np.squeeze(dataTrain[i]))
#   plt.colorbar()
#   plt.show()
#   plt.waitforbuttonpress()
# exit()


## Build the model....

nKernSize = 3
nFilters = (8, 16, 32)
# nFilters = (8,)
nConvLayers = len(nFilters)
nRows = 32
nCols = nRows*2
nChan = 1

RNN = layers.GRU   # Try LSTM, GRU, or SimpleRNN
HIDDEN_SIZE = 256
DECODE_LAYERS = 1

# x = np.round(np.random.uniform(size=(nSamples, nRows, nCols, nChan)))
# # May need to flip the input images such that space-to-sequence conversion it top-bottom rather than left-right,
# # due to way the layers.core.Reshape works.  Or build a transpose layer/operation?
# x = x.transpose(0,2,1,3)

bUseSavedModel = True

if bUseSavedModel:
    model = keras.models.load_model('model_dotnum_120_seed1.h5')
    startEpoch = 120
else:
    ## Build the model
    startEpoch = 0
    print('\nBuilding the model...')
    model = Sequential()
    for i in range(nConvLayers):

        model.add(layers.normalization.BatchNormalization(input_shape=dataAll.shape[1:]))
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
                                                input_shape=dataAll.shape[1:]))
        model.add(layers.pooling.MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))


    # Reshape, converting columns to timesteps....
    # (batch_size, timesteps/cols, rows)
    inputSeqLen = dataAll.shape[1] / 2**nConvLayers
    rnnInputDim = dataAll.shape[2] / 2**nConvLayers * nFilters[-1]
    model.add(layers.core.Reshape((inputSeqLen, rnnInputDim)))

    #================================================
    # TODO: Add a dense layer here, with dropout?
    #================================================


    # "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
    # Note: In a situation where your input sequences have a variable length,
    # use input_shape=(None, num_feature).
    model.add(RNN(HIDDEN_SIZE,
                  input_shape=(inputSeqLen, rnnInputDim),
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

    ## Set training loss functions and additional metrics (e.g., accuracy metric)
    start = time.time()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    end = time.time()
    print("\tDone: " + str(end - start) + "seconds")


model.summary()

## Train the model
print('\nTraining the model...')
start = time.time()
history = History()
model.fit(dataTrain, labelsTrain, batch_size=32, epochs=150, verbose=1, callbacks=[history], validation_split=0.0, validation_data=(dataVal, labelsVal), shuffle=True, class_weight=None, sample_weight=None, initial_epoch=startEpoch)
end = time.time()
print("\nTotal training duration: " + str((end - start)/60) + " minutes.")

# # list all data in history
# print(history.history.keys())

## Save the model architecture and weights
json_string = model.to_json()
open('model_dotnum_150_seed2_arch.json', 'w').write(json_string)
model.save_weights('model_dotnum_150_seed2_weights.h5')


## Save loss and accuracy values
a = {'acc': history.history['acc'], 'val_acc': history.history['val_acc'], 'loss': history.history['loss'], 'val_loss': history.history['val_loss']}
with open('results.pkl', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Plot accuracy training history
plt.figure(1)
plt.subplot(1,2,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# Plot loss training history
plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()


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

    plt.show()
    plt.waitforbuttonpress()
