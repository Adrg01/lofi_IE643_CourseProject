from distutils.version import StrictVersion

import os
import sklearn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import keras
import cv2
import numpy as np
import time
import sys
import threading
import math
import callbacks
#from frame_utils import *
np.random.seed(123)
from PIL import Image
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Concatenate, BatchNormalization
from keras.layers import Lambda, Input, concatenate
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, LambdaCallback
#from keras.backend import tf as ktf
from keras import backend as K
from keras.utils import to_categorical
from keras_contrib.losses import DSSIMObjective

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

ZERO_PADDING = 0
SYMMETRIC_PADDING = 1
EXPLICIT = 2
EXPLICIT_IN = 3
mode_names = ["zero", "reflect", "aware", "aware_in"]
mode = EXPLICIT
model_path = "models/debayering_" + mode_names[mode] + "_128_p4.hdf5"
result_path = "../test/debayering/" + mode_names[mode] + "_128_p4.png"
truth_path = "../test/debayering/truth_128_p4.png"
original_path = "../test/debayering/orig_128_p4.png"
error_path = "../test/debayering/" + mode_names[mode] + "_error_128_p4_abs_16.png"
training_phase = 1

load_previous_model = False
make_video = False
data_path = "/home/vivekm/imagenet/"
example_path = "/20171121_150028_947_0"
input_extension = '.png'
output_extension = '.png'
batch_size = 1
dim_x = 128
dim_y = 128
in_channels = 1
out_channels = 3
training_size = 800000
validation_size = 200000
test_size = 200000
epoch_dumper = 100
leading_zeros = 4
epochs = 30
learning_rate = 0.0001

class DataGenerator(object):
    def __init__(self, phase):
        'Initialization'
        self.batch_size = batch_size
        self.phase = phase

    def generate(self, data_path):
        # Infinite loop
        while 1:
            # Compute the list of data files.
            directories = os.listdir(data_path)
            files = []
            if self.phase != 'training':
                files = directories
                files = [os.path.join(data_path, file) for file in files]
            else:
                for directory in directories:
                    dir_path = os.path.join(data_path, directory)
                    files += [dir_path + '/' + file for file in os.listdir(dir_path)]
            # Compute how many batches we'll iterate over each epoch.
            total_batches = int(len(files) / batch_size)
            # Used to sample from the training or test data.
            random_list = np.random.permutation(int(len(files)))
            for batch_id in range(total_batches):
                # Find the video folders for the batch.
                list_IDs_temp = [files[k] for k in random_list[batch_id * batch_size:(batch_id + 1) * batch_size]]
                # Generate data.
                X, y = self.__data_generation(list_IDs_temp, data_path)
                yield X, y

    def __data_generation(self, list_IDs_temp, data_path):
        # Initialization.
        X = np.empty((batch_size, dim_x, dim_y, in_channels))
        y = np.empty((batch_size, dim_x, dim_y, out_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            image_path = list_IDs_temp[i]
            x_data = cv2.imread(image_path)
            if x is not None:
              try:
                x_data = cv2.resize(x_data, (dim_y, dim_x), interpolation=cv2.INTER_AREA)
                y_data = x_data / 255.0
                x_data = np.expand_dims(cv2.cvtColor(x_data, cv2.COLOR_BGR2GRAY), axis=2) / 255.0
                x_data[::2, ::2, 0] = y_data[::2, ::2, 1]
                x_data[::2, 1::2, 0] = y_data[::2, 1::2, 2]
                x_data[1::2, ::2, 0] = y_data[1::2, ::2, 0]
                x_data[1::2, 1::2, 0] = y_data[1::2, 1::2, 1]
            # Store input.
                X[i, :, :, :] = np.expand_dims(x_data, axis=0)
            # Store output.
                y[i, :, :, :] = np.expand_dims(y_data, axis=0)
              except Exception as e:
                    print(str(x))
            else:
                X[i, :, :, :] = X[i-1, :, :, :]
                y[i, :, :, :] = y[i-1, :, :, :]
        return X, y

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        print("Time: " + str(self.times[-1]))

def save_image16(npdata, filename, display=True):
    if display:
        print("Save path: " + filename)
    npdata = (npdata * 65535).astype(np.uint16)
    # Save the data.
    with open(filename, 'wb') as file:
        writer = png.Writer(width=npdata.shape[1], height=npdata.shape[0], bitdepth=16)
        # Convert npdata to the Python list of lists expected by the png writer.
        npdata_2list = npdata.reshape(npdata.shape[0], npdata.shape[1] * npdata.shape[2]).tolist()
        writer.write(file, npdata_2list)

def load_image16(filename, display=False):
    reader = png.Reader(filename)
    _, _, pixels, metadata = reader.read()
    #print(metadata)
    image_2d = np.vstack(list(map(np.uint16, pixels)))
    image_3d = np.reshape(image_2d, [dim_y, dim_x, 3], order='C')
    npdata = (image_3d / 65535.0).astype(np.float32)
    return npdata

def log_process(epoch, logs):
    print("Task: debayering; mode: " + mode_names[mode] + ".")

def make_constant(x, shape):
    initializer = keras.initializers.Constant(value=1)
    alpha = initializer([1, shape[1], shape[2], shape[3]])
    return alpha

def make_variable(x, shape):
    bias = K.variable(np.zeros([shape[3]], dtype=np.float32))
    return bias

def bias_add(v):
    return K.bias_add(v[0], v[1])

def constant_divide(v):
    return v[0] / v[1]

def symmetric_pad(x, rows, columns):
    paddings = tf.constant([[0, 0], rows, columns, [0, 0]])
    return tf.pad(x, paddings, "SYMMETRIC")

def boundary_learned_convolution2d(x, out, kernel, strides, mode):
    if mode == 3:
        return boundary_learned_convolution2d_in(x, out, kernel, strides)
    if mode == 2:
        return boundary_learned_convolution2d_std(x, out, kernel, strides)
    elif mode == 1:
        paddings = [1, 1] if strides % 2 == 1 else [1, 0]
        x = Lambda(symmetric_pad, arguments={'rows':paddings, 'columns':paddings})(x)
        x = Convolution2D(out, kernel_size=kernel, use_bias=False, strides=strides, padding='valid')(x)
        return x
    else:
        x = Convolution2D(out, kernel_size=kernel, use_bias=True, strides=strides, padding='same')(x)
        return x

def boundary_learned_convolution2d_std(x, out, kernel, strides):
    convolution_layer = Convolution2D(out, kernel_size=kernel, use_bias=False, strides=strides, padding='valid')
    print("bconv")
    if x.shape[1] == 1 or x.shape[2] == 1:
        convolution_layer_ = Convolution2D(out, kernel_size=kernel, use_bias=True, strides=strides, padding='same')
        return convolution_layer_(x)

    # Corners.
    convolution_layer_top_left = Convolution2D(out, kernel_size=kernel, use_bias=False, strides=strides, padding='same')
    convolution_layer_top_right = Convolution2D(out, kernel_size=kernel, use_bias=False, strides=strides, padding='same')
    convolution_layer_bottom_left = Convolution2D(out, kernel_size=kernel, use_bias=False, strides=strides, padding='same')
    convolution_layer_bottom_right = Convolution2D(out, kernel_size=kernel, use_bias=False, strides=strides, padding='same')

    if x.shape[2] == 2 or x.shape[1] == 2:
        if strides % 2 == 0:
            convolution_layer_ = Convolution2D(out, kernel_size=kernel, use_bias=True, strides=strides, padding='same')
            return convolution_layer_(x)

        top_left = Lambda(lambda y: y[:, 0:2, 0:2, :])(x)
        top_left = convolution_layer_top_left(top_left)
        top_left = Lambda(lambda y: y[:, 0:1, 0:1, :])(top_left)
        bottom_left = Lambda(lambda y: y[:, -2:, 0:2, :])(x)
        bottom_left = convolution_layer_bottom_left(bottom_left)
        bottom_left = Lambda(lambda y: y[:, -1:, 0:1, :])(bottom_left)
        top_right = Lambda(lambda y: y[:, 0:2, -2:, :])(x)
        top_right = convolution_layer_top_right(top_right)
        top_right = Lambda(lambda y: y[:, 0:1, -1:, :])(top_right)
        bottom_right = Lambda(lambda y: y[:, -2:, -2:, :])(x)
        bottom_right = convolution_layer_bottom_right(bottom_right)
        bottom_right = Lambda(lambda y: y[:, -1:, -1:, :])(bottom_right)

        top = concatenate([top_left, top_right], axis=2)
        bottom = concatenate([bottom_left, bottom_right], axis=2)
        x = concatenate([top, bottom], axis=1)

        output_shape = convolution_layer_top_left.output_shape
        bias = Lambda(make_variable, arguments={'shape':output_shape})(x)
        convolution_layer_top_left.trainable_weights.extend([bias])
        x = Lambda(bias_add)([x, bias])
        return x

    # Edges.
    convolution_layer_top = Convolution2D(out, kernel_size=kernel, use_bias=False, strides=strides, padding='same')
    convolution_layer_bottom = Convolution2D(out, kernel_size=kernel, use_bias=False, strides=strides, padding='same')
    convolution_layer_left = Convolution2D(out, kernel_size=kernel, use_bias=False, strides=strides, padding='same')
    convolution_layer_right = Convolution2D(out, kernel_size=kernel, use_bias=False, strides=strides, padding='same')

    # Edges.
    if strides % 2 == 1:
        top = Lambda(lambda y: y[:, 0:2, :, :])(x)
        top = convolution_layer_top(top)
        top = Lambda(lambda y: y[:, 0:1, 1:-1, :])(top)
        left = Lambda(lambda y: y[:, :, 0:2, :])(x)
        left = convolution_layer_left(left)
        left = Lambda(lambda y: y[:, 1:-1, 0:1, :])(left)
        bottom = Lambda(lambda y: y[:, -2:, :, :])(x)
        bottom = convolution_layer_bottom(bottom)
        bottom = Lambda(lambda y: y[:, -1:, 1:-1, :])(bottom)
        right = Lambda(lambda y: y[:, :, -2:, :])(x)
        right = convolution_layer_right(right)
        right = Lambda(lambda y: y[:, 1:-1, -1:, :])(right)
    else:
        top = Lambda(lambda y: y[:, 0:2, :, :])(x)
        top = convolution_layer_top(top)
        top = Lambda(lambda y: y[:, 0:1, 1:, :])(top)
        left = Lambda(lambda y: y[:, :, 0:2, :])(x)
        left = convolution_layer_left(left)
        left = Lambda(lambda y: y[:, 1:, 0:1, :])(left)
    # Corners.
    top_left = Lambda(lambda y: y[:, 0:2, 0:2, :])(x)
    top_left = convolution_layer_top_left(top_left)
    top_left = Lambda(lambda y: y[:, 0:1, 0:1, :])(top_left)
    if strides % 2 == 1:
        bottom_left = Lambda(lambda y: y[:, -2:, 0:2, :])(x)
        bottom_left = convolution_layer_bottom_left(bottom_left)
        bottom_left = Lambda(lambda y: y[:, -1:, 0:1, :])(bottom_left)
        top_right = Lambda(lambda y: y[:, 0:2, -2:, :])(x)
        top_right = convolution_layer_top_right(top_right)
        top_right = Lambda(lambda y: y[:, 0:1, -1:, :])(top_right)
        bottom_right = Lambda(lambda y: y[:, -2:, -2:, :])(x)
        bottom_right = convolution_layer_bottom_right(bottom_right)
        bottom_right = Lambda(lambda y: y[:, -1:, -1:, :])(bottom_right)

    x = convolution_layer(x)
    # Corners to edges.
    left = concatenate([top_left, left], axis=1)
    if strides % 2 == 1:
        left = concatenate([left, bottom_left], axis=1)
        right = concatenate([top_right, right], axis=1)
        right = concatenate([right, bottom_right], axis=1)
    # Edges to body.
    x = concatenate([top, x], axis=1)
    if strides % 2 == 1:
        x = concatenate([x, bottom], axis=1)
        x = concatenate([left, x], axis=2)
        x = concatenate([x, right], axis=2)
    else:
        x = concatenate([left, x], axis=2)

    output_shape = convolution_layer_top.output_shape
    bias = Lambda(make_variable, arguments={'shape':output_shape})(x)
    convolution_layer_top.trainable_weights.extend([bias])
    x = Lambda(bias_add)([x, bias])

    return x

def boundary_learned_convolution2d_in(x, out, kernel, strides):
    convolution_layer = Convolution2D(out, kernel_size=kernel, use_bias=False, strides=strides, padding='valid')
    print("bconv_in")
    if x.shape[1] == 1 or x.shape[2] == 1:
        convolution_layer_ = Convolution2D(out, kernel_size=kernel, use_bias=True, strides=strides, padding='same')
        return convolution_layer_(x)

    padding = 'same' if x.shape[2] == 2 or x.shape[1] == 2 else 'valid'
    # Corners.
    convolution_layer_top_left = Convolution2D(out, kernel_size=kernel, use_bias=False, strides=strides, padding=padding)
    convolution_layer_top_right = Convolution2D(out, kernel_size=kernel, use_bias=False, strides=strides, padding=padding)
    convolution_layer_bottom_left = Convolution2D(out, kernel_size=kernel, use_bias=False, strides=strides, padding=padding)
    convolution_layer_bottom_right = Convolution2D(out, kernel_size=kernel, use_bias=False, strides=strides, padding=padding)

    if x.shape[2] == 2 or x.shape[1] == 2:
        if strides % 2 == 0:
            convolution_layer_ = Convolution2D(out, kernel_size=kernel, use_bias=True, strides=strides, padding='same')
            return convolution_layer_(x)

        top_left = Lambda(lambda y: y[:, 0:2, 0:2, :])(x)
        top_left = convolution_layer_top_left(top_left)
        top_left = Lambda(lambda y: y[:, 0:1, 0:1, :])(top_left)
        bottom_left = Lambda(lambda y: y[:, -2:, 0:2, :])(x)
        bottom_left = convolution_layer_bottom_left(bottom_left)
        bottom_left = Lambda(lambda y: y[:, -1:, 0:1, :])(bottom_left)
        top_right = Lambda(lambda y: y[:, 0:2, -2:, :])(x)
        top_right = convolution_layer_top_right(top_right)
        top_right = Lambda(lambda y: y[:, 0:1, -1:, :])(top_right)
        bottom_right = Lambda(lambda y: y[:, -2:, -2:, :])(x)
        bottom_right = convolution_layer_bottom_right(bottom_right)
        bottom_right = Lambda(lambda y: y[:, -1:, -1:, :])(bottom_right)

        top = concatenate([top_left, top_right], axis=2)
        bottom = concatenate([bottom_left, bottom_right], axis=2)
        x = concatenate([top, bottom], axis=1)

        output_shape = convolution_layer_top_left.output_shape
        bias = Lambda(make_variable, arguments={'shape':output_shape})(x)
        convolution_layer_top_left.trainable_weights.extend([bias])
        x = Lambda(bias_add)([x, bias])
        return x

    # Edges.
    convolution_layer_top = Convolution2D(out, kernel_size=kernel, use_bias=False, strides=strides, padding='valid')
    convolution_layer_bottom = Convolution2D(out, kernel_size=kernel, use_bias=False, strides=strides, padding='valid')
    convolution_layer_left = Convolution2D(out, kernel_size=kernel, use_bias=False, strides=strides, padding='valid')
    convolution_layer_right = Convolution2D(out, kernel_size=kernel, use_bias=False, strides=strides, padding='valid')

    if strides % 2 == 1:
        top = Lambda(lambda y: y[:, 0:3, :, :])(x)
        top = convolution_layer_top(top)
        left = Lambda(lambda y: y[:, :, 0:3, :])(x)
        left = convolution_layer_left(left)
        bottom = Lambda(lambda y: y[:, -3:, :, :])(x)
        bottom = convolution_layer_bottom(bottom)
        right = Lambda(lambda y: y[:, :, -3:, :])(x)
        right = convolution_layer_right(right)
    else:
        top = Lambda(lambda y: y[:, 0:3, :, :])(x)
        top = convolution_layer_top(top)
        left = Lambda(lambda y: y[:, :, 0:3, :])(x)
        left = convolution_layer_left(left)
    # Corners.
    top_left = Lambda(lambda y: y[:, 0:3, 0:3, :])(x)
    top_left = convolution_layer_top_left(top_left)
    if strides % 2 == 1:
        bottom_left = Lambda(lambda y: y[:, -3:, 0:3, :])(x)
        bottom_left = convolution_layer_bottom_left(bottom_left)
        top_right = Lambda(lambda y: y[:, 0:3, -3:, :])(x)
        top_right = convolution_layer_top_right(top_right)
        bottom_right = Lambda(lambda y: y[:, -3:, -3:, :])(x)
        bottom_right = convolution_layer_bottom_right(bottom_right)

    x = convolution_layer(x)
    # Corners to edges.
    left = concatenate([top_left, left], axis=1)
    if strides % 2 == 1:
        left = concatenate([left, bottom_left], axis=1)
        right = concatenate([top_right, right], axis=1)
        right = concatenate([right, bottom_right], axis=1)
    # Edges to body.
    x = concatenate([top, x], axis=1)
    if strides % 2 == 1:
        x = concatenate([x, bottom], axis=1)
        x = concatenate([left, x], axis=2)
        x = concatenate([x, right], axis=2)
    else:
        x = concatenate([left, x], axis=2)

    output_shape = convolution_layer_top.output_shape
    bias = Lambda(make_variable, arguments={'shape':output_shape})(x)
    convolution_layer_top.trainable_weights.extend([bias])
    x = Lambda(bias_add)([x, bias])

    return x

inputs = Input(shape=(dim_y, dim_x, in_channels))
x = inputs
modes = ["zero-padding", "symmetric-padding", "explicit", "explicit-in"]
print("Mode: %s" % modes[mode])
# Used for skip-connections.
encoder_layers = []
# Layer depth power.
power = 4
# Zero padding.
padding = 'same'
# Default activation type.
activation = 'relu'
# Encoder.
encoder_n = 7
for i in range(encoder_n):
    kernel = (3, 3)
    strides = 1 if i == 0 else 2
    # 32, 32, 64, 64, 128, 128
    channels = 2 ** (int(i / 2) + power) if i > 0 else 2 ** power
    print(channels)
    x = boundary_learned_convolution2d(x, channels, kernel, strides, mode)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    # Store layer.
    encoder_layers.append(x)

# Fully connected emulation through convolutions.
flat_n = 1
for i in range(flat_n):
    kernel = (1, 1)
    strides = 1
    # 128 * 2, 128 * 2
    channels = 2 ** (power + 4)
    print(channels)
    x = boundary_learned_convolution2d(x, channels, kernel, strides, mode)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

# Decoder.
decoder_n = 6
for i in range(decoder_n):
    # Resize-convolution.
    kernel = (3, 3)
    strides = 1
    size = [int(encoder_layers[decoder_n - i - 1].shape[1]), int(encoder_layers[decoder_n - i - 1].shape[2])]
    # 128, 64, 64, 32, 32
    channels = 2 ** (int((decoder_n - i - 1) / 2) + power)
    print(channels)
    x = Lambda(lambda y: tf.image.resize_images(y, size))(x)
    x =  boundary_learned_convolution2d(x, channels, kernel, strides, mode)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    # Concatenation.
    x = concatenate([x, encoder_layers[decoder_n - i - 1]], axis=3)
    # 128, 64, 64, 32, 1
    channels = 2 ** (int((decoder_n - i - 1) / 2) + power) if i != decoder_n - 1 else out_channels
    x =  boundary_learned_convolution2d(x, channels, kernel, strides, mode)
    activation = activation if i != decoder_n - 1 else 'sigmoid'
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

model = Model(inputs=inputs, outputs=x)
model.compile(keras.optimizers.Adam(lr=learning_rate), keras.losses.mean_squared_error, metrics=[DSSIMObjective()])

if not training_phase:
    print("Beginning test phase.")
    model.load_weights(model_path)
    global_score = 0
    global_score2 = 0
    global_score3 = 0
    global_score4 = 0
    count = 0
    count2 = 0
    mean_error_batch = np.zeros((batch_size, dim_y, dim_x, out_channels), dtype=np.float32)
    for batch in DataGenerator(phase='validation').generate(data_path + "test"):
        result = model.predict(batch[0], batch_size=batch_size)
        score = np.sum(np.square(batch[1] - result))
        score2 = model.evaluate(batch[0], batch[1])
        global_score += score
        global_score2 += score2[1]
        global_score3 += score2[0]
        global_score4 += 10 * math.log10(1.0 / score2[0])
        mean_error_batch += np.abs(batch[1] - result)
        count += 1
        if count % 100 == 0:
            print("Computed: " + str(count) + " batches.")
    global_score /= (count // batch_size)
    global_score2 /= (count // batch_size)
    global_score3 /= (count // batch_size)
    global_score4 /= (count // batch_size)
    mean_error_batch /= (count // batch_size)
    mean_error = np.zeros((dim_y, dim_x, out_channels), dtype=np.float32)
    for batch_id in range(batch_size):
        mean_error += mean_error_batch[batch_id, :, :, :] / batch_size
    save_image16(mean_error, error_path)
    print("Global L2 loss: " + str(global_score))
    print("Global DSSIM loss: " + str(global_score2))
    print("Norm. L2 loss: " + str(global_score3))
    print("PSNR: " + str(global_score4))
    print("Mean error saved.")

else:
    if load_previous_model:
        model.load_weights(model_path)
    # Periodically save checkpoints.
    checkpoint_path = model_path
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True)
    logger = LambdaCallback(on_epoch_begin=log_process)
    time_callback = TimeHistory()
    #"Disparity, mode: " + mode_names[mode] + "."
    callbacks_list = [checkpoint, logger, time_callback]

    # Train the model.
    model.fit_generator(DataGenerator(phase='training').generate(data_path + "train"),
                        steps_per_epoch=training_size // batch_size // epoch_dumper,
                        epochs=epochs,
                        callbacks=callbacks_list,
                        verbose=1,
                        validation_data=DataGenerator(phase='validation').generate(data_path + "val"),
                        validation_steps=validation_size // batch_size // epoch_dumper)

    model.save_weights("./models/segmentation_last-epoch_{0}.hdf5".format(epochs))
