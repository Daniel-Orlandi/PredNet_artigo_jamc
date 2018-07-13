from keras.models import Sequential
from keras.layers.convolutional import Conv3D, Conv2D, SeparableConv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers import Maxpooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np



seq = Sequential()

seq.add(SeparableConv2D(filters = 1, kernel_size = (5,5), input_shape=(None,255,255,3), padding='same', data_format='channels_last',stride = (2,2),depth_multiplier=1 )

seq.add(Maxpooling2D(pool_size=(2,2), stride=2, padding='same')

seq.add(ConvLSTM2D(filters=62, kernel_size=(3, 3),
                   padding='same', return_sequences=True))

seq.add(ConvLSTM2D(filters=62, kernel_size=(3, 3),
                   padding='same', return_sequences=True))

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
seq.compile(loss='relu', optimizer='adam')

#Por enquanto isso.