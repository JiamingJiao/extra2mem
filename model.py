#!/usr/bin/env python
# -*- coding:  utf-8 -*-

import numpy as np
import math
import functools
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, UpSampling2D, Dropout, BatchNormalization, Flatten, Dense, MaxPooling2D
from keras.layers import Conv3D, UpSampling3D, Lambda, Add, Activation
from keras.layers import TimeDistributed, ConvLSTM2D
# from keras.layers import MaxPooling3D, Reshape, Permute, ZeroPadding3D, Bidirectional
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import Constant


class Networks(object):
    def __init__(self, imgSize=(256, 256), channels=1, gKernels=64, dKernels=64, gKernelSize=3, temporalDepth=None, activationG=None):
        self.imgSize = imgSize
        self.imgRows = imgSize[0]
        self.imgCols = imgSize[1]
        self.channels = channels
        self.gKernels = gKernels
        self.dKernels = dKernels
        self.temporalDepth = temporalDepth
        self.activationG = activationG
        self.gKernelSize = gKernelSize

    def uNet(self):
        inputs = Input((self.imgRows, self.imgCols, self.channels))
        encoder1 = Conv2D(self.gKernels, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal', name='encoder1')(inputs)
        encoder2 = Conv2D(self.gKernels*2, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder1)
        encoder2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder2)
        encoder2 = LeakyReLU(alpha=0.2, name='encoder2')(encoder2)
        encoder3 = Conv2D(self.gKernels*4, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder2)
        encoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder3)
        encoder3 = LeakyReLU(alpha=0.2, name='encoder3')(encoder3)
        encoder4 = Conv2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder3)
        encoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder4)
        encoder4 = LeakyReLU(alpha=0.2, name='encoder4')(encoder4)
        encoder5 = Conv2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder4)
        encoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder5)
        encoder5 = LeakyReLU(alpha=0.2, name='encoder5')(encoder5)
        encoder6 = Conv2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder5)
        encoder6 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder6)
        encoder6 = LeakyReLU(alpha=0.2, name='encoder6')(encoder6)
        encoder7 = Conv2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder6)
        encoder7 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder7)
        encoder7 = LeakyReLU(alpha=0.2, name='encoder7')(encoder7)
        encoder8 = Conv2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder7)
        encoder8 = LeakyReLU(alpha=0.2, name='encoder8')(encoder8)

        decoder1 = Conv2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(encoder8))
        decoder1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder1)
        decoder1 = Dropout(0.5, name='decoder1')(decoder1)
        connection1 = Concatenate(axis=-1, name='connection1')([decoder1, encoder7])
        decoder2 = Conv2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(connection1))
        decoder2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder2)
        decoder2 = Dropout(0.5, name='decoder2')(decoder2)
        connection2 = Concatenate(axis=-1, name='connection2')([decoder2, encoder6])
        decoder3 = Conv2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(connection2))
        decoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder3)
        decoder3 = Dropout(0.5, name='decoder3')(decoder3)
        connection3 = Concatenate(axis=-1, name='connection3')([decoder3, encoder5])
        decoder4 = Conv2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(connection3))
        decoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False, name='decoder4')(decoder4)
        connection4 = Concatenate(axis=-1, name='connection4')([decoder4, encoder4])
        decoder5 = Conv2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(connection4))
        decoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False, name='decoder5')(decoder5)
        connection5 = Concatenate(axis=-1, name='connection5')([decoder5, encoder3])
        decoder6 = Conv2D(self.gKernels*4, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(connection5))
        decoder6 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False, name='decoder6')(decoder6)
        connection6 = Concatenate(axis=-1, name='connection6')([decoder6, encoder2])
        decoder7 = Conv2D(self.gKernels*2, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(connection6))
        decoder7 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False, name='decoder7')(decoder7)
        connection7 = Concatenate(axis=-1, name='connection7')([decoder7, encoder1])
        decoder8 = Conv2D(self.gKernels, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal', name='decoder8')(UpSampling2D(size=(2, 2))
                                                                                                                                               (connection7))
        decoder9 = Conv2D(1, 1, activation=self.activationG, name='decoder9')(decoder8)
        model = Model(inputs=inputs, outputs=decoder9, name='uNet')
        return model

    def uNet5(self):
        inputs = Input((self.imgRows, self.imgCols, self.channels))  # 32x32

        encoder1 = Conv2D(self.gKernels, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(inputs)
        encoder1 = LeakyReLU(alpha=0.2, name='encoder1')(encoder1)  # 16x16

        encoder2 = Conv2D(self.gKernels*2, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder1)
        encoder2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder2)
        encoder2 = LeakyReLU(alpha=0.2, name='encoder2')(encoder2)  # 8x8

        encoder3 = Conv2D(self.gKernels*4, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder2)
        encoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder3)
        encoder3 = LeakyReLU(alpha=0.2, name='encoder3')(encoder3)  # 4x4

        encoder4 = Conv2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder3)
        encoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder4)
        encoder4 = LeakyReLU(alpha=0.2, name='encoder4')(encoder4)  # 2x2

        encoder5 = Conv2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder4)
        # encoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder5)
        encoder5 = LeakyReLU(alpha=0.2, name='encoder5')(encoder5)  # 1x1

        decoder1 = Conv2D(self.gKernels*8, 1, activation='relu', padding='valid', kernel_initializer='he_normal')(encoder5)
        decoder1 = UpSampling2D(size=2, name='decoder1')(decoder1)  # 2x2
        decoder1 = Dropout(0.5)(decoder1)
        connection1 = Concatenate(axis=-1, name='connection1')([decoder1, encoder4])

        decoder2 = Conv2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(connection1)
        decoder2 = UpSampling2D(size=2, name='decoder2')(decoder2)  # 4x4
        decoder2 = Dropout(0.5)(decoder2)
        connection2 = Concatenate(axis=-1, name='connection2')([decoder2, encoder3])

        decoder3 = Conv2D(self.gKernels*4, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(connection2)
        decoder3 = UpSampling2D(size=2, name='decoder3')(decoder3)  # 8x8
        decoder3 = Dropout(0.5)(decoder3)
        connection3 = Concatenate(axis=-1, name='connection3')([decoder3, encoder2])

        decoder4 = Conv2D(self.gKernels*2, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(connection3)
        decoder4 = UpSampling2D(size=2, name='decoder4')(decoder4)  # 16x16
        # decoder4 = Dropout(0.5)(decoder4)
        connection4 = Concatenate(axis=-1, name='connection4')([decoder4, encoder1])

        decoder5 = Conv2D(self.gKernels, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(connection4)
        decoder5 = UpSampling2D(size=2, name='decoder5')(decoder5)  # 32x32
        # decoder5 = Dropout(0.5)(decoder5)

        decoder6 = Conv2D(int(self.gKernels/2), self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(decoder5)
        decoder6 = Conv2D(int(self.gKernels/4), self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(decoder6)
        decoder6 = Conv2D(1, self.gKernelSize, activation=self.activationG, padding='same', kernel_initializer='he_normal')(decoder6)

        model = Model(inputs=inputs, outputs=decoder6, name='uNet5')
        return model

    def uNet5_2(self):
        inputs = Input((self.imgRows, self.imgCols, self.channels))  # 32x32
        
        encoder1 = Conv2D(self.gKernels, self.gKernelSize, padding='same', kernel_initializer='he_normal')(inputs)
        encoder1 = LeakyReLU(alpha=0.2)(encoder1)
        encoder1 = Conv2D(self.gKernels, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder1)
        encoder1 = LeakyReLU(alpha=0.2, name='encoder1')(encoder1)  # 16x16

        encoder2 = Conv2D(self.gKernels*2, self.gKernelSize, padding='same', kernel_initializer='he_normal')(encoder1)
        encoder2 = LeakyReLU(alpha=0.2)(encoder2)
        encoder2 = Conv2D(self.gKernels*2, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder2)
        encoder2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder2)
        encoder2 = LeakyReLU(alpha=0.2, name='encoder2')(encoder2)  # 8x8

        encoder3 = Conv2D(self.gKernels*4, self.gKernelSize, padding='same', kernel_initializer='he_normal')(encoder2)
        encoder3 = LeakyReLU(alpha=0.2)(encoder3)
        encoder3 = Conv2D(self.gKernels*4, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder3)
        encoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder3)
        encoder3 = LeakyReLU(alpha=0.2, name='encoder3')(encoder3)  # 4x4

        encoder4 = Conv2D(self.gKernels*8, self.gKernelSize, padding='same', kernel_initializer='he_normal')(encoder3)
        encoder4 = LeakyReLU(alpha=0.2)(encoder4)
        encoder4 = Conv2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder4)
        encoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder4)
        encoder4 = LeakyReLU(alpha=0.2, name='encoder4')(encoder4)  # 2x2

        encoder5 = Conv2D(self.gKernels*8, self.gKernelSize, padding='same', kernel_initializer='he_normal')(encoder4)
        encoder5 = LeakyReLU(alpha=0.2)(encoder5)
        encoder5 = Conv2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder5)
        # encoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder5)
        encoder5 = LeakyReLU(alpha=0.2, name='encoder5')(encoder5)  # 1x1

        decoder1 = Conv2D(self.gKernels*8, 1, activation='relu', padding='valid', kernel_initializer='he_normal')(encoder5)
        decoder1 = Conv2D(self.gKernels*8, 1, activation='relu', padding='valid', kernel_initializer='he_normal')(decoder1)
        decoder1 = UpSampling2D(size=2, name='decoder1')(decoder1)  # 2x2
        decoder1 = Dropout(0.5)(decoder1)
        connection1 = Concatenate(axis=-1, name='connection1')([decoder1, encoder4])

        decoder2 = Conv2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(connection1)
        decoder2 = Conv2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(decoder2)
        decoder2 = UpSampling2D(size=2, name='decoder2')(decoder2)  # 4x4
        decoder2 = Dropout(0.5)(decoder2)
        connection2 = Concatenate(axis=-1, name='connection2')([decoder2, encoder3])

        decoder3 = Conv2D(self.gKernels*4, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(connection2)
        decoder3 = Conv2D(self.gKernels*4, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(decoder3)
        decoder3 = UpSampling2D(size=2, name='decoder3')(decoder3)  # 8x8
        decoder3 = Dropout(0.5)(decoder3)
        connection3 = Concatenate(axis=-1, name='connection3')([decoder3, encoder2])

        decoder4 = Conv2D(self.gKernels*2, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(connection3)
        decoder4 = Conv2D(self.gKernels*2, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(decoder4)
        decoder4 = UpSampling2D(size=2, name='decoder4')(decoder4)  # 16x16
        # decoder4 = Dropout(0.5)(decoder4)
        connection4 = Concatenate(axis=-1, name='connection4')([decoder4, encoder1])

        decoder5 = Conv2D(self.gKernels, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(connection4)
        decoder5 = Conv2D(self.gKernels, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(decoder5)
        decoder5 = UpSampling2D(size=2, name='decoder5')(decoder5)  # 32x32
        # decoder5 = Dropout(0.5)(decoder5)

        decoder6 = Conv2D(int(self.gKernels/2), self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(decoder5)
        decoder6 = Conv2D(int(self.gKernels/4), self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(decoder6)
        decoder6 = Conv2D(1, self.gKernelSize, activation=self.activationG, padding='same', kernel_initializer='he_normal')(decoder6)

        model = Model(inputs=inputs, outputs=decoder6, name='uNet5_2')
        return model

    def uNet3d(self):
        inputs = Input((self.temporalDepth, self.imgRows, self.imgCols, self.channels))
        encoder1 = Conv3D(filters=self.gKernels, kernel_size=self.gKernelSize, strides=(1, 2, 2), padding='same', kernel_initializer='he_normal')(inputs)
        encoder2 = Conv3D(filters=self.gKernels*2, kernel_size=self.gKernelSize, strides=(1, 2, 2), padding='same', kernel_initializer='he_normal')(encoder1)
        encoder2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder2)
        encoder2 = LeakyReLU(alpha=0.2)(encoder2)
        encoder3 = Conv3D(filters=self.gKernels*4, kernel_size=self.gKernelSize, strides=(1, 2, 2), padding='same', kernel_initializer='he_normal')(encoder2)
        encoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder3)
        encoder3 = LeakyReLU(alpha=0.2)(encoder3)
        encoder4 = Conv3D(filters=self.gKernels*8, kernel_size=self.gKernelSize, strides=(1, 2, 2), padding='same', kernel_initializer='he_normal')(encoder3)
        encoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder4)
        encoder4 = LeakyReLU(alpha=0.2)(encoder4)
        encoder5 = Conv3D(filters=self.gKernels*8, kernel_size=self.gKernelSize, strides=(1, 2, 2), padding='same', kernel_initializer='he_normal')(encoder4)
        encoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder5)
        encoder5 = LeakyReLU(alpha=0.2)(encoder5)
        encoder6 = Conv3D(filters=self.gKernels*8, kernel_size=self.gKernelSize, strides=(1, 2, 2), padding='same', kernel_initializer='he_normal')(encoder5)
        encoder6 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder6)
        encoder6 = LeakyReLU(alpha=0.2)(encoder6)
        encoder7 = Conv3D(filters=self.gKernels*8, kernel_size=self.gKernelSize, strides=(1, 2, 2), padding='same', kernel_initializer='he_normal')(encoder6)
        encoder7 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder7)
        encoder7 = LeakyReLU(alpha=0.2)(encoder7)
        encoder8 = Conv3D(filters=self.gKernels*8, kernel_size=self.gKernelSize, strides=(1, 2, 2), padding='same', kernel_initializer='he_normal')(encoder7)
        encoder8 = LeakyReLU(alpha=0.2)(encoder8)
        
        decoder1 = Conv3D(self.gKernels*8, kernel_size=self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(1, 2, 2))
                                                                                                                                            (encoder8))
        decoder1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder1)
        decoder1 = Dropout(0.5)(decoder1)
        connection1 = Concatenate(axis=-1)([decoder1, encoder7])
        decoder2 = Conv3D(self.gKernels*8, kernel_size=self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(1, 2, 2))
                                                                                                                                            (connection1))
        decoder2=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder2)
        decoder2 = Dropout(0.5)(decoder2)
        connection2 = Concatenate(axis=-1)([decoder2, encoder6])
        decoder3 = Conv3D(self.gKernels*8, kernel_size=self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(1, 2, 2))
                                                                                                                                            (connection2))
        decoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder3)
        decoder3 = Dropout(0.5)(decoder3)
        connection3 = Concatenate(axis=-1)([decoder3, encoder5])
        decoder4 = Conv3D(self.gKernels*8, kernel_size=self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(1, 2, 2))
                                                                                                                                            (connection3))
        decoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder4)
        connection4 = Concatenate(axis=-1)([decoder4, encoder4])
        decoder5 = Conv3D(self.gKernels*8, kernel_size=self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(1, 2, 2))
                                                                                                                                            (connection4))
        decoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder5)
        connection5 = Concatenate(axis=-1)([decoder5, encoder3])
        decoder6 = Conv3D(self.gKernels*4, kernel_size=self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(1, 2, 2))
                                                                                                                                            (connection5))
        decoder6 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder6)
        connection6 = Concatenate(axis=-1)([decoder6, encoder2])
        decoder7 = Conv3D(self.gKernels*2, kernel_size=self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(1, 2, 2))
                                                                                                                                            (connection6))
        decoder7 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder7)
        connection7 = Concatenate(axis=-1)([decoder7, encoder1])
        decoder8 = Conv3D(self.gKernels, kernel_size=self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(1, 2, 2))
                                                                                                                                          (connection7))
        decoder9 = Conv3D(1, kernel_size=(self.temporalDepth, 1, 1), activation='relu', padding='valid', kernel_initializer='he_normal')(decoder8)
        squeezed10 = Lambda(squeeze, output_shape=(self.imgRows, self.imgCols, self.channels), arguments={'layer': 1})(decoder9)
        decoder11 = Conv2D(1, kernel_size=(1, 1), activation=self.activationG, padding='valid', kernel_initializer='he_normal')(squeezed10)
        model = Model(inputs=inputs, outputs=decoder11, name='uNet3d')
        return model

    def uNet3d5(self):
        inputs = Input((self.temporalDepth, self.imgRows, self.imgCols, self.channels))  # 32x32

        encoder1 = Conv3D(self.gKernels, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(inputs)
        encoder1 = LeakyReLU(alpha=0.2, name='encoder1')(encoder1)  # 16x16

        encoder2 = Conv3D(self.gKernels*2, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder1)
        encoder2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder2)
        encoder2 = LeakyReLU(alpha=0.2, name='encoder2')(encoder2)  # 8x8

        encoder3 = Conv3D(self.gKernels*4, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder2)
        encoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder3)
        encoder3 = LeakyReLU(alpha=0.2, name='encoder3')(encoder3)  # 4x4

        encoder4 = Conv3D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder3)
        encoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder4)
        encoder4 = LeakyReLU(alpha=0.2, name='encoder4')(encoder4)  # 2x2

        encoder5 = Conv3D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder4)
        # encoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder5)
        encoder5 = LeakyReLU(alpha=0.2, name='encoder5')(encoder5)  # 1x1

        decoder1 = Conv3D(self.gKernels*8, 1, activation='relu', padding='valid', kernel_initializer='he_normal')(encoder5)
        decoder1 = UpSampling3D(size=2, name='decoder1')(decoder1)  # 2x2
        decoder1 = Dropout(0.5)(decoder1)
        connection1 = Concatenate(axis=-1, name='connection1')([decoder1, encoder4])

        decoder2 = Conv3D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(connection1)
        decoder2 = UpSampling3D(size=2, name='decoder2')(decoder2)  # 4x4
        decoder2 = Dropout(0.5)(decoder2)
        connection2 = Concatenate(axis=-1, name='connection2')([decoder2, encoder3])

        decoder3 = Conv3D(self.gKernels*4, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(connection2)
        decoder3 = UpSampling3D(size=2, name='decoder3')(decoder3)  # 8x8
        decoder3 = Dropout(0.5)(decoder3)
        connection3 = Concatenate(axis=-1, name='connection3')([decoder3, encoder2])

        decoder4 = Conv3D(self.gKernels*2, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(connection3)
        decoder4 = UpSampling3D(size=2, name='decoder4')(decoder4)  # 16x16
        # decoder4 = Dropout(0.5)(decoder4)
        connection4 = Concatenate(axis=-1, name='connection4')([decoder4, encoder1])

        decoder5 = Conv3D(self.gKernels, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(connection4)
        decoder5 = UpSampling3D(size=2, name='decoder5')(decoder5)  # 32x32
        # decoder5 = Dropout(0.5)(decoder5)

        decoder6 = Conv3D(int(self.gKernels/2), self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(decoder5)
        decoder6 = Conv3D(int(self.gKernels/4), self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(decoder6)
        decoder6 = Conv3D(1, self.gKernelSize, activation=self.activationG, padding='same', kernel_initializer='he_normal')(decoder6)

        model = Model(inputs=inputs, outputs=decoder6, name='uNet3d5')
        return model

    def uNet3d5_2(self):
        inputs = Input((self.temporalDepth, self.imgRows, self.imgCols, self.channels))  # 32x32

        encoder1 = Conv3D(self.gKernels, self.gKernelSize, padding='same', kernel_initializer='he_normal')(inputs)
        # encoder1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder1)
        encoder1 = LeakyReLU(alpha=0.2)(encoder1)
        encoder1 = Conv3D(self.gKernels, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder1)
        encoder1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder1)
        feature_mapping1 = Conv3D(self.gKernels, 1, strides=2, padding='same', kernel_initializer=Constant(1/int(inputs.shape[-1])))
        feature_mapping1.trainable = False
        shortcut1 = feature_mapping1(inputs)
        shortcut1 = Add()([shortcut1, encoder1])
        encoder1 = LeakyReLU(alpha=0.2, name='encoder1')(shortcut1)  # 16x16

        encoder2 = Conv3D(self.gKernels*2, self.gKernelSize, padding='same', kernel_initializer='he_normal')(encoder1)
        encoder2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder2)
        encoder2 = LeakyReLU(alpha=0.2)(encoder2)
        encoder2 = Conv3D(self.gKernels*2, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder2)
        encoder2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder2)
        feature_mapping2 = Conv3D(self.gKernels*2, 1, strides=2, padding='same', kernel_initializer=Constant(1/int(encoder1.shape[-1])))
        feature_mapping2.trainable = False
        shortcut2 = feature_mapping2(encoder1)
        shortcut2 = Add()([shortcut2, encoder2])
        encoder2 = LeakyReLU(alpha=0.2, name='encoder2')(shortcut2)  # 8x8

        encoder3 = Conv3D(self.gKernels*4, self.gKernelSize, padding='same', kernel_initializer='he_normal')(encoder2)
        encoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder3)
        encoder3 = LeakyReLU(alpha=0.2)(encoder3)
        encoder3 = Conv3D(self.gKernels*4, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder3)
        encoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder3)
        feature_mapping3 = Conv3D(self.gKernels*4, 1, strides=2, padding='same', kernel_initializer=Constant(1/int(encoder2.shape[-1])))
        feature_mapping3.trainable = False
        shortcut3 = feature_mapping3(encoder2)
        shortcut3 = Add()([shortcut3, encoder3])
        encoder3 = LeakyReLU(alpha=0.2, name='encoder3')(shortcut3)  # 4x4

        encoder4 = Conv3D(self.gKernels*8, self.gKernelSize, padding='same', kernel_initializer='he_normal')(encoder3)
        encoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder4)
        encoder4 = LeakyReLU(alpha=0.2)(encoder4)
        encoder4 = Conv3D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder4)
        encoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder4)
        feature_mapping4 = Conv3D(self.gKernels*8, 1, strides=2, padding='same', kernel_initializer=Constant(1/int(encoder3.shape[-1])))
        feature_mapping4.trainable = False
        shortcut4 = feature_mapping4(encoder3)
        shortcut4 = Add()([shortcut4, encoder4])
        encoder4 = LeakyReLU(alpha=0.2, name='encoder4')(shortcut4)  # 2x2

        encoder5 = Conv3D(self.gKernels*8, self.gKernelSize, padding='same', kernel_initializer='he_normal')(encoder4)
        # encoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder5)
        encoder5 = LeakyReLU(alpha=0.2)(encoder5)
        encoder5 = Conv3D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal')(encoder5)
        # encoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder5)
        feature_mapping5 = Conv3D(self.gKernels*8, 1, strides=2, padding='same', kernel_initializer=Constant(1/int(encoder4.shape[-1])))
        feature_mapping5.trainable = False
        shortcut5 = feature_mapping5(encoder4)
        shortcut5 = Add()([shortcut5, encoder5])
        encoder5 = LeakyReLU(alpha=0.2, name='encoder5')(shortcut5)  # 1x1

        decoder1 = Conv3D(self.gKernels*8, 1, activation='relu', padding='valid', kernel_initializer='he_normal')(encoder5)
        decoder1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder1)
        decoder1 = Conv3D(self.gKernels*8, 1, padding='valid', kernel_initializer='he_normal')(decoder1)
        decoder1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder1)
        shortcut6 = Add()([encoder5, decoder1])
        decoder1 = Activation('relu')(shortcut6)
        decoder1 = UpSampling3D(size=2, name='decoder1')(decoder1)  # 2x2
        decoder1 = Dropout(0.5)(decoder1)
        connection1 = Concatenate(axis=-1, name='connection1')([decoder1, encoder4])

        decoder2 = Conv3D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(connection1)
        decoder2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder2)
        decoder2 = Conv3D(self.gKernels*8, self.gKernelSize, padding='same', kernel_initializer='he_normal')(decoder2)
        decoder2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder2)
        feature_mapping7 = Conv3D(self.gKernels*8, 1, padding='same', kernel_initializer=Constant(1/int(connection1.shape[-1])))
        feature_mapping7.trainable = False
        shortcut7 = feature_mapping7(connection1)
        shortcut7 = Add()([shortcut7, decoder2])
        decoder2 = Activation('relu')(shortcut7)
        decoder2 = UpSampling3D(size=2, name='decoder2')(decoder2)  # 4x4
        decoder2 = Dropout(0.5)(decoder2)
        connection2 = Concatenate(axis=-1, name='connection2')([decoder2, encoder3])

        decoder3 = Conv3D(self.gKernels*4, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(connection2)
        decoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder3)
        decoder3 = Conv3D(self.gKernels*4, self.gKernelSize, padding='same', kernel_initializer='he_normal')(decoder3)
        decoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder3)
        feature_mapping8 = Conv3D(self.gKernels*4, 1, padding='same', kernel_initializer=Constant(1/int(connection2.shape[-1])))
        feature_mapping8.trainable = False
        shortcut8 = feature_mapping8(connection2)
        shortcut8 = Add()([shortcut8, decoder3])
        decoder3 = Activation('relu')(shortcut8)
        decoder3 = UpSampling3D(size=2, name='decoder3')(decoder3)  # 8x8
        decoder3 = Dropout(0.5)(decoder3)
        connection3 = Concatenate(axis=-1, name='connection3')([decoder3, encoder2])

        decoder4 = Conv3D(self.gKernels*2, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(connection3)
        decoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder4)
        decoder4 = Conv3D(self.gKernels*2, self.gKernelSize, padding='same', kernel_initializer='he_normal')(decoder4)
        decoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder4)
        feature_mapping9 = Conv3D(self.gKernels*2, 1, padding='same', kernel_initializer=Constant(1/int(connection3.shape[-1])))
        feature_mapping9.trainable = False
        shortcut9 = feature_mapping9(connection3)
        shortcut9 = Add()([shortcut9, decoder4])
        decoder4 = Activation('relu')(shortcut9)
        decoder4 = UpSampling3D(size=2, name='decoder4')(decoder4)  # 16x16
        # decoder4 = Dropout(0.5)(decoder4)
        connection4 = Concatenate(axis=-1, name='connection4')([decoder4, encoder1])

        decoder5 = Conv3D(self.gKernels, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(connection4)
        decoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder5)
        decoder5 = Conv3D(self.gKernels, self.gKernelSize, padding='same', kernel_initializer='he_normal')(decoder5)
        decoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder5)
        feature_mapping10 = Conv3D(self.gKernels, 1, padding='same', kernel_initializer=Constant(1/int(connection4.shape[-1])))
        feature_mapping10.trainable = False
        shortcut10 = feature_mapping10(connection4)
        shortcut10 = Add()([shortcut10, decoder5])
        decoder5 = Activation('relu')(shortcut10)
        decoder5 = UpSampling3D(size=2, name='decoder5')(decoder5)  # 32x32
        # decoder5 = Dropout(0.5)(decoder5)

        decoder6 = Conv3D(int(self.gKernels), self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(decoder5)
        decoder6 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder6)
        decoder6 = Conv3D(int(self.gKernels/2), self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(decoder6)
        # shortcut11 = Add()([decoder5, decoder6])
        # decoder6 = relu(shortcut11)
        decoder6 = Conv3D(1, self.gKernelSize, activation=self.activationG, padding='same', kernel_initializer='he_normal')(decoder6)

        model = Model(inputs=inputs, outputs=decoder6, name='uNet3d5_2')
        return model

    def convLstm(self):
        inputs = Input((self.temporalDepth, self.imgRows, self.imgCols, self.channels))  # 256x256

        encoder1 = TimeDistributed(Conv2D(self.gKernels, self.gKernelSize, strides=2, padding='same', activation='linear', kernel_initializer='he_normal', name='encoder1'),
                                   input_shape=(self.temporalDepth, self.imgRows, self.imgCols, self.channels))(inputs)  # output 128x128
        # activation of the 1st layer?

        encoder2 = TimeDistributed(Conv2D(self.gKernels*2, self.gKernelSize, strides=2, padding='same', activation='linear', kernel_initializer='he_normal'),
                                   input_shape=(self.temporalDepth, self.imgRows, self.imgCols, self.channels))(encoder1)  # output 64x64
        encoder2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder2)
        encoder2 = LeakyReLU(alpha=0.2, name='encoder2')(encoder2)

        encoder3 = TimeDistributed(Conv2D(self.gKernels*4, self.gKernelSize, strides=2, padding='same', activation='linear', kernel_initializer='he_normal'),
                                   input_shape=(self.temporalDepth, self.imgRows, self.imgCols, self.channels))(encoder2)  # output 32x32
        encoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder3)
        encoder3 = LeakyReLU(alpha=0.2, name='encoder3')(encoder3)

        encoder4 = TimeDistributed(Conv2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', activation='linear', kernel_initializer='he_normal'),
                                   input_shape=(self.temporalDepth, self.imgRows, self.imgCols, self.channels))(encoder3)  # output 16x16
        encoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder4)
        encoder4 = LeakyReLU(alpha=0.2, name='encoder4')(encoder4)

        encoder5 = TimeDistributed(Conv2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', activation='linear', kernel_initializer='he_normal'),
                                   input_shape=(self.temporalDepth, self.imgRows, self.imgCols, self.channels))(encoder4)  # output 8x8
        encoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder5)
        encoder5 = LeakyReLU(alpha=0.2, name='encoder5')(encoder5)

        encoder6 = TimeDistributed(Conv2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', activation='linear', kernel_initializer='he_normal'),
                                   input_shape=(self.temporalDepth, self.imgRows, self.imgCols, self.channels))(encoder5)  # output 4x4
        encoder6 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder6)
        encoder6 = LeakyReLU(alpha=0.2, name='encoder6')(encoder6)

        encoder7 = TimeDistributed(Conv2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', activation='linear', kernel_initializer='he_normal'),
                                   input_shape=(self.temporalDepth, self.imgRows, self.imgCols, self.channels))(encoder6)  # output 2x2
        encoder7 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder7)
        encoder7 = LeakyReLU(alpha=0.2, name='encoder7')(encoder7)

        encoder8 = TimeDistributed(Conv2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', activation='linear', kernel_initializer='he_normal'),
                                   input_shape=(self.temporalDepth, self.imgRows, self.imgCols, self.channels))(encoder7)  # output 1x1
        encoder8 = LeakyReLU(alpha=0.2, name='encoder8')(encoder8)

        decoder1 = ConvLSTM2D(self.gKernels*8, 1, activation='relu', padding='valid', kernel_initializer='he_normal', return_sequences=True)(encoder8)
        decoder1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder1)
        decoder1 = UpSampling3D(size=(1, 2, 2))(decoder1)
        decoder1 = Dropout(0.5, name='decoder1')(decoder1)
        connection1 = Concatenate(axis=-1, name='connection1')([decoder1, encoder7])

        decoder2 = ConvLSTM2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True)(connection1)
        decoder2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder2)
        decoder2 = UpSampling3D(size=(1, 2, 2))(decoder2)
        decoder2 = Dropout(0.5, name='decoder2')(decoder2)
        connection2 = Concatenate(axis=-1, name='connection2')([decoder2, encoder6])

        decoder3 = ConvLSTM2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True)(connection2)
        decoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder3)
        decoder3 = UpSampling3D(size=(1, 2, 2))(decoder3)
        decoder3 = Dropout(0.5, name='decoder3')(decoder3)
        connection3 = Concatenate(axis=-1, name='connection3')([decoder3, encoder5])

        decoder4 = ConvLSTM2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=False)(connection3)
        decoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False, name='decoder4')(decoder4)
        decoder4 = UpSampling2D(size=(2, 2))(decoder4)
        encoder4Last = Lambda(sliceSqueeze, output_shape=encoder4.get_shape().as_list()[2: 5], arguments={'begin': self.temporalDepth-1, 'length': 1, 'layer': 1})(encoder4)
        connection4 = Concatenate(axis=-1, name='connection4')([decoder4, encoder4Last])

        decoder5 = Conv2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(connection4)
        decoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False, name='decoder5')(decoder5)
        decoder5 = UpSampling2D(size=(2, 2))(decoder5)
        encoder3Last = Lambda(sliceSqueeze, output_shape=encoder3.get_shape().as_list()[2: 5], arguments={'begin': self.temporalDepth-1, 'length': 1, 'layer': 1})(encoder3)
        connection5 = Concatenate(axis=-1, name='connection5')([decoder5, encoder3Last])

        decoder6 = Conv2D(self.gKernels*4, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(connection5)
        decoder6 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False, name='decoder6')(decoder6)
        decoder6 = UpSampling2D(size=(2, 2))(decoder6)
        encoder2Last = Lambda(sliceSqueeze, output_shape=encoder2.get_shape().as_list()[2: 5], arguments={'begin': self.temporalDepth-1, 'length': 1, 'layer': 1})(encoder2)
        connection6 = Concatenate(axis=-1, name='connection6')([decoder6, encoder2Last])

        decoder7 = Conv2D(self.gKernels*2, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(connection6)
        decoder7 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False, name='decoder7')(decoder7)
        decoder7 = UpSampling2D(size=(2, 2))(decoder7)
        encoder1Last = Lambda(sliceSqueeze, output_shape=encoder1.get_shape().as_list()[2: 5], arguments={'begin': self.temporalDepth-1, 'length': 1, 'layer': 1})(encoder1)
        connection7 = Concatenate(axis=-1, name='connection7')([decoder7, encoder1Last])

        decoder8 = Conv2D(self.gKernels, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal', name='decoder8')(connection7)
        decoder8 = UpSampling2D(size=(2, 2))(decoder8)
        decoder9 = Conv2D(1, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(decoder8)
        decoder10 = Conv2D(1, kernel_size=1, activation=self.activationG, padding='valid', kernel_initializer='he_normal')(decoder9)
        model = Model(inputs=inputs, outputs=decoder10, name='convLstm')
        return model

    def convLstm5(self):
        inputs = Input((self.temporalDepth, self.imgRows, self.imgCols, self.channels))  # 32x32

        encoder1 = TimeDistributed(Conv2D(self.gKernels, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal'))(inputs)  # output 16x16
        encoder1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder1)
        encoder1 = LeakyReLU(0.2, name='encoder1')(encoder1)

        encoder2 = TimeDistributed(Conv2D(self.gKernels*2, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal'))(encoder1)  # output 8x8
        encoder2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder2)
        encoder2 = LeakyReLU(0.2, name='encoder2')(encoder2)

        encoder3 = TimeDistributed(Conv2D(self.gKernels*4, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal'))(encoder2)  # output 4x4
        encoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder3)
        encoder3 = LeakyReLU(0.2, name='encoder3')(encoder3)

        encoder4 = TimeDistributed(Conv2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal'))(encoder3)  # output 2x2
        encoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder4)
        encoder4 = LeakyReLU(0.2, name='encoder4')(encoder4)

        encoder5 = TimeDistributed(Conv2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal'))(encoder4)  # output 1x1
        # encoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder5)
        encoder5 = LeakyReLU(0.2, name='encoder5')(encoder5)

        decoder1 = TimeDistributed(Conv2D(self.gKernels*8, 1, padding='valid', activation='relu', kernel_initializer='he_normal'))(encoder5)
        # decoder1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder1)
        decoder1 = UpSampling3D(size=(1, 2, 2), name='decoder1')(decoder1)  # output 2x2
        decoder1 = Dropout(0.5)(decoder1)
        connection1 = Concatenate(axis=-1, name='connection1')([decoder1, encoder4])

        decoder2 = TimeDistributed(Conv2D(self.gKernels*8, self.gKernelSize, padding='same', activation='relu', kernel_initializer='he_normal'))(connection1)
        decoder2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder2)
        decoder2 = UpSampling3D(size=(1, 2, 2), name='decoder2')(decoder2)  # output 4x4
        decoder2 = Dropout(0.5)(decoder2)
        connection2 = Concatenate(axis=-1, name='connection2')([decoder2, encoder3])

        decoder3 = TimeDistributed(Conv2D(self.gKernels*4, self.gKernelSize, padding='same', activation='relu', kernel_initializer='he_normal'))(connection2)
        decoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder3)
        decoder3 = UpSampling3D(size=(1, 2, 2), name='decoder3')(decoder3)  # output 8x8
        decoder3 = Dropout(0.5)(decoder3)
        connection3 = Concatenate(axis=-1, name='connection3')([decoder3, encoder2])

        decoder4 = TimeDistributed(Conv2D(self.gKernels*2, self.gKernelSize, padding='same', activation='relu', kernel_initializer='he_normal'))(connection3)
        decoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder4)
        decoder4 = UpSampling3D(size=(1, 2, 2), name='decoder4')(decoder4)  # output 16x16
        # decoder4 = Dropout(0.5)(decoder4)
        connection4 = Concatenate(axis=-1, name='connection4')([decoder4, encoder1])

        decoder5 = TimeDistributed(Conv2D(self.gKernels, self.gKernelSize, padding='same', activation='relu', kernel_initializer='he_normal'))(connection4)
        decoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder5)
        decoder5 = UpSampling3D(size=(1, 2, 2), name='decoder5')(decoder5)  # output 32x32
        # decoder5 = Dropout(0.5)(decoder5)

        decoder6 = ConvLSTM2D(self.gKernels, self.gKernelSize, padding='same', activation='relu', kernel_initializer='he_normal', return_sequences=False)(decoder5)
        decoder6 = Conv2D(int(self.gKernels/2), self.gKernelSize, padding='same', activation='relu', kernel_initializer='he_normal')(decoder6)
        decoder6 = Conv2D(int(self.gKernels/4), self.gKernelSize, padding='same', activation='relu', kernel_initializer='he_normal')(decoder6)
        # decoder6 = ConvLSTM2D(int(self.gKernels/2), self.gKernelSize, padding='same', activation='relu', kernel_initializer='he_normal', return_sequences=True)(decoder6)
        # decoder6 = ConvLSTM2D(int(self.gKernels/4), self.gKernelSize, padding='same', activation='relu', kernel_initializer='he_normal', return_sequences=False)(decoder6)

        decoder7 = Conv2D(1, self.gKernelSize, padding='same', activation='relu', kernel_initializer='he_normal')(decoder6)
        decoder7 = Conv2D(1, 1, activation=self.activationG, padding='valid', kernel_initializer='he_normal')(decoder7)

        model = Model(inputs=inputs, outputs=decoder7, name='convLstm5')
        return model

    def seqConv5(self):
        inputs = Input((self.temporalDepth, self.imgRows, self.imgCols, self.channels))  # 32x32

        encoder1 = TimeDistributed(Conv2D(self.gKernels, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal'))(inputs)  # output 16x16
        encoder1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder1)
        encoder1 = LeakyReLU(0.2, name='encoder1')(encoder1)

        encoder2 = TimeDistributed(Conv2D(self.gKernels*2, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal'))(encoder1)  # output 8x8
        encoder2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder2)
        encoder2 = LeakyReLU(0.2, name='encoder2')(encoder2)

        encoder3 = TimeDistributed(Conv2D(self.gKernels*4, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal'))(encoder2)  # output 4x4
        encoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder3)
        encoder3 = LeakyReLU(0.2, name='encoder3')(encoder3)

        encoder4 = TimeDistributed(Conv2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal'))(encoder3)  # output 2x2
        encoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder4)
        encoder4 = LeakyReLU(0.2, name='encoder4')(encoder4)

        encoder5 = TimeDistributed(Conv2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', kernel_initializer='he_normal'))(encoder4)  # output 1x1
        # encoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder5)
        encoder5 = LeakyReLU(0.2, name='encoder5')(encoder5)

        decoder1 = TimeDistributed(Conv2D(self.gKernels*8, 1, padding='valid', activation='relu', kernel_initializer='he_normal'))(encoder5)
        # decoder1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder1)
        decoder1 = UpSampling3D(size=(1, 2, 2), name='decoder1')(decoder1)  # output 2x2
        decoder1 = Dropout(0.5)(decoder1)
        connection1 = Concatenate(axis=-1, name='connection1')([decoder1, encoder4])

        decoder2 = TimeDistributed(Conv2D(self.gKernels*8, self.gKernelSize, padding='same', activation='relu', kernel_initializer='he_normal'))(connection1)
        decoder2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder2)
        decoder2 = UpSampling3D(size=(1, 2, 2), name='decoder2')(decoder2)  # output 4x4
        decoder2 = Dropout(0.5)(decoder2)
        connection2 = Concatenate(axis=-1, name='connection2')([decoder2, encoder3])

        decoder3 = TimeDistributed(Conv2D(self.gKernels*4, self.gKernelSize, padding='same', activation='relu', kernel_initializer='he_normal'))(connection2)
        decoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder3)
        decoder3 = UpSampling3D(size=(1, 2, 2), name='decoder3')(decoder3)  # output 8x8
        decoder3 = Dropout(0.5)(decoder3)
        connection3 = Concatenate(axis=-1, name='connection3')([decoder3, encoder2])

        decoder4 = TimeDistributed(Conv2D(self.gKernels*2, self.gKernelSize, padding='same', activation='relu', kernel_initializer='he_normal'))(connection3)
        decoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder4)
        decoder4 = UpSampling3D(size=(1, 2, 2), name='decoder4')(decoder4)  # output 16x16
        # decoder4 = Dropout(0.5)(decoder4)
        connection4 = Concatenate(axis=-1, name='connection4')([decoder4, encoder1])

        decoder5 = TimeDistributed(Conv2D(self.gKernels, self.gKernelSize, padding='same', activation='relu', kernel_initializer='he_normal'))(connection4)
        decoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder5)
        decoder5 = UpSampling3D(size=(1, 2, 2), name='decoder5')(decoder5)  # output 32x32
        # decoder5 = Dropout(0.5)(decoder5)

        # decoder6 = Bidirectional(ConvLSTM2D(self.gKernels, self.gKernelSize, padding='same', activation='relu', kernel_initializer='he_normal', return_sequences=True))/
        #                         (decoder5)
        decoder6 = ConvLSTM2D(self.gKernels, self.gKernelSize, padding='same', activation='relu', kernel_initializer='he_normal', return_sequences=True,
                              recurrent_dropout=0.5, recurrent_regularizer=keras.regularizers.l1(0.01))(decoder5)
        # outputs of two directions are concatenated
        # decoder6 = TimeDistributed(Conv2D(int(self.gKernels/2), self.gKernelSize, padding='same', activation='relu', kernel_initializer='he_normal'))(decoder6)
        decoder6 = TimeDistributed(Conv2D(int(self.gKernels/4), self.gKernelSize, padding='same', activation='relu', kernel_initializer='he_normal'))(decoder6)
        
        decoder7 = TimeDistributed(Conv2D(1, self.gKernelSize, padding='same', activation='relu', kernel_initializer='he_normal'))(decoder6)
        decoder7 = TimeDistributed(Conv2D(1, 1, activation=self.activationG, padding='valid', kernel_initializer='he_normal'))(decoder7)
        model = Model(inputs=inputs, outputs=decoder7, name='seqConv5')
        return model

    def straight3(self):
        inputs = Input((self.imgRows, self.imgCols, self.channels))
        conv1 = Conv2D(self.dKernels, 3, padding='same', kernel_initializer='he_normal')(inputs)
        pool1 = MaxPooling2D((4, 4))(conv1)
        conv1 = LeakyReLU(alpha=0.2)(pool1)
        conv2 = Conv2D(self.dKernels*2, 3, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(conv2)
        pool2 = MaxPooling2D((4, 4))(conv2)
        conv2 = LeakyReLU(alpha=0.2)(pool2)
        conv3 = Conv2D(self.dKernels*4, 3, padding='same', kernel_initializer='he_normal')(conv2)
        conv3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)
        conv3 = LeakyReLU(alpha=0.2)(pool3)
        flatten4 = Flatten()(conv3)
        dense4 = Dense(self.dKernels*16)(flatten4)
        dense4 = LeakyReLU(alpha=0.2)(dense4)
        drop4 = Dropout(0.5)(dense4)
        dense5 = Dense(self.dKernels*16)(drop4)
        dense5 = LeakyReLU(alpha=0.2)(dense5)
        drop5 = Dropout(0.5)(dense5)
        probability = Dense(1, activation='linear')(drop5)
        model = Model(inputs=inputs, outputs=probability, name='straight3')
        return model

    def vgg16(self):
        inputs = Input((self.imgRows, self.imgCols, self.channels*2))
        conv1 = Conv2D(self.dKernels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = Conv2D(self.dKernels, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)
        conv2 = Conv2D(self.dKernels*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = Conv2D(self.dKernels*2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)
        conv3 = Conv2D(self.dKernels*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = Conv2D(self.dKernels*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = Conv2D(self.dKernels*4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)
        conv4 = Conv2D(self.dKernels*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(self.dKernels*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = Conv2D(self.dKernels*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        pool4 = MaxPooling2D((2, 2))(conv4)
        conv5 = Conv2D(self.dKernels*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(self.dKernels*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = Conv2D(self.dKernels*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        pool5 = MaxPooling2D((2, 2))(conv5)
        flatten6 = Flatten()(pool5)
        dense6 = Dense(self.dKernels*64, activation='relu')(flatten6)
        drop6 = Dropout(0.5)(dense6)
        dense7 = Dense(self.dKernels*64, activation='relu')(drop6)
        drop7 = Dropout(0.5)(dense7)
        probability = Dense(1, activation='linear')(drop7)
        model = Model(inputs=inputs, outputs=probability, name='VGG16')
        return model


class Gan(object):
    def __init__(self, imgRows=256, imgCols=256, channels=1, netDName=None, netGName=None, temporalDepth=None, gKernels=64, dKernels=64, gKernelSize=4,
                 activationG='relu', lossFuncG='mae', gradientPenaltyWeight=10, lossDWeight=0.01, learningRateG=1e-4, learningRateD=1e-4, beta1=0.9, beta2=0.999,
                 batchSize=10):
        self.imgRows = imgRows
        self.imgCols = imgCols
        self.channels = channels
        self.temporalDepth = temporalDepth
        self.netGName = netGName
        self.netDName = netDName
        self.lossFuncG = lossFuncG
        self.learningRateG = learningRateG
        self.learningRateD = learningRateD
        self.activationG = activationG
        self.batchSize = batchSize
        self.network = Networks(imgSize=(self.imgRows, self.imgCols), channels=self.channels, gKernels=gKernels, dKernels=dKernels, gKernelSize=gKernelSize,
                                temporalDepth=self.temporalDepth, activationG=self.activationG)
        if self.netDName == 'straight3':
            self.netD = self.network.straight3()
        elif self.netDName == 'VGG16':
            self.netD = self.network.vgg16()
        # add gradient penalty on D
        real = Input((self.imgRows, self.imgCols, self.channels))
        if self.netGName == 'uNet':
            self.netG = self.network.uNet()
            self.netG.trainable = False
            inputsGForGradient = Input((self.imgRows, self.imgCols, self.channels))
            outputsGForGradient = self.netG(inputsGForGradient)
            realPair = Concatenate(axis=-1)([inputsGForGradient, real])
            fakePair = Concatenate(axis=-1)([inputsGForGradient, outputsGForGradient])
        elif self.netGName == 'uNet3d':
            self.netG = self.network.uNet3d()
            inputsGForGradient = Input((self.temporalDepth, self.imgRows, self.imgCols, self.channels))
            self.netG.trainable = False
            inputsGForGradient = Input((self.temporalDepth, self.imgRows, self.imgCols, self.channels))
            outputsGForGradient = self.netG(inputsGForGradient)
        outputsDOnReal = self.netD(realPair)
        outputsDOnFake = self.netD(fakePair)
        averagedRealFake = Lambda(self.randomlyWeightedAverage, output_shape=(self.imgRows, self.imgCols, self.channels*2))([realPair, fakePair])
        outputsDOnAverage = self.netD(averagedRealFake)
        gradientPenaltyLoss = functools.partial(calculateGradientPenaltyLoss, samples=averagedRealFake, weight=gradientPenaltyWeight)
        gradientPenaltyLoss.__name__ = 'gradientPenalty'
        self.penalizedNetD = Model(inputs=[inputsGForGradient, real], outputs=[outputsDOnReal, outputsDOnFake, outputsDOnAverage])
        wassersteinDistance.__name__ = 'wassertein'
        self.penalizedNetD.compile(optimizer=Adam(lr=self.learningRateD, beta_1=beta1, beta_2=beta2), loss=[wassersteinDistance, wassersteinDistance,
                                                                                                            gradientPenaltyLoss])
        print(self.penalizedNetD.metrics_names)
        # build adversarial network
        self.netG.trainable = True
        self.netD.trainable = False
        if self.netGName == 'uNet':
            inputsA = Input((self.imgRows, self.imgCols, self.channels))
            outputsG = self.netG(inputsA)
            inputsD = Concatenate(axis=-1)([inputsA, outputsG])
        elif self.netGName == 'uNet3d':
            self.netG = self.network.uNet3d()
            inputsA = Input((self.temporalDepth, self.imgRows, self.imgCols, self.channels))
            outputsG = self.netG(inputsA)
            temporalMid = math.floor(self.temporalDepth/2.0 + 0.1)
            middleLayerOfInputs = Lambda(slice3d, output_shape=(1, self.imgRows, self.imgCols, self.channels), arguments={'begin': temporalMid, 'length': 1})(inputsA)
            middleLayerOfInputs = Lambda(squeeze, output_shape=(self.imgRows, self.imgCols, self.channels), arguments={'layer': 1})(middleLayerOfInputs)
            inputsD = Concatenate(axis=-1)([middleLayerOfInputs, outputsG])
        outputsD = self.netD(inputsD)
        self.netA = Model(inputs=inputsA, outputs=[outputsG, outputsD], name='netA')
        self.netA.compile(optimizer=Adam(lr=self.learningRateG, beta_1=beta1, beta_2=beta2), loss=[lossFuncG, wassersteinDistance], loss_weights=[1, lossDWeight])
        print(self.netA.metrics_names)
        self.netG.compile(optimizer=Adam(lr=self.learningRateG), loss=self.lossFuncG, metrics=[self.lossFuncG])
        self.netG.summary()
        self.netD.summary()
        self.penalizedNetD.summary()
        self.netA.summary()

    def randomlyWeightedAverage(self, src):
        weights = K.random_uniform((self.batchSize, 1, 1, 1), minval=0., maxval=1.)
        dst = (weights*src[0]) + ((1-weights)*src[1])
        return dst


def squeeze(src, layer):
    dst = tf.squeeze(src, [layer])
    return dst


def slice3d(src, begin, length):
    # srcShape = src.shape.as_list()
    # middleLayer = math.floor(srcShape[1]/2.0 + 0.1)
    dst = tf.slice(src, [0, begin, 0, 0, 0], [-1, length, -1, -1, -1])
    return dst


def sliceSqueeze(src, begin, length, layer):
    sliced = slice3d(src, begin, length)
    dst = squeeze(sliced, layer)
    return dst


def pad3d(src, gKernelSize):
    borderSize = math.floor(gKernelSize/2 + 0.1)
    paddingSize = tf.constant([[0, 0], [1, 0], [2, borderSize], [3, borderSize], [4, 0]])
    dst = tf.pad(src, paddingSize, 'CONSTANT')
    return dst


def wassersteinDistance(src1, src2):
    dst = K.mean(src1*src2)
    return dst


def calculateGradientPenaltyLoss(true, prediction, samples, weight):
    gradients = K.gradients(prediction, samples)[0]
    gradientsSqr = K.square(gradients)
    gradientsSqrSum = K.sum(gradientsSqr, axis=np.arange(1, len(gradientsSqr.shape)))
    gradientsL2Norm = K.sqrt(gradientsSqrSum)
    penalty = weight*K.square(1-gradientsL2Norm)
    averagePenalty = K.mean(penalty, axis=0)
    return averagePenalty
