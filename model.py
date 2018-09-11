#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
import os
import glob
import math
import functools
import datetime
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import *
from keras.layers import Input, Concatenate, Conv2D, UpSampling2D, Dropout, BatchNormalization, Flatten, Dense, MaxPooling2D
from keras.layers import Conv3D, UpSampling3D, MaxPooling3D, Reshape, Permute, Lambda, ZeroPadding3D
from keras.layers import ConvLSTM2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler
from keras import backend
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical

import dataProc

class networks(object):
    def __init__(self, imgRows=256, imgCols=256, channels=1, gKernels=64, dKernels=64, gKernelSize=4, temporalDepth=None, activationG=None, **kwargs):
        self.imgRows = imgRows
        self.imgCols = imgCols
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

        decoder1 = Conv2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(encoder8))
        decoder1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder1)
        decoder1 = Dropout(0.5, name='decoder1')(decoder1)
        connection1 = Concatenate(axis=-1, name='connection1')([decoder1, encoder7])
        decoder2 = Conv2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(connection1))
        decoder2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder2)
        decoder2 = Dropout(0.5, name='decoder2')(decoder2)
        connection2 = Concatenate(axis=-1, name='connection2')([decoder2, encoder6])
        decoder3 = Conv2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(connection2))
        decoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder3)
        decoder3 = Dropout(0.5, name='decoder3')(decoder3)
        connection3 = Concatenate(axis=-1, name='connection3')([decoder3, encoder5])
        decoder4 = Conv2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(connection3))
        decoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False, name='decoder4')(decoder4)
        connection4 = Concatenate(axis=-1, name='connection4')([decoder4, encoder4])
        decoder5 = Conv2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(connection4))
        decoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False, name='decoder5')(decoder5)
        connection5 = Concatenate(axis=-1, name='connection5')([decoder5, encoder3])
        decoder6 = Conv2D(self.gKernels*4, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(connection5))
        decoder6 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False, name='decoder6')(decoder6)
        connection6 = Concatenate(axis=-1, name='connection6')([decoder6, encoder2])
        decoder7 = Conv2D(self.gKernels*2, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(connection6))
        decoder7 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False, name='decoder7')(decoder7)
        connection7 = Concatenate(axis=-1, name='connection7')([decoder7, encoder1])
        decoder8 = Conv2D(self.gKernels, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal', name='decoder8')(UpSampling2D(size = (2,2))(connection7))
        decoder9 = Conv2D(1, 1, activation=self.activationG, name='decoder9')(decoder8)
        model = Model(inputs=inputs, outputs=decoder9, name='uNet')
        return model

    def uNet3D(self):
        inputs = Input((self.temporalDepth, self.imgRows, self.imgCols, self.channels))
        encoder1 = Conv3D(filters=self.gKernels, kernel_size=(self.temporalDepth, 4, 4), strides=(1, 2, 2), \
        padding='same', kernel_initializer='he_normal')(inputs)
        encoder2 = Conv3D(filters=self.gKernels*2, kernel_size=(self.temporalDepth, 4, 4), strides=(1, 2, 2), \
        padding='same', kernel_initializer='he_normal')(encoder1)
        encoder2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder2)
        encoder2 = LeakyReLU(alpha=0.2)(encoder2)
        encoder3 = Conv3D(filters=self.gKernels*4, kernel_size=(self.temporalDepth, 4, 4), strides=(1, 2, 2), \
        padding='same', kernel_initializer='he_normal')(encoder2)
        encoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder3)
        encoder3 = LeakyReLU(alpha=0.2)(encoder3)
        encoder4 = Conv3D(filters=self.gKernels*8, kernel_size=(self.temporalDepth, 4, 4), strides=(1, 2, 2), \
        padding='same', kernel_initializer='he_normal')(encoder3)
        encoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder4)
        encoder4 = LeakyReLU(alpha=0.2)(encoder4)
        encoder5 = Conv3D(filters=self.gKernels*8, kernel_size=(self.temporalDepth, 4, 4), strides=(1, 2, 2), \
        padding='same', kernel_initializer='he_normal')(encoder4)
        encoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder5)
        encoder5 = LeakyReLU(alpha=0.2)(encoder5)
        encoder6 = Conv3D(filters=self.gKernels*8, kernel_size=(self.temporalDepth, 4, 4), strides=(1, 2, 2), \
        padding='same', kernel_initializer='he_normal')(encoder5)
        encoder6 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder6)
        encoder6 = LeakyReLU(alpha=0.2)(encoder6)
        encoder7 = Conv3D(filters=self.gKernels*8, kernel_size=(self.temporalDepth, 4, 4), strides=(1, 2, 2), \
        padding='same', kernel_initializer='he_normal')(encoder6)
        encoder7 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder7)
        encoder7 = LeakyReLU(alpha=0.2)(encoder7)
        encoder8 = Conv3D(filters=self.gKernels*8, kernel_size=(self.temporalDepth, 4, 4), strides=(1, 2, 2), \
        padding='same', kernel_initializer='he_normal')(encoder7)
        encoder8 = LeakyReLU(alpha=0.2)(encoder8)
        decoder1 = Conv3D(self.gKernels*8, kernel_size=(self.temporalDepth, 4, 4), activation='relu', \
        padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(1,2,2))(encoder8))
        decoder1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder1)
        decoder1 = Dropout(0.5)(decoder1)
        connection1 = Concatenate(axis=-1)([decoder1, encoder7])
        decoder2 = Conv3D(self.gKernels*8, kernel_size=(self.temporalDepth, 4, 4), activation='relu', \
        padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(1,2,2))(connection1))
        decoder2=BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder2)
        decoder2 = Dropout(0.5)(decoder2)
        connection2 = Concatenate(axis=-1)([decoder2, encoder6])
        decoder3 = Conv3D(self.gKernels*8, kernel_size=(self.temporalDepth, 4, 4), activation='relu', \
        padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(1,2,2))(connection2))
        decoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder3)
        decoder3 = Dropout(0.5)(decoder3)
        connection3 = Concatenate(axis=-1)([decoder3, encoder5])
        decoder4 = Conv3D(self.gKernels*8, kernel_size=(self.temporalDepth, 4, 4), activation='relu', \
        padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(1,2,2))(connection3))
        decoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder4)
        connection4 = Concatenate(axis=-1)([decoder4, encoder4])
        decoder5 = Conv3D(self.gKernels*8, kernel_size=(self.temporalDepth, 4, 4), activation='relu', \
        padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(1,2,2))(connection4))
        decoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder5)
        connection5 = Concatenate(axis=-1)([decoder5, encoder3])
        decoder6 = Conv3D(self.gKernels*4, kernel_size=(self.temporalDepth, 4, 4), activation='relu', \
        padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(1,2,2))(connection5))
        decoder6 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder6)
        connection6 = Concatenate(axis=-1)([decoder6, encoder2])
        decoder7 = Conv3D(self.gKernels*2, kernel_size=(self.temporalDepth, 4, 4), activation='relu', \
        padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(1,2,2))(connection6))
        decoder7 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder7)
        connection7 = Concatenate(axis=-1)([decoder7, encoder1])
        decoder8 = Conv3D(self.gKernels, kernel_size=(self.temporalDepth, 4, 4), activation='relu', \
        padding='same', kernel_initializer='he_normal')(UpSampling3D(size=(1,2,2))(connection7))
        decoder9 = Conv3D(1, kernel_size=(self.temporalDepth, 4, 4), activation='relu', padding='same', kernel_initializer='he_normal')(decoder8)
        decoder10 = Conv3D(1, kernel_size=(self.temporalDepth, 1, 1), activation=self.activationG, padding='valid', kernel_initializer='he_normal')(decoder9)
        squeezed10 = Lambda(squeeze, output_shape=(self.imgRows, self.imgCols, self.channels), arguments={'layer':1})(decoder10)
        model = Model(inputs=inputs, outputs=squeezed10, name='uNet3D')
        return model
    
    def convLstm(self):
        inputs = Input((self.temporalDepth, self.imgRows, self.imgCols, self.channels))
        encoder1 = ConvLSTM2D(self.gKernels, self.gKernelSize, strides=2, padding='same', activation='linear', kernel_initializer='he_normal', name='encoder1',
        return_sequences=True)(inputs)
        encoder2 = ConvLSTM2D(self.gKernels*2, self.gKernelSize, strides=2, padding='same', activation='linear', kernel_initializer='he_normal', return_sequences=True)(encoder1)
        encoder2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder2)
        encoder2 = LeakyReLU(alpha=0.2, name='encoder2')(encoder2)
        encoder3 = ConvLSTM2D(self.gKernels*4, self.gKernelSize, strides=2, padding='same', activation='linear', kernel_initializer='he_normal', return_sequences=True)(encoder2)
        encoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder3)
        encoder3 = LeakyReLU(alpha=0.2, name='encoder3')(encoder3)
        encoder4 = ConvLSTM2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', activation='linear', kernel_initializer='he_normal', return_sequences=True)(encoder3)
        encoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder4)
        encoder4 = LeakyReLU(alpha=0.2, name='encoder4')(encoder4)
        encoder5 = ConvLSTM2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', activation='linear', kernel_initializer='he_normal', return_sequences=True)(encoder4)
        encoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder5)
        encoder5 = LeakyReLU(alpha=0.2, name='encoder5')(encoder5)
        encoder6 = ConvLSTM2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', activation='linear', kernel_initializer='he_normal', return_sequences=True)(encoder5)
        encoder6 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder6)
        encoder6 = LeakyReLU(alpha=0.2, name='encoder6')(encoder6)
        encoder7 = ConvLSTM2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', activation='linear', kernel_initializer='he_normal', return_sequences=True)(encoder6)
        encoder7 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(encoder7)
        encoder7 = LeakyReLU(alpha=0.2, name='encoder7')(encoder7)
        encoder8 = ConvLSTM2D(self.gKernels*8, self.gKernelSize, strides=2, padding='same', activation='linear', kernel_initializer='he_normal', return_sequences=True)(encoder7)
        encoder8 = LeakyReLU(alpha=0.2, name='encoder8')(encoder8)

        decoder1 = ConvLSTM2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True) \
        (UpSampling3D(size=(1, 2, 2))(encoder8))
        decoder1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder1)
        decoder1 = Dropout(0.5, name='decoder1')(decoder1)
        connection1 = Concatenate(axis=-1, name='connection1')([decoder1, encoder7])
        decoder2 = ConvLSTM2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True) \
        (UpSampling3D(size=(1, 2, 2))(connection1))
        decoder2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder2)
        decoder2 = Dropout(0.5, name='decoder2')(decoder2)
        connection2 = Concatenate(axis=-1, name='connection2')([decoder2, encoder6])
        decoder3 = ConvLSTM2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True) \
        (UpSampling3D(size=(1, 2, 2))(connection2))
        decoder3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(decoder3)
        decoder3 = Dropout(0.5, name='decoder3')(decoder3)
        connection3 = Concatenate(axis=-1, name='connection3')([decoder3, encoder5])
        decoder4 = ConvLSTM2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True) \
        (UpSampling3D(size=(1, 2, 2))(connection3))
        decoder4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False, name='decoder4')(decoder4)
        connection4 = Concatenate(axis=-1, name='connection4')([decoder4, encoder4])
        decoder5 = ConvLSTM2D(self.gKernels*8, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True) \
        (UpSampling3D(size=(1, 2, 2))(connection4))
        decoder5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False, name='decoder5')(decoder5)
        connection5 = Concatenate(axis=-1, name='connection5')([decoder5, encoder3])
        decoder6 = ConvLSTM2D(self.gKernels*4, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True) \
        (UpSampling3D(size=(1, 2, 2))(connection5))
        decoder6 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False, name='decoder6')(decoder6)
        connection6 = Concatenate(axis=-1, name='connection6')([decoder6, encoder2])
        decoder7 = ConvLSTM2D(self.gKernels*2, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True) \
        (UpSampling3D(size=(1, 2, 2))(connection6))
        decoder7 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False, name='decoder7')(decoder7)
        connection7 = Concatenate(axis=-1, name='connection7')([decoder7, encoder1])
        decoder8 = ConvLSTM2D(self.gKernels, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=True, name='decoder8') \
        (UpSampling3D(size = (1, 2, 2))(connection7))
        decoder9 = ConvLSTM2D(1, self.gKernelSize, activation='relu', padding='same', kernel_initializer='he_normal', return_sequences=False)(decoder8)
        decoder10 = Conv2D(1, kernel_size=(self.temporalDepth, 1, 1), activation=self.activationG, padding='valid', kernel_initializer='he_normal')(decoder9)
        squeezed10 = Lambda(squeeze, output_shape=(self.imgRows, self.imgCols, self.channels), arguments={'layer':1})(decoder10)
        model = Model(inputs=inputs, outputs=squeezed10, name='convLstm')
        return model

    def straight3(self):
        inputs = Input((self.imgRows, self.imgCols, self.channels))
        conv1 = Conv2D(self.dKernels, 3, padding='same', kernel_initializer='he_normal')(inputs)
        pool1 = MaxPooling2D((4, 4))(conv1)
        conv1 = LeakyReLU(alpha=0.2)(pool1)
        conv2 = Conv2D(self.dKernels*2, 3, padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale = False)(conv2)
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
        pool3 = MaxPooling2D((2,2))(conv3)
        conv4 = Conv2D(self.dKernels*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(self.dKernels*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = Conv2D(self.dKernels*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        pool4 = MaxPooling2D((2,2))(conv4)
        conv5 = Conv2D(self.dKernels*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(self.dKernels*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = Conv2D(self.dKernels*8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        pool5 = MaxPooling2D((2,2))(conv5)
        flatten6 = Flatten()(pool5)
        dense6 = Dense(self.dKernels*64, activation='relu')(flatten6)
        drop6 = Dropout(0.5)(dense6)
        dense7 = Dense(self.dKernels*64, activation='relu')(drop6)
        drop7 = Dropout(0.5)(dense7)
        probability = Dense(1, activation='linear')(drop7)
        model = Model(inputs=inputs, outputs=probability, name='VGG16')
        return model

class GAN(object):
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
        self.network = networks(imgRows=self.imgRows, imgCols=self.imgCols, channels=self.channels, gKernels=gKernels, dKernels=dKernels, gKernelSize=gKernelSize,
        temporalDepth=self.temporalDepth,activationG=self.activationG)
        if self.netDName == 'straight3':
            self.netD = self.network.straight3()
        elif self.netDName == 'VGG16':
            self.netD = self.network.vgg16()
        #add gradient penalty on D
        real = Input((self.imgRows, self.imgCols, self.channels))
        if self.netGName == 'uNet':
            self.netG = self.network.uNet()
            self.netG.trainable = False
            inputsGForGradient = Input((self.imgRows, self.imgCols, self.channels))
            outputsGForGradient = self.netG(inputsGForGradient)
            realPair = Concatenate(axis=-1)([inputsGForGradient, real])
            fakePair = Concatenate(axis=-1)([inputsGForGradient, outputsGForGradient])
        elif self.netGName == 'uNet3D':
            self.netG = self.network.uNet3D()
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
        self.penalizedNetD = Model(inputs = [inputsGForGradient, real], outputs = [outputsDOnReal, outputsDOnFake, outputsDOnAverage])
        wassersteinDistance.__name__ = 'wassertein'
        self.penalizedNetD.compile(optimizer=Adam(lr=self.learningRateD, beta_1=beta1, beta_2=beta2), loss=[wassersteinDistance, wassersteinDistance,
        gradientPenaltyLoss])
        print(self.penalizedNetD.metrics_names)
        #build adversarial network
        self.netG.trainable = True
        self.netD.trainable = False
        if self.netGName == 'uNet':
            inputsA = Input((self.imgRows, self.imgCols, self.channels))
            outputsG = self.netG(inputsA)
            inputsD = Concatenate(axis=-1)([inputsA, outputsG])
        elif self.netGName == 'uNet3D':
            self.netG = self.network.uNet3D()
            inputsA = Input((self.temporalDepth, self.imgRows, self.imgCols, self.channels))
            outputsG = self.netG(inputsA)
            middleLayerOfInputs = Lambda(slice, output_shape=(1, self.imgRows, self.imgCols, self.channels))(inputsA)
            middleLayerOfInputs = Lambda(squeeze, output_shape=(self.imgRows, self.imgCols, self.channels), arguments={'layer':1})(middleLayerOfInputs)
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

    # ((number of training samples*validation split ratio)/batch size) should be an integer
    def trainGAN(self, pEcgDir, memDir, modelDir, epochsNum=100, valSplit=0.2, continueTrain=False, pretrainedGPath=None, pretrainedDPath=None,
    approximateData=True, trainingRatio=5, earlyStoppingPatience=10):
        if self.activationG == 'tanh':
            dataRange = [-1., 1.]
        else:
            dataRange = [0., 1.]
        pEcgRaw = dataProc.loadData(srcDir=pEcgDir, resize=1, normalization=1, normalizationRange=dataRange, approximateData=approximateData)
        if self.netGName == 'uNet':
            pEcgSequence = pEcgRaw.reshape((pEcgRaw.shape[0], self.imgRows, self.imgCols, self.channels))
        elif self.netGName == 'uNet3D':
            pEcgSequence = dataProc.create3DData(pEcgRaw, temporalDepth = self.temporalDepth)
            pEcgSequence = pEcgSequence.reshape((pEcgSequence.shape[0], self.temporalDepth, self.imgRows, self.imgCols, self.channels))
        memRaw = dataProc.loadData(srcDir=memDir, resize=1, normalization=1, normalizationRange=dataRange, approximateData=approximateData)
        memSequence = memRaw.reshape((memRaw.shape[0], self.imgRows, self.imgCols, self.channels))
        print('traing data loaded')

        trainingDataLength = math.floor((1-valSplit)*memSequence.shape[0]+0.1)
        lossRecorder = np.ndarray((math.floor(trainingDataLength/self.batchSize + 0.1)*epochsNum, 2), dtype=np.float32)
        lossCounter = 0
        minLossG = np.inf
        weightsDPath = modelDir + 'netD_latest.h5'
        weightsGPath = modelDir + 'netG_latest.h5'
        if continueTrain == True:
            self.netG.load_weights(pretrainedGPath)
            self.netD.load_weights(pretrainedDPath)
        labelReal = np.ones((self.batchSize), dtype=np.float32)
        labelFake = -np.ones((self.batchSize), dtype=np.float32)
        dummyMem = np.zeros((self.batchSize), dtype=np.float32)
        earlyStoppingCounter = 0
        print('begin to train GAN')

        for currentEpoch in range(0, epochsNum):
            beginingTime = datetime.datetime.now()
            [pEcgTrain, pEcgVal, memTrain, memVal] = dataProc.splitTrainAndVal(pEcgSequence, memSequence, valSplit)
            for currentBatch in range(0, trainingDataLength, self.batchSize):
                pEcgLocal = pEcgTrain[currentBatch:currentBatch+self.batchSize, :]
                memLocal = memTrain[currentBatch:currentBatch+self.batchSize, :]
                randomIndexes = np.random.randint(low=0, high=trainingDataLength-self.batchSize-1, size=trainingRatio, dtype=np.int32)
                for i in range(0, trainingRatio):
                    pEcgForD = pEcgTrain[randomIndexes[i]:randomIndexes[i]+self.batchSize]
                    memForD = memTrain[randomIndexes[i]:randomIndexes[i]+self.batchSize]
                    lossD = self.penalizedNetD.train_on_batch([pEcgForD, memForD], [labelReal, labelFake, dummyMem])
                lossA = self.netA.train_on_batch(pEcgLocal, [memLocal, labelReal])
            #validate the model
            lossVal = self.netG.evaluate(x=pEcgVal, y=memVal, batch_size=self.batchSize, verbose=0)
            lossRecorder[lossCounter, 0] = lossA[0]
            lossRecorder[lossCounter, 1] = lossVal[0]
            lossCounter += 1
            if (minLossG > lossVal[0]):                
                self.netG.save_weights(weightsGPath, overwrite=True)
                self.netD.save_weights(weightsDPath, overwrite=True)
                minLossG = lossVal[0]
                earlyStoppingCounter = 0
            earlyStoppingCounter += 1
            displayLoss(lossD, lossA, lossVal, beginingTime, currentEpoch+1)
            if earlyStoppingCounter == earlyStoppingPatience:
                print('early stopping')
                break
        np.save(modelDir + 'loss', lossRecorder)
        print('training completed')

    def diminishElectrodes(self, extraPathList, memPath, modelPath, epochsNum=100, netGOnlyEpochs=0, valSplit=0.2, continueTrain=False, pretrainedGPath=None, pretrainedDPath=None,
    approximateData=True, trainingRatio=5, earlyStoppingPatience=10):
        steps = len(extraPathList)
        if continueTrain == True:
            self.netG.load_weights(pretrainedGPath)
            self.netD.load_weights(pretrainedDPath)
        for i in range(0, steps):
            currentModelPath = modelPath + 'model_%04d/'%i
            if not os.path.exists(currentModelPath):
                os.makedirs(currentModelPath)
            if i == 0:
                isFirstStep = True
                previousGPath = None
                previousDPath = None
            else:
                isFirstStep = False
                previousGPath = modelPath + 'model_%04d/netG_latest.h5'%(i-1)
                previousDPath = modelPath + 'model_%04d/netD_latest.h5'%(i-1)
            self.trainGAN(pEcgDir=extraPathList[i], memDir=memPath, modelDir=currentModelPath, epochsNum=epochsNum, valSplit=valSplit,
            continueTrain = not isFirstStep, pretrainedGPath=previousGPath, pretrainedDPath=previousDPath, approximateData=approximateData,
            trainingRatio=trainingRatio, earlyStoppingPatience=earlyStoppingPatience)

    def randomlyWeightedAverage(self, src):
        weights = K.random_uniform((self.batchSize, 1, 1, 1), minval=0., maxval=1.)
        dst = (weights*src[0]) + ((1-weights)*src[1])
        return dst


def trainG(pEcgPath, memPath, modelPath, imgRows=256, imgCols=256, channels=1, netGName='uNet', activationG='relu', temporalDepth=None, gKernels=64, gKernelSize=4,
epochsNum=100, lossFuncG='mae', batchSize=10, learningRateG=1e-4, earlyStoppingPatience=10, valSplit=0.2, approximateData=True):
    network = networks(imgRows=imgRows, imgCols=imgCols, channels=channels, gKernels=gKernels, gKernelSize=gKernelSize, temporalDepth=temporalDepth,
    activationG=activationG)
    if activationG == 'tanh':
        dataRange = [-1., 1.]
    else:
        dataRange = [0., 1.]
    extraRaw = dataProc.loadData(srcDir=pEcgPath, resize=1, normalization=1, normalizationRange=dataRange, approximateData=approximateData)
    if netGName == 'uNet':
        netG = network.uNet()
        extraSequence = np.ndarray((extraRaw.shape[0], imgRows, imgCols, channels), dtype=np.float32)
        extraSequence = extraRaw.reshape((extraRaw.shape[0], imgRows, imgCols, channels))
    elif netGName == 'uNet3D':
        netG = network.uNet3D()
        extraSequence = np.ndarray((extraRaw.shape[0], temporalDepth, imgRows, imgCols, channels), dtype=np.float32)
        extraRaw = dataProc.create3DData(extraRaw, temporalDepth=temporalDepth)
        extraSequence = extraRaw.reshape((extraSequence.shape[0], temporalDepth, imgRows, imgCols, channels))
    memRaw = dataProc.loadData(srcDir=memPath, resize=1, normalization=1, normalizationRange=dataRange, approximateData=approximateData)
    memSequence = np.ndarray((memRaw.shape[0], imgRows, imgCols, channels), dtype=np.float32)
    memSequence = memRaw.reshape((memRaw.shape[0], imgRows, imgCols, channels))
    netG.compile(optimizer=Adam(lr=learningRateG), loss=lossFuncG, metrics=[lossFuncG])
    netG.summary()
    checkpointer = ModelCheckpoint(modelPath+'netG_latest.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
    earlyStopping = EarlyStopping(patience=earlyStoppingPatience, verbose=1)
    print('begin to train netG')
    historyG = netG.fit(x=extraSequence, y=memSequence, batch_size=batchSize, epochs=epochsNum, verbose=2, shuffle=True, validation_split=valSplit,
    callbacks=[checkpointer, earlyStopping])

def squeeze(src, layer):
    dst = tf.squeeze(src, [layer])
    return dst

def slice(src):
    srcShape = src.shape.as_list()
    middleLayer = math.floor(srcShape[1]/2.0)
    dst = tf.slice(src, [0, middleLayer, 0, 0, 0], [-1, 1, -1, -1, -1])
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
    averagePenalty = K.mean(penalty, axis = 0)
    return averagePenalty

def displayLoss(lossD, lossA, lossVal, beginingTime, epoch):
    lossValStr = ' - lossVal: ' + '%.6f'%lossVal[0]
    lossDStr = ' - lossD: ' + lossD[0].astype(np.str) + ' '
    lossDOnRealStr = ' - lossD_real: ' + lossD[1].astype(np.str) + ' '
    lossDOnFakeStr = ' - lossD_fake: ' + lossD[2].astype(np.str) + ' '
    lossDOnPenalty = ' - penalty: ' + lossD[3].astype(np.str) + ' '
    lossAStr = ' - lossA: ' + lossA[0].astype(np.str) + ' '
    lossGStr = ' - lossG: ' + lossA[1].astype(np.str) + ' '
    lossDInA = ' - lossD: ' + lossA[2].astype(np.str) + ' '
    endTime = datetime.datetime.now()
    trainingTime = endTime - beginingTime
    msg = ' - %d'%trainingTime.total_seconds() + 's' + ' - epoch: ' + '%d '%(epoch) + lossDStr + lossDOnRealStr + lossDOnFakeStr \
    + lossDOnPenalty + lossAStr + lossGStr + lossDInA + lossValStr
    print(msg)
