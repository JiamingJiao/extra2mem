from keras.models import Model
from keras.layers import Input, Conv3D, Conv3DTranspose, Concatenate, Dropout, BatchNormalization, MaxPooling3D, Add, Activation
from keras.layers.advanced_activations import LeakyReLU


def activate(x, name):
    if name == 'relu':
        x = Activation('relu')(x)
    elif name == 'leakyRelu':
        x = LeakyReLU(alpha=0.2)(x)
    return x


def resBlock(x, kernels, activation_name, kernel_size=3, strides=1):
    shortcut = x
    x = Conv3D(kernels, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(x)
    x = activate(x, activation_name)
    x = Conv3D(kernels, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(x)
    x = Add()([shortcut, x])
    x = activate(x, activation_name)
    return x


def getModel(temporal_depth, img_rows, img_cols, channels, depth, kernels, max_kernel_multiplier=16, activation='sigmoid'):
    inputs = Input((temporal_depth, img_rows, img_cols, channels))  # 64
    x = inputs
    connection = []
    for k in range(0, depth+1):
        if 2**k > max_kernel_multiplier:
            kernels_multiplier = max_kernel_multiplier
        else:
            kernels_multiplier = 2**k
        x = Conv3D(kernels * kernels_multiplier, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(x)
        x = activate(x, 'leakyRelu')
        x = resBlock(x, kernels * kernels_multiplier, 'leakyRelu')
        connection.append(x)
        if k < depth:
            x = MaxPooling3D(2)(x)
        # x = getResBlock(x, kernels * kernels_multiplier, 'leakyRelu')

    for k in range(depth-1, -1, -1):
        # x = UpSampling3D(2)(x)
        if 2**k > max_kernel_multiplier:
            kernels_multiplier = max_kernel_multiplier
        else:
            kernels_multiplier = 2**k
        x = Conv3DTranspose(kernels * kernels_multiplier, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(x)
        x = activate(x, 'relu')
        x = Concatenate(axis=-1)([connection[k], x])
        if k > depth-3:
            x = Dropout(0.5)(x)
        x = Conv3D(kernels * kernels_multiplier, 3, padding='same', kernel_initializer='he_normal')(x)
        x = resBlock(x, kernels * kernels_multiplier, 'relu')
    
    out_layer = Conv3D(1, 1, activation=activation, padding='same', kernel_initializer='he_normal')(x)
    model = Model(inputs=inputs, outputs=out_layer, name='resUnet3d')
    return model


# def getModel(temporal_depth, img_rows, img_cols, channels, kernels, activation):
#     inputs = Input((temporal_depth, img_rows, img_cols, channels))  # 64

#     encoder1 = getResBlock(inputs, kernels)  # 64

#     encoder2 = MaxPooling3D(2)(encoder1)  # 32
#     encoder2 = getResBlock(encoder2, kernels*2)

#     encoder3 = MaxPooling3D(2)(encoder2)  # 16
#     encoder3 = getResBlock(encoder3, kernels*4)

#     encoder4 = MaxPooling3D(2)(encoder3)  # 8
#     encoder4 = getResBlock(encoder4, kernels*8)

#     encoder5 = MaxPooling3D(2)(encoder4)  # 4
#     encoder5 = getResBlock(encoder5, kernels*16)
    
#     middle = MaxPooling3D(2)(encoder5)  # 2
#     middle = getResBlock(middle, kernels*32)

#     decoder1 = UpSampling3D(2)(middle)  # 4
#     connection1 = Concatenate(axis=-1)([encoder5, decoder1])
#     decoder1 = Dropout(0.5)(connection1)
#     decoder1 = getResBlock(decoder1, kernels*16)

#     decoder2 = UpSampling3D(2)(decoder1)  # 8
#     connection2 = Concatenate(axis=-1)([encoder4, decoder2])
#     decoder2 = Dropout(0.5)(connection2)
#     decoder2 = getResBlock(decoder2, kernels*8)

#     decoder3 = UpSampling3D(2)(decoder2)  # 16
#     connection3 = Concatenate(axis=-1)([encoder3, decoder3])
#     decoder3 = Dropout(0.5)(connection3)
#     decoder3 = getResBlock(decoder3, kernels*4)

#     decoder4 = UpSampling3D(2)(decoder3)  # 32
#     connection4 = Concatenate(axis=-1)([encoder2, decoder4])
#     decoder4 = getResBlock(connection4, kernels*2)

#     decoder5 = UpSampling3D(2)(decoder4)  # 64
#     connection5 = Concatenate(axis=-1)([encoder1, decoder5])
#     decoder5 = getResBlock(connection5, kernels)

#     out_layer = Conv3D(1, 1, activation=activation, padding='same', kernel_initializer='he_normal')(decoder5)

#     model = Model(inputs=inputs, outputs=out_layer, name='resUnet3d')
#     return model
