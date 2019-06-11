from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, Dropout, BatchNormalization, MaxPooling2D, Add, Activation


def resBlock(x, kernels, kernel_size=3, strides=1):
    shortcut = x
    x = Conv2D(kernels, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(x)
    x = Activation('relu')(x)
    x = Conv2D(kernels, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(x)
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
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
        x = Conv2D(kernels * kernels_multiplier, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(x)
        x = Activation('relu')(x)
        x = resBlock(x, kernels * kernels_multiplier, )
        connection.append(x)
        if k < depth:
            x = MaxPooling2D(2)(x)

    for k in range(depth-1, -1, -1):
        if 2**k > max_kernel_multiplier:
            kernels_multiplier = max_kernel_multiplier
        else:
            kernels_multiplier = 2**k
        x = Conv2DTranspose(kernels * kernels_multiplier, 3, strides=2, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.0001, center=False, scale=False)(x)
        x = Activation('relu')(x)
        x = Concatenate(axis=-1)([connection[k], x])
        if k > depth-3:
            x = Dropout(0.5)(x)
        x = Conv2D(kernels * kernels_multiplier, 3, padding='same', kernel_initializer='he_normal')(x)
        x = resBlock(x, kernels * kernels_multiplier)
    
    out_layer = Conv2D(1, 1, activation=activation, padding='same', kernel_initializer='he_normal')(x)
    model = Model(inputs=inputs, outputs=out_layer, name='resUnet2d')
    return model
