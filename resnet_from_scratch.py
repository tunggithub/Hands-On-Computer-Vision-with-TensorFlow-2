import tensorflow as tf
import functools

from tensorflow.keras.layers import (
    Input, Activation, Dense, Flatten, Conv2D, MaxPooling2D,
     GlobalAveragePooling2D, AveragePooling2D, BatchNormalization, add
)

import tensorflow.keras.regularizers as regulizers


class ConvWithBatchNorm(Conv2D):
    def __init__(self, activation='relu', name='convbn', **kwargs):
        self.activation = Activation(activation, name=name+'_act') if activation is not None else None

        super().__init__(activation=None, name=name + '_c', **kwargs)
        self.batch_norm = BatchNormalization(axis=-1, name=name + '_bn')
    
    def call(self, inputs, training=None):
        x = super().call(inputs)
        x = self.batch_norm(x, training=training)
        if self.activation is not None:
            x = self.activation(x)
        return x
    
class ResidualMerge(tf.keras.layers.Layer):

    def __init__(self, name='block', **kwargs):
        super().__init__(name=name)
        self.shotcut = None
        self.kwargs = kwargs

    def build(self, input_shape):
        x_shape = input_shape[0]
        x_residual_shape = input_shape[1]
        if x_shape[1] == x_residual_shape[1] and \
            x_shape[2] == x_residual_shape[2] and \
            x_shape[3] == x_residual_shape[3]:
            self.shotcut = functools.partial(tf.identity, name=self.name + '_shortcut')
        
        else:
            strides = (
                int(round(x_shape[1] / x_residual_shape[1])),
                int(round(x_shape[2] / x_residual_shape[2]))
            )
            x_residual_channels = x_residual_shape[3]
            self.shotcut = ConvWithBatchNorm(
                filters = x_residual_channels,
                kernel_size = (1,1),
                strides = strides,
                activation=None,
                name=self.name + '_shortcut_c', **self.kwargs
            )
        
    def call(self, inputs):
        x, x_residual = inputs

        x_shorcut = self.shotcut(x)
        x_merge = add([x_shorcut, x_residual])
        return x_merge


class BasicResidualBlock(tf.keras.Model):
    def __init__(self, filters = 16, kernel_size=1, strides=1, activation='relu', 
                 kernel_initializer='he_normal', kernel_regularizer=regulizers.l2(1e-4), 
                 name='res_basic', **kwargs):
        super().__init__(name=name, **kwargs)

        self.conv_0 = ConvWithBatchNorm(
            filters = filters, kernel_size=1, activation=activation, padding='valid',
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
            strides=1, name=name + 'cb_0', **kwargs
        )

        self.conv_1 = ConvWithBatchNorm(
            filters=filters, kernel_size=kernel_size, activation=activation, padding='same',
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
            strides=strides, name=name + '_cb_1', **kwargs
        )

        self.conv_2 = ConvWithBatchNorm(
            filters=4 * filters, kernel_size=1, activation=None, padding='valid',
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
            strides=1, name=name + '_cb_2', **kwargs)
        
        self.merge = ResidualMerge(
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, 
            name=name)
        
        self.activation = Activation(activation, name=name + '_act')
    
    def call(self, inputs, training=None):
        x = inputs
        
        x_residual = self.conv_0(x, training=training)
        x_residual = self.conv_1(x_residual, training=training)
        x_residual = self.conv_2(x_residual, training=training)

        x_merge = self.merge([x, x_residual])
        x_merge = self.activation(x_merge)

        return x_merge

class ResidualBlockWithBottleneck(tf.keras.Model):
    def __init__(self, filters=16, kernel_size=1, strides=1, activation='relu',
                 kernel_initializer='he_normal', kernel_regularizer=regulizers.l2(1e-4),
                 name='res_basic', **kwargs):
        super().__init__(name=name)
        self.conv_0 = ConvWithBatchNorm(
            filters=filters, kernel_size=1, activation=activation, padding='valid',
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
            strides=1, name=name + '_cb_1', **kwargs
        )

        self.conv_1 = ConvWithBatchNorm(
            filters=filters, kernel_size=kernel_size, activation=activation, padding='same',
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
            strides=strides, name=name + '_cb_1', **kwargs
        )

        self.conv_2 = ConvWithBatchNorm(
            filters=4 * filters, kernel_size=1, activation=None, padding='valid',
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
            strides=1, name=name + '_cb_2', **kwargs
        )

        self.merge = ResidualMerge(
            kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, 
            name=name)
        
        self.activation = Activation(activation, name=name + '_act')
    
    def call(self, inputs, training=None):
        x = inputs

        x_residual = self.conv_0(x, training=training)
        x_residual = self.conv_1(x_residual, training=training)
        x_residual = self.conv_2(x_residual, training=training)

        x_merge = self.merge([x, x_residual])
        x_merge = self.activation(x_merge)
        return x_merge
    
class ResidualMacroBlock(tf.keras.models.Sequential):
    def __init__(self, block_class=ResidualBlockWithBottleneck, repetitions=3,
                 filters=16, kernel_size=1, strides=1, activation='relu',
                 kernel_initializer='he_normal', kernel_regularizer=regulizers.l2(1e-4),
                 name='res_macroblock', **kwargs):
        super().__init__(
            [
                block_class(
                    filters=filters, kernel_size=kernel_size, activation=activation,
                    strides=strides if i == 0 else 1, name=f"{name}_{i}",
                    kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer
                ) for i in range(repetitions)
            ]
        )

class ResNet(tf.keras.models.Sequential):
    def __init__(self, input_shape, num_classes=1000, block_class=ResidualBlockWithBottleneck, 
                 repetitions=(2,2,2,2), kernel_initializer='he_normal', kernel_regularizer=regulizers.l2(1e-4), 
                 name='resnet'):
        
        filters = 64
        strides = 2

        super().__init__(
            [
                Input(shape=input_shape, name='input'),
                ConvWithBatchNorm(
                    filters=filters, kernel_size=7, activation='relu', padding='same', strides=strides,
                    kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer,
                    name='conv'
                ),
                MaxPooling2D(pool_size=3, strides=strides, padding='same', name='max_pool')
            ]
            +
            [
                ResidualMacroBlock(
                    block_class=block_class, repetitions=repeat, filters=min(filters * (2 ** i), 1024),
                    kernel_size=3, activation='relu', strides=strides if i!=0 else 1, 
                    name=f'block_{i}', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer
                ) for i, repeat in enumerate(repetitions)
            ]
            +
            [GlobalAveragePooling2D(name='avg_pool'),
             Dense(units=num_classes, kernel_initializer=kernel_initializer, activation='softmax')
            ], name=name
        )

class ResNet18(ResNet):
    def __init__(self, input_shape, num_classes=1000, name='resnet18',
                 kernel_initializer='he_normal', kernel_regularizer=regulizers.l2(1e-4)):
        super().__init__(input_shape, num_classes, 
                         block_class=BasicResidualBlock, repetitions=(2, 2, 2, 2),
                         kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
        


        
ResNet18(input_shape=(33,33,3)).summary()