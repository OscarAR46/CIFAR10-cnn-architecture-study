"""
Custom layer implementations for advanced architectures
"""

import tensorflow as tf
import keras
from keras import layers
import numpy as np

class DropBlock(layers.Layer):
    """DropBlock regularization for CNNs"""
    
    def __init__(self, drop_rate=0.1, block_size=7, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = drop_rate
        self.block_size = block_size
        
    def call(self, inputs, training=None):
        if not training or self.drop_rate == 0:
            return inputs
            
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        channels = tf.shape(inputs)[3]
        
        # Calculate gamma
        height_float = tf.cast(height, tf.float32)
        width_float = tf.cast(width, tf.float32)
        block_size_float = tf.cast(self.block_size, tf.float32)
        
        gamma = (self.drop_rate * height_float * width_float / 
                (block_size_float * block_size_float) / 
                ((height_float - block_size_float + 1) * 
                 (width_float - block_size_float + 1)))
        
        # Sample mask - create a smaller mask first
        mask_height = height - self.block_size + 1
        mask_width = width - self.block_size + 1
        
        # Ensure mask dimensions are at least 1
        mask_height = tf.maximum(mask_height, 1)
        mask_width = tf.maximum(mask_width, 1)
        
        mask = tf.random.uniform(
            shape=[batch_size, mask_height, mask_width, channels],
            dtype=tf.float32
        )
        mask = tf.cast(mask < gamma, dtype=tf.float32)
        
        # Max pool to create block mask
        mask = tf.nn.max_pool(
            mask,
            ksize=[1, self.block_size, self.block_size, 1],
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        
        # Resize mask to match input dimensions exactly
        mask = tf.image.resize(mask, [height, width], method='nearest')
        
        # Invert mask (1 - mask so that 1 = keep, 0 = drop)
        mask = 1.0 - mask
        
        # Normalize output
        mask_sum = tf.reduce_sum(mask)
        mask_sum = tf.maximum(mask_sum, 1.0)
        
        # Apply mask and scale
        outputs = inputs * mask * tf.cast(tf.size(mask), tf.float32) / mask_sum
        outputs = tf.cast(outputs, inputs.dtype)
        
        return outputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'drop_rate': self.drop_rate,
            'block_size': self.block_size
        })
        return config


class StochasticDepth(layers.Layer):
    """Stochastic depth for residual networks"""
    
    def __init__(self, survival_probability=0.8, **kwargs):
        super().__init__(**kwargs)
        self.survival_prob = survival_probability
        
    def call(self, inputs, training=None):
        if not training:
            return inputs
            
        batch_size = tf.shape(inputs)[0]
        random_tensor = self.survival_prob
        random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
        binary_tensor = tf.floor(random_tensor)
        
        output = inputs * binary_tensor / self.survival_prob
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({'survival_probability': self.survival_prob})
        return config


def channel_attention(x, reduction_ratio=8):
    """Squeeze-and-excitation attention block"""
    channels = x.shape[-1]
    
    # Global average pooling
    gap = layers.GlobalAveragePooling2D()(x)
    
    # Squeeze and excitation
    fc1 = layers.Dense(
        channels // reduction_ratio, 
        activation='relu',
        kernel_initializer='he_normal', 
        use_bias=False
    )(gap)
    
    fc2 = layers.Dense(
        channels, 
        activation='sigmoid',
        kernel_initializer='he_normal', 
        use_bias=False
    )(fc1)
    
    # Reshape and scale
    scale = layers.Reshape((1, 1, channels))(fc2)
    
    return layers.Multiply()([x, scale])


def residual_block(x, filters, stride=1, use_bottleneck=False, 
                   use_stochastic_depth=False, survival_prob=0.8):
    """Flexible residual block implementation"""
    shortcut = x
    
    if use_bottleneck:
        # Bottleneck architecture
        x = layers.Conv2D(filters // 4, 1, use_bias=False,
                         kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters // 4, 3, strides=stride, padding='same',
                         use_bias=False, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters, 1, use_bias=False,
                         kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
    else:
        # Standard residual block
        x = layers.Conv2D(filters, 3, strides=stride, padding='same',
                         use_bias=False, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False,
                         kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
    
    # Adjust shortcut if needed
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False,
                                kernel_initializer='he_normal')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Apply stochastic depth
    if use_stochastic_depth:
        x = StochasticDepth(survival_probability=survival_prob)(x)
    
    return layers.Activation('relu')(layers.Add()([x, shortcut]))


def dense_block(x, growth_rate=32, n_layers=4):
    """DenseNet-style dense block"""
    features = [x]
    
    for _ in range(n_layers):
        y = layers.BatchNormalization()(x)
        y = layers.Activation('relu')(y)
        y = layers.Conv2D(growth_rate * 4, 1, use_bias=False,
                         kernel_initializer='he_normal')(y)
        
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        y = layers.Conv2D(growth_rate, 3, padding='same', use_bias=False,
                         kernel_initializer='he_normal')(y)
        
        features.append(y)
        x = layers.Concatenate()(features)
    
    return x


def inception_module(x, filters):
    """Inception module with multiple paths"""
    # 1x1 branch
    branch1x1 = layers.Conv2D(filters, 1, padding='same', activation='relu',
                             kernel_initializer='he_normal')(x)
    
    # 3x3 branch
    branch3x3 = layers.Conv2D(filters, 1, activation='relu',
                             kernel_initializer='he_normal')(x)
    branch3x3 = layers.Conv2D(filters, 3, padding='same', activation='relu',
                             kernel_initializer='he_normal')(branch3x3)
    
    # 5x5 branch (factorized as two 3x3)
    branch5x5 = layers.Conv2D(filters, 1, activation='relu',
                             kernel_initializer='he_normal')(x)
    branch5x5 = layers.Conv2D(filters, 3, padding='same', activation='relu',
                             kernel_initializer='he_normal')(branch5x5)
    branch5x5 = layers.Conv2D(filters, 3, padding='same', activation='relu',
                             kernel_initializer='he_normal')(branch5x5)
    
    # Pool branch
    branch_pool = layers.MaxPooling2D(3, strides=1, padding='same')(x)
    branch_pool = layers.Conv2D(filters, 1, padding='same', activation='relu',
                               kernel_initializer='he_normal')(branch_pool)
    
    return layers.Concatenate()([branch1x1, branch3x3, branch5x5, branch_pool])