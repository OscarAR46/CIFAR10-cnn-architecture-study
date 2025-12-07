"""
Model architecture definitions for CIFAR-10
"""

import tensorflow as tf
import keras
from keras import layers, models, regularizers
import config
from custom_layers import (
    DropBlock, StochasticDepth, channel_attention, 
    residual_block, dense_block, inception_module
)

def create_baseline_cnn(input_shape=None, num_classes=None):
    """Simple baseline CNN"""
    if input_shape is None:
        input_shape = config.DATASET_CONFIG['input_shape']
    if num_classes is None:
        num_classes = config.DATASET_CONFIG['num_classes']
    
    cfg = config.MODEL_CONFIGS['baseline']
    
    model = models.Sequential([
        # Block 1
        layers.Conv2D(cfg['filters'][0], 3, padding='same', 
                     kernel_initializer='he_normal',
                     use_bias=False, input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(cfg['filters'][0], 3, padding='same', 
                     kernel_initializer='he_normal', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(cfg['dropout']),
        
        # Block 2
        layers.Conv2D(cfg['filters'][1], 3, padding='same', 
                     kernel_initializer='he_normal', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(cfg['filters'][1], 3, padding='same', 
                     kernel_initializer='he_normal', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(cfg['dropout'] * 1.5),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(cfg['dense_units'], kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(cfg['dropout'] * 2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def create_vgg_style(input_shape=None, num_classes=None):
    """VGG-inspired architecture"""
    if input_shape is None:
        input_shape = config.DATASET_CONFIG['input_shape']
    if num_classes is None:
        num_classes = config.DATASET_CONFIG['num_classes']
    
    cfg = config.MODEL_CONFIGS['vgg_style']
    reg = regularizers.l2(cfg['l2_reg'])
    
    model = models.Sequential()
    
    # Block 1: 64 filters
    model.add(layers.Conv2D(cfg['filters'][0], 3, padding='same', 
                           kernel_regularizer=reg,
                           kernel_initializer='he_normal', 
                           use_bias=False, input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('gelu'))
    model.add(layers.Conv2D(cfg['filters'][0], 3, padding='same', 
                           kernel_regularizer=reg,
                           kernel_initializer='he_normal', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('gelu'))
    model.add(layers.MaxPooling2D(2))
    
    # Block 2: 128 filters
    model.add(layers.Conv2D(cfg['filters'][1], 3, padding='same', 
                           kernel_regularizer=reg,
                           kernel_initializer='he_normal', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('gelu'))
    model.add(layers.Conv2D(cfg['filters'][1], 3, padding='same', 
                           kernel_regularizer=reg,
                           kernel_initializer='he_normal', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('gelu'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Dropout(cfg['dropout']))
    
    # Block 3: 256 filters
    for _ in range(3):
        model.add(layers.Conv2D(cfg['filters'][2], 3, padding='same', 
                               kernel_regularizer=reg,
                               kernel_initializer='he_normal', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('gelu'))
    model.add(layers.MaxPooling2D(2))
    model.add(layers.Dropout(cfg['dropout']))
    
    # Dense layers
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(cfg['dense_units'][0], kernel_regularizer=reg, 
                          kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('gelu'))
    model.add(layers.Dropout(cfg['dropout'] + 0.2))
    model.add(layers.Dense(cfg['dense_units'][1], kernel_regularizer=reg, 
                          kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('gelu'))
    model.add(layers.Dropout(cfg['dropout'] + 0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model


def create_resnet(input_shape=None, num_classes=None):
    """ResNet-inspired architecture"""
    if input_shape is None:
        input_shape = config.DATASET_CONFIG['input_shape']
    if num_classes is None:
        num_classes = config.DATASET_CONFIG['num_classes']
    
    cfg = config.MODEL_CONFIGS['resnet']
    
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(cfg['initial_filters'], 3, padding='same', 
                     kernel_initializer='he_normal', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Residual blocks with progressive feature maps
    filters = cfg['initial_filters']
    survival_prob = 1.0 - cfg['stochastic_depth_rate']
    
    for i, num_blocks in enumerate(cfg['num_blocks']):
        for j in range(num_blocks):
            stride = 2 if (i > 0 and j == 0) else 1
            x = residual_block(x, filters, stride=stride,
                             use_stochastic_depth=True, 
                             survival_prob=survival_prob)
            survival_prob *= 0.95  # Decay survival probability
        filters *= 2
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)


def create_attention_cnn(input_shape=None, num_classes=None):
    """CNN with channel attention mechanisms"""
    if input_shape is None:
        input_shape = config.DATASET_CONFIG['input_shape']
    if num_classes is None:
        num_classes = config.DATASET_CONFIG['num_classes']
    
    cfg = config.MODEL_CONFIGS['attention']
    
    inputs = layers.Input(shape=input_shape)
    
    x = inputs
    for i, filters in enumerate(cfg['filters']):
        # Conv block
        x = layers.Conv2D(filters, 3, padding='same', 
                         kernel_initializer='he_normal', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('gelu')(x)
        x = layers.Conv2D(filters, 3, padding='same', 
                         kernel_initializer='he_normal', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('gelu')(x)
        
        # Add attention
        x = channel_attention(x, reduction_ratio=cfg['reduction_ratio'])
        
        # Pooling and regularization
        x = layers.MaxPooling2D(2)(x)
        if i > 0:
            x = DropBlock(drop_rate=cfg['dropblock_rate'] * (i+1), block_size=3)(x)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)


def create_densenet(input_shape=None, num_classes=None):
    """DenseNet-style architecture"""
    if input_shape is None:
        input_shape = config.DATASET_CONFIG['input_shape']
    if num_classes is None:
        num_classes = config.DATASET_CONFIG['num_classes']
    
    cfg = config.MODEL_CONFIGS['densenet']
    
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(64, 3, padding='same', use_bias=False,
                     kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Dense blocks with transition layers
    for i in range(3):
        x = dense_block(x, growth_rate=cfg['growth_rate'], 
                       n_layers=cfg['n_layers_per_block'])
        
        if i < 2:  # Add transition layer except for last block
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            # Compression
            n_filters = int(x.shape[-1] * cfg['compression'])
            x = layers.Conv2D(n_filters, 1, use_bias=False)(x)
            x = layers.AveragePooling2D(2)(x)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)


def create_inception(input_shape=None, num_classes=None):
    """Inception-inspired architecture"""
    if input_shape is None:
        input_shape = config.DATASET_CONFIG['input_shape']
    if num_classes is None:
        num_classes = config.DATASET_CONFIG['num_classes']
    
    cfg = config.MODEL_CONFIGS['inception']
    
    inputs = layers.Input(shape=input_shape)
    
    # Stem
    x = layers.Conv2D(32, 3, padding='same', activation='relu',
                     kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Inception modules with progressive filters
    for i, filters in enumerate(cfg['filters']):
        x = inception_module(x, filters)
        x = inception_module(x, filters)
        
        if i < len(cfg['filters']) - 1:
            x = layers.MaxPooling2D(2)(x)
    
    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)


def create_wide_resnet(input_shape=None, num_classes=None):
    """Wide ResNet implementation"""
    if input_shape is None:
        input_shape = config.DATASET_CONFIG['input_shape']
    if num_classes is None:
        num_classes = config.DATASET_CONFIG['num_classes']
    
    cfg = config.MODEL_CONFIGS['wide_resnet']
    
    def wide_basic(x, filters, stride=1):
        """Wide residual block"""
        shortcut = x
        
        # BN-ReLU-Conv ordering
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Adjust shortcut if needed
        if stride != 1 or x.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False,
                                    kernel_initializer='he_normal')(x)
        
        x = layers.Conv2D(filters, 3, strides=stride, padding='same', 
                         use_bias=False, kernel_initializer='he_normal')(x)
        
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = DropBlock(drop_rate=cfg['dropout'], block_size=3)(x)
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False,
                         kernel_initializer='he_normal')(x)
        
        return layers.Add()([shortcut, x])
    
    inputs = layers.Input(shape=input_shape)
    
    # Build network
    n = (cfg['depth'] - 4) // 6
    x = layers.Conv2D(16, 3, padding='same', use_bias=False,
                     kernel_initializer='he_normal')(inputs)
    
    # Group 1
    for _ in range(n):
        x = wide_basic(x, 16 * cfg['width_factor'], stride=1)
    
    # Group 2
    x = wide_basic(x, 32 * cfg['width_factor'], stride=2)
    for _ in range(n-1):
        x = wide_basic(x, 32 * cfg['width_factor'], stride=1)
    
    # Group 3
    x = wide_basic(x, 64 * cfg['width_factor'], stride=2)
    for _ in range(n-1):
        x = wide_basic(x, 64 * cfg['width_factor'], stride=1)
    
    # Classification head
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                          kernel_initializer='he_normal')(x)
    
    return models.Model(inputs, outputs)


# Model factory
MODEL_FACTORY = {
    'baseline': create_baseline_cnn,
    'vgg_style': create_vgg_style,
    'resnet': create_resnet,
    'attention': create_attention_cnn,
    'densenet': create_densenet,
    'inception': create_inception,
    'wide_resnet': create_wide_resnet
}

def get_model(model_name, **kwargs):
    """Factory method to get model by name"""
    if model_name not in MODEL_FACTORY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_FACTORY.keys())}")
    
    return MODEL_FACTORY[model_name](**kwargs)