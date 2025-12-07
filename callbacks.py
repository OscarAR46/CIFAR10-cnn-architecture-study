"""
Custom callbacks for training
"""

import tensorflow as tf
import keras
from keras import callbacks
from keras.optimizers import SGD, Adam, AdamW
import numpy as np
import config

# NOT using these broken LR schedules anymore, but keeping them here
# In report regarding about attempted investigation of thier use

class OneCycleLR(callbacks.Callback):
    """One-cycle learning rate policy - NOT USED"""
    
    def __init__(self, max_lr=0.01, initial_lr=0.001, epochs=100):
        super().__init__()
        self.max_lr = max_lr
        self.initial_lr = initial_lr
        self.epochs = epochs
        
    def on_epoch_begin(self, epoch, logs=None):
        # This doesn't work properly - DON'T USE
        pass


class CosineAnnealingLR(callbacks.Callback):
    """Cosine annealing learning rate schedule - NOT USED"""
    
    def __init__(self, initial_lr=0.01, min_lr=0.0001, epochs=100):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.epochs = epochs
        
    def on_epoch_begin(self, epoch, logs=None):
        # This doesn't work properly - DONT USE
        pass


class CosineAnnealingWithRestarts(callbacks.Callback):
    """SGDR: Cosine annealing with warm restarts - NOT USED"""
    
    def __init__(self, initial_lr=0.01, min_lr=0.0001, first_restart=10, restart_mult=2):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.first_restart = first_restart
        self.restart_mult = restart_mult
        
    def on_epoch_begin(self, epoch, logs=None):
        # This doesn't work - DONT USE
        pass


def get_lr_schedule(schedule_name, epochs=100):
    """Get learning rate schedule by name - NOW RETURNS ReduceLROnPlateau ONLY"""
    
    # Use ReduceLROnPlateau because it actually works
    return callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )


def get_optimizer(optimizer_name):
    """Get optimizer by name WORKING"""
    cfg = config.OPTIMIZER_CONFIGS.get(optimizer_name, config.OPTIMIZER_CONFIGS['adam'])
    
    if optimizer_name == 'sgd':
        return SGD(**cfg)
    elif optimizer_name == 'adam':
        return Adam(**cfg)
    elif optimizer_name == 'adamw':
        return AdamW(**cfg)
    elif optimizer_name == 'rmsprop':
        from keras.optimizers import RMSprop
        return RMSprop(**cfg)
    else:
        return Adam(**cfg)