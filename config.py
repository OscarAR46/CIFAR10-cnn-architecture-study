"""
Configuration settings for CIFAR-10 CNN experiments
All hyperparameters and experiment settings centralized here
"""

import os
from datetime import datetime

# Create output directories
OUTPUT_DIRS = ['outputs', 'outputs/models', 'outputs/plots', 'outputs/results', 'outputs/logs']
for dir_path in OUTPUT_DIRS:
    os.makedirs(dir_path, exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    'num_classes': 10,
    'input_shape': (32, 32, 3),
    'class_names': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck'],
    'validation_split': 0.1,
    'random_seed': 42
}

# Training defaults
TRAINING_DEFAULTS = {
    'batch_size': 128,
    'epochs': 100,
    'verbose': 1,
    'use_augmentation': True,
    'use_mixup': False,
    'mixup_alpha': 1.0  # Reduced for stability
}

# Model configurations
MODEL_CONFIGS = {
    'baseline': {
        'filters': [32, 64],
        'dropout': 0.25,
        'dense_units': 128
    },
    'vgg_style': {
        'filters': [64, 128, 256],
        'dropout': 0.25,
        'l2_reg': 0.0001,
        'dense_units': [512, 256]
    },
    'resnet': {
        'initial_filters': 64,
        'num_blocks': [2, 2, 2, 2],
        'stochastic_depth_rate': 0.1
    },
    'attention': {
        'filters': [64, 128, 256],
        'reduction_ratio': 8,
        'dropblock_rate': 0.15  # Reduced for stability
    },
    'densenet': {
        'growth_rate': 32,
        'n_layers_per_block': 6,
        'compression': 0.5
    },
    'inception': {
        'filters': [32, 64, 128],
        'pool_proj': 32
    },
    'wide_resnet': {
        'depth': 16,
        'width_factor': 8,
        'dropout': 0.25  # Reduced for stability
    }
}

# Optimizer configurations - THSE ARE GOOD
OPTIMIZER_CONFIGS = {
    'sgd': {
        'learning_rate': 0.01,
        'momentum': 0.9,
        'nesterov': True
    },
    'adam': {
        'learning_rate': 0.001,
        'beta_1': 0.9,
        'beta_2': 0.999
    },
    'adamw': {
        'learning_rate': 0.001,
        'weight_decay': 1e-4
    },
    'rmsprop': {
        'learning_rate': 0.001,
        'rho': 0.9
    }
}

# Learning rate schedule configurations - NOT USED NOW
LR_SCHEDULE_CONFIGS = {
    'cosine': {
        'initial_lr': 0.01,
        'min_lr': 0.0001
    },
    'one_cycle': {
        'initial_lr': 0.001,
        'max_lr': 0.01
    },
    'cosine_restarts': {
        'initial_lr': 0.01,
        'min_lr': 0.0001,
        'first_restart': 10,
        'restart_mult': 2
    },
    'exponential': {
        'initial_lr': 0.01,
        'decay_rate': 0.96,
        'decay_steps': 100
    }
}

# Augmentation settings
AUGMENTATION_CONFIGS = {
    'basic': {
        'rotation_range': 15,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'horizontal_flip': True,
        'zoom_range': 0.1
    },
    'advanced': {
        'rotation_range': 20,
        'width_shift_range': 0.125,
        'height_shift_range': 0.125,
        'horizontal_flip': True,
        'zoom_range': 0.15,
        'shear_range': 0.1,
        'brightness_range': [0.9, 1.1]
    },
    'cutout': {
        'n_holes': 1,
        'length': 16
    }
}

# Experiment settings - UPDATED FOR BETTER RESULTS
EXPERIMENT_CONFIGS = {
    'regularization_study': {
        'dropout_rates': [0.0, 0.2, 0.3],
        'l2_values': [0, 1e-4, 1e-3],
        'epochs': 40
    },
    'batch_size_study': {
        'batch_sizes': [64, 128, 256],
        'epochs': 40
    },
    'optimizer_comparison': {
        'optimizers': ['sgd', 'adam', 'adamw'],
        'epochs': 20 # Dont change works fine
    },
    'ensemble': {
        'n_models': 3,
        'model_types': ['vgg_style', 'resnet', 'attention'],
        'voting': 'soft'
    }
}

# Optuna hyperparameter search space
OPTUNA_SEARCH_SPACE = {
    'n_conv_blocks': (2, 4),
    'base_filters': [32, 64],
    'kernel_size': [3, 5],
    'dropout_rate': (0.1, 0.4),
    'l2_regularization': (1e-5, 1e-3),
    'use_batch_norm': [True, False],
    'use_attention': [False],  # Disabled for stability
    'optimizer': ['Adam'],  # Only Adam works reliably
    'learning_rate': (5e-4, 5e-3),
    'n_dense_units': [128, 256, 512]
}

# PCA visualization settings
PCA_CONFIG = {
    'n_components': 50,
    'n_samples_viz': 5000,
    'perplexity': 30
}

# Timestamp for this run
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")