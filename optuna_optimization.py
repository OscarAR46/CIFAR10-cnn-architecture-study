"""
Optuna-based hyperparameter optimization
"""

import optuna
from optuna.integration import TFKerasPruningCallback
import tensorflow as tf
import keras
from keras import layers, models, regularizers
from keras.optimizers import SGD, Adam, AdamW
import numpy as np
import config
from custom_layers import channel_attention
from data_utils import DataPreprocessor

class HyperparameterOptimizer:
    """Optuna hyperparameter optimization"""
    
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.best_model = None
        
    def create_model(self, trial):
        """Create model based on trial suggestions"""
        
        # Architecture hyperparameters
        n_blocks = trial.suggest_int('n_conv_blocks', *config.OPTUNA_SEARCH_SPACE['n_conv_blocks'])
        base_filters = trial.suggest_categorical('base_filters', config.OPTUNA_SEARCH_SPACE['base_filters'])
        kernel_size = trial.suggest_categorical('kernel_size', config.OPTUNA_SEARCH_SPACE['kernel_size'])
        
        # Regularization
        dropout_rate = trial.suggest_float('dropout_rate', *config.OPTUNA_SEARCH_SPACE['dropout_rate'])
        l2_reg = trial.suggest_float('l2_regularization', *config.OPTUNA_SEARCH_SPACE['l2_regularization'], log=True)
        use_batch_norm = trial.suggest_categorical('use_batch_norm', config.OPTUNA_SEARCH_SPACE['use_batch_norm'])
        use_attention = trial.suggest_categorical('use_attention', config.OPTUNA_SEARCH_SPACE['use_attention'])
        
        # Build model
        model = models.Sequential()
        filters = base_filters
        
        # First convolution
        model.add(layers.Conv2D(filters, kernel_size, padding='same',
                               kernel_regularizer=regularizers.l2(l2_reg),
                               kernel_initializer='he_normal',
                               use_bias=not use_batch_norm,
                               input_shape=config.DATASET_CONFIG['input_shape']))
        if use_batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        
        # Additional conv blocks
        for block in range(n_blocks - 1):
            if block > 0:
                filters *= 2
                model.add(layers.MaxPooling2D(2))
                model.add(layers.Dropout(dropout_rate))
            
            # Two conv layers per block
            for _ in range(2):
                model.add(layers.Conv2D(filters, kernel_size, padding='same',
                                       kernel_regularizer=regularizers.l2(l2_reg),
                                       kernel_initializer='he_normal',
                                       use_bias=not use_batch_norm))
                if use_batch_norm:
                    model.add(layers.BatchNormalization())
                model.add(layers.Activation('relu'))
            
            # Add attention if suggested
            if use_attention and block == n_blocks - 2:
                pass
        
        # Classification head
        model.add(layers.GlobalAveragePooling2D())
        
        n_dense = trial.suggest_categorical('n_dense_units', config.OPTUNA_SEARCH_SPACE['n_dense_units'])
        model.add(layers.Dense(n_dense, kernel_regularizer=regularizers.l2(l2_reg),
                              kernel_initializer='he_normal'))
        if use_batch_norm:
            model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(dropout_rate + 0.1))
        
        model.add(layers.Dense(config.DATASET_CONFIG['num_classes'], activation='softmax'))
        
        return model
    
    def objective(self, trial):
        """Objective function for optimization"""
        
        # Clear session to free memory
        tf.keras.backend.clear_session()
        
        # Create model
        model = self.create_model(trial)
        
        # Select optimizer
        opt_name = trial.suggest_categorical('optimizer', config.OPTUNA_SEARCH_SPACE['optimizer'])
        lr = trial.suggest_float('learning_rate', *config.OPTUNA_SEARCH_SPACE['learning_rate'], log=True)
        
        if opt_name == 'SGD':
            momentum = trial.suggest_float('momentum', 0.8, 0.99)
            optimizer = SGD(learning_rate=lr, momentum=momentum, nesterov=True)
        elif opt_name == 'Adam':
            optimizer = Adam(learning_rate=lr)
        else:
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
            optimizer = AdamW(learning_rate=lr, weight_decay=weight_decay)
        
        # Compile
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            TFKerasPruningCallback(trial, 'val_accuracy')
        ]
        
        # Train with or without augmentation
        use_aug = trial.suggest_categorical('use_augmentation', [True, False])
        
        if use_aug:
            datagen = DataPreprocessor().get_augmentation_generator()
            datagen.fit(self.x_train)
            
            history = model.fit(
                datagen.flow(self.x_train, self.y_train, batch_size=64),
                epochs=30,
                validation_data=(self.x_val, self.y_val),
                callbacks=callbacks_list,
                verbose=0
            )
        else:
            history = model.fit(
                self.x_train, self.y_train,
                batch_size=64,
                epochs=30,
                validation_data=(self.x_val, self.y_val),
                callbacks=callbacks_list,
                verbose=0
            )
        
        # Get best validation accuracy
        best_val_acc = max(history.history['val_accuracy'])
        
        # Save best model
        if self.best_model is None or best_val_acc > self.best_model['accuracy']:
            self.best_model = {
                'model': model,
                'accuracy': best_val_acc,
                'params': trial.params
            }
        
        return best_val_acc
    
    def optimize(self, n_trials=50, timeout=3600):
        """Run optimization"""
        
        # Create study
        sampler = optuna.samplers.TPESampler(seed=config.DATASET_CONFIG['random_seed'])
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            )
        )
        
        print(f"Starting Optuna optimization with {n_trials} trials...")
        print(f"Timeout: {timeout} seconds")
        
        # Optimize
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            catch=(ValueError,)
        )
        
        # Print results
        print(f"\nBest trial value: {study.best_value:.4f}")
        print("\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        return study