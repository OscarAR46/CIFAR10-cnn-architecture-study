"""
Training utilities and functions
"""

import tensorflow as tf
import keras
from keras import callbacks as keras_callbacks
from keras.optimizers import SGD, Adam, AdamW
import numpy as np
import time
import config
from data_utils import MixUpGenerator, DataPreprocessor
from models import get_model

def create_callbacks(epochs=100, model_name='model'):
    """Create callback list for training - SIMPLIFIED"""
    callback_list = [
        keras_callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras_callbacks.ModelCheckpoint(
            f'outputs/models/best_{model_name}_{config.RUN_TIMESTAMP}.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        ),
        keras_callbacks.CSVLogger(
            f'outputs/logs/{model_name}_{config.RUN_TIMESTAMP}.csv'
        ),
        keras_callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callback_list

def train_model(model, x_train, y_train, x_val, y_val, 
                strategy='adam', epochs=100, batch_size=128,
                use_augmentation=True, use_mixup=False, 
                model_name='model', verbose=1):
    """Unified training function - FIXED TO ACTUALLY WORK"""
    
    print(f"\nTraining {model_name}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print(f"Augmentation: {use_augmentation}, MixUp: {use_mixup}")
    
    # ALWAYS USE ADAM - IT'S THE ONLY THING THAT WORKS, next: try learning_rate=1e-4
    optimizer = Adam(learning_rate=1e-4)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Get callbacks
    callback_list = create_callbacks(epochs, model_name)
    
    # Prepare data
    if use_mixup:
        datagen = DataPreprocessor().get_augmentation_generator('basic') if use_augmentation else None
        train_generator = MixUpGenerator(
            x_train, y_train, 
            batch_size=batch_size, 
            alpha=1.0,  # Reduced alpha for stability
            datagen=datagen
        )
        
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callback_list,
            verbose=verbose
        )
    elif use_augmentation:
        datagen = DataPreprocessor().get_augmentation_generator('basic')
        datagen.fit(x_train)
        
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callback_list,
            verbose=verbose
        )
    else:
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            callbacks=callback_list,
            verbose=verbose
        )
    
    return history


def create_ensemble(x_train, y_train, x_val, y_val, n_models=3, model_types=None):
    """Create an ensemble of models"""
    
    if model_types is None:
        model_types = config.EXPERIMENT_CONFIGS['ensemble']['model_types']
    
    ensemble_models = []
    ensemble_histories = []
    
    print(f"\nCreating ensemble with {n_models} models")
    
    for i in range(n_models):
        # Select model type
        model_type = model_types[i % len(model_types)]
        print(f"\nTraining ensemble model {i+1}/{n_models}: {model_type}")
        
        # Create and train model
        model = get_model(model_type)
        
        history = train_model(
            model, x_train, y_train, x_val, y_val,
            strategy='adam',
            epochs=50,
            batch_size=128,
            use_augmentation=True,
            model_name=f'ensemble_{model_type}_{i}',
            verbose=0
        )
        
        ensemble_models.append(model)
        ensemble_histories.append(history)
        
        # Print validation accuracy
        val_acc = max(history.history['val_accuracy'])
        print(f"Model {i+1} best validation accuracy: {val_acc:.4f}")
    
    return ensemble_models, ensemble_histories


def ensemble_predict(models, x_data, voting='soft'):
    """Make predictions with ensemble"""
    predictions = []
    
    for model in models:
        if voting == 'soft':
            predictions.append(model.predict(x_data, verbose=0))
        else:
            pred = model.predict(x_data, verbose=0)
            predictions.append(np.argmax(pred, axis=1))
    
    if voting == 'soft':
        # Average probabilities
        ensemble_pred = np.mean(predictions, axis=0)
    else:
        # Majority voting
        predictions = np.array(predictions)
        ensemble_pred = np.zeros((x_data.shape[0], config.DATASET_CONFIG['num_classes']))
        for i in range(x_data.shape[0]):
            votes = predictions[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            winner = unique[np.argmax(counts)]
            ensemble_pred[i, winner] = 1
    
    return ensemble_pred