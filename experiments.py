"""
Experiment runners for systematic studies
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, models
from keras.optimizers import Adam
import time
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import config
from models import get_model, create_vgg_style
from training import train_model
from data_utils import DataPreprocessor
import gc

def cleanup_memory():
    """Clean up memory between experiments"""
    tf.keras.backend.clear_session()
    gc.collect()

def run_architecture_comparison(x_train, y_train, x_val, y_val, x_test, y_test):
    """Compare different architectures USE ADAM"""
    
    architectures = [
        ('baseline', 60),
        ('vgg_style', 80),
        ('resnet', 80),
        ('attention', 80),
        ('densenet', 80),
        ('inception', 70),
        ('wide_resnet', 80)
    ]
    
    results = []
    
    for arch_name, epochs in architectures:
        print(f"\n{'='*60}")
        print(f"Testing {arch_name} architecture")
        print(f"{'='*60}")
        
        # Clear memory before each model
        cleanup_memory()
        
        # Create model
        model = get_model(arch_name)
        params = model.count_params()
        print(f"Parameters: {params:,}")
        
        # Train with ADAM (the only thing that works!!!!)
        start_time = time.time()
        history = train_model(
            model, x_train, y_train, x_val, y_val,
            strategy='adam',  # ALWAYS USE ADAM
            epochs=epochs,
            batch_size=128,
            use_augmentation=True,
            use_mixup=False,  # Disable mixup for stability
            model_name=arch_name,
            verbose=0
        )
        training_time = time.time() - start_time
        
        # Evaluate
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        
        # Inference time
        start_time = time.time()
        _ = model.predict(x_test[:100], verbose=0)
        inference_time = (time.time() - start_time) / 100
        
        results.append({
            'Architecture': arch_name,
            'Parameters': params,
            'Test_Accuracy': test_acc,
            'Training_Time': training_time,
            'Inference_Time': inference_time,
            'Best_Val_Acc': max(history.history['val_accuracy']),
            'Epochs': epochs
        })
        
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Training Time: {training_time:.2f}s")
        print(f"Inference Time: {inference_time*1000:.2f}ms")
    
    return pd.DataFrame(results)


def run_regularization_study(x_train, y_train, x_val, y_val, x_test, y_test):
    """Study different regularization techniques - FIXED"""
    
    results = []
    cfg = config.EXPERIMENT_CONFIGS['regularization_study']
    
    print("\n" + "="*60)
    print("REGULARIZATION STUDY")
    print("="*60)
    
    # Test Dropout rates with baseline model
    for dropout in cfg['dropout_rates']:
        print(f"\nTesting dropout rate: {dropout}")
        
        cleanup_memory()
        
        # Create baseline model with specific dropout
        model = models.Sequential([
            layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(32, 32, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(dropout),
            
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(min(dropout * 1.5, 0.99) if dropout > 0 else 0),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(min(dropout * 2, 0.99) if dropout > 0 else 0),
            layers.Dense(10, activation='softmax')
        ])
        
        # Use Adam optimizer directly
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train with basic augmentation
        datagen = DataPreprocessor().get_augmentation_generator('basic')
        datagen.fit(x_train)
        
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=128),
            epochs=40,  # Increased epochs this is optimal no.
            validation_data=(x_val, y_val),
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ],
            verbose=0
        )
        
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        
        results.append({
            'Type': 'Dropout',
            'Value': dropout,
            'Test_Accuracy': test_acc,
            'Val_Accuracy': max(history.history['val_accuracy'])
        })
        
        print(f"Test Accuracy: {test_acc:.4f}")
    
    # Test L2 regularization
    for l2_val in cfg['l2_values']:
        print(f"\nTesting L2 regularization: {l2_val}")
        
        cleanup_memory()
        
        # Create model with L2
        from keras import regularizers
        
        model = models.Sequential([
            layers.Conv2D(32, 3, padding='same', activation='relu', 
                         kernel_regularizer=regularizers.l2(l2_val),
                         input_shape=(32, 32, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, 3, padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(l2_val)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.2),
            
            layers.Conv2D(64, 3, padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(l2_val)),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(l2_val)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Dropout(0.3),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l2(l2_val)),
            layers.Dropout(0.4),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        datagen = DataPreprocessor().get_augmentation_generator('basic')
        datagen.fit(x_train)
        
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=128),
            epochs=40,
            validation_data=(x_val, y_val),
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ],
            verbose=0
        )
        
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        
        results.append({
            'Type': 'L2',
            'Value': l2_val,
            'Test_Accuracy': test_acc,
            'Val_Accuracy': max(history.history['val_accuracy'])
        })
        
        print(f"Test Accuracy: {test_acc:.4f}")
    
    return pd.DataFrame(results)


def run_optimizer_comparison(x_train, y_train, x_val, y_val, x_test, y_test):
    """Compare different optimizers - THIS ONE ALREADY WORKS"""
    
    results = []
    cfg = config.EXPERIMENT_CONFIGS['optimizer_comparison']
    
    print("\n" + "="*60)
    print("OPTIMIZER COMPARISON")
    print("="*60)
    
    for opt_name in cfg['optimizers']:
        print(f"\nTesting {opt_name} optimizer")
        
        cleanup_memory()
        
        model = get_model('baseline')
        
        # Get optimizer
        if opt_name == 'sgd':
            from keras.optimizers import SGD
            optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        elif opt_name == 'adam':
            optimizer = Adam(learning_rate=0.001)
        else:  # adamw
            from keras.optimizers import AdamW
            optimizer = AdamW(learning_rate=0.001, weight_decay=1e-4)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = model.fit(
            x_train, y_train,
            batch_size=128,
            epochs=cfg['epochs'],
            validation_data=(x_val, y_val),
            verbose=0
        )
        
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        
        results.append({
            'Optimizer': opt_name,
            'Test_Accuracy': test_acc,
            'Best_Val_Accuracy': max(history.history['val_accuracy']),
            'Convergence_Epoch': np.argmax(history.history['val_accuracy']) + 1
        })
        
        print(f"Test Accuracy: {test_acc:.4f}")
    
    return pd.DataFrame(results)


def run_batch_size_study(x_train, y_train, x_val, y_val, x_test, y_test):
    """Study effect of different batch sizes - FIXED"""
    
    results = []
    cfg = config.EXPERIMENT_CONFIGS['batch_size_study']
    
    print("\n" + "="*60)
    print("BATCH SIZE STUDY")
    print("="*60)
    
    for batch_size in cfg['batch_sizes']:
        print(f"\nTesting batch size: {batch_size}")
        
        cleanup_memory()
        
        model = get_model('baseline')
        
        # Always use Adam
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        start_time = time.time()
        
        # Train with augmentation
        datagen = DataPreprocessor().get_augmentation_generator('basic')
        datagen.fit(x_train)
        
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs=40,  # Increased epochs do not change
            validation_data=(x_val, y_val),
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
            ],
            verbose=0
        )
        
        training_time = time.time() - start_time
        
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        
        results.append({
            'Batch_Size': batch_size,
            'Test_Accuracy': test_acc,
            'Training_Time': training_time,
            'Best_Val_Accuracy': max(history.history['val_accuracy'])
        })
        
        print(f"Test Accuracy: {test_acc:.4f}, Time: {training_time:.2f}s")
    
    return pd.DataFrame(results)


def run_pca_analysis(x_train, y_train, x_test, y_test):
    """Analyze data using PCA - THIS ONE IS FINE"""
    
    print("\n" + "="*60)
    print("PCA ANALYSIS")
    print("="*60)
    
    cleanup_memory()
    
    # Fit PCA on training data
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    n_samples = min(config.PCA_CONFIG['n_samples_viz'], len(x_train))
    
    pca = PCA(n_components=config.PCA_CONFIG['n_components'])
    pca.fit(x_train_flat[:n_samples])
    
    # Transform data
    x_train_pca = pca.transform(x_train_flat)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    x_test_pca = pca.transform(x_test_flat)
    
    # Calculate explained variance
    explained_var = np.cumsum(pca.explained_variance_ratio_)
    
    print(f"Components needed for 90% variance: {np.argmax(explained_var >= 0.9) + 1}")
    print(f"Components needed for 95% variance: {np.argmax(explained_var >= 0.95) + 1}")
    print(f"Components needed for 99% variance: {np.argmax(explained_var >= 0.99) + 1}")
    
    # Train simple model on PCA features
    pca_model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(config.PCA_CONFIG['n_components'],)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(config.DATASET_CONFIG['num_classes'], activation='softmax')
    ])
    
    pca_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nTraining model on PCA features...")
    history = pca_model.fit(
        x_train_pca, y_train,
        batch_size=128,
        epochs=50,
        validation_split=0.1,
        verbose=0
    )
    
    test_loss, test_acc = pca_model.evaluate(x_test_pca, y_test, verbose=0)
    print(f"Test accuracy with PCA features: {test_acc:.4f}")
    
    return {
        'pca_model': pca,
        'explained_variance': explained_var,
        'test_accuracy': test_acc,
        'n_components': config.PCA_CONFIG['n_components']
    }