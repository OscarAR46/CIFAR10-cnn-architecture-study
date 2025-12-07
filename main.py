"""
Main execution script for CIFAR-10 CNN experiments
Module: CETM26 - Machine Learning and Data Mining
Assignment 2: Research Project
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from sklearn.metrics import precision_score, recall_score, f1_score
import gc

# GPU memory configuration for Colab
print("Configuring GPU memory")
tf.keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU Available: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU available, using CPU")

# Import project modules
import config
from data_utils import DataPreprocessor
from models import get_model
from training import train_model, create_ensemble, ensemble_predict
from experiments import (
    run_architecture_comparison,
    run_regularization_study,
    run_optimizer_comparison,
    run_batch_size_study,
    run_pca_analysis,
    cleanup_memory
)
from visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_comparison_results,
    plot_pca_variance
)
from optuna_optimization import HyperparameterOptimizer

# Set random seeds
np.random.seed(config.DATASET_CONFIG['random_seed'])
tf.random.set_seed(config.DATASET_CONFIG['random_seed'])


def evaluate_model(model, x_test, y_test, model_name="Model"):
    """Comprehensive model evaluation"""
    
    # Basic evaluation
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    # Detailed predictions
    y_pred = model.predict(x_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Additional metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
    
    # Inference time
    import time
    start_time = time.time()
    _ = model.predict(x_test[:100], verbose=0)
    inference_time = (time.time() - start_time) / 100
    
    # Model size
    model_size_mb = model.count_params() * 4 / (1024 * 1024)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS - {model_name}")
    print(f"{'='*60}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Inference Time: {inference_time*1000:.2f} ms/sample")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print(f"Parameters: {model.count_params():,}")
    
    # Generate confusion matrix for evaluated model
    plot_confusion_matrix(y_true_classes, y_pred_classes,
                         class_names=config.DATASET_CONFIG['class_names'],
                         model_name=model_name)
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'inference_time': inference_time,
        'model_size': model_size_mb,
        'parameters': model.count_params(),
        'y_pred': y_pred_classes,
        'y_true': y_true_classes
    }


def run_architecture_comparison_with_plots(x_train, y_train, x_val, y_val, x_test, y_test):
    """Compare different architectures with visualization"""
    
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
        
        # Train with ADAM
        start_time = time.time()
        history = train_model(
            model, x_train, y_train, x_val, y_val,
            strategy='adam',
            epochs=epochs,
            batch_size=128,
            use_augmentation=True,
            use_mixup=False,
            model_name=arch_name,
            verbose=0
        )
        training_time = time.time() - start_time
        
        # Plot training history for this model
        plot_training_history(history, model_name=arch_name)
        
        # Evaluate
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        
        # Generate confusion matrix for this model
        y_pred = model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        plot_confusion_matrix(y_true_classes, y_pred_classes,
                            class_names=config.DATASET_CONFIG['class_names'],
                            model_name=arch_name)
        
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


def create_ensemble_with_plots(x_train, y_train, x_val, y_val, x_test, y_test, n_models=3, model_types=None):
    """Create an ensemble of models with visualization"""
    
    if model_types is None:
        model_types = config.EXPERIMENT_CONFIGS['ensemble']['model_types']
    
    ensemble_models = []
    ensemble_histories = []
    
    print(f"\nCreating ensemble with {n_models} models")
    
    for i in range(n_models):
        # Select model type
        model_type = model_types[i % len(model_types)]
        print(f"\nTraining ensemble model {i+1}/{n_models}: {model_type}")
        
        # Create and thetrain model
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
        
        # Plot training history for ensemble member
        plot_training_history(history, model_name=f'ensemble_{model_type}_{i}')
        
        ensemble_models.append(model)
        ensemble_histories.append(history)
        
        # Print validation accuracy
        val_acc = max(history.history['val_accuracy'])
        print(f"Model {i+1} best validation accuracy: {val_acc:.4f}")
    
    return ensemble_models, ensemble_histories


def main():
    """Main execution function"""
    
    print("="*80)
    print("CIFAR-10 CNN CLASSIFICATION EXPERIMENTS")
    print(f"Timestamp: {config.RUN_TIMESTAMP}")
    print("="*80)
    
    # Load and preprocess data
    print("\n" + "="*60)
    print("DATA LOADING AND PREPROCESSING")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    x_train, y_train, x_val, y_val, x_test, y_test = preprocessor.load_and_preprocess(
        apply_zca=False,
        apply_pca=True  # Enable PCA for analysis
    )
    
    # Initialize results storage
    all_results = []
    
    # ====================================================================
    # 1. BASELINE EXPERIMENTS
    # ====================================================================
    
    print("\n" + "="*60)
    print("BASELINE EXPERIMENTS")
    print("="*60)
    
    # Regularization study
    reg_results = run_regularization_study(x_train, y_train, x_val, y_val, x_test, y_test)
    reg_results.to_csv(f'outputs/results/regularization_results_{config.RUN_TIMESTAMP}.csv', index=False)
    cleanup_memory()
    
    # Optimizer comparison
    opt_results = run_optimizer_comparison(x_train, y_train, x_val, y_val, x_test, y_test)
    opt_results.to_csv(f'outputs/results/optimizer_results_{config.RUN_TIMESTAMP}.csv', index=False)
    cleanup_memory()
    
    # Batch size study
    batch_results = run_batch_size_study(x_train, y_train, x_val, y_val, x_test, y_test)
    batch_results.to_csv(f'outputs/results/batch_size_results_{config.RUN_TIMESTAMP}.csv', index=False)
    cleanup_memory()
    
    # ====================================================================
    # 2. PCA ANALYSIS
    # ====================================================================
    
    print("\n" + "="*60)
    print("PCA ANALYSIS")
    print("="*60)
    
    pca_results = run_pca_analysis(x_train, y_train, x_test, y_test)
    plot_pca_variance(pca_results['explained_variance'])
    
    # Save PCA results
    with open(f'outputs/results/pca_analysis_{config.RUN_TIMESTAMP}.json', 'w') as f:
        json.dump({
            'n_components': pca_results['n_components'],
            'test_accuracy': float(pca_results['test_accuracy']),
            'explained_variance_90': int(np.argmax(pca_results['explained_variance'] >= 0.9) + 1),
            'explained_variance_95': int(np.argmax(pca_results['explained_variance'] >= 0.95) + 1),
            'explained_variance_99': int(np.argmax(pca_results['explained_variance'] >= 0.99) + 1)
        }, f, indent=2)
    cleanup_memory()
    
    # ====================================================================
    # 3. ARCHITECTURE COMPARISON
    # ====================================================================
    
    arch_results = run_architecture_comparison_with_plots(x_train, y_train, x_val, y_val, x_test, y_test)
    arch_results.to_csv(f'outputs/results/architecture_comparison_{config.RUN_TIMESTAMP}.csv', index=False)
    
    # Plot comparisons
    plot_comparison_results(arch_results)
    cleanup_memory()
    
    # ====================================================================
    # 4. ENSEMBLE METHODS (WITH ALL NECC PLOTS)
    # ====================================================================
    
    print("\n" + "="*60)
    print("ENSEMBLE METHODS")
    print("="*60)
    
    ensemble_models, ensemble_histories = create_ensemble_with_plots(
        x_train, y_train, x_val, y_val, x_test,
        n_models=config.EXPERIMENT_CONFIGS['ensemble']['n_models']
    )
    
    # Evaluate ensemble with soft voting
    ensemble_pred = ensemble_predict(ensemble_models, x_test, voting='soft')
    ensemble_acc = np.mean(np.argmax(ensemble_pred, axis=1) == np.argmax(y_test, axis=1))
    
    # Generate confusion matrix for soft voting ensemble
    ensemble_pred_classes = np.argmax(ensemble_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    plot_confusion_matrix(y_true_classes, ensemble_pred_classes,
                        class_names=config.DATASET_CONFIG['class_names'],
                        model_name="Ensemble_Soft_Voting")
    
    print(f"\nEnsemble Test Accuracy (soft voting): {ensemble_acc:.4f}")
    
    # Try hard voting
    ensemble_pred_hard = ensemble_predict(ensemble_models, x_test, voting='hard')
    ensemble_acc_hard = np.mean(np.argmax(ensemble_pred_hard, axis=1) == np.argmax(y_test, axis=1))
    
    # Generate confusion matrix for hard voting ensemble
    ensemble_pred_hard_classes = np.argmax(ensemble_pred_hard, axis=1)
    plot_confusion_matrix(y_true_classes, ensemble_pred_hard_classes,
                        class_names=config.DATASET_CONFIG['class_names'],
                        model_name="Ensemble_Hard_Voting")
    
    print(f"Ensemble Test Accuracy (hard voting): {ensemble_acc_hard:.4f}")
    cleanup_memory()
    
    # ====================================================================
    # 5. OPTUNA HYPERPARAMETER OPTIMIZATION
    # ====================================================================
    
    print("\n" + "="*60)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    
    optimizer = HyperparameterOptimizer(x_train, y_train, x_val, y_val)
    study = optimizer.optimize(n_trials=20, timeout=1800)  # 30 minutes timeout
    
    if optimizer.best_model:
        best_model = optimizer.best_model['model']
        print(f"\nRetraining best model with full epochs...")
        
        # Retrain with more epochs
        history = train_model(
            best_model, x_train, y_train, x_val, y_val,
            strategy='one_cycle',
            epochs=100,
            use_augmentation=True,
            model_name='optuna_best',
            verbose=0
        )
        
        # ADD: Plot training history for Optuna best model
        plot_training_history(history, model_name="Optuna_Best")
        
        # Evaluate INCLUDE matrix generation
        optuna_metrics = evaluate_model(best_model, x_test, y_test, "Optuna_Optimized")
        
        # Save best parameters
        with open(f'outputs/results/optuna_best_params_{config.RUN_TIMESTAMP}.json', 'w') as f:
            json.dump(optimizer.best_model['params'], f, indent=2)
    
    cleanup_memory()
    
    # ====================================================================
    # 6. FINAL RESULTS SUMMARY
    # ====================================================================
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Results saved to outputs/results/")
    print(f"Plots saved to outputs/plots/")
    print(f"Models saved to outputs/models/")
    
    # Create final summary
    summary = {
        'timestamp': config.RUN_TIMESTAMP,
        'best_architecture': arch_results.iloc[0]['Architecture'] if len(arch_results) > 0 else 'N/A',
        'best_test_accuracy': float(arch_results.iloc[0]['Test_Accuracy']) if len(arch_results) > 0 else 0,
        'ensemble_accuracy_soft': float(ensemble_acc),
        'ensemble_accuracy_hard': float(ensemble_acc_hard),
        'pca_accuracy': float(pca_results['test_accuracy']),
        'optuna_best_accuracy': float(optimizer.best_model['accuracy']) if optimizer.best_model else None
    }
    
    with open(f'outputs/results/final_summary_{config.RUN_TIMESTAMP}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nFinal Summary:")
    for key, value in summary.items():
        if value is not None:
            print(f"  {key}: {value}")
    
    # List all generated plots
    print("\nGenerated Plots:")
    if os.path.exists('outputs/plots'):
        plot_files = sorted(os.listdir('outputs/plots'))
        for plot_file in plot_files:
            print(f"  - {plot_file}")
    
    return summary

if __name__ == "__main__":
    print("\nStarting CIFAR-10 CNN Classification Experiments")
    print("This will take time. Wait\n")
    
    results = main()
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)