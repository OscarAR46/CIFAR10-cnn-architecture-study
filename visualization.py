"""
Visualization utilities for results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import config

def plot_training_history(history, model_name="Model"):
    """Plot training and validation curves"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    # Accuracy plot
    axes[0].plot(epochs, history.history['accuracy'], 'b-', 
                label='Training', linewidth=2)
    axes[0].plot(epochs, history.history['val_accuracy'], 'r-', 
                label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].set_title(f'{model_name} - Model Accuracy', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(epochs, history.history['loss'], 'b-', 
                label='Training', linewidth=2)
    axes[1].plot(epochs, history.history['val_loss'], 'r-', 
                label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].set_title(f'{model_name} - Model Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = f'outputs/plots/{model_name.replace(" ", "_")}_history.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=None, model_name="Model"):
    """Plot confusion matrix heatmap"""
    
    if class_names is None:
        class_names = config.DATASET_CONFIG['class_names']
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Proportion'})
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    
    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    save_path = f'outputs/plots/{model_name.replace(" ", "_")}_confusion.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm


def plot_comparison_results(results_df, metric='Test_Accuracy'):
    """Plot comparison of different models"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sort by metric
    results_df = results_df.sort_values(metric, ascending=False)
    
    # Bar plot of accuracy
    models = results_df.iloc[:, 0].values  # First column should be model names
    accuracies = results_df[metric].values
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    bars = axes[0].bar(range(len(models)), accuracies, color=colors)
    
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].set_ylabel(metric.replace('_', ' '), fontsize=11)
    axes[0].set_title(f'Model Performance Comparison', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Efficiency plot (if inference time available)
    if 'Inference_Time' in results_df.columns:
        axes[1].scatter(results_df['Inference_Time']*1000, 
                       results_df[metric], s=100, alpha=0.6)
        
        for i, model in enumerate(models):
            axes[1].annotate(model, 
                           (results_df['Inference_Time'].iloc[i]*1000,
                            results_df[metric].iloc[i]),
                           fontsize=8, ha='right')
        
        axes[1].set_xlabel('Inference Time (ms)', fontsize=11)
        axes[1].set_ylabel(metric.replace('_', ' '), fontsize=11)
        axes[1].set_title('Accuracy vs Inference Speed Trade-off', fontsize=12)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_pca_variance(explained_variance):
    """Plot PCA explained variance"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    n_components = len(explained_variance)
    
    # Individual explained variance
    axes[0].bar(range(1, n_components + 1), 
               explained_variance[0] if len(explained_variance) == 1 else np.diff(np.concatenate([[0], explained_variance])),
               color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Principal Component', fontsize=11)
    axes[0].set_ylabel('Explained Variance Ratio', fontsize=11)
    axes[0].set_title('Individual Explained Variance', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Cumulative explained variance
    axes[1].plot(range(1, n_components + 1), explained_variance, 
                'b-', linewidth=2, marker='o', markersize=4)
    axes[1].axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
    axes[1].axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
    axes[1].set_xlabel('Number of Components', fontsize=11)
    axes[1].set_ylabel('Cumulative Explained Variance', fontsize=11)
    axes[1].set_title('Cumulative Explained Variance', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/plots/pca_variance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig