"""
Data utilities for CIFAR-10 dataset
Handles loading, preprocessing, augmentation
"""

import numpy as np
import tensorflow as tf
import keras
from keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import utils
from sklearn.decomposition import PCA
import config

np.random.seed(config.DATASET_CONFIG['random_seed'])
tf.random.set_seed(config.DATASET_CONFIG['random_seed'])

class DataPreprocessor:
    """Main data preprocessing class"""
    
    def __init__(self):
        self.class_names = config.DATASET_CONFIG['class_names']
        self.pca_model = None
        
    def load_and_preprocess(self, apply_zca=False, apply_pca=False):
        """Load CIFAR-10 and apply preprocessing"""
        print("Loading CIFAR-10 dataset...")
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        # Normalize pixel values
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Create validation split
        val_split = config.DATASET_CONFIG['validation_split']
        val_size = int(len(x_train) * val_split)
        
        x_val = x_train[-val_size:]
        y_val = y_train[-val_size:]
        x_train = x_train[:-val_size]
        y_train = y_train[:-val_size]
        
        # Apply ZCA whitening if requested
        if apply_zca:
            print("Applying ZCA whitening...")
            x_train = self._apply_zca_whitening(x_train)
            x_val = self._apply_zca_whitening(x_val)
            x_test = self._apply_zca_whitening(x_test)
        
        # Apply PCA for visualization/analysis if requested
        if apply_pca:
            print("Extracting PCA features for analysis...")
            self._fit_pca(x_train)
        
        # Convert to categorical
        num_classes = config.DATASET_CONFIG['num_classes']
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_val = keras.utils.to_categorical(y_val, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        
        print(f"Training samples: {len(x_train)}")
        print(f"Validation samples: {len(x_val)}")
        print(f"Test samples: {len(x_test)}")
        
        return x_train, y_train, x_val, y_val, x_test, y_test
    
    def _apply_zca_whitening(self, x_data, epsilon=1e-5):
        """Apply ZCA whitening"""
        orig_shape = x_data.shape
        x_flat = x_data.reshape(x_data.shape[0], -1)
        
        mean = np.mean(x_flat, axis=0)
        x_centered = x_flat - mean
        
        cov = np.dot(x_centered.T, x_centered) / x_centered.shape[0]
        U, S, V = np.linalg.svd(cov)
        
        zca_matrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T))
        x_zca = np.dot(x_centered, zca_matrix)
        
        return x_zca.reshape(orig_shape)
    
    def _fit_pca(self, x_train):
        """Fit PCA model for feature extraction"""
        # Flatten images for PCA
        x_flat = x_train.reshape(x_train.shape[0], -1)
        
        # Fit PCA
        n_components = min(config.PCA_CONFIG['n_components'], x_flat.shape[1])
        self.pca_model = PCA(n_components=n_components)
        self.pca_model.fit(x_flat[:config.PCA_CONFIG['n_samples_viz']])
        
        # Calculate explained variance
        explained_var = np.cumsum(self.pca_model.explained_variance_ratio_)
        print(f"PCA with {n_components} components explains {explained_var[-1]:.2%} of variance")
        
        return self.pca_model
    
    def extract_pca_features(self, x_data):
        """Extract PCA features from data"""
        if self.pca_model is None:
            raise ValueError("PCA model not fitted. Run load_and_preprocess with apply_pca=True first")
        
        x_flat = x_data.reshape(x_data.shape[0], -1)
        return self.pca_model.transform(x_flat)
    
    def get_augmentation_generator(self, level='basic'):
        """Get data augmentation generator"""
        aug_config = config.AUGMENTATION_CONFIGS.get(level, config.AUGMENTATION_CONFIGS['basic'])
        
        return ImageDataGenerator(
            rotation_range=aug_config.get('rotation_range', 15),
            width_shift_range=aug_config.get('width_shift_range', 0.1),
            height_shift_range=aug_config.get('height_shift_range', 0.1),
            horizontal_flip=aug_config.get('horizontal_flip', True),
            zoom_range=aug_config.get('zoom_range', 0.15),
            shear_range=aug_config.get('shear_range', 0),
            brightness_range=aug_config.get('brightness_range', None),
            fill_mode='reflect'
        )


def mixup_batch(x, y, alpha=2.0):
    """Apply MixUp augmentation to a batch"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = len(x)
    index = np.random.permutation(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutout_batch(images, n_holes=1, length=16):
    """Apply CutOut augmentation to a batch"""
    batch_size, h, w, c = images.shape
    mask = np.ones((batch_size, h, w, c), np.float32)
    
    for n in range(batch_size):
        for _ in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)
            
            mask[n, y1:y2, x1:x2, :] = 0.
    
    return images * mask


class MixUpGenerator(keras.utils.Sequence):
    """Custom generator for MixUp augmentation"""
    
    def __init__(self, x, y, batch_size=32, alpha=2.0, datagen=None, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.alpha = alpha
        self.datagen = datagen
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.x))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x_batch = self.x[indexes]
        y_batch = self.y[indexes]
        
        # Apply standrd augmentation
        if self.datagen:
            for i in range(len(x_batch)):
                x_batch[i] = self.datagen.random_transform(x_batch[i])
        
        # Apply MixUp
        if self.alpha > 0:
            x_batch, y_a, y_b, lam = mixup_batch(x_batch, y_batch, self.alpha)
            y_batch = lam * y_a + (1 - lam) * y_b
        
        return x_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)