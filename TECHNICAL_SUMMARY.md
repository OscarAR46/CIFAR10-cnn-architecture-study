# Technical Summary

This document summarizes the methodology, experiments, and key findings from a systematic study of CNN architectures for CIFAR-10 image classification.

## Research Questions

1. How do different CNN architectural paradigms (depth vs. width, residual vs. dense connections) perform on small-image classification?
2. What regularization strategies are most effective when combined with batch normalization?
3. How do modern optimizers (Adam, AdamW, SGD) compare in practice?
4. Can automated hyperparameter optimization discover configurations that outperform manual tuning?

## Methodology

### Experimental Design

The study followed a systematic approach with five experimental tracks, each isolating specific variables while controlling others:

**Track 1 - Baseline Studies**: Established performance benchmarks with a simple 2-block CNN (592K parameters), providing a reference point for architectural improvements.

**Track 2 - Architecture Comparison**: Implemented seven architectures spanning major CNN paradigms—VGG-style depth, ResNet residual connections, DenseNet dense connectivity, Inception multi-scale features, attention mechanisms, and Wide ResNet width scaling.

**Track 3 - Regularization Analysis**: Tested dropout rates (0, 0.2, 0.3) crossed with L2 regularization strengths (0, 1e-4, 1e-3) on the VGG-style architecture to isolate regularization effects.

**Track 4 - Training Dynamics**: Compared optimizers (SGD with momentum, Adam, AdamW) and batch sizes (64, 128, 256) to understand their interaction with architecture.

**Track 5 - Automated HPO**: Applied Optuna with Tree-structured Parzen Estimators to explore the hyperparameter space systematically.

### Implementation Details

All experiments used consistent protocols: 45K/5K/10K train/validation/test splits, pixel normalization to [0,1], random seed 42 for reproducibility, and early stopping with patience of 15 epochs. Data augmentation included horizontal flips, rotations (±15°), width/height shifts (10%), and zoom (10%). Training was performed on Google Colab L4 GPU with TensorFlow 2.19.0.

## Key Findings

### Architecture Performance

Wide ResNet achieved the highest accuracy (90.71%) with 11.0M parameters, validating the hypothesis that network width matters more than depth for small 32×32 images. The architecture's DropBlock regularization and width factor of 8 allowed it to learn richer feature representations without overfitting.

VGG-style (87.15%) demonstrated that classical architectures remain competitive when properly regularized with GELU activations and global average pooling. Its strong performance with only 2.0M parameters suggests diminishing returns from architectural complexity on CIFAR-10.

ResNet and DenseNet underperformed expectations (84.99% and 84.19% respectively), likely because their designs optimize for ImageNet-scale images where gradient flow through 100+ layers is critical. For 32×32 images requiring only 3-4 downsampling stages, these mechanisms add overhead without proportional benefit.

### Regularization Insights

The most surprising finding was that zero dropout consistently outperformed dropout rates of 0.2 and 0.3 across all architectures. With batch normalization present, adding dropout appeared to interfere with batch statistics, degrading both training stability and final accuracy. L2 regularization at 1e-3 provided modest benefits without the negative interaction effects of dropout.

### Optimizer Comparison

AdamW (79.88%) significantly outperformed both Adam (76.24%) and SGD with momentum (74.45%) on the VGG-style baseline. The decoupled weight decay in AdamW proved more effective than L2 regularization embedded in standard SGD, supporting recent theoretical work on the importance of separating gradient descent from weight regularization.

SGD's underperformance contradicted common wisdom that it generalizes better than adaptive methods. This may be explained by the relatively short training runs (100 epochs) and learning rate schedule (ReduceLROnPlateau) used—SGD typically requires longer training with carefully tuned warmup and cosine annealing schedules to reach its potential.

### Ensemble Methods

Soft voting across VGG-style, ResNet, and Attention CNN achieved 86.04% accuracy—only marginally better than individual models. This modest improvement suggests the architectures learned similar underlying features, limiting the diversity benefit that ensemble methods rely on. Future work might explore ensembles of architecturally dissimilar models (e.g., CNNs with transformers).

### Hyperparameter Optimization

Optuna's best configuration discovered several unexpected preferences: 5×5 kernels (contradicting the VGG trend toward 3×3), batch normalization disabled (suggesting it conflicted with other regularization), and dropout at 0.343 (higher than manual experiments found optimal). The retrained model achieved 81.72% accuracy—competitive but not exceeding manually-tuned architectures, indicating that architecture choice dominates hyperparameter selection for this task.

## Computational Efficiency

Inference time analysis revealed important trade-offs: Baseline CNN was fastest (19.48ms) but least accurate; Wide ResNet achieved the best accuracy-to-speed ratio (90.71% at 36.73ms); DenseNet was slowest (109.49ms) due to dense connectivity overhead despite having only 1.3M parameters. For deployment scenarios, VGG-style offers an attractive balance of 87.15% accuracy with moderate inference time (42.51ms) and reasonable model size (2.0M parameters).

## Limitations and Future Work

Several learning rate schedulers (cosine annealing, one-cycle, cosine with restarts) were implemented but produced training failures, likely due to interaction effects with batch normalization and the specific optimizer configurations. The code preserves these implementations with documentation for future debugging.

The study was constrained to CIFAR-10's 32×32 resolution—findings may not transfer to higher-resolution datasets where depth and skip connections become more critical. Future work could explore knowledge distillation from Wide ResNet to smaller architectures, investigate the batch normalization-dropout interaction more systematically, and extend the architecture comparison to transformer-based approaches.

## Reproducibility

All experiments are reproducible using the provided codebase with random seed 42. Training logs, result CSVs, and visualization plots are included in the outputs directory. Model weights are excluded due to size (50MB+ each) but can be regenerated by running main.py with GPU access.