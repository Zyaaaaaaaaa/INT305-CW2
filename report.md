# 2. Model Training and Performance Analysis

## 2.1 Base CNN vs Improved CNN Results

### 2.1.1 Base CNN Model
The baseline CNN achieved moderate performance on the MNIST dataset:

- Training dynamics over 10 epochs:
  - Initial accuracy: 92.61% (Epoch 1)
  - Peak accuracy: 94.63% (Epoch 7)
  - Final accuracy: 92.56% (Epoch 10)
- Training showed significant fluctuations
- Loss values varied considerably during training

### 2.1.2 Improved CNN Performance
We implemented three improvements to the base CNN:

1. **Deep CNN**:
   - Added additional convolutional layers
   - Best accuracy: 97.40%
   - More stable training curve

2. **BatchNorm CNN**:
   - Integrated batch normalization layers
   - Best accuracy: 99.09%
   - Fastest convergence rate

3. **Wide CNN**:
   - Increased channel width
   - Best accuracy: 99.41%
   - Highest overall performance

## 2.2 Performance Analysis

### 2.2.1 Comparative Results
| Model | Best Accuracy | Final Accuracy | Training Stability |
|-------|---------------|----------------|-------------------|
| Base CNN | 94.63% | 92.56% | Moderate |
| Deep CNN | 97.40% | 97.15% | Good |
| BatchNorm CNN | 99.09% | 98.98% | Excellent |
| Wide CNN | 99.41% | 99.27% | Very Good |

### 2.2.2 Analysis of Improvements
- BatchNorm significantly stabilized training
- Width expansion improved feature representation
- Deep architecture enhanced feature extraction
- All improvements showed substantial gains over baseline

# 3. MobileViT: A Hybrid Approach for Performance Enhancement

## 3.1 Proposed Method

We propose using MobileViT, a hybrid architecture combining CNNs and Vision Transformers, to improve classification performance while maintaining model efficiency.

### 3.1.1 Architecture Overview
- Initial CNN layers for local feature extraction
- Transformer blocks for global feature modeling
- Lightweight design with efficient attention mechanisms
- Balanced parameter count for resource efficiency

## 3.2 Implementation Details

Our MobileViT implementation includes:
```python
# Key architecture components
- Conv layers: 32 -> 64 -> 128 channels
- Transformer blocks with 4 attention heads
- Dropout rate: 0.1
- Learning rate: 0.001
- Batch size: 128
```

## 3.3 Comparative Analysis

### 3.3.1 Performance Comparison
| Model | Best Accuracy | Parameter Efficiency | Training Time |
|-------|---------------|---------------------|---------------|
| CNN (BatchNorm) | 99.09% | Moderate | Fast |
| CNN (Wide) | 99.41% | Low | Fast |
| MobileViT | 97.70% | High | Moderate |

### 3.3.2 Key Findings
1. **Performance Characteristics**:
   - MobileViT achieved competitive accuracy (97.70%)
   - More stable training compared to base CNN
   - Better generalization capabilities

2. **Efficiency Analysis**:
   - Balanced parameter usage
   - Effective feature extraction
   - Good scalability potential

3. **Trade-offs**:
   - Slightly lower accuracy than specialized CNNs
   - Better parameter efficiency
   - More flexible architecture for transfer learning

## 3.4 Discussion

The MobileViT approach offers several advantages:

1. **Architecture Benefits**:
   - Combines local and global feature processing
   - Efficient parameter utilization
   - Flexible design for different tasks

2. **Performance Trade-offs**:
   - Competitive accuracy with fewer parameters
   - More stable training process
   - Better generalization potential

3. **Future Improvements**:
   - Fine-tuning attention mechanisms
   - Optimizing architecture dimensions
   - Exploring hybrid training strategies

## 3.5 Conclusion

While MobileViT didn't achieve the highest accuracy (97.70% vs 99.41% for Wide CNN), it demonstrates several important advantages:
- Better parameter efficiency
- More stable training
- Potential for transfer learning

These characteristics make it a promising approach for scenarios where resource efficiency and model flexibility are important considerations alongside raw performance metrics.