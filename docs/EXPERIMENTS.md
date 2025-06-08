# Experiment Protocols and Analysis Guide

A comprehensive guide for running systematic experiments and analyzing results with the inverted transformer implementation.

## 📋 Table of Contents

1. [Experiment Design](#experiment-design)
2. [Standard Protocols](#standard-protocols)
3. [Comparative Analysis](#comparative-analysis)
4. [Result Interpretation](#result-interpretation)
5. [Visualization and Reporting](#visualization-and-reporting)
6. [Advanced Experiments](#advanced-experiments)

## 🎯 Experiment Design

### Research Questions

Our implementation enables investigation of several key research questions:

1. **Architecture Comparison**: How do inverted transformers compare to classic transformers for stock prediction?
2. **Feature Importance**: What's the impact of different market features (price, volume, transactions)?
3. **Multivariate Modeling**: Do inverted transformers better capture cross-feature correlations?
4. **Scalability**: How do different models perform with varying sequence lengths and model sizes?
5. **Efficiency**: What are the trade-offs between accuracy and computational efficiency?

### Experimental Variables

#### Independent Variables
- **Model Architecture**: iTransformer, iInformer, iReformer, Transformer, Informer
- **Feature Sets**: Single (close), Multi (close+volume+transactions)
- **Sequence Length**: 60, 120, 240 minutes
- **Prediction Horizon**: 15, 30, 60 minutes
- **Model Size**: Small (256), Medium (512), Large (1024) dimensions

#### Dependent Variables
- **Accuracy**: MSE, MAE, MAPE
- **Training Efficiency**: Time per epoch, convergence speed
- **Memory Usage**: Peak GPU/CPU memory
- **Attention Patterns**: Feature vs temporal focus

## 📊 Standard Protocols

### Protocol 1: Basic Model Comparison

**Objective**: Compare inverted vs classic transformers on stock prediction

**Setup**:
```bash
# Run all models with standard configuration
python run_all_experiments.py
```

**Configuration**:
- Models: iTransformer, iInformer, Transformer, Informer
- Features: close, volume, transactions
- Sequence: 60 → 15 minutes
- Epochs: 20
- Batch size: 32

**Expected Results**:
- iTransformer > Transformer (multivariate advantage)
- iInformer ≈ iTransformer (efficiency vs accuracy)
- Multi-feature > Single-feature

### Protocol 2: Feature Ablation Study

**Objective**: Understand the impact of different market features

**Setup**:
```bash
# Single feature
python train.py --model iTransformer --features close

# Two features  
python train.py --model iTransformer --features close,volume

# Three features
python train.py --model iTransformer --features close,volume,transactions
```

**Analysis**:
- Compare MSE reduction with additional features
- Analyze attention patterns for feature importance
- Evaluate computational overhead

### Protocol 3: Sequence Length Analysis

**Objective**: Determine optimal input sequence length

**Setup**:
```bash
# Short sequences
python train.py --model iTransformer --seq-len 30 --pred-len 15

# Medium sequences  
python train.py --model iTransformer --seq-len 60 --pred-len 15

# Long sequences
python train.py --model iTransformer --seq-len 120 --pred-len 15
```

**Metrics**:
- Accuracy vs sequence length
- Training time vs sequence length
- Memory usage vs sequence length

### Protocol 4: Prediction Horizon Study

**Objective**: Evaluate performance across different prediction horizons

**Setup**:
```bash
# Short-term prediction
python train.py --model iTransformer --seq-len 60 --pred-len 15

# Medium-term prediction
python train.py --model iTransformer --seq-len 120 --pred-len 30

# Long-term prediction  
python train.py --model iTransformer --seq-len 240 --pred-len 60
```

**Analysis**:
- Accuracy degradation with longer horizons
- Model confidence and uncertainty
- Feature importance changes over time

## 🔬 Comparative Analysis

### Model Performance Matrix

Create a comprehensive comparison across all dimensions:

| Model | Features | Seq→Pred | MSE | MAE | Time/Epoch | Memory |
|-------|----------|----------|-----|-----|------------|--------|
| iTransformer | close,vol,trans | 60→15 | ? | ? | ? | ? |
| iInformer | close,vol,trans | 60→15 | ? | ? | ? | ? |
| Transformer | close,vol,trans | 60→15 | ? | ? | ? | ? |
| ... | ... | ... | ... | ... | ... | ... |

### Statistical Significance Testing

```python
import scipy.stats as stats
import numpy as np

# Compare model performances
model_a_mse = [0.045, 0.043, 0.047, 0.044, 0.046]  # Multiple runs
model_b_mse = [0.052, 0.051, 0.053, 0.050, 0.054]

# Paired t-test
t_stat, p_value = stats.ttest_rel(model_a_mse, model_b_mse)
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

# Effect size (Cohen's d)
pooled_std = np.sqrt((np.var(model_a_mse) + np.var(model_b_mse)) / 2)
cohens_d = (np.mean(model_a_mse) - np.mean(model_b_mse)) / pooled_std
print(f"Cohen's d: {cohens_d:.4f}")
```

### Attention Pattern Analysis

```python
# Extract and analyze attention patterns
import torch
import matplotlib.pyplot as plt

def analyze_attention_patterns(model, data_loader):
    """Extract and visualize attention patterns"""
    model.eval()
    attention_weights = []
    
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(batch, return_attention=True)
            attention_weights.append(outputs['attention'])
    
    # Average attention across batches
    avg_attention = torch.stack(attention_weights).mean(0)
    
    # Visualize
    plt.figure(figsize=(10, 8))
    plt.imshow(avg_attention.cpu().numpy(), cmap='Blues')
    plt.title('Average Attention Patterns')
    plt.xlabel('Features')
    plt.ylabel('Features')
    plt.colorbar()
    plt.show()
    
    return avg_attention
```

## 📈 Result Interpretation

### Performance Metrics

#### Accuracy Metrics
- **MSE (Mean Squared Error)**: Primary metric for regression
  - Lower is better
  - Sensitive to outliers
  - Units: squared price units

- **MAE (Mean Absolute Error)**: Robust alternative
  - Lower is better  
  - Less sensitive to outliers
  - Units: price units

- **MAPE (Mean Absolute Percentage Error)**: Scale-independent
  - Lower is better
  - Interpretable as percentage error
  - Can be unstable with values near zero

#### Efficiency Metrics
- **Training Time**: Time per epoch in seconds
- **Memory Usage**: Peak GPU memory in GB
- **Convergence Speed**: Epochs to reach best validation loss

### Expected Performance Ranges

Based on financial time series literature:

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| **MAPE** | < 2% | 2-5% | 5-10% | > 10% |
| **MSE** | < 0.01 | 0.01-0.05 | 0.05-0.1 | > 0.1 |
| **MAE** | < 0.5 | 0.5-2.0 | 2.0-5.0 | > 5.0 |

*Note: Actual ranges depend on data scale and normalization*

### Interpretation Guidelines

#### Model Comparison
1. **Significant Improvement**: > 10% MSE reduction + p < 0.05
2. **Marginal Improvement**: 5-10% MSE reduction
3. **Equivalent Performance**: < 5% difference
4. **Consider Efficiency**: Training time and memory trade-offs

#### Feature Analysis
1. **High Impact Features**: > 15% MSE reduction when added
2. **Medium Impact Features**: 5-15% MSE reduction
3. **Low Impact Features**: < 5% MSE reduction
4. **Redundant Features**: No improvement or degradation

## 📊 Visualization and Reporting

### Automated Report Generation

```python
def generate_experiment_report(results_dir):
    """Generate comprehensive experiment report"""
    
    # Load all experiment results
    results = load_all_results(results_dir)
    
    # Create performance comparison
    create_performance_matrix(results)
    
    # Generate learning curves
    plot_learning_curves(results)
    
    # Create attention heatmaps
    plot_attention_patterns(results)
    
    # Generate prediction examples
    plot_prediction_samples(results)
    
    # Statistical analysis
    perform_statistical_tests(results)
    
    # Save comprehensive report
    save_report(results, "experiment_report.html")
```

### Key Visualizations

#### 1. Performance Comparison
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Performance heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(performance_matrix, annot=True, cmap='RdYlGn_r')
plt.title('Model Performance Comparison (MSE)')
plt.xlabel('Feature Sets')
plt.ylabel('Models')
plt.show()
```

#### 2. Learning Curves
```python
# Training progress comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for model in models:
    plt.plot(model.train_losses, label=model.name)
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 3, 2)
for model in models:
    plt.plot(model.val_losses, label=model.name)
plt.title('Validation Loss')
plt.legend()

plt.subplot(1, 3, 3)
for model in models:
    plt.plot(model.test_metrics, label=model.name)
plt.title('Test Performance')
plt.legend()

plt.tight_layout()
plt.show()
```

#### 3. Prediction Quality
```python
# Sample predictions
plt.figure(figsize=(15, 10))

for i, model in enumerate(models):
    plt.subplot(2, 3, i+1)
    plt.plot(true_values, label='True', alpha=0.7)
    plt.plot(model.predictions, label='Predicted', alpha=0.7)
    plt.title(f'{model.name} Predictions')
    plt.legend()

plt.tight_layout()
plt.show()
```

### Report Template

```markdown
# Experiment Report: [Date]

## Executive Summary
- **Best Model**: iTransformer with multi-features
- **Key Finding**: 23% MSE improvement over baseline
- **Recommendation**: Deploy iTransformer for production

## Methodology
- Models tested: 5 architectures
- Feature sets: 2 configurations  
- Evaluation: 5-fold cross-validation
- Metrics: MSE, MAE, MAPE

## Results
### Performance Summary
[Performance table]

### Statistical Analysis
[Significance tests]

### Efficiency Analysis
[Training time and memory usage]

## Conclusions
[Key insights and recommendations]

## Appendix
[Detailed results and code]
```

## 🔬 Advanced Experiments

### Experiment 1: Hyperparameter Optimization

```bash
# Grid search over key parameters
for lr in 0.0001 0.0005 0.001; do
  for d_model in 256 512 1024; do
    for n_heads in 4 8 16; do
      python train.py \
        --model iTransformer \
        --learning-rate $lr \
        --d-model $d_model \
        --n-heads $n_heads \
        --experiment-name "hp_${lr}_${d_model}_${n_heads}"
    done
  done
done
```

### Experiment 2: Robustness Testing

```bash
# Test with different data splits
python train.py --train-size 5 --test-size 3 --val-size 2
python train.py --train-size 10 --test-size 3 --val-size 2  
python train.py --train-size 15 --test-size 3 --val-size 2

# Test with different stocks
python train.py --stocks "AAPL,MSFT"
python train.py --stocks "GOOGL,AMZN"
python train.py --stocks "JPM,BAC"
```

### Experiment 3: Attention Analysis

```python
def attention_analysis_experiment():
    """Comprehensive attention pattern analysis"""
    
    # Train models
    itransformer = train_model("iTransformer")
    transformer = train_model("Transformer")
    
    # Extract attention patterns
    itr_attention = extract_attention(itransformer)
    tr_attention = extract_attention(transformer)
    
    # Compare patterns
    compare_attention_patterns(itr_attention, tr_attention)
    
    # Feature importance analysis
    analyze_feature_importance(itr_attention)
    
    # Temporal vs feature attention
    compare_attention_types(itr_attention, tr_attention)
```

### Experiment 4: Transfer Learning

```bash
# Pre-train on large dataset
python train.py \
  --model iTransformer \
  --train-size 20 \
  --save-checkpoint pretrained_model.pth

# Fine-tune on specific stocks
python train.py \
  --model iTransformer \
  --resume-checkpoint pretrained_model.pth \
  --stocks "AAPL" \
  --learning-rate 0.0001 \
  --train-epochs 10
```

## 📝 Experiment Checklist

### Before Running Experiments
- [ ] Data quality check (no missing files, correct format)
- [ ] Environment setup (GPU availability, dependencies)
- [ ] Baseline results (simple models for comparison)
- [ ] Resource planning (time and compute requirements)

### During Experiments  
- [ ] Monitor training progress (TensorBoard)
- [ ] Check for overfitting (validation curves)
- [ ] Resource usage monitoring (memory, GPU utilization)
- [ ] Intermediate result validation

### After Experiments
- [ ] Statistical significance testing
- [ ] Result reproducibility verification
- [ ] Performance vs efficiency analysis
- [ ] Documentation and reporting
- [ ] Code and data archival

## 🎯 Success Criteria

### Technical Success
- [ ] All models train without errors
- [ ] Reproducible results across runs
- [ ] Statistically significant improvements
- [ ] Reasonable training times (< 2 hours per experiment)

### Scientific Success
- [ ] Clear performance hierarchy established
- [ ] Feature importance quantified
- [ ] Attention patterns analyzed and interpreted
- [ ] Practical recommendations derived

### Practical Success
- [ ] Best model identified for deployment
- [ ] Trade-offs between accuracy and efficiency understood
- [ ] Scalability limitations documented
- [ ] Future research directions identified 
