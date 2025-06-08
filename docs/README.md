# Stock Market Forecasting with iTransformer - Usage Guide

A comprehensive guide for using the inverted transformer implementation for stock market time series forecasting.

## 📋 Table of Contents

1. [Training Interface](#training-interface)
2. [Configuration Management](#configuration-management)
3. [Data Format and Preparation](#data-format-and-preparation)
4. [Running Experiments](#running-experiments)
5. [Model Architectures](#model-architectures)
6. [Results and Analysis](#results-and-analysis)
7. [Advanced Usage](#advanced-usage)
8. [Troubleshooting](#troubleshooting)

## 🎯 Training Interface

### Primary Interface: train.py

The main training interface uses Click for an intuitive command-line experience:

```bash
# Basic usage
python train.py --model iTransformer --features close,volume,transactions

# Full example
python train.py \
  --model iTransformer \
  --features close,volume,transactions \
  --train-epochs 20 \
  --batch-size 32 \
  --learning-rate 0.0005 \
  --seq-len 60 \
  --pred-len 15 \
  --data-dir dataset \
  --resume-checkpoint checkpoints/best_model.pth
```

### Compatible Interface: run.py

For compatibility with original iTransformer scripts:

```bash
python run.py \
  --is_training 1 \
  --model_id STOCK_MULTI \
  --model iTransformer \
  --data stock \
  --features close,volume,transactions \
  --seq_len 60 \
  --pred_len 15 \
  --train_epochs 20
```

### Key Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--model` | Model architecture | `iTransformer` | `iTransformer`, `iInformer`, `Transformer` |
| `--features` | Input features (comma-separated) | `close,volume,transactions` | `close` or `close,volume` |
| `--train-epochs` | Number of training epochs | `20` | `50` |
| `--batch-size` | Training batch size | `32` | `64` |
| `--learning-rate` | Learning rate | `0.0005` | `0.001` |
| `--seq-len` | Input sequence length | `60` | `120` |
| `--pred-len` | Prediction length | `15` | `30` |
| `--data-dir` | Data directory path | `dataset` | `data/stocks` |

## ⚙️ Configuration Management

### Config File: configs.py

The `StockPredictionConfig` class manages all parameters:

```python
from configs import StockPredictionConfig

# Default configuration
config = StockPredictionConfig()

# Custom configuration
config = StockPredictionConfig(
    model="iTransformer",
    features=["close", "volume"],
    train_epochs=50,
    batch_size=64
)
```

### Parameter Categories

#### Data Parameters
- `data_dir`: Path to data files
- `features`: List of features to use
- `train_size`, `test_size`, `val_size`: Number of files per split
- `stocks`: Specific stocks to include (None = all)

#### Model Parameters  
- `model`: Architecture name
- `d_model`: Model dimension (512)
- `n_heads`: Attention heads (8)
- `e_layers`: Encoder layers (4)
- `d_ff`: Feed-forward dimension (2048)
- `dropout`: Dropout rate (0.2)

#### Training Parameters
- `batch_size`: Batch size (32)
- `learning_rate`: Learning rate (0.0005)
- `train_epochs`: Training epochs (20)
- `patience`: Early stopping patience (5)

#### Sequence Parameters
- `seq_len`: Input sequence length (60)
- `pred_len`: Prediction length (15)
- `label_len`: Label length for teacher forcing (30)

## 📊 Data Format and Preparation

### Required CSV Format

Your data files should contain these columns:

```csv
ticker,window_start,close,volume,transactions,open,high,low
AAPL,1640995200000000000,150.25,1000000,5000,149.80,150.50,149.70
AAPL,1640995260000000000,150.30,1100000,5100,150.25,150.45,150.15
...
```

### Column Descriptions

- `ticker`: Stock symbol (string)
- `window_start`: Timestamp in nanoseconds (int64)
- `close`: Close price (float)
- `volume`: Trading volume (float)  
- `transactions`: Number of transactions (float)
- `open`, `high`, `low`: OHLC data (optional)

### Data Directory Structure

```
dataset/
├── train_file_001.csv
├── train_file_002.csv
├── ...
├── test_file_001.csv
└── test_file_002.csv
```

### Global Normalization

The system automatically calculates global statistics from training files:

```python
# Automatic in dataset creation
global_mean, global_std = calculate_global_stats(
    train_files, 
    features=['close', 'volume', 'transactions']
)
```

## 🧪 Running Experiments

### Quick Start Experiments

```bash
# 1. Single feature (close price only)
python train.py --model iTransformer --features close --train-epochs 5

# 2. Multi-feature
python train.py --model iTransformer --features close,volume,transactions

# 3. Model comparison
python train.py --model iInformer --features close,volume,transactions
python train.py --model Transformer --features close,volume,transactions
```

### Automated Experiment Suite

```bash
# Run all model and feature combinations
python run_all_experiments.py
```

This runs:
- 5 inverted models (iTransformer, iInformer, iReformer, iFlowformer, iFlashformer)
- 2 feature sets (single: close, multi: close,volume,transactions)
- = 10 total experiments

### Bash Script Experiments

```bash
# Single feature experiments
bash scripts/single_feature/iTransformer.sh
bash scripts/single_feature/iInformer.sh
bash scripts/single_feature/Transformer.sh

# Multi-feature experiments  
bash scripts/multi_feature/iTransformer.sh
bash scripts/multi_feature/iInformer.sh

# Comparative analysis
bash scripts/comparative/model_comparison.sh
```

### Custom Experiments

```bash
# Longer sequences
python train.py \
  --model iTransformer \
  --seq-len 120 \
  --pred-len 30 \
  --label-len 60

# Different model size
python train.py \
  --model iTransformer \
  --d-model 256 \
  --n-heads 4 \
  --e-layers 6

# Specific stocks
python train.py \
  --model iTransformer \
  --stocks "AAPL,MSFT,GOOGL"
```

## 🏗️ Model Architectures

### Inverted Models (Primary)

#### iTransformer
- **Core inverted architecture**
- Features become tokens, attention across features
- Best for multivariate correlation modeling

#### iInformer  
- **ProbSparse attention** for efficiency
- Good performance with lower computational cost
- Recommended for large-scale experiments

#### iReformer
- **LSH attention** for memory efficiency  
- Handles very long sequences
- Good for memory-constrained environments

#### iFlowformer
- **Flow-based attention** mechanism
- Novel attention computation
- Experimental architecture

#### iFlashformer
- **Flash attention** implementation
- Optimized for modern GPUs
- Fast training and inference

### Classic Models (Comparison)

#### Transformer
- **Standard transformer** for baseline
- Time steps as tokens
- Good for temporal pattern learning

#### Informer
- **ProbSparse attention** on time dimension
- Classic architecture with efficiency improvements

### Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| **Best Performance** | iTransformer | Superior multivariate modeling |
| **Efficiency** | iInformer | Good performance/cost ratio |
| **Large Datasets** | iReformer | Memory efficient |
| **GPU Optimization** | iFlashformer | Optimized attention |
| **Baseline Comparison** | Transformer | Standard reference |

## 📈 Results and Analysis

### Output Structure

```
myTransformers/
├── checkpoints/           # Model checkpoints
│   ├── best_model.pth    # Best validation model
│   └── checkpoint_*.pth  # Training checkpoints
├── logs/                 # Training logs
│   ├── train.log        # Text logs
│   └── events.out.*     # TensorBoard events
├── figures/             # Generated plots
│   ├── training_*.png   # Training curves
│   ├── predictions_*.png # Sample predictions
│   └── attention_*.png  # Attention visualizations
└── results/             # Experiment results
    ├── metrics.json     # Performance metrics
    └── predictions.csv  # Detailed predictions
```

### Key Metrics

1. **Accuracy Metrics**
   - MSE (Mean Squared Error)
   - MAE (Mean Absolute Error) 
   - MAPE (Mean Absolute Percentage Error)

2. **Training Metrics**
   - Training/Validation loss curves
   - Learning rate schedule
   - Training time per epoch

3. **Model Analysis**
   - Attention pattern visualization
   - Feature importance analysis
   - Prediction quality examples

### TensorBoard Monitoring

```bash
# Start TensorBoard
tensorboard --logdir logs/

# View at http://localhost:6006
```

### Results Comparison

```python
# Load and compare results
import pandas as pd
import json

# Load metrics
with open('results/metrics.json', 'r') as f:
    metrics = json.load(f)

print(f"Test MSE: {metrics['test_mse']:.4f}")
print(f"Test MAE: {metrics['test_mae']:.4f}")
```

## 🔧 Advanced Usage

### Resuming Training

```bash
# Resume from checkpoint
python train.py \
  --model iTransformer \
  --resume-checkpoint checkpoints/checkpoint_epoch_10.pth
```

### Memory Management

```bash
# Limit training samples for testing
python train.py \
  --model iTransformer \
  --max-train-samples 1000 \
  --max-test-samples 200
```

### Multi-GPU Training

```bash
# Use multiple GPUs
python train.py \
  --model iTransformer \
  --use-multi-gpu \
  --device-ids "0,1,2,3"
```

### Custom Loss Functions

```bash
# Different loss types
python train.py \
  --model iTransformer \
  --loss-type "adaptive"    # or "mse", "mae", "huber"
```

### Debugging Mode

```bash
# Quick test mode
python train.py \
  --model iTransformer \
  --quick-test             # 1 epoch, 10 iterations, minimal data
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Memory Errors
```bash
# Reduce batch size
python train.py --batch-size 16

# Limit data samples  
python train.py --max-train-samples 5000
```

#### 2. CUDA Errors
```bash
# Force CPU mode
python train.py --use-gpu False

# Specific GPU
python train.py --gpu 1
```

#### 3. Data Loading Issues
```bash
# Check data directory
python train.py --data-dir /path/to/data

# Verbose logging
python train.py --log-level DEBUG
```

#### 4. Model Compatibility
```bash
# List available models
python -c "from configs import StockPredictionConfig; print(StockPredictionConfig().available_models)"
```

### Performance Optimization

1. **Batch Size**: Start with 32, increase if memory allows
2. **Sequence Length**: Balance between context and memory  
3. **Model Size**: Start with defaults, scale based on data size
4. **Features**: More features usually improve performance
5. **Learning Rate**: Use learning rate scheduling for better convergence

### Getting Help

1. Check logs in `logs/train.log`
2. Use `--help` flag for parameter descriptions
3. Review configuration in `configs.py`
4. Check TensorBoard for training progress
5. Validate data format matches requirements

## 📚 API Reference

### Key Classes

#### StockPredictionConfig
```python
config = StockPredictionConfig(
    model="iTransformer",
    features=["close", "volume"],
    seq_len=60,
    pred_len=15
)
```

#### Exp_Stock_Forecast
```python
exp = Exp_Stock_Forecast(config)
exp.train(setting="experiment_1")
exp.test(setting="experiment_1")
```

#### StockDataset
```python
dataset = StockDataset(
    file_paths=["data.csv"],
    features=["close", "volume"],
    seq_len=60,
    pred_len=15
)
```

### Key Functions

#### Training
```python
from train import main
main(config)  # Run training with config
```

#### Data Processing
```python
from stock_dataset import calculate_global_stats
mean, std = calculate_global_stats(train_files, features)
```
