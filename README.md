# Stock Market Forecasting with Inverted Transformers

A **custom implementation** of inverted transformer architectures specifically designed for stock market time series forecasting, **inspired by and based on** the paper ["iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"](https://arxiv.org/abs/2310.06625).

> **Note**: This is a **derivative work** that adapts the iTransformer concept for stock market data. This is not the official implementation of the original paper.

## 🎯 Overview

This project implements and evaluates inverted transformer architectures specifically optimized for stock market prediction using intraday minute-bar data. Unlike traditional transformers that treat time steps as tokens, inverted transformers treat **features as tokens**, enabling better multivariate correlation modeling - crucial for financial time series.

### Key Features

- **Stock Market Specialized**: Custom implementation optimized for OHLCV market data
- **True Inverted Architecture**: Features (price, volume, transactions) become tokens instead of time steps
- **Sliding Window & Full Day Modes**: Flexible data processing for different prediction scenarios
- **Multiple Model Support**: iTransformer, iInformer, iReformer, iFlowformer, and classic models for comparison
- **Memory-Efficient Data Loading**: Processes large datasets (20+ years of minute data) without memory overflow
- **Comprehensive CLI Interface**: Easy experimentation with extensive configuration options
- **Advanced Training Features**: TensorBoard integration, checkpointing, early stopping, multiple loss functions

## 🏗️ Architecture

### Inverted vs Traditional Transformers

| Aspect | Traditional Transformer | **Inverted Transformer (our adaptation)** |
|--------|------------------------|----------------------------------|
| **Tokens** | Time steps | **Features (close, volume, transactions)** |
| **Attention** | Across time | **Across features (multivariate correlations)** |
| **Embedding** | All features at timestep t → vector | **Time series of feature f → vector** |
| **Market Advantage** | Temporal patterns | **Cross-asset correlations, volume-price relationships** |

### Supported Models

- **iTransformer**: Core inverted transformer with feature-based attention
- **iInformer**: Inverted ProbSparse attention for efficiency  
- **iReformer**: Inverted LSH attention for memory efficiency
- **iFlowformer**: Inverted flow-based attention
- **iFlashformer**: Inverted flash attention implementation
- **Classic Models**: Transformer, Informer, Reformer for ablation studies

## 📁 Project Structure

```
myTransformers/
├── README.md                    # This file
├── train.py                     # Main CLI training interface with 50+ options
├── configs.py                   # Centralized configuration management
├── exp_stock_forecasting.py     # Experiment orchestration and model training
├── stock_dataset.py             # Memory-efficient sliding window data loading
├── models/                      # Custom iTransformer implementations
│   └── iTransformer.py         # Stock-optimized iTransformer model
├── utils/                       # Utilities
│   ├── metrics.py              # Evaluation metrics
│   ├── loss.py                 # Custom loss functions (directional, etc.)
│   ├── visualization.py        # Plotting and analysis tools
│   └── logger.py              # Training and experiment logging
├── results/                    # Experiment outputs and metrics
├── checkpoints/                # Model checkpoints
├── logs/                       # Training logs and TensorBoard data
└── figures/                    # Generated prediction plots
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies (create requirements.txt based on your environment)
pip install torch pandas numpy scikit-learn tqdm click tensorboard matplotlib seaborn
```

### 2. Data Preparation

Ensure your stock data CSV files contain these columns:
- `ticker`: Stock symbol (optional if one ticker per file)
- `window_start`: Timestamp (nanoseconds or standard datetime)
- `close`: Close price
- `volume`: Trading volume  
- `transactions`: Number of transactions
- Additional OHLC columns supported

### 3. Run Your First Experiment

```bash
# Quick test with sliding window mode (default)
python train.py --model iTransformer --features close --quick-test

# Full experiment with multiple features
python train.py --model iTransformer \
    --features close,volume,transactions \
    --seq-len 60 --pred-len 30 \
    --train-epochs 20 \
    --mode sliding_window

# Full day mode (variable length sequences)
python train.py --model iTransformer \
    --features close,volume,transactions \
    --mode full_day \
    --train-epochs 20
```

### 4. Monitor Training

The training automatically starts TensorBoard:
```bash
# TensorBoard starts automatically on localhost:6006
# Or manually: tensorboard --logdir=./logs
```

## 💡 Key Differences from Original iTransformer Paper

### Our Adaptations for Stock Markets

1. **Financial Data Specialization**: 
   - Optimized for OHLCV minute-bar data
   - Handles missing values and market gaps
   - Custom time features (minute, hour, day-of-week cycles)

2. **Sliding Window Implementation**:
   - Creates 361 sequences from 450 daily datapoints
   - No cross-ticker contamination
   - Comprehensive evaluation across all possible windows

3. **Memory Efficiency**: 
   - Processes 20+ years of data without memory issues
   - File-by-file processing with chunking
   - Global normalization across entire dataset

4. **Advanced Training Pipeline**:
   - Multiple loss functions (MSE, MAE, directional loss)
   - Learning rate scheduling and early stopping
   - Comprehensive checkpointing and resumable training

5. **Evaluation Framework**:
   - Extensive metrics (MSE, MAE, MAPE, RMSE, MSPE)
   - Visualization of predictions vs actuals
   - Per-ticker and aggregate performance analysis

### Implementation Highlights

- **True Feature Tokenization**: Each market feature becomes a token with its own time series embedding
- **Market Correlation Modeling**: Attention across price, volume, and transaction patterns
- **Flexible Sequence Modes**: Both sliding window and full-day prediction scenarios
- **Production-Grade**: Robust error handling, logging, and monitoring

## 📊 Understanding the Evaluation

### What Test Results Represent

When using sliding window mode with 20 years of data:
- **Per ticker per day**: ~361 sliding windows (450 - 60 - 30 + 1)
- **Total evaluation scale**: Millions of 60→30 minute predictions
- **Coverage**: All possible market conditions, times of day, volatility regimes
- **Statistical robustness**: Aggregated performance across comprehensive scenarios

### Metrics Interpretation
- **MSE/MAE**: Average prediction error across all sliding windows
- **MAPE**: Percentage error relative to actual prices
- **Comprehensive**: Not cherry-picked - includes all market conditions

## 🔬 Research Applications

### Experiment Types Available

1. **Architecture Comparison**: `--model iTransformer` vs `--model Transformer`
2. **Feature Analysis**: `--features close` vs `--features close,volume,transactions`
3. **Sequence Modes**: `--mode sliding_window` vs `--mode full_day`
4. **Loss Functions**: `--loss-type mse` vs `--loss-type directional`
5. **Prediction Horizons**: `--pred-len 15` vs `--pred-len 60`

### Available CLI Options

The `train.py` script provides 50+ configuration options including:
- Model architecture parameters (`--d-model`, `--n-heads`, `--e-layers`)
- Training parameters (`--learning-rate`, `--batch-size`, `--train-epochs`)
- Data parameters (`--stocks`, `--features`, `--seq-len`, `--pred-len`)
- Advanced options (checkpointing, logging, TensorBoard, validation)

## 🤝 Contributing

This is a research-focused implementation designed for:
- Experimenting with inverted transformers on financial data
- Comparing different architectural approaches
- Evaluating feature combinations and prediction horizons
- Extending to new financial datasets and timeframes

## 📄 Citation

This work is based on the original iTransformer paper. If you use this implementation, please cite the original paper:

```bibtex
@article{liu2023itransformer,
  title={iTransformer: Inverted Transformers Are Effective for Time Series Forecasting},
  author={Liu, Yong and Hu, Tengge and Zhang, Haoran and Wu, Haixu and Wang, Shiyu and Ma, Lintao and Long, Mingsheng},
  journal={arXiv preprint arXiv:2310.06625},
  year={2023}
}
```

**Note**: This repository contains a custom implementation adapted for stock market forecasting. For the official implementation of the paper, please refer to the authors' original repository.

## 📞 Support

For questions about this specific implementation:
- Check the CLI help: `python train.py --help`
- Review configuration options in `configs.py`
- Examine example usage in the training logs

This implementation focuses specifically on stock market applications of inverted transformers and may differ significantly from other iTransformer implementations. 
