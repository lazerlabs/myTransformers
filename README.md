# Stock Market Forecasting with Inverted Transformers

A comprehensive implementation of inverted transformer architectures for stock market time series forecasting, based on the paper ["iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"](https://arxiv.org/abs/2310.06625).

## 🎯 Overview

This project implements and evaluates inverted transformer architectures for stock market prediction. Unlike traditional transformers that treat time steps as tokens, inverted transformers treat **features as tokens**, enabling better multivariate correlation modeling - crucial for financial time series.

### Key Features

- **True Inverted Architecture**: Features (price, volume, transactions) become tokens instead of time steps
- **Multivariate Correlation Modeling**: Superior handling of cross-feature relationships
- **Multiple Model Support**: iTransformer, iInformer, iReformer, iFlowformer, and classic models for comparison
- **Memory-Efficient Data Loading**: Processes large datasets without memory overflow
- **Comprehensive Experiment Framework**: Easy comparison across models and configurations

## 🏗️ Architecture

### Inverted vs Traditional Transformers

| Aspect | Traditional Transformer | **Inverted Transformer (ours)** |
|--------|------------------------|----------------------------------|
| **Tokens** | Time steps | **Features (price, volume, etc.)** |
| **Attention** | Across time | **Across features (multivariate)** |
| **Embedding** | All features at timestep t → vector | **Time series of feature f → vector** |
| **Advantage** | Temporal patterns | **Multivariate correlations** |

### Supported Models

- **iTransformer**: Core inverted transformer with feature-based attention
- **iInformer**: Inverted ProbSparse attention for efficiency  
- **iReformer**: Inverted LSH attention for memory efficiency
- **iFlowformer**: Inverted flow-based attention
- **iFlashformer**: Inverted flash attention implementation
- **Classic Models**: Transformer, Informer, Reformer for comparison

## 📁 Project Structure

```
myTransformers/
├── README.md                    # This file
├── train.py                     # Advanced CLI training interface  
├── run.py                       # Compatible interface for scripts
├── run_all_experiments.py       # Automated experiment runner
├── configs.py                   # Configuration management
├── exp_stock_forecasting.py     # Experiment orchestration
├── stock_dataset.py             # Memory-efficient data loading
├── docs/                        # Documentation
│   ├── README.md               # Detailed usage guide  
│   ├── model_size.md           # Model architecture and scaling
│   └── EXPERIMENTS.md          # Experiment protocols
├── scripts/                     # Experiment scripts
│   ├── single_feature/         # Close price experiments
│   ├── multi_feature/          # Multi-feature experiments  
│   └── comparative/            # Model comparison
├── models/                      # Model architectures
├── layers/                      # Transformer components
├── data_provider/              # Data preprocessing
├── utils/                      # Utilities and visualization
├── results/                    # Experiment outputs
├── checkpoints/                # Model checkpoints
└── figures/                    # Generated plots
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

Ensure your stock data is in the `dataset/` directory with CSV files containing:
- `ticker`: Stock symbol
- `window_start`: Timestamp (nanoseconds)
- `close`: Close price
- `volume`: Trading volume  
- `transactions`: Number of transactions

### 3. Run Your First Experiment

```bash
# Quick test - iTransformer with close price only
python train.py --model iTransformer --features close --train-epochs 5

# Multi-feature experiment
python train.py --model iTransformer --features close,volume,transactions --train-epochs 20
```

### 4. Automated Experiments

```bash
# Run all model comparisons
python run_all_experiments.py

# Or use bash scripts
bash scripts/single_feature/iTransformer.sh
bash scripts/comparative/model_comparison.sh
```

## 💡 Key Differences from Original iTransformer

### Enhanced Features

1. **Stock Market Specialization**: Optimized for financial time series with proper handling of trading features
2. **Memory Efficiency**: Processes large datasets file-by-file to avoid memory issues
3. **Global Normalization**: Consistent scaling across all data splits
4. **Comprehensive Evaluation**: Automated comparison across multiple models and feature sets
5. **Production Ready**: Robust error handling, checkpointing, and resumable training

### Implementation Highlights

- **True Feature Inversion**: Each feature becomes a token with its own time series embedding
- **Attention Across Features**: Models correlations between price, volume, and transaction patterns
- **Flexible Architecture**: Easy to add new models and compare with classics
- **Scalable Data Loading**: Handles datasets too large for memory

## 📊 Expected Results

Based on the iTransformer paper and our stock market adaptation:

### Performance Hierarchy
1. **iTransformer (Multi-feature)** → Best multivariate modeling
2. **iInformer (Multi-feature)** → Good performance with efficiency  
3. **iTransformer (Single-feature)** → Strong baseline
4. **Classic Transformer** → Comparison baseline

### Key Insights
- **Multi-feature > Single-feature**: Volume and transaction data improve predictions
- **Inverted > Classic**: Better multivariate correlation modeling
- **Feature attention**: More interpretable than temporal attention for trading

## 🔬 Research Applications

### Experiment Types

1. **Architecture Comparison**: Inverted vs Classic transformers
2. **Feature Analysis**: Impact of different market features
3. **Prediction Horizons**: Short-term (15min) to long-term (60min) forecasting
4. **Model Scaling**: Effect of model size on performance

### Evaluation Metrics

- **Accuracy**: MSE, MAE, MAPE
- **Efficiency**: Training time, memory usage
- **Interpretability**: Attention pattern analysis
- **Robustness**: Performance across different stocks and time periods

## 📖 Documentation

- [`docs/README.md`](docs/README.md): Detailed usage guide and API reference
- [`docs/model_size.md`](docs/model_size.md): Model architecture and parameter scaling
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): Experiment protocols and analysis

## 🤝 Contributing

This implementation is designed for research and experimentation. Key areas for extension:

- Additional inverted architectures
- New financial features and data sources  
- Advanced evaluation metrics
- Hyperparameter optimization frameworks

## 📄 Citation

If you use this implementation in your research, please cite:

```bibtex
@article{liu2023itransformer,
  title={iTransformer: Inverted Transformers Are Effective for Time Series Forecasting},
  author={Liu, Yong and Hu, Tengge and Zhang, Haoran and Wu, Haixu and Wang, Shiyu and Ma, Lintao and Long, Mingsheng},
  journal={arXiv preprint arXiv:2310.06625},
  year={2023}
}
```

## 📞 Support

For questions about the implementation or experiments, please refer to the documentation in the `docs/` directory or open an issue. 
