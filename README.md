# Stock Market Forecasting with iTransformer

This project implements a specialized version of the iTransformer architecture for stock market price prediction. The implementation is based on the paper ["iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"](https://arxiv.org/abs/2310.06625).

## Overview

The iTransformer architecture inverts the traditional Transformer architecture by treating features as the sequence dimension and timestamps as the feature dimension. This inversion makes it particularly effective for time series forecasting tasks, including stock market prediction.

## Project Structure

```
├── data_provider/      # Data loading and preprocessing
├── layers/            # Core transformer layers
├── models/           # Model architecture
├── utils/            # Utility functions
└── figures/          # Generated visualizations
```

## Data Preparation

### Input Features
- Close price
- Volume
- Additional technical indicators
- All features are normalized using z-score normalization to ensure consistent scale

### Data Processing
- Time series data is split into training, validation, and test sets
- Sliding window approach for sequence generation
- Missing values are handled through forward filling

## Model Architecture

### Tokenization
- Feature-first approach where each feature dimension is treated as a token
- Temporal information is encoded in the feature dimension

### Embedding
- Linear embedding layer to project input features
- Positional encoding to maintain temporal order
- Feature embedding dimension: 512

### Transformer Architecture
- Multi-head self-attention mechanism
- 4 transformer encoder layers
- 8 attention heads per layer
- Feed-forward network dimension: 2048
- Dropout rate: 0.1

### Output Layer
- Multi-layer perceptron for final predictions
- Outputs predictions for both price and volume

## Training and Evaluation

### Training Process
- Adam optimizer with learning rate scheduling
- Loss function: Mean Squared Error (MSE)
- Early stopping based on validation loss
- Batch size: 32

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

## Results Visualization

The project includes various visualization tools:
- Price prediction plots
- Volume prediction plots
- Learning rate curves
- Training metrics visualization

## Training Scripts

The project includes two main training scripts:

### train.py
- Main training script for full model training
- Uses full model capacity (d_model = 512)
- Uses all available training data
- Runs for 10 epochs by default
- Saves the model in `./checkpoints` directory
- Suitable for production model training

### test_run.py
- Quick testing script for development
- Uses reduced model capacity (d_model = 128)
- Uses only 1 file for training
- Runs for 5 epochs
- More verbose logging and error handling
- Suitable for testing changes and debugging

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your data in the required format

3. Run training:
```bash
python train.py
```

4. Run predictions:
```bash
python test_run.py
```

## References

- Original iTransformer paper: [arXiv:2310.06625](https://arxiv.org/abs/2310.06625) 