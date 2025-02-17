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
The model processes the following features for each stock:
- Volume
- Close price
- Number of transactions
- Features are normalized independently for each stock in each file using z-score normalization

### Data Processing
- Each file is processed independently
- Stocks are handled independently within each file
- Each stock's time series contributes multiple sequences based on sliding windows
- Minimum required length for a stock's data is sequence_length + prediction_length
- Statistics (mean, std) are calculated per stock per file for normalization
- Missing values are handled through forward filling

### Data Loading Strategy
- Files are processed one at a time to minimize memory usage
- Each stock in each file can contribute multiple sequences
- Sequences are created using a sliding window approach
- The total number of sequences is the sum of available sequences from all stocks in all files
- Batch creation is dynamic, potentially mixing sequences from different stocks and files

## Model Architecture

### Tokenization
- Feature-first approach where each feature dimension is treated as a token
- Temporal information is encoded in the feature dimension

### Embedding
- Linear embedding layer to project input features
- Feature embedding for each feature type
- Temporal gradient information added to capture trends
- Feature embedding dimension: 512

### Transformer Architecture
- Multi-head self-attention mechanism
- 4 transformer encoder layers
- 16 attention heads per layer (increased from 8)
- Feed-forward network dimension: 2048
- Dropout rate: 0.2 (increased from 0.1 for better regularization)

### Output Layer
- Linear projection layer for final predictions
- Outputs predictions for all features

## Training and Evaluation

### Training Process
- Adam optimizer with learning rate scheduling
- Cosine annealing learning rate schedule
- Loss function: Mean Squared Error (MSE) with optional directional loss
- Early stopping based on validation loss (configurable patience)
- Batch size: 32
- Warm-up epochs: 1

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

## Results Visualization

The project includes various visualization tools:
- Price prediction plots for each stock
- Learning rate curves
- Training metrics visualization
- Embeddings visualization

## Training Scripts

### test_run.py
- Quick testing script for development
- Configurable sequence length and prediction horizon
- Uses reduced model capacity for faster iteration
- Saves embeddings for visualization
- More verbose logging and error handling
- Automatically selects best available device (CUDA, MPS, or CPU)

### train.py
- Main training script for full model training
- Uses full model capacity
- Processes all available training data
- Supports early stopping
- Includes learning rate scheduling
- Saves checkpoints and visualizations

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your data in CSV format with columns:
   - ticker: Stock symbol
   - window_start: Timestamp
   - volume: Trading volume
   - close: Closing price
   - transactions: Number of transactions

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

## Detailed implementation notes
I'll analyze the project structure and answer your questions:

### 1. Data Pipeline Flow

The data flows through the following stages:

a) **Data Loading**:
- Starts in `StockDataset` class which loads CSV files
- Features: volume, close price, and transactions
- Creates sequences using sliding window approach

```3:15:myTransformer/data_provider/data_loader.py
from torch.utils.data import Dataset, DataLoader

class StockDataset(Dataset):
    def __init__(self, data_paths, seq_len, pred_len, label_len=None, stocks=None, features=None):
        """
        Initialize dataset for iTransformer
        
        Args:
            data_paths: list of paths to CSV files containing stock data
            seq_len: length of input sequence
            pred_len: length of prediction sequence
            label_len: length of label sequence (defaults to pred_len if None)
            stocks: list of stock symbols to use (if None, use all stocks)
```


b) **Embedding**:
- Data shape transformation: `[Batch, Time, Features] -> [Batch, Features, Time]`
- Uses `DataEmbedding_inverted` which combines:
  - Value embedding (linear projection)
  - Feature embedding (learned embeddings for each feature type)

```4:24:myTransformer/layers/Embed.py
class DataEmbedding_inverted(nn.Module):
    """
    Data Embedding for stock market data
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.feature_embedding = nn.Embedding(3, d_model)  # 3 features: close, volume, transactions
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x: [Batch, Time, Features] -> [Batch, Features, Time]
        x = x.permute(0, 2, 1)
        
        # Create feature indices tensor
        feature_indices = torch.arange(x.size(1), device=x.device)
        
        # Get feature embeddings
        feature_emb = self.feature_embedding(feature_indices)
        
        # Combine value and feature embeddings
```


c) **Transformer Processing**:
- Encoder-only architecture
- Multi-head self-attention
- Feed-forward network
- Layer normalization

```31:54:myTransformer/models/iTransformer.py

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, 
                            configs.factor, 
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ), 
                        configs.d_model, 
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

```


### 2. Data Manipulation

Yes, there are several data manipulations:

1. **Normalization**:
- Z-score normalization per feature and stock
- Stores mean and std for later denormalization

```89:95:myTransformer/data_provider/data_loader.py
            stock_df = df[df['ticker'] == stock]
            
            # Interpolate missing values if any
            stock_df = stock_df.sort_values('window_start')
            stock_df[self.features] = stock_df[self.features].interpolate(method='linear')
            
            # Process features
```


2. **Missing Value Handling**:
- Linear interpolation for missing values

```142:143:myTransformer/data_provider/data_loader.py
        r_end = r_begin + self.pred_len  # Target sequence should be pred_len long
        
```


3. **Time Feature Engineering**:
- Extracts hour and minute from timestamps
- Creates time features for temporal information

```54:67:myTransformer/data_provider/data_loader.py
            # Store timestamps for visualization
            self.timestamps = df['window_start'].unique()
            # Convert timestamps to hour of day and add as time features
            df['hour'] = df['window_start'].dt.hour
            df['minute'] = df['window_start'].dt.minute
            self.time_features = ['hour', 'minute']
        else:
            # If no timestamp, use dummy time features
            df['hour'] = 0
            df['minute'] = 0
            self.time_features = ['hour', 'minute']
            self.timestamps = pd.date_range('2000-01-01', periods=len(df), freq='1min')
        
        # Store the last timestamp
```


### 3. Loss Function

Looking through the code, I don't actually see an explicit loss function defined. According to the README, it should be using Mean Squared Error (MSE), but this isn't implemented in the shared code. This would need to be defined in the training script.

### 4. Network Topology

Current architecture:
- **Input Dimension**: Variable (based on number of features)
- **Embedding**: 
  - Dimension: 512 (as per README)
  - Combines value and feature embeddings

- **Transformer**:
  - 4 encoder layers
  - 8 attention heads
  - Feed-forward dimension: 2048
  - Dropout: 0.1

```25:54:myTransformer/models/iTransformer.py
            configs.seq_len, 
            configs.d_model, 
            configs.embed, 
            configs.freq,
            configs.dropout
        )

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, 
                            configs.factor, 
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ), 
                        configs.d_model, 
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

```


### Differences from LLM Transformers

1. **Architecture**:
   - iTransformer: Encoder-only, inverted feature-time dimensions
   - LLM: Encoder-decoder or decoder-only, sequential token processing

2. **Attention Mechanism**:
   - iTransformer: Treats features as sequence dimension, time as feature dimension
   - LLM: Processes tokens sequentially with causal masking

3. **Input Processing**:
   - iTransformer: Continuous numerical values with feature embeddings
   - LLM: Discrete tokens with learned token embeddings

4. **Output**:
   - iTransformer: Continuous value prediction
   - LLM: Token probability distribution

5. **Sequence Length**:
   - iTransformer: Typically shorter sequences (time series)
   - LLM: Much longer sequences (thousands of tokens)

This architecture is specifically optimized for time series forecasting by inverting the traditional transformer architecture to better handle feature relationships across time.
