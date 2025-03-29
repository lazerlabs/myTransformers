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
- Features are normalized using global mean/std statistics calculated **only** from the training dataset files (Z-score normalization).

### Data Processing
- Each file is processed independently
- Stocks are handled independently within each file
- Each stock's time series contributes multiple sequences based on sliding windows
- Minimum required length for a stock's data is sequence_length + prediction_length
- Global statistics (mean, std) are calculated across all training files and applied consistently to train, validation, and test sets.
- Missing values are handled by skipping sequences containing NaNs during data loading.

### Data Loading Strategy
- Files are processed one at a time to minimize memory usage
- Each stock in each file can contribute multiple sequences
- Sequences are created using a sliding window approach
- The total number of sequences is the sum of available sequences from all stocks in all files
- Batch creation is dynamic, potentially mixing sequences from different stocks and files

### Data Splitting
- Split level: File-based splitting
- Split strategy: Files are sorted chronologically, then assigned sequentially (earliest to train, latest to test).
- Default split sizes:
  - Test set: 1 file
  - Validation set: 2 files
  - Training set: Configurable via train_size (default 5 files)
- Minimum requirement: Need at least (test_size + val_size + 1) files
- Each file can contain multiple stocks and sequences
- Sequence creation:
  - Each stock's data is split into sequences using sliding windows
  - Minimum required length per stock: sequence_length + prediction_length
  - Only stocks with sufficient data length are included
  - Each valid stock contributes multiple sequences based on its length

### Data Processing Details
- **Global Normalization**:
  - Mean and standard deviation are calculated across **all training files** for each feature.
  - These global statistics are then applied using Z-score normalization to the training, validation, and test sets.
- Sequence generation:
  - Input sequence length: 60 minutes (configurable)
  - Prediction length: 15 minutes (configurable)
  # - Label length: 30 minutes (for teacher forcing) # Removed - Not applicable to encoder-only model
- Features processed:
  - Volume
  - Close price
  - Number of transactions

## Model Architecture

### Tokenization and Input Processing
- Each feature (volume, close, transactions) is treated as a token
- Input shape: [Batch, Stocks, Time, Features]
- Features are normalized using global training set statistics (Z-score).
- Temporal information is encoded using time features (see below) and positional encodings.

### Embedding Layer
The embedding system (`DataEmbedding_inverted`) combines:
- Value embedding: Linear projection of input features
- Temporal embedding: Captures gradient information
- Feature embedding: Learned embeddings for 3 features (close, volume, transactions)
- Positional encoding: Sinusoidal with learnable scale parameter
- Additional components:
  - Layer normalization
  - Dropout (configurable, default 0.1)
  - Temporal gradient computation with scale normalization

### Transformer Architecture
- Encoder-only architecture (no decoder)
- Core components:
  - Multi-head self-attention mechanism
  - Feed-forward network
  - Layer normalization
- Default configuration:
  - Model dimension (d_model): 512
  - Number of heads (n_heads): 8
  - Number of encoder layers (e_layers): 4
  - Feed-forward dimension (d_ff): 2048
  - Dropout rate: 0.2
  - Activation: GELU

### Loss Functions
Multiple loss functions available through `StockPredictionLoss`:
- MSE (Mean Squared Error)
- Squared MAE (Mean Absolute Error)
- Huber Loss (delta=1.0)
- Asymmetric Loss (alpha=1.5)
- Directional Loss
  - Combines value accuracy and direction accuracy
  - Configurable direction weight (default: 0.2)
- Adaptive Scale Loss
  - Handles different scales of price movements
  - Configurable alpha (0.3) and beta (2.0)

## Training Configuration

### Default Parameters
- Sequence length: 60 (1 hour of minute data)
- Prediction length: 15 (predict next 15 minutes)
# - Label length: 30 (for teacher forcing) # Removed - Not applicable
- Batch size: 64
- Learning rate: 5e-4
- Training epochs: 20
- Early stopping patience: 5

### Learning Rate Scheduling
- Scheduler: Cosine annealing
- Minimum learning rate: 1e-5
- Warmup epochs: 2
- Decay factor: 0.1
- Scheduler patience: 3

### Data Processing
- Chunk size: 10,000 rows for efficient memory usage
- Per-stock statistics computation
- Features processed: volume, close price, transactions
- Automatic hardware selection (CUDA/MPS/CPU)

## Training and Evaluation

### Training Process
- Adam optimizer with learning rate scheduling
- Loss function is configurable via `configs.py` (see `loss_type` and `loss_kwargs`). Default is "adaptive". See `utils/loss.py` for all options.
- Early stopping based on validation loss
- Batch size: 32 (Note: `configs.py` default is 64)
- Sequence length: 60 minutes
- Prediction length: 60 minutes
# - Label length: 60 minutes (for teacher forcing # Removed - Not applicable

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
- Reduced model configuration:
  - Model dimension: 256
  - Sequence length: 60 minutes
  - Prediction length: 60 minutes
  # - Label length: 60 minutes # Removed - Not applicable
  - Directional loss weight: 0.3
- Includes embedding visualization
- More verbose logging
- Automatically selects best available device

### train.py
- Main training script for production
- Full model configuration:
  - Model dimension: 512 (default)
  - Uses default sequence/prediction lengths
  - Directional loss weight: 0.8
- More emphasis on directional prediction
- Detailed experiment naming
- Supports keyboard interrupt for early stopping

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
- Z-score normalization applied using global mean/std calculated **only** from the training dataset files.
- These fixed stats are applied to train, validation, and test sets.

``` (Code snippet reference might be outdated due to changes)
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
- Extracts minute-of-hour, hour-of-day (cyclical sin/cos), and day-of-week (cyclical sin/cos) from timestamps.
- Creates `x_mark` and `y_mark` tensors with these features.

``` (Code snippet reference might be outdated due to changes)
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

The loss function is dynamically selected based on the `loss_type` parameter in `configs.py`, using the `get_loss_function` utility in `utils/loss.py`. The default is currently set to `"adaptive"`. The training script (`exp_stock_forecasting.py`) now correctly uses this configured loss.

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

## Hardware Support
The model automatically selects the best available hardware:
- CUDA GPU if available
- Apple Silicon MPS if available
- CPU as fallback
