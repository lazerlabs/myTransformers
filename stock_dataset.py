import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
from configs import StockPredictionConfig

class StockDataset(Dataset):
    """Custom dataset for stock market data"""
    def __init__(self, file_path, tickers=['AAPL', 'MSFT', 'JPM', 'JNJ', 'AXP'], 
                 seq_len=60, pred_len=30, scale=True):
        """
        Args:
            file_path (str): Path to the CSV file
            tickers (list): List of stock tickers to include
            seq_len (int): Input sequence length
            pred_len (int): Prediction sequence length
            scale (bool): Whether to apply standardization
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        self.feature_names = ['volume', 'close', 'transactions']
        
        # Read data
        df = pd.read_csv(file_path)
        
        # Filter selected tickers
        df = df[df['ticker'].isin(tickers)]
        
        # Convert timestamp to datetime and sort
        df['window_start'] = pd.to_datetime(df['window_start'], unit='ns')
        df = df.sort_values(['ticker', 'window_start'])
        
        # Store timestamps for later use
        self.timestamps = df[df['ticker'] == tickers[0]]['window_start'].values
        
        # Store the last timestamp for visualization
        self.last_timestamp = df['window_start'].max()
        
        # Find minimum length across all tickers to ensure equal lengths
        min_length = float('inf')
        ticker_data_dict = {}
        for ticker in tickers:
            ticker_df = df[df['ticker'] == ticker]
            ticker_data = ticker_df[self.feature_names].values
            min_length = min(min_length, len(ticker_data))
            ticker_data_dict[ticker] = ticker_data
        
        # Ensure minimum length is sufficient for sequence
        required_length = seq_len + pred_len
        if min_length < required_length:
            raise ValueError(f"Some stocks have less than {required_length} data points")
        
        # Trim all sequences to the same length
        self.data = []
        self.scalers = []  # List of lists: [stock][feature]
        for ticker in tickers:
            ticker_data = ticker_data_dict[ticker][:min_length]  # Trim to minimum length
            
            # Normalize if requested
            if scale:
                # Create a scaler for each feature
                stock_scalers = []
                for feature_idx in range(len(self.feature_names)):
                    scaler = StandardScaler()
                    feature_data = ticker_data[:, feature_idx].reshape(-1, 1)
                    ticker_data[:, feature_idx] = scaler.fit_transform(feature_data).ravel()
                    stock_scalers.append(scaler)
                self.scalers.append(stock_scalers)
            else:
                self.scalers.append([None] * len(self.feature_names))
            
            self.data.append(ticker_data)
        
        # Convert to numpy array
        self.data = np.array(self.data)  # Shape: [num_stocks, sequence_length, num_features]
        
        # Store tickers for reference
        self.tickers = tickers

    def __len__(self):
        return self.data.shape[1] - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        # Get sequences
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data[:, s_begin:s_end, :]  # [num_stocks, seq_len, features]
        seq_y = self.data[:, r_begin:r_end, :]  # [num_stocks, pred_len, features]

        # Create empty mark tensors
        batch_x_mark = torch.zeros((seq_x.shape[0], seq_x.shape[1], 1))  # [num_stocks, seq_len, 1]
        batch_y_mark = torch.zeros((seq_y.shape[0], seq_y.shape[1], 1))  # [num_stocks, pred_len, 1]

        return (
            torch.FloatTensor(seq_x),  # [batch, seq_len, features]
            torch.FloatTensor(seq_y),  # [batch, pred_len, features]
            torch.FloatTensor(batch_x_mark),  # [batch, seq_len, 1]
            torch.FloatTensor(batch_y_mark)   # [batch, pred_len, 1]
        )

    def get_last_timestamp(self):
        """Return the last timestamp in the dataset"""
        return self.last_timestamp
        
    def denormalize(self, data, stock_idx, feature_idx=None):
        """Denormalize the data if scaling was applied
        
        Args:
            data: numpy array of shape [..., features] or single feature
            stock_idx: index of the stock to denormalize
            feature_idx: if provided, denormalize only this feature
        Returns:
            denormalized data with the same shape
        """
        if not self.scale or self.scalers[stock_idx][0] is None:
            return data
            
        # Reshape data to 2D for inverse transform
        original_shape = data.shape
        if len(original_shape) == 1:
            # Single feature data
            data_2d = data.reshape(-1, 1)
            denormalized = self.scalers[stock_idx][feature_idx].inverse_transform(data_2d)
            return denormalized.reshape(original_shape)
        else:
            # Multiple feature data
            result = np.zeros_like(data)
            for i in range(len(self.feature_names)):
                feature_data = data[..., i].reshape(-1, 1)
                result[..., i] = self.scalers[stock_idx][i].inverse_transform(feature_data).ravel()
            return result

    def get_timestamps(self, start_idx, length):
        """Get a sequence of timestamps starting from start_idx"""
        return self.timestamps[start_idx:start_idx + length]

def create_dataloader(file_path=None, batch_size=32, seq_len=60, pred_len=30, scale=True, 
                     tickers=['AAPL', 'MSFT', 'JPM', 'JNJ', 'AXP'], config=None):
    """Create DataLoader for stock market data
    
    Args:
        file_path (str, optional): Path to CSV file. If config is provided, this is ignored.
        batch_size (int, optional): Batch size. Defaults to 32.
        seq_len (int, optional): Input sequence length. Defaults to 60.
        pred_len (int, optional): Prediction sequence length. Defaults to 30.
        scale (bool, optional): Whether to apply standardization. Defaults to True.
        tickers (list, optional): List of stock tickers. Defaults to ['AAPL', 'MSFT', 'JPM', 'JNJ', 'AXP'].
        config (StockPredictionConfig, optional): Configuration object. If provided, its values take precedence.
    """
    # Use config values if provided, otherwise use default parameters
    if config is not None:
        file_path = file_path or config.train_data_path  # Use provided file_path or config path
        batch_size = config.batch_size
        seq_len = config.seq_len
        pred_len = config.pred_len
        scale = config.scale
        tickers = config.stocks
    
    if file_path is None:
        raise ValueError("Either file_path or config must be provided")

    dataset = StockDataset(
        file_path=file_path,
        tickers=tickers,
        seq_len=seq_len,
        pred_len=pred_len,
        scale=scale
    )
    
    # Adjust batch_size to account for multiple stocks
    effective_batch_size = batch_size // len(tickers)
    if effective_batch_size == 0:
        effective_batch_size = 1
    
    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,  # Adjusted batch size
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    return dataset, dataloader

if __name__ == "__main__":
    # Example usage
    # Get data directory from config
    config = StockPredictionConfig()
    file_path = os.path.join(config.data_dir, "2024-11-01.csv")  # Use a sample file
    
    # Create dataloader
    dataset, dataloader = create_dataloader(
        file_path=file_path,
        batch_size=32,
        seq_len=60,  # 1 hour of minute data
        pred_len=30, # Predict next 30 minutes
        scale=True  # Use standardization
    )
    
    # Print dataset info
    print(f"Dataset size: {len(dataset)}")
    
    # Get a batch
    for batch_x, batch_y, batch_x_mark, batch_y_mark in dataloader:
        print(f"Batch shapes:")
        print(f"Input (x): {batch_x.shape}")
        print(f"Target (y): {batch_y.shape}")
        break 