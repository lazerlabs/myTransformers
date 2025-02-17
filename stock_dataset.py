import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
from configs import StockPredictionConfig
from tqdm import tqdm

class StockDataset(Dataset):
    """Custom dataset for stock market data with memory-efficient loading"""
    def __init__(self, file_paths, tickers=None, seq_len=60, pred_len=30, scale=True, features=None):
        """
        Args:
            file_paths (str or list): Path(s) to the CSV file(s)
            tickers (list, optional): List of stock tickers to include
            seq_len (int): Input sequence length
            pred_len (int): Prediction sequence length
            scale (bool): Whether to apply standardization
            features (list): List of features to use
        """
        print(f"StockDataset.__init__ - Received tickers: {tickers}")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        self.features = features or ['volume', 'close', 'transactions']
        
        # Convert single file path to list
        if isinstance(file_paths, str):
            file_paths = [file_paths]
            
        print("\nProcessing data files...")
        self.data_chunks = []  # Will store {ticker: sequences} for each file
        total_sequences = 0
        
        for file_path in file_paths:
            print(f"\nProcessing {os.path.basename(file_path)}:")
            
            # Read and process file in chunks
            chunks = pd.read_csv(file_path, chunksize=10000)
            file_data = {}  # {ticker: DataFrame} for this file
            
            for chunk in tqdm(chunks, desc="Processing data"):
                # Group by ticker
                for ticker, group in chunk.groupby('ticker'):
                    if ticker not in file_data:
                        file_data[ticker] = []
                    file_data[ticker].append(group)
            
            # Process each ticker's data
            valid_tickers = 0
            file_sequences = 0
            
            for ticker, chunks in file_data.items():
                # Combine chunks and sort by time
                ticker_data = pd.concat(chunks).sort_values('window_start').reset_index(drop=True)
                
                if len(ticker_data) >= (self.seq_len + self.pred_len):
                    # Compute statistics
                    feature_data = ticker_data[self.features].values
                    mean = feature_data.mean(axis=0)
                    std = feature_data.std(axis=0)
                    
                    # Create sequences
                    sequences = []
                    for i in range(len(ticker_data) - self.seq_len - self.pred_len + 1):
                        seq = feature_data[i:i + self.seq_len + self.pred_len]
                        if self.scale:
                            seq = (seq - mean) / std
                        sequences.append(seq)
                    
                    if sequences:
                        self.data_chunks.append({
                            'ticker': ticker,
                            'sequences': sequences,
                            'mean': mean,
                            'std': std
                        })
                        valid_tickers += 1
                        file_sequences += len(sequences)
            
            print(f"Valid tickers in file: {valid_tickers}")
            print(f"Sequences in file: {file_sequences}")
            total_sequences += file_sequences
        
        self.total_sequences = total_sequences
        print(f"\nTotal sequences across all files: {total_sequences}")

    def __len__(self):
        return self.total_sequences
        
    def __getitem__(self, idx):
        # Find which chunk contains this index
        current_idx = 0
        for chunk in self.data_chunks:
            if idx < current_idx + len(chunk['sequences']):
                # Found the right chunk
                sequence = chunk['sequences'][idx - current_idx]
                
                # Split into x and y
                x = sequence[:self.seq_len]
                y = sequence[self.seq_len:]
                
                # Create time features
                x_mark = np.zeros((self.seq_len, 1))
                y_mark = np.zeros((self.pred_len, 1))
                
                # Add batch dimension
                x = np.expand_dims(x, axis=0)
                y = np.expand_dims(y, axis=0)
                x_mark = np.expand_dims(x_mark, axis=0)
                y_mark = np.expand_dims(y_mark, axis=0)
                
                return (
                    torch.FloatTensor(x),
                    torch.FloatTensor(x_mark),
                    torch.FloatTensor(y),
                    torch.FloatTensor(y_mark)
                )
            
            current_idx += len(chunk['sequences'])
        
        raise IndexError("Index out of range")

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
            for i in range(len(self.features)):
                feature_data = data[..., i].reshape(-1, 1)
                result[..., i] = self.scalers[stock_idx][i].inverse_transform(feature_data).ravel()
            return result

    def get_timestamps(self, start_idx, length):
        """Get a sequence of timestamps starting from start_idx"""
        return self.timestamps[start_idx:start_idx + length]

def create_dataloader(file_path=None, batch_size=32, seq_len=60, pred_len=30, scale=True, 
                     tickers=None, features=None, config=None):
    print(f"create_dataloader - Initial tickers: {tickers}")
    
    # Use config values if provided, otherwise use default parameters
    if config is not None:
        file_path = file_path or config.train_data_path
        batch_size = config.batch_size
        seq_len = config.seq_len
        pred_len = config.pred_len
        scale = config.scale
        print(f"create_dataloader - Config stocks: {config.stocks}")
        tickers = config.stocks
        print(f"create_dataloader - After config tickers: {tickers}")
        features = config.features

    dataset = StockDataset(
        file_paths=file_path,
        tickers=tickers,  # This should be None
        seq_len=seq_len,
        pred_len=pred_len,
        scale=scale,
        features=features
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # Use full batch size
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