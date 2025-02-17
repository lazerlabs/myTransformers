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
    def __init__(self, file_paths, tickers=None, seq_len=60, pred_len=30, 
                 scale=True, features=None, label_len=None):
        """
        Args:
            file_paths (str or list): Path(s) to the CSV file(s)
            tickers (list, optional): List of stock tickers to include
            seq_len (int): Input sequence length
            pred_len (int): Prediction sequence length
            scale (bool): Whether to apply standardization
            features (list): List of features to use
            label_len (int, optional): Length of label sequence
        """
        print(f"StockDataset.__init__ - Received tickers: {tickers}")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len if label_len is not None else pred_len
        self.scale = scale
        self.features = features or ['volume', 'close', 'transactions']

       
        # Convert single file path to list
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        self.file_paths = file_paths
            
        # First pass: collect metadata and compute statistics
        print("\nAnalyzing data files...")
        self.metadata = []  # Store metadata for each file/stock combination
        total_sequences = 0
        
        for file_path in file_paths:
            print(f"\nProcessing {os.path.basename(file_path)}:")
            
            # First pass: get all unique tickers from the file
            all_tickers = set()
            chunks = pd.read_csv(file_path, chunksize=10000)
            for chunk in tqdm(chunks, desc="Finding all tickers"):
                # Convert to string and filter out NaN values
                chunk_tickers = chunk['ticker'].astype(str)
                chunk_tickers = chunk_tickers[chunk_tickers != 'nan'].unique()
                all_tickers.update(chunk_tickers)
            
            # Convert to sorted list, ensuring all are strings and no 'nan'
            available_tickers = sorted([str(t) for t in all_tickers if str(t).lower() != 'nan'])
            
            # Debug print
            print(f"Total unique tickers found: {len(available_tickers)}")
            print(f"Sample of tickers: {available_tickers[:20]}")  # Show first 20 sorted tickers
            
            # Filter tickers if specified
            if tickers is not None:
                available_tickers = [t for t in tickers if t in available_tickers]
                print(f"Filtered to {len(available_tickers)} specified tickers: {available_tickers}")
            else:
                print(f"Using all {len(available_tickers)} tickers")
            
            # Initialize statistics
            file_stats = {}
            for ticker in available_tickers:
                file_stats[ticker] = {
                    'count': 0,
                    'sum': np.zeros(len(self.features)),
                    'sum_sq': np.zeros(len(self.features))
                }
            
            # Second pass: compute statistics more efficiently
            chunks = pd.read_csv(file_path, chunksize=10000)
            for chunk in tqdm(chunks, desc="Computing statistics"):
                # Group by ticker and compute stats for all tickers in chunk at once
                grouped = chunk.groupby('ticker')[self.features]
                counts = grouped.count().iloc[:, 0]  # Count of rows per ticker
                sums = grouped.sum()
                squared = chunk[self.features] ** 2
                sum_squares = squared.groupby(chunk['ticker']).sum()
                
                # Update statistics for each ticker found in this chunk
                for ticker in counts.index:
                    if ticker in file_stats:
                        file_stats[ticker]['count'] += counts[ticker]
                        file_stats[ticker]['sum'] += sums.loc[ticker].values
                        file_stats[ticker]['sum_sq'] += sum_squares.loc[ticker].values
            
            # Debug print
            print("\nStatistics summary:")
            #for ticker in available_tickers:
            #    print(f"{ticker}: {file_stats[ticker]['count']} rows")
            
            # Compute final statistics and store metadata
            for ticker in available_tickers:
                stats = file_stats[ticker]
                if stats['count'] >= (self.seq_len + self.pred_len):
                    mean = stats['sum'] / stats['count']
                    var = (stats['sum_sq'] / stats['count']) - (mean ** 2)
                    std = np.sqrt(var)
                    
                    sequences = stats['count'] - self.seq_len - self.pred_len + 1
                    total_sequences += sequences
                    
                    self.metadata.append({
                        'file_path': file_path,
                        'ticker': ticker,
                        'start_idx': len(self.metadata),
                        'sequences': sequences,
                        'stats': {'mean': mean, 'std': std}
                    })
            
            print(f"Valid tickers in file: {len([s for s in file_stats.keys() if file_stats[s]['count'] >= (self.seq_len + self.pred_len)])}")
            print(f"Total sequences: {total_sequences}")
        
        if not self.metadata:
            raise ValueError("No valid data found in any of the files")
        
        self.total_sequences = total_sequences
        print(f"\nDataset initialized with {len(self.file_paths)} files")
        print(f"Total available sequences: {self.total_sequences}")

    def __len__(self):
        return self.total_sequences
        
    def __getitem__(self, idx):
        """Get a sequence starting at idx"""
        # Find which file/stock contains this index
        for meta in self.metadata:
            if idx < meta['sequences']:
                # Found the right file and stock
                # Read only the needed portion of the file
                ticker_df = pd.read_csv(
                    meta['file_path'],
                    skiprows=lambda x: x > 0 and x < idx,  # Skip rows we don't need
                    nrows=self.seq_len + self.pred_len     # Read only rows we need
                )
                ticker_df = ticker_df[ticker_df['ticker'] == meta['ticker']]
                
                # Get the sequence
                sequence = ticker_df[self.features].values
                
                # Normalize if needed
                if self.scale:
                    sequence = (sequence - meta['stats']['mean']) / meta['stats']['std']
                
                # Split into x and y
                x = sequence[:self.seq_len]
                y = sequence[self.seq_len:self.seq_len + self.pred_len]
                
                # Create time features (dummy for now)
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
            
            idx -= meta['sequences']
        
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
    
    # Adjust batch_size if using multiple stocks
    if tickers:
        effective_batch_size = batch_size // len(tickers)
        if effective_batch_size == 0:
            effective_batch_size = 1
    else:
        effective_batch_size = batch_size
    
    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
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