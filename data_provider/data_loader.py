import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import os

class StockDataset(Dataset):
    def __init__(self, data_paths, seq_len, pred_len, label_len=None, stocks=None, features=None):
        """
        Initialize dataset for iTransformer
        
        Args:
            data_paths: list of paths to CSV files containing stock data
            seq_len: length of input sequence
            pred_len: length of prediction sequence
            label_len: length of label sequence (defaults to pred_len if None)
            stocks: list of stock symbols to use (if None, use all stocks in each file)
            features: list of features to use (if None, use default)
        """
        self.data_paths = data_paths
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len if label_len is not None else pred_len
        self.min_required_length = seq_len + pred_len
        
        # Default features if none specified
        if features is None:
            features = ['volume', 'close', 'transactions']
        self.features = features
        
        # Process each file to get available data
        print("\nProcessing files:")
        self.file_data = []  # Store metadata for each file
        total_sequences = 0
        
        for file_path in data_paths:
            file_name = os.path.basename(file_path)
            print(f"\nProcessing {file_name}:")
            
            # Read the file
            df = pd.read_csv(file_path)
            
            # Get available stocks in this file
            file_stocks = df['ticker'].unique() if stocks is None else [s for s in stocks if s in df['ticker'].unique()]
            
            # Process each stock in this file
            file_stock_data = []
            file_total_timestamps = 0
            file_total_sequences = 0
            
            for stock in file_stocks:
                stock_df = df[df['ticker'] == stock]
                if len(stock_df) >= self.min_required_length:
                    # Calculate statistics for this stock
                    stock_data = stock_df[self.features].values
                    stats = {
                        'mean': stock_data.mean(axis=0),
                        'std': stock_data.std(axis=0),
                        'length': len(stock_df)
                    }
                    
                    sequences = len(stock_df) - self.min_required_length + 1
                    file_total_timestamps += len(stock_df)
                    file_total_sequences += sequences
                    
                    # Store metadata for this stock
                    file_stock_data.append({
                        'ticker': stock,
                        'stats': stats,
                        'sequences': sequences
                    })
            
            if file_stock_data:
                self.file_data.append({
                    'path': file_path,
                    'stocks': file_stock_data
                })
                total_sequences += file_total_sequences
                
                # Print summary for this file
                print(f"  Total rows: {len(df)}")
                print(f"  Available stocks: {len(file_stocks)}")
                print(f"  Valid stocks: {len(file_stock_data)}")
                print(f"  Total timestamps: {file_total_timestamps}")
                print(f"  Total sequences: {file_total_sequences}")
        
        if not self.file_data:
            raise ValueError("No valid data found in any of the files")
            
        self.total_sequences = total_sequences
        print(f"\nDataset initialized with {len(self.file_data)} files")
        print(f"Total available sequences across all files: {self.total_sequences}")

    def __len__(self):
        return self.total_sequences
        
    def __getitem__(self, idx):
        """Get a sequence starting at idx"""
        # Find which file and stock contains this index
        current_idx = 0
        for file_info in self.file_data:
            for stock_info in file_info['stocks']:
                if current_idx + stock_info['sequences'] > idx:
                    # Found the right file and stock
                    pos_in_stock = idx - current_idx
                    
                    # Read the data
                    df = pd.read_csv(file_info['path'])
                    stock_df = df[df['ticker'] == stock_info['ticker']]
                    
                    # Get the sequence
                    start_idx = pos_in_stock
                    end_idx = start_idx + self.seq_len + self.pred_len
                    sequence = stock_df[self.features].values[start_idx:end_idx]
                    
                    # Normalize
                    sequence = (sequence - stock_info['stats']['mean']) / stock_info['stats']['std']
                    
                    # Split into x and y
                    x = sequence[:self.seq_len]
                    y = sequence[self.seq_len:self.seq_len + self.pred_len]
                    
                    # Create time features (dummy for now)
                    x_mark = np.zeros((self.seq_len, 1))
                    y_mark = np.zeros((self.pred_len, 1))
                    
                    # Add batch and stock dimensions (1 stock at a time)
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
                    
                current_idx += stock_info['sequences']
        
        raise IndexError("Index out of range")

def create_data_loaders(config):
    """Create train/val/test data loaders"""
    train_dataset = StockDataset(
        config.train_files,
        config.seq_len,
        config.pred_len,
        label_len=config.label_len,
        stocks=config.stocks,
        features=config.features
    )
    
    test_dataset = StockDataset(
        config.test_files,
        config.seq_len,
        config.pred_len,
        label_len=config.label_len,
        stocks=config.stocks,
        features=config.features
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # Keep single process for now
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader 