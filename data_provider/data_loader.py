import numpy as np
import pandas as pd
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
            features: list of features to use (if None, use default)
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len if label_len is not None else pred_len
        self.min_required_length = seq_len + pred_len
        
        # Default features if none specified
        if features is None:
            features = ['volume', 'close', 'transactions']
        self.features = features
        
        # Initialize arrays and store lengths
        self.data = []
        self.time_data = []
        self.stock_lengths = {}
        self.feature_stats = {}  # Store mean and std for each stock and feature
        
        # Process all files
        all_data = []
        for data_path in data_paths:
            df = pd.read_csv(data_path)
            if 'ticker' not in df.columns:
                raise ValueError(f"CSV file {data_path} must contain a 'ticker' column")
            all_data.append(df)
        
        # Concatenate all data and sort by timestamp
        df = pd.concat(all_data, ignore_index=True)
        
        # Get all available stocks if none specified
        if stocks is None:
            stocks = sorted(df['ticker'].unique())
        self.stocks = stocks
        
        # Sort by window_start timestamp
        if 'window_start' in df.columns:
            df['window_start'] = pd.to_datetime(df['window_start'])
            df = df.sort_values('window_start')
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
        self.last_timestamp = self.timestamps[-1]
        
        # Find minimum length across all stocks
        min_stock_length = float('inf')
        for stock in self.stocks:
            stock_df = df[df['ticker'] == stock]
            if len(stock_df) == 0:
                raise ValueError(f"Stock {stock} not found in data files")
            min_stock_length = min(min_stock_length, len(stock_df))
        
        # Adjust sequence lengths if necessary
        if min_stock_length < self.min_required_length:
            print(f"Warning: Adjusting sequence lengths due to data availability")
            print(f"Minimum available length: {min_stock_length}")
            # Reserve 1/3 of available data for prediction
            self.pred_len = min_stock_length // 3
            self.seq_len = min_stock_length - self.pred_len
            print(f"Adjusted seq_len: {self.seq_len}, pred_len: {self.pred_len}")
        
        # Process each stock
        for stock in self.stocks:
            stock_df = df[df['ticker'] == stock]
            
            # Interpolate missing values if any
            stock_df = stock_df.sort_values('window_start')
            stock_df[self.features] = stock_df[self.features].interpolate(method='linear')
            
            # Process features
            stock_data = stock_df[self.features].values
            
            # Store statistics for denormalization
            self.feature_stats[stock] = {
                'mean': stock_data.mean(axis=0),
                'std': stock_data.std(axis=0)
            }
            
            # Normalize each feature independently
            stock_data = (stock_data - self.feature_stats[stock]['mean']) / self.feature_stats[stock]['std']
            self.data.append(stock_data)
            
            # Process time features
            time_data = stock_df[self.time_features].values
            self.time_data.append(time_data)
            
            self.stock_lengths[stock] = len(stock_data)
        
        self.data = np.array(self.data, dtype=object)  # Store as object array to handle different lengths
        self.time_data = np.array(self.time_data, dtype=object)
        self.min_length = min(self.stock_lengths.values())
        
        print(f"Loaded {len(data_paths)} files with {self.min_length} timestamps for {len(self.stocks)} stocks")
        print(f"Using sequence length: {self.seq_len}, prediction length: {self.pred_len}")
    
    def denormalize(self, data, stock_idx):
        """Denormalize data for a specific stock"""
        stock = self.stocks[stock_idx]
        return data * self.feature_stats[stock]['std'] + self.feature_stats[stock]['mean']
    
    def __len__(self):
        # Use the shortest stock's length to determine number of sequences
        # Need enough data for input sequence and prediction sequence
        return max(0, self.min_length - self.seq_len - self.pred_len + 1)
    
    def __getitem__(self, index):
        """
        Returns:
            x: Input sequence [n_stocks, seq_len, features]
            x_mark: Time features for input sequence [n_stocks, seq_len, time_features]
            y: Target sequence [n_stocks, pred_len, features]
            y_mark: Time features for target sequence [n_stocks, pred_len, time_features]
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len  # Target sequence should be pred_len long
        
        # Get sequences for all stocks
        x = np.stack([stock_data[s_begin:s_end] for stock_data in self.data])
        y = np.stack([stock_data[r_begin:r_end] for stock_data in self.data])
        
        # Get time features
        x_mark = np.stack([time_data[s_begin:s_end] for time_data in self.time_data])
        y_mark = np.stack([time_data[r_begin:r_end] for time_data in self.time_data])
        
        return x, x_mark, y, y_mark

    def get_last_timestamp(self):
        """Return the last timestamp in the dataset"""
        return self.last_timestamp

def create_data_loaders(config):
    """
    Create train/val/test data loaders
    
    Args:
        config: StockPredictionConfig object
    """
    # Create datasets
    train_dataset = StockDataset(
        config.train_files,  # Use all training files
        config.seq_len,
        config.pred_len,
        label_len=config.label_len,
        stocks=config.stocks,
        features=config.features
    )
    
    test_dataset = StockDataset(
        config.test_files,  # Use all test files
        config.seq_len,
        config.pred_len,
        label_len=config.label_len,
        stocks=config.stocks,
        features=config.features
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader 