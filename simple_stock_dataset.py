"""
Simple Stock Dataset - Returns-Only Implementation

A clean, simplified version of the stock dataset that:
- Always uses returns-based preprocessing (no normalization options)
- Much simpler API and implementation
- Better suited for financial time series data
- No global statistics tracking needed
"""

from typing import List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
from tqdm import tqdm
import warnings

# Import the recursive CSV finding function from utils
from file_utils import find_csv_files


class SimpleStockDataset(Dataset):
    """
    Simplified stock dataset that always uses returns-based preprocessing.
    
    Key advantages of returns-based approach:
    - Stable range: returns typically ∈ [-0.1, +0.1] 
    - Perfect reconstruction: P_t = P_0 × ∏(1 + r_i)
    - Neural network friendly: centered around 0, consistent variance
    - No parameter tracking: no global statistics needed
    - Stationary: returns are typically stationary, prices are not
    """

    def __init__(self,
                 file_paths: Union[str, List[str]],
                 tickers: Optional[List[str]] = None,
                 seq_len: int = 60,
                 pred_len: int = 30,
                 features: Optional[List[str]] = None,
                 mode: str = 'full_day',
                 interpolate_max_missing: int = 3):
        """
        Args:
            file_paths: Path(s) to CSV file(s) or directory(ies)
            tickers: List of stock tickers to include (None = all)
            seq_len: Input sequence length (for sliding_window mode)
            pred_len: Prediction sequence length
            features: List of features to use (default: ['close'])
            mode: 'sliding_window' or 'full_day'
            interpolate_max_missing: Max consecutive NaNs to interpolate
        """
        print(f"SimpleStockDataset - Mode: {mode}")
        print(f"SimpleStockDataset - Using returns-based preprocessing")
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features or ['close']
        self.mode = mode
        self.interpolate_max_missing = interpolate_max_missing
        
        # Store first values for each sequence to enable denormalization
        self.first_values_ = {}  # {sequence_id: first_values_array}
        self.first_values = []  # List of first values for each sequence (for visualization)
        
        # Validate mode
        if mode not in ['sliding_window', 'full_day']:
            raise ValueError(f"mode must be 'sliding_window' or 'full_day', got: {mode}")

        # Find all CSV files
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        expanded_file_paths = []
        for path in file_paths:
            found_files = find_csv_files(path)
            expanded_file_paths.extend(found_files)
        
        if not expanded_file_paths:
            warnings.warn("No CSV files found in the provided paths.")
            
        print(f"Processing {len(expanded_file_paths)} CSV file(s) in {mode} mode...")
        
        self.all_sequences = []
        self.max_input_length = 0
        
        # Process all files
        for file_path in expanded_file_paths:
            self._process_file(file_path, tickers)
            
        self.total_sequences = len(self.all_sequences)
        print(f"Total sequences loaded: {self.total_sequences}")
        
        if self.mode == 'full_day' and self.total_sequences > 0:
            print(f"Maximum input sequence length: {self.max_input_length}")

    def _process_file(self, file_path: str, tickers: Optional[List[str]]):
        """Process a single CSV file."""
        print(f"Processing {os.path.basename(file_path)}...")
        
        try:
            # Read in chunks for memory efficiency
            chunks = pd.read_csv(file_path, chunksize=10000)
        except Exception as e:
            warnings.warn(f"Error reading {file_path}: {e}. Skipping.")
            return
            
        file_data = {}  # {ticker: DataFrame}
        
        for chunk in chunks:
            # Convert timestamps
            if 'window_start' in chunk.columns:
                try:
                    chunk['window_start'] = pd.to_datetime(chunk['window_start'], unit='ns')
                except Exception as e:
                    warnings.warn(f"Error converting timestamps: {e}")
                    continue
            
            # Group by ticker or use filename
            if 'ticker' in chunk.columns:
                for ticker, group in chunk.groupby('ticker'):
                    if tickers is not None and ticker not in tickers:
                        continue
                    if ticker not in file_data:
                        file_data[ticker] = []
                    if self._validate_chunk(group, ticker, file_path):
                        file_data[ticker].append(group)
            else:
                # Use filename as ticker
                ticker = os.path.splitext(os.path.basename(file_path))[0]
                if ticker not in file_data:
                    file_data[ticker] = []
                if self._validate_chunk(chunk, ticker, file_path):
                    file_data[ticker].append(chunk)
        
        # Process each ticker's data
        valid_tickers = 0
        for ticker, ticker_chunks in file_data.items():
            if not ticker_chunks:
                continue
                
            try:
                # Combine and sort
                ticker_data = pd.concat(ticker_chunks).sort_values('window_start').reset_index(drop=True)
                
                # Extract features
                feature_data = ticker_data[self.features].values.astype(np.float32)
                timestamps = ticker_data['window_start'].values
                
                # Handle missing values
                feature_data = self._handle_missing_values(feature_data, ticker)
                if feature_data is None:
                    continue
                
                # Create sequences
                sequences = self._create_sequences(feature_data, timestamps, ticker)
                if sequences:
                    self.all_sequences.extend(sequences)
                    valid_tickers += 1
                    
            except Exception as e:
                warnings.warn(f"Error processing ticker {ticker}: {e}")
                continue
                
        print(f"Valid tickers processed: {valid_tickers}")

    def _validate_chunk(self, chunk: pd.DataFrame, ticker: str, file_path: str) -> bool:
        """Validate that chunk has required columns."""
        missing_features = [f for f in self.features if f not in chunk.columns]
        if missing_features:
            warnings.warn(f"Skipping ticker {ticker} - missing features: {missing_features}")
            return False
            
        if 'window_start' not in chunk.columns:
            warnings.warn(f"Skipping ticker {ticker} - missing 'window_start' column")
            return False
            
        return True

    def _handle_missing_values(self, feature_data: np.ndarray, ticker: str) -> Optional[np.ndarray]:
        """Handle missing values with interpolation."""
        if not np.isnan(feature_data).any():
            return feature_data
            
        # Check for long consecutive NaN stretches
        nan_mask = np.isnan(feature_data).any(axis=1)
        consecutive_nans = self._find_consecutive_groups(nan_mask)
        
        for group in consecutive_nans:
            if len(group) > self.interpolate_max_missing:
                warnings.warn(f"Ticker {ticker} has {len(group)} consecutive NaNs (max: {self.interpolate_max_missing})")
                return None
        
        # Interpolate
        result = feature_data.copy()
        for col in range(feature_data.shape[1]):
            col_data = pd.Series(feature_data[:, col])
            col_data = col_data.interpolate(method='linear', limit=self.interpolate_max_missing)
            col_data = col_data.fillna(method='ffill').fillna(method='bfill')
            result[:, col] = col_data.values
            
        if np.isnan(result).any():
            warnings.warn(f"Ticker {ticker} still has NaNs after interpolation")
            return None
            
        return result

    def _find_consecutive_groups(self, mask: np.ndarray) -> List[List[int]]:
        """Find consecutive True groups in boolean mask."""
        groups = []
        current_group = []
        
        for i, is_true in enumerate(mask):
            if is_true:
                current_group.append(i)
            else:
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    
        if current_group:
            groups.append(current_group)
            
        return groups

    def _create_sequences(self, feature_data: np.ndarray, timestamps: np.ndarray, ticker: str) -> List[Tuple]:
        """Create sequences using returns-based preprocessing."""
        # Need at least one extra point to calculate returns
        min_length = self.pred_len + 2  # +1 for returns calculation, +1 for minimum data
        
        if len(feature_data) < min_length:
            return []
            
        # Calculate returns for entire series
        returns_data, first_values = self._calculate_returns(feature_data)
        
        if self.mode == 'sliding_window':
            return self._create_sliding_window_sequences(returns_data, timestamps, ticker, first_values)
        else:  # full_day
            return self._create_full_day_sequence(returns_data, timestamps, ticker, first_values)

    def _calculate_returns(self, feature_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate returns (rate of change) for feature data."""
        if len(feature_data) < 2:
            raise ValueError("Need at least 2 data points to calculate returns")
            
        first_values = feature_data[0].copy()
        returns_data = np.zeros((len(feature_data) - 1, feature_data.shape[1]), dtype=np.float32)
        
        for i in range(1, len(feature_data)):
            prev_values = feature_data[i-1]
            curr_values = feature_data[i]
            
            # Avoid division by zero
            prev_values_safe = np.where(np.abs(prev_values) < 1e-10, 1e-10, prev_values)
            returns_data[i-1] = (curr_values - prev_values) / prev_values_safe
            
        return returns_data, first_values

    def _create_sliding_window_sequences(self, returns_data: np.ndarray, timestamps: np.ndarray, 
                                       ticker: str, first_values: np.ndarray) -> List[Tuple]:
        """Create sliding window sequences."""
        min_length = self.seq_len + self.pred_len
        if len(returns_data) < min_length:
            return []
            
        sequences = []
        for i in range(len(returns_data) - min_length + 1):
            seq_data = returns_data[i:i + self.seq_len + self.pred_len]
            seq_start_time = timestamps[i + 1]  # +1 because returns start from second timestamp
            
            # Store first values for denormalization
            sequence_id = f"{ticker}_{i}"
            self.first_values_[sequence_id] = first_values.copy()
            self.first_values.append(first_values.copy())  # For visualization
            
            sequences.append((seq_data, seq_start_time, ticker))
            
        return sequences

    def _create_full_day_sequence(self, returns_data: np.ndarray, timestamps: np.ndarray,
                                ticker: str, first_values: np.ndarray) -> List[Tuple]:
        """Create full day sequence."""
        if len(returns_data) < self.pred_len:
            return []
            
        total_length = len(returns_data)
        input_length = total_length - self.pred_len
        
        # Track max length for padding
        self.max_input_length = max(self.max_input_length, input_length)
        
        seq_data = returns_data
        seq_start_time = timestamps[1]  # Start from second timestamp
        
        # Store first values
        sequence_id = f"{ticker}_fullday"
        self.first_values_[sequence_id] = first_values.copy()
        self.first_values.append(first_values.copy())  # For visualization
        
        return [(seq_data, seq_start_time, ticker, input_length)]

    def __len__(self) -> int:
        return self.total_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sequence."""
        if idx >= self.total_sequences:
            raise IndexError("Index out of range")

        if self.mode == 'sliding_window':
            sequence_data, start_timestamp, ticker = self.all_sequences[idx]
            x_data = sequence_data[:self.seq_len]
            y_data = sequence_data[self.seq_len:]
            attention_mask = np.ones(self.seq_len, dtype=np.float32)
            actual_seq_len = self.seq_len
            
        else:  # full_day
            sequence_data, start_timestamp, ticker, input_length = self.all_sequences[idx]
            x_data = sequence_data[:input_length]
            y_data = sequence_data[input_length:input_length + self.pred_len]
            
            # Create attention mask and pad
            attention_mask = np.zeros(self.max_input_length, dtype=np.float32)
            attention_mask[:input_length] = 1.0
            
            if len(x_data) < self.max_input_length:
                padding = np.zeros((self.max_input_length - len(x_data), x_data.shape[1]), dtype=np.float32)
                x_data = np.concatenate([x_data, padding], axis=0)
                
            actual_seq_len = input_length

        # Generate time features
        x_mark, y_mark = self._generate_time_features(start_timestamp, actual_seq_len)
        
        return (
            torch.from_numpy(x_data.astype(np.float32)),
            torch.from_numpy(x_mark),
            torch.from_numpy(y_data.astype(np.float32)),
            torch.from_numpy(y_mark),
            torch.from_numpy(attention_mask)
        )

    def _generate_time_features(self, start_timestamp: pd.Timestamp, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate time features for input and target sequences."""
        try:
            # Input time features
            x_time_index = pd.date_range(start=start_timestamp, periods=seq_len, freq='min')
            x_mark = get_time_features(x_time_index)
            
            # Pad if needed for full_day mode
            if self.mode == 'full_day' and len(x_mark) < self.max_input_length:
                padding = np.zeros((self.max_input_length - len(x_mark), 5), dtype=np.float32)
                x_mark = np.concatenate([x_mark, padding], axis=0)
            
            # Target time features
            y_start_timestamp = x_time_index[-1] + pd.Timedelta(minutes=1)
            y_time_index = pd.date_range(start=y_start_timestamp, periods=self.pred_len, freq='min')
            y_mark = get_time_features(y_time_index)
            
        except Exception as e:
            warnings.warn(f"Error generating time features: {e}")
            # Return zero features as fallback
            if self.mode == 'full_day':
                x_mark = np.zeros((self.max_input_length, 5), dtype=np.float32)
            else:
                x_mark = np.zeros((self.seq_len, 5), dtype=np.float32)
            y_mark = np.zeros((self.pred_len, 5), dtype=np.float32)
            
        return x_mark, y_mark

    def get_sequence_info(self, idx: int) -> Tuple[str, pd.Timestamp]:
        """Get ticker and timestamp for a sequence."""
        if idx >= self.total_sequences:
            raise IndexError("Index out of range")
            
        sequence_tuple = self.all_sequences[idx]
        if len(sequence_tuple) >= 4:  # full_day
            _, timestamp, ticker, _ = sequence_tuple
        else:  # sliding_window
            _, timestamp, ticker = sequence_tuple
            
        return ticker, timestamp

    def denormalize(self, data: Union[np.ndarray, torch.Tensor], 
                   sequence_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Convert returns back to actual prices using stored first values.
        
        Args:
            data: Returns data to denormalize
            sequence_indices: List of sequence indices to get appropriate first values
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
            
        if not hasattr(self, 'first_values_') or not self.first_values_:
            warnings.warn("No first values stored for denormalization")
            return data
            
        available_first_values = list(self.first_values_.values())
        
        if len(data.shape) == 2:  # Single sequence
            seq_idx = sequence_indices[0] if sequence_indices else 0
            first_val_idx = seq_idx % len(available_first_values)
            first_values = available_first_values[first_val_idx]
            return self._reconstruct_prices(data, first_values)
            
        elif len(data.shape) == 3:  # Batch of sequences
            batch_size = data.shape[0]
            reconstructed = np.zeros_like(data)
            
            for batch_idx in range(batch_size):
                seq_idx = sequence_indices[batch_idx] if sequence_indices and batch_idx < len(sequence_indices) else batch_idx
                first_val_idx = seq_idx % len(available_first_values)
                first_values = available_first_values[first_val_idx]
                reconstructed[batch_idx] = self._reconstruct_prices(data[batch_idx], first_values)
                
            return reconstructed
        else:
            warnings.warn(f"Unexpected data shape for denormalization: {data.shape}")
            return data

    def _reconstruct_prices(self, returns: np.ndarray, first_values: np.ndarray) -> np.ndarray:
        """Reconstruct prices from returns: price[t] = price[t-1] * (1 + return[t])"""
        seq_len, n_features = returns.shape
        reconstructed = np.zeros_like(returns)
        
        # Ensure first_values matches features
        if len(first_values) != n_features:
            if len(first_values) < n_features:
                padding = np.full(n_features - len(first_values), 100.0)  # Default price
                first_values = np.concatenate([first_values, padding])
            else:
                first_values = first_values[:n_features]
        
        current_values = first_values.copy()
        for t in range(seq_len):
            current_values = current_values * (1 + returns[t])
            reconstructed[t] = current_values.copy()
            
        return reconstructed


def get_time_features(time_index: pd.DatetimeIndex) -> np.ndarray:
    """Extract normalized time features."""
    minute = time_index.minute.values.astype(np.float32) / 59.0
    hour_sin = np.sin(2 * np.pi * time_index.hour.values / 24.0).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * time_index.hour.values / 24.0).astype(np.float32)
    dayofweek_sin = np.sin(2 * np.pi * time_index.dayofweek.values / 7.0).astype(np.float32)
    dayofweek_cos = np.cos(2 * np.pi * time_index.dayofweek.values / 7.0).astype(np.float32)
    
    return np.stack([minute, hour_sin, hour_cos, dayofweek_sin, dayofweek_cos], axis=-1)


def create_simple_dataloader(file_paths: Union[str, List[str]],
                           batch_size: int = 32,
                           seq_len: int = 60,
                           pred_len: int = 30,
                           tickers: Optional[List[str]] = None,
                           features: Optional[List[str]] = None,
                           shuffle: bool = True,
                           mode: str = 'full_day',
                           interpolate_max_missing: int = 3,
                           max_samples: Optional[int] = None) -> Tuple[SimpleStockDataset, Optional[DataLoader]]:
    """
    Create a SimpleStockDataset and DataLoader with returns-based preprocessing.
    
    Args:
        file_paths: Path(s) to CSV file(s) or directory(ies)
        batch_size: Batch size for DataLoader
        seq_len: Input sequence length
        pred_len: Prediction length
        tickers: List of tickers to include (None = all)
        features: List of features (default: ['close'])
        shuffle: Whether to shuffle data
        mode: 'sliding_window' or 'full_day'
        interpolate_max_missing: Max consecutive NaNs to interpolate
        max_samples: Limit dataset size for testing
        
    Returns:
        Tuple of (dataset, dataloader) - dataloader is None if dataset is empty
    """
    print(f"Creating SimpleStockDataset with returns-based preprocessing...")
    
    dataset = SimpleStockDataset(
        file_paths=file_paths,
        tickers=tickers,
        seq_len=seq_len,
        pred_len=pred_len,
        features=features or ['close'],
        mode=mode,
        interpolate_max_missing=interpolate_max_missing
    )
    
    # Apply sample limit if specified
    if max_samples is not None and len(dataset) > max_samples:
        print(f"Limiting dataset from {len(dataset)} to {max_samples} samples")
        dataset.all_sequences = dataset.all_sequences[:max_samples]
        dataset.total_sequences = len(dataset.all_sequences)
    
    if len(dataset) == 0:
        warnings.warn("Dataset is empty - returning None for dataloader")
        return dataset, None
    
    # Use drop_last=False for small datasets
    use_drop_last = len(dataset) >= batch_size
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=use_drop_last
    )
    
    return dataset, dataloader


if __name__ == "__main__":
    # Example usage
    print("Testing SimpleStockDataset...")
    
    # This would need to be updated with actual file paths
    test_files = ["path/to/test/file.csv"]  # Update this
    
    dataset, dataloader = create_simple_dataloader(
        file_paths=test_files,
        batch_size=32,
        seq_len=60,
        pred_len=15,
        features=['close'],
        mode='full_day'
    )
    
    if dataloader:
        print(f"Dataset size: {len(dataset)}")
        
        # Test a batch
        for batch_x, batch_x_mark, batch_y, batch_y_mark, attention_mask in dataloader:
            print(f"Batch shapes:")
            print(f"  Input (x): {batch_x.shape}")
            print(f"  Input time features: {batch_x_mark.shape}")
            print(f"  Target (y): {batch_y.shape}")
            print(f"  Target time features: {batch_y_mark.shape}")
            print(f"  Attention mask: {attention_mask.shape}")
            break
    else:
        print("No data available") 
