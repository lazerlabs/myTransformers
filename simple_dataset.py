"""
Simple Stock Dataset - Returns-Only Implementation

A clean, simplified version that always uses returns-based preprocessing.
"""

from typing import List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import warnings
from tqdm import tqdm

# Import the recursive CSV finding function from utils
from file_utils import find_csv_files


class SimpleStockDataset(Dataset):
    """Simplified stock dataset that always uses returns-based preprocessing."""

    def __init__(self,
                 file_paths: Union[str, List[str]],
                 tickers: Optional[List[str]] = None,
                 seq_len: int = 60,
                 pred_len: int = 30,
                 features: Optional[List[str]] = None,
                 mode: str = 'full_day'):
        """
        Args:
            file_paths: Path(s) to CSV file(s) or directory(ies)
            tickers: List of stock tickers to include (None = all)
            seq_len: Input sequence length
            pred_len: Prediction sequence length
            features: List of features to use (default: ['close'])
            mode: 'sliding_window' or 'full_day'
        """
        print(f"SimpleStockDataset - Mode: {mode}")
        print(f"SimpleStockDataset - Using returns-based preprocessing")
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features or ['close']
        self.mode = mode
        
        # Store first values for denormalization
        self.first_values_ = {}
        
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
            
        print(f"Processing {len(expanded_file_paths)} CSV file(s)...")
        
        self.all_sequences = []
        self.max_input_length = 0
        
        # Process all files
        for file_path in expanded_file_paths:
            self._process_file(file_path, tickers)
            
        self.total_sequences = len(self.all_sequences)
        print(f"Total sequences loaded: {self.total_sequences}")

    def _process_file(self, file_path: str, tickers: Optional[List[str]]):
        """Process a single CSV file."""
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            warnings.warn(f"Error reading {file_path}: {e}")
            return
            
        # Convert timestamps
        if 'window_start' in df.columns:
            try:
                df['window_start'] = pd.to_datetime(df['window_start'], unit='ns')
            except Exception as e:
                warnings.warn(f"Error converting timestamps: {e}")
                return
        
        # Process by ticker or treat whole file as one ticker
        if 'ticker' in df.columns:
            for ticker, group in df.groupby('ticker'):
                if tickers is not None and ticker not in tickers:
                    continue
                self._process_ticker_data(group, ticker)
        else:
            ticker = os.path.splitext(os.path.basename(file_path))[0]
            self._process_ticker_data(df, ticker)

    def _process_ticker_data(self, data: pd.DataFrame, ticker: str):
        """Process data for a single ticker."""
        try:
            # Validate required columns
            missing_features = [f for f in self.features if f not in data.columns]
            if missing_features or 'window_start' not in data.columns:
                return
                
            # Sort by time and extract features
            data = data.sort_values('window_start').reset_index(drop=True)
            feature_data = data[self.features].values.astype(np.float32)
            timestamps = data['window_start'].values
            
            # Handle missing values (simple approach)
            if np.isnan(feature_data).any():
                warnings.warn(f"NaNs found in {ticker}, skipping")
                return
            
            # Create sequences
            sequences = self._create_sequences(feature_data, timestamps, ticker)
            if sequences:
                self.all_sequences.extend(sequences)
                
        except Exception as e:
            warnings.warn(f"Error processing ticker {ticker}: {e}")

    def _create_sequences(self, feature_data: np.ndarray, timestamps: np.ndarray, ticker: str) -> List[Tuple]:
        """Create sequences using returns-based preprocessing."""
        min_length = self.pred_len + 2
        if len(feature_data) < min_length:
            return []
            
        # Calculate returns
        returns_data, first_values = self._calculate_returns(feature_data)
        
        if self.mode == 'sliding_window':
            return self._create_sliding_sequences(returns_data, timestamps, ticker, first_values)
        else:
            return self._create_full_day_sequences(returns_data, timestamps, ticker, first_values)

    def _calculate_returns(self, feature_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate returns (rate of change)."""
        first_values = feature_data[0].copy()
        returns_data = np.zeros((len(feature_data) - 1, feature_data.shape[1]), dtype=np.float32)
        
        for i in range(1, len(feature_data)):
            prev_values = feature_data[i-1]
            curr_values = feature_data[i]
            prev_values_safe = np.where(np.abs(prev_values) < 1e-10, 1e-10, prev_values)
            returns_data[i-1] = (curr_values - prev_values) / prev_values_safe
            
        return returns_data, first_values

    def _create_sliding_sequences(self, returns_data: np.ndarray, timestamps: np.ndarray, 
                                ticker: str, first_values: np.ndarray) -> List[Tuple]:
        """Create sliding window sequences."""
        min_length = self.seq_len + self.pred_len
        if len(returns_data) < min_length:
            return []
            
        sequences = []
        for i in range(len(returns_data) - min_length + 1):
            seq_data = returns_data[i:i + self.seq_len + self.pred_len]
            seq_start_time = timestamps[i + 1]
            
            sequence_id = f"{ticker}_{i}"
            self.first_values_[sequence_id] = first_values.copy()
            sequences.append((seq_data, seq_start_time, ticker))
            
        return sequences

    def _create_full_day_sequences(self, returns_data: np.ndarray, timestamps: np.ndarray,
                                 ticker: str, first_values: np.ndarray) -> List[Tuple]:
        """Create full day sequence."""
        if len(returns_data) < self.pred_len:
            return []
            
        total_length = len(returns_data)
        input_length = total_length - self.pred_len
        self.max_input_length = max(self.max_input_length, input_length)
        
        sequence_id = f"{ticker}_fullday"
        self.first_values_[sequence_id] = first_values.copy()
        
        return [(returns_data, timestamps[1], ticker, input_length)]

    def __len__(self) -> int:
        return self.total_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sequence."""
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
            
            # Pad and create mask
            attention_mask = np.zeros(self.max_input_length, dtype=np.float32)
            attention_mask[:input_length] = 1.0
            
            if len(x_data) < self.max_input_length:
                padding = np.zeros((self.max_input_length - len(x_data), x_data.shape[1]), dtype=np.float32)
                x_data = np.concatenate([x_data, padding], axis=0)
                
            actual_seq_len = input_length

        # Generate simple time features (zeros for now)
        if self.mode == 'full_day':
            x_mark = np.zeros((self.max_input_length, 5), dtype=np.float32)
        else:
            x_mark = np.zeros((self.seq_len, 5), dtype=np.float32)
        y_mark = np.zeros((self.pred_len, 5), dtype=np.float32)
        
        return (
            torch.from_numpy(x_data.astype(np.float32)),
            torch.from_numpy(x_mark),
            torch.from_numpy(y_data.astype(np.float32)),
            torch.from_numpy(y_mark),
            torch.from_numpy(attention_mask)
        )

    def denormalize(self, data: Union[np.ndarray, torch.Tensor], 
                   sequence_indices: Optional[List[int]] = None) -> np.ndarray:
        """Convert returns back to actual prices."""
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
            
        if not self.first_values_:
            warnings.warn("No first values stored for denormalization")
            return data
            
        available_first_values = list(self.first_values_.values())
        
        if len(data.shape) == 3:  # Batch of sequences
            batch_size = data.shape[0]
            reconstructed = np.zeros_like(data)
            
            for batch_idx in range(batch_size):
                seq_idx = sequence_indices[batch_idx] if sequence_indices and batch_idx < len(sequence_indices) else batch_idx
                first_val_idx = seq_idx % len(available_first_values)
                first_values = available_first_values[first_val_idx]
                reconstructed[batch_idx] = self._reconstruct_prices(data[batch_idx], first_values)
                
            return reconstructed
        else:
            return data

    def _reconstruct_prices(self, returns: np.ndarray, first_values: np.ndarray) -> np.ndarray:
        """Reconstruct prices from returns."""
        seq_len, n_features = returns.shape
        reconstructed = np.zeros_like(returns)
        
        if len(first_values) != n_features:
            if len(first_values) < n_features:
                padding = np.full(n_features - len(first_values), 100.0)
                first_values = np.concatenate([first_values, padding])
            else:
                first_values = first_values[:n_features]
        
        current_values = first_values.copy()
        for t in range(seq_len):
            current_values = current_values * (1 + returns[t])
            reconstructed[t] = current_values.copy()
            
        return reconstructed


def create_simple_dataloader(file_paths: Union[str, List[str]],
                           batch_size: int = 32,
                           seq_len: int = 60,
                           pred_len: int = 30,
                           tickers: Optional[List[str]] = None,
                           features: Optional[List[str]] = None,
                           shuffle: bool = True,
                           mode: str = 'full_day',
                           max_samples: Optional[int] = None) -> Tuple[SimpleStockDataset, Optional[DataLoader]]:
    """Create a SimpleStockDataset and DataLoader."""
    print(f"Creating SimpleStockDataset with returns-based preprocessing...")
    
    dataset = SimpleStockDataset(
        file_paths=file_paths,
        tickers=tickers,
        seq_len=seq_len,
        pred_len=pred_len,
        features=features or ['close'],
        mode=mode
    )
    
    if max_samples is not None and len(dataset) > max_samples:
        print(f"Limiting dataset from {len(dataset)} to {max_samples} samples")
        dataset.all_sequences = dataset.all_sequences[:max_samples]
        dataset.total_sequences = len(dataset.all_sequences)
    
    if len(dataset) == 0:
        warnings.warn("Dataset is empty")
        return dataset, None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=len(dataset) >= batch_size
    )
    
    return dataset, dataloader 
