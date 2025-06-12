from typing import List, Optional, Tuple, Union # Added typing imports
import pandas as pd
# Removed duplicate pandas import
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
from configs import StockPredictionConfig
from tqdm import tqdm
import warnings

class StockDataset(Dataset):
    """
    Custom dataset for stock market data with memory-efficient loading and global normalization.

    Memory Efficiency:
        - Only one CSV file is processed at a time, in chunks, to avoid loading all data into memory at once.
        - For each file, data is grouped by ticker in memory only for the duration of processing that file.
        - All valid sequences are stored in memory for random access, as required by PyTorch's Dataset interface.
        - This design ensures scalability for large numbers of files and tickers, as specified in REFACTOR_PLAN.md.

    Note:
        - If a single file contains a very large amount of data for a single ticker, memory usage may still be high for that ticker during processing.
        - The global statistics calculation (see calculate_global_stats) does load all data for stats at once, which is documented separately.
    """

    # Type hints for attributes
    seq_len: int
    pred_len: int
    scale: bool
    features: List[str]
    mean_: Optional[np.ndarray]
    std_: Optional[np.ndarray]
    all_sequences: List[Tuple[np.ndarray, pd.Timestamp, str]] # List of (sequence_data, start_timestamp, ticker)
    total_sequences: int
    mode: str
    interpolate_max_missing: int
    max_input_length: int

    def __init__(self,
                 file_paths: Union[str, List[str]],
                 tickers: Optional[List[str]] = None,
                 seq_len: int = 60,
                 pred_len: int = 30,
                 scale: bool = True,
                 features: Optional[List[str]] = None,
                 global_mean: Optional[np.ndarray] = None,
                 global_std: Optional[np.ndarray] = None,
                 mode: str = 'full_day',
                 interpolate_max_missing: int = 3):
        """
        Initializes the StockDataset.

        Args:
            file_paths (Union[str, List[str]]): Path(s) to the CSV file(s) for this dataset split.
            tickers (list, optional): List of stock tickers to include.
            seq_len (int): Input sequence length (used for sliding_window mode).
            pred_len (int): Prediction sequence length.
            scale (bool): Whether to apply standardization using global stats.
            features (list): List of features to use.
            global_mean (np.ndarray, optional): Global mean calculated from the training set.
            global_std (np.ndarray, optional): Global standard deviation calculated from the training set.
            mode (str): Mode of operation ('sliding_window' or 'full_day').
            interpolate_max_missing (int): Maximum number of consecutive NaNs to interpolate.
        """
        print(f"StockDataset.__init__ - Mode: {mode}")
        print(f"StockDataset.__init__ - Received tickers: {tickers}")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        self.mean_ = global_mean # Store global stats
        self.std_ = global_std   # Store global stats
        self.features = features or ['volume', 'close', 'transactions']
        self.mode = mode
        self.interpolate_max_missing = interpolate_max_missing

        if self.scale and (self.mean_ is None or self.std_ is None):
             warnings.warn("Scaling is enabled, but global_mean or global_std were not provided. Data will not be scaled.")
             self.scale = False # Disable scaling if stats are missing
        elif self.scale:
             print("Global mean/std provided. Scaling enabled.")
             # Ensure stats have correct shape
             expected_shape = (len(self.features),)
             if self.mean_.shape != expected_shape or self.std_.shape != expected_shape:
                 raise ValueError(f"global_mean/std shape mismatch. Expected {expected_shape}, got mean: {self.mean_.shape}, std: {self.std_.shape}")

        # Validation for mode
        if mode not in ['sliding_window', 'full_day']:
            raise ValueError(f"mode must be 'sliding_window' or 'full_day', got: {mode}")

        # Convert single file path to list
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        print(f"\nProcessing {len(file_paths)} data file(s) for dataset in {mode} mode...")
        self.all_sequences = []
        self.max_input_length = 0  # Track maximum input sequence length for padding

        # First pass: collect all sequences and find max length
        for file_path in file_paths:
            print(f"\nProcessing {os.path.basename(file_path)}:")

            # Read and process file in chunks
            try:
                # Read CSV without parsing dates first, then convert nanosecond timestamps
                chunks = pd.read_csv(file_path, chunksize=10000)
            except FileNotFoundError:
                warnings.warn(f"File not found: {file_path}. Skipping.")
                continue
            except Exception as e: # Catch other potential pd.read_csv errors
                 warnings.warn(f"Error reading {file_path}: {e}. Skipping.")
                 continue

            file_data = {}  # {ticker: DataFrame} for this file

            for chunk in tqdm(chunks, desc=f"Reading {os.path.basename(file_path)}"):
                # Convert nanosecond timestamps to datetime
                if 'window_start' in chunk.columns:
                    try:
                        # Convert nanosecond timestamps to datetime
                        chunk['window_start'] = pd.to_datetime(chunk['window_start'], unit='ns')
                    except Exception as e:
                        warnings.warn(f"Error converting timestamps in {os.path.basename(file_path)}: {e}. Skipping file.")
                        break
                
                # If ticker column exists, group by it. Otherwise, treat the whole file as one ticker.
                if 'ticker' in chunk.columns:
                    # Group by ticker
                    for ticker, group in chunk.groupby('ticker'):
                        if tickers is not None and ticker not in tickers: # Filter tickers if provided
                            continue
                        if ticker not in file_data:
                            file_data[ticker] = []
                        # Ensure features exist and handle potential missing columns gracefully
                        if not all(feat in group.columns for feat in self.features):
                            warnings.warn(f"Skipping ticker {ticker} in {os.path.basename(file_path)} due to missing feature columns ({[f for f in self.features if f not in group.columns]}).")
                            continue
                        # Ensure window_start exists
                        if 'window_start' not in group.columns:
                            warnings.warn(f"Skipping ticker {ticker} in {os.path.basename(file_path)} due to missing 'window_start' column.")
                            continue
                        file_data[ticker].append(group)
                else:
                    # Use filename as ticker if 'ticker' column is missing
                    # Extract a clean name from the file path, e.g., "my_stock.csv" -> "my_stock"
                    ticker_from_file = os.path.splitext(os.path.basename(file_path))[0]
                    
                    if ticker_from_file not in file_data:
                        file_data[ticker_from_file] = []

                    group = chunk
                    # Ensure features exist in the chunk
                    if not all(feat in group.columns for feat in self.features):
                        warnings.warn(f"Skipping file {os.path.basename(file_path)} due to missing feature columns ({[f for f in self.features if f not in group.columns]}).")
                        continue
                    # Ensure window_start exists
                    if 'window_start' not in group.columns:
                        warnings.warn(f"Skipping file {os.path.basename(file_path)} due to missing 'window_start' column.")
                        continue
                    file_data[ticker_from_file].append(group)

            # Process each ticker's data
            valid_tickers = 0
            file_sequences = 0

            for ticker, ticker_chunks in tqdm(file_data.items(), desc="Processing tickers"):
                if not ticker_chunks: continue # Skip if no data after filtering

                # Combine chunks and sort by time
                try:
                    ticker_data = pd.concat(ticker_chunks).sort_values('window_start').reset_index(drop=True)
                except Exception as e:
                    warnings.warn(f"Error processing data for ticker {ticker}: {e}. Skipping.")
                    continue

                # Extract feature data
                try:
                    feature_data = ticker_data[self.features].values.astype(np.float32)
                except KeyError as e:
                    warnings.warn(f"Missing feature {e} for ticker {ticker}. Skipping.")
                    continue

                # Handle missing values with interpolation
                feature_data = self._handle_missing_values(feature_data, ticker)
                if feature_data is None:
                    continue

                timestamps = ticker_data['window_start'].values

                if self.mode == 'sliding_window':
                    sequences_for_ticker = self._create_sliding_window_sequences(
                        feature_data, timestamps, ticker)
                elif self.mode == 'full_day':
                    sequences_for_ticker = self._create_full_day_sequence(
                        feature_data, timestamps, ticker)

                if sequences_for_ticker:
                    self.all_sequences.extend(sequences_for_ticker)
                    valid_tickers += 1
                    file_sequences += len(sequences_for_ticker)

            print(f"Valid tickers processed in file: {valid_tickers}")
            print(f"Sequences added from file: {file_sequences}")

        self.total_sequences = len(self.all_sequences)
        print(f"\nTotal sequences loaded for this dataset split: {self.total_sequences}")
        
        if self.total_sequences == 0:
            warnings.warn("Dataset created with 0 sequences. Check file paths, ticker lists, and sequence length requirements.")
        
        # Print max input length for full_day mode
        if self.mode == 'full_day' and self.total_sequences > 0:
            print(f"Maximum input sequence length: {self.max_input_length}")

    def _handle_missing_values(self, feature_data: np.ndarray, ticker: str) -> Optional[np.ndarray]:
        """Handle missing values through interpolation with max consecutive limit."""
        if not np.isnan(feature_data).any():
            return feature_data

        # Check for consecutive NaN stretches longer than max allowed
        nan_mask = np.isnan(feature_data).any(axis=1)  # Any NaN in any feature for this timestep
        
        # Find consecutive NaN groups
        consecutive_groups = []
        current_group = []
        
        for i, is_nan in enumerate(nan_mask):
            if is_nan:
                current_group.append(i)
            else:
                if current_group:
                    consecutive_groups.append(current_group)
                    current_group = []
        
        if current_group:  # Handle case where data ends with NaNs
            consecutive_groups.append(current_group)

        # Check if any group exceeds max allowed
        for group in consecutive_groups:
            if len(group) > self.interpolate_max_missing:
                warnings.warn(f"Ticker {ticker} has {len(group)} consecutive NaNs (max allowed: {self.interpolate_max_missing}). Skipping ticker.")
                return None

        # Interpolate missing values
        result = feature_data.copy()
        for col in range(feature_data.shape[1]):
            col_data = pd.Series(feature_data[:, col])
            col_data = col_data.interpolate(method='linear', limit=self.interpolate_max_missing)
            
            # Handle any remaining NaNs (e.g., at start/end)
            col_data = col_data.fillna(method='ffill').fillna(method='bfill')
            result[:, col] = col_data.values

        # Final check
        if np.isnan(result).any():
            warnings.warn(f"Ticker {ticker} still has NaNs after interpolation. Skipping.")
            return None

        return result

    def _create_sliding_window_sequences(self, feature_data: np.ndarray, timestamps: np.ndarray, ticker: str) -> List[Tuple]:
        """Create sliding window sequences (existing logic)."""
        if len(feature_data) < (self.seq_len + self.pred_len):
            return []

        sequences = []
        for i in range(len(feature_data) - self.seq_len - self.pred_len + 1):
            seq_data = feature_data[i : i + self.seq_len + self.pred_len]
            seq_start_time = timestamps[i]

            if self.scale:
                seq_data = (seq_data - self.mean_) / (self.std_ + 1e-7)

            sequences.append((seq_data, seq_start_time, ticker))

        return sequences

    def _create_full_day_sequence(self, feature_data: np.ndarray, timestamps: np.ndarray, ticker: str) -> List[Tuple]:
        """Create one sequence using all available data for the ticker."""
        if len(feature_data) < self.pred_len:
            # Filter out tickers with insufficient data (< pred_len data points)
            return []

        # Use all available data except the last pred_len points
        # The last pred_len points become the prediction target
        total_length = len(feature_data)
        input_length = total_length - self.pred_len
        
        # Track maximum input length for global padding
        self.max_input_length = max(self.max_input_length, input_length)
        
        seq_data = feature_data[:total_length]  # All data for input + target
        seq_start_time = timestamps[0]

        if self.scale:
            seq_data = (seq_data - self.mean_) / (self.std_ + 1e-7)

        # Store the actual input length with the sequence
        sequences = [(seq_data, seq_start_time, ticker, input_length)]
        
        return sequences

    def __len__(self) -> int:
        return self.total_sequences # Use the length of the sequence list

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx >= self.total_sequences:
            raise IndexError("Index out of range")

        if self.mode == 'sliding_window':
            # Existing logic
            sequence_data, start_timestamp, ticker = self.all_sequences[idx]
            x_data = sequence_data[:self.seq_len]
            y_data = sequence_data[self.seq_len:]
            
            # Create attention mask (all ones for sliding window since no padding)
            attention_mask = np.ones(self.seq_len, dtype=np.float32)
            
        elif self.mode == 'full_day':
            # New logic for full day with padding
            sequence_data, start_timestamp, ticker, input_length = self.all_sequences[idx]
            x_data = sequence_data[:input_length]  # Original (unpadded) input
            y_data = sequence_data[input_length:input_length + self.pred_len]   # Target data
            
            # Create attention mask: 1 for real data, 0 for padding
            attention_mask = np.zeros(self.max_input_length, dtype=np.float32)
            attention_mask[:input_length] = 1.0  # Mark real data positions
            
            # Pad x_data to max_input_length if needed
            if len(x_data) < self.max_input_length:
                padding = np.zeros((self.max_input_length - len(x_data), x_data.shape[1]), dtype=np.float32)
                x_data = np.concatenate([x_data, padding], axis=0)
            
            # Update seq_len for time feature generation
            actual_seq_len = input_length
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Generate time features
        try:
            if self.mode == 'full_day':
                x_time_index = pd.date_range(start=start_timestamp, periods=actual_seq_len, freq='min')
                
                # Pad time features to match padded sequence length
                x_mark = get_time_features(x_time_index)
                if len(x_mark) < self.max_input_length:
                    # Pad time features with zeros
                    time_padding = np.zeros((self.max_input_length - len(x_mark), 5), dtype=np.float32)
                    x_mark = np.concatenate([x_mark, time_padding], axis=0)
                    
                y_start_timestamp = x_time_index[-1] + pd.Timedelta(minutes=1)
            else:
                x_time_index = pd.date_range(start=start_timestamp, periods=self.seq_len, freq='min')
                x_mark = get_time_features(x_time_index)
                y_start_timestamp = x_time_index[-1] + pd.Timedelta(minutes=1)
                
            y_time_index = pd.date_range(start=y_start_timestamp, periods=self.pred_len, freq='min')
            y_mark = get_time_features(y_time_index)
            
        except Exception as e:
            warnings.warn(f"Error generating time index for sequence {idx}: {e}. Returning zero time features.")
            if self.mode == 'full_day':
                seq_len_for_features = self.max_input_length
            else:
                seq_len_for_features = self.seq_len
            x_mark = np.zeros((seq_len_for_features, 5), dtype=np.float32)
            y_mark = np.zeros((self.pred_len, 5), dtype=np.float32)
            
            return (
                torch.from_numpy(x_data),
                torch.from_numpy(x_mark),
                torch.from_numpy(y_data),
                torch.from_numpy(y_mark),
                torch.from_numpy(attention_mask)
            )

        return (
            torch.from_numpy(x_data),
            torch.from_numpy(x_mark),
            torch.from_numpy(y_data),
            torch.from_numpy(y_mark),
            torch.from_numpy(attention_mask)
        )

    def get_last_timestamp(self) -> Optional[pd.Timestamp]:
        if self.all_sequences:
            return self.all_sequences[-1][1]  # Return timestamp from last sequence tuple
        return None
    
    def get_ticker_for_sequence(self, idx: int) -> str:
        """Get the ticker symbol for a specific sequence index."""
        if idx >= self.total_sequences:
            raise IndexError("Index out of range")
        return self.all_sequences[idx][2]  # Return ticker from tuple
    
    def get_sequence_info(self, idx: int) -> Tuple[str, pd.Timestamp]:
        """Get ticker and timestamp for a specific sequence index."""
        if idx >= self.total_sequences:
            raise IndexError("Index out of range")
        
        # Handle both modes: sliding_window (3-tuple) and full_day (4-tuple)
        sequence_tuple = self.all_sequences[idx]
        if len(sequence_tuple) >= 4:
            # full_day mode: (sequence_data, start_timestamp, ticker, input_length)
            _, timestamp, ticker, _ = sequence_tuple
        else:
            # sliding_window mode: (sequence_data, start_timestamp, ticker)
            _, timestamp, ticker = sequence_tuple
        return ticker, timestamp

    def denormalize(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Denormalize the data using the stored global mean and std.

        Args:
            data (np.ndarray): Numpy array of shape [..., features] containing normalized data.

        Returns:
            np.ndarray: Denormalized data with the same shape.
        """
        if not self.scale:
            # If scaling was never applied or disabled due to missing stats
            return data
        if self.mean_ is None or self.std_ is None:
             warnings.warn("Attempting to denormalize but global stats are missing.")
             return data # Cannot denormalize

        # Ensure data is numpy array
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()

        # Apply inverse transformation: original = (scaled * std) + mean
        # Broadcasting should handle different input shapes (e.g., [batch, seq, features] or [seq, features])
        try:
            denormalized_data = (data * (self.std_ + 1e-7)) + self.mean_
        except Exception as e:
             warnings.warn(f"Error during denormalization: {e}. Check data shape {data.shape} against stats shape {self.mean_.shape}, {self.std_.shape}")
             return data # Return original data on error

        return denormalized_data


    def get_timestamps(self, start_idx, length):
        """Get a sequence of timestamps starting from start_idx"""
        # Note: self.timestamps is not currently set anywhere
        return self.timestamps[start_idx:start_idx + length] # Note: self.timestamps is not currently set anywhere

# Function to extract and normalize features
def get_time_features(time_index):
    # Normalize minute: 0-59 -> 0-1
    minute = time_index.minute.values.astype(np.float32) / 59.0
    # Cyclical encoding for hour (better)
    hour_sin = np.sin(2 * np.pi * time_index.hour.values / 24.0).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * time_index.hour.values / 24.0).astype(np.float32)
    # Cyclical encoding for day of week (better)
    dayofweek_sin = np.sin(2 * np.pi * time_index.dayofweek.values / 7.0).astype(np.float32)
    dayofweek_cos = np.cos(2 * np.pi * time_index.dayofweek.values / 7.0).astype(np.float32)

    # Stack features: [minute, hour_sin, hour_cos, dayofweek_sin, dayofweek_cos]
    # Shape: [seq_len, num_time_features=5]
    return np.stack([minute, hour_sin, hour_cos, dayofweek_sin, dayofweek_cos], axis=-1)


def calculate_global_stats(file_paths, features, tickers=None):
    """
    Calculates the global mean and standard deviation across all specified files and tickers.

    WARNING:
        - This function loads all relevant data from all specified files into memory at once to compute statistics.
        - For very large datasets, this may cause high memory usage.
        - This is only used for statistics calculation, not for training or inference.

    Args:
        file_paths (list): List of paths to the training CSV files.
        features (list): List of feature names to calculate stats for.
        tickers (list, optional): List of stock tickers to include. Defaults to None (all tickers).

    Returns:
        tuple: (global_mean, global_std) as numpy arrays, or (None, None) if no data.
    Args:
        file_paths (list): List of paths to the training CSV files.
        features (list): List of feature names to calculate stats for.
        tickers (list, optional): List of stock tickers to include. Defaults to None (all tickers).

    Returns:
        tuple: (global_mean, global_std) as numpy arrays, or (None, None) if no data.
    """
    print(f"\nCalculating global statistics from {len(file_paths)} training file(s)...")
    all_feature_data = []

    for file_path in file_paths:
        print(f"Processing {os.path.basename(file_path)} for stats...")
        try:
            # Read the entire file for stats calculation (consider memory for very large files)
            df = pd.read_csv(file_path)
            
            # Convert nanosecond timestamps if window_start column exists
            if 'window_start' in df.columns:
                try:
                    df['window_start'] = pd.to_datetime(df['window_start'], unit='ns')
                except Exception as ts_error:
                    warnings.warn(f"Stats calculation: Error converting timestamps in {os.path.basename(file_path)}: {ts_error}. Continuing without timestamp conversion.")
                    
        except FileNotFoundError:
            warnings.warn(f"Stats calculation: File not found: {file_path}. Skipping.")
            continue
        except Exception as e:
            warnings.warn(f"Stats calculation: Error reading {file_path}: {e}. Skipping.")
            continue

        # Filter tickers if specified
        if tickers:
            df = df[df['ticker'].isin(tickers)]

        if df.empty:
            continue

        # Check for required features
        if not all(feat in df.columns for feat in features):
            warnings.warn(f"Stats calculation: Skipping {os.path.basename(file_path)} due to missing features.")
            continue

        # Extract and append feature data, handling potential NaNs
        feature_data = df[features].values.astype(np.float32)
        if np.isnan(feature_data).any():
             nan_rows = np.isnan(feature_data).any(axis=1)
             warnings.warn(f"Stats calculation: Found NaNs in {os.path.basename(file_path)}. Excluding {nan_rows.sum()} rows with NaNs.")
             feature_data = feature_data[~nan_rows] # Exclude rows with any NaNs

        if feature_data.shape[0] > 0:
            all_feature_data.append(feature_data)

    if not all_feature_data:
        warnings.warn("No valid data found to calculate global statistics.")
        return None, None

    # Concatenate all data and calculate mean/std
    all_feature_data = np.concatenate(all_feature_data, axis=0)
    global_mean = np.mean(all_feature_data, axis=0)
    global_std = np.std(all_feature_data, axis=0)

    print(f"Global Mean: {global_mean}")
    print(f"Global Std: {global_std}")

    # Check for zero std dev
    if np.any(global_std < 1e-7):
        zero_std_features = [features[i] for i, std in enumerate(global_std) if std < 1e-7]
        warnings.warn(f"Features with near-zero standard deviation found: {zero_std_features}. Scaling might be unstable for these.")

    return global_mean, global_std


def create_dataloader(file_paths: Union[str, List[str]],
                     batch_size: int = 32,
                     seq_len: int = 60,
                     pred_len: int = 30,
                     scale: bool = True,
                     tickers: Optional[List[str]] = None,
                     features: Optional[List[str]] = None,
                     global_mean: Optional[np.ndarray] = None,
                     global_std: Optional[np.ndarray] = None,
                     shuffle: bool = True,
                     mode: str = 'full_day',
                     interpolate_max_missing: int = 3,
                     max_samples: Optional[int] = None) -> Tuple[StockDataset, Optional[DataLoader]]:
    """
    Creates a StockDataset and DataLoader for the given file paths, applying global normalization if specified.

    Memory Efficiency:
        The underlying StockDataset processes one file at a time, in chunks, and does not load all CSVs into memory at once.
        Only the final list of valid sequences is kept in memory for random access, ensuring scalability for large datasets.

    Args:
        file_paths (Union[str, List[str]]): Path(s) to the CSV file(s) for this dataloader.
        batch_size (int): Batch size for the DataLoader.
        seq_len (int): Input sequence length.
        pred_len (int): Prediction sequence length.
        scale (bool): Whether to enable scaling using global_mean and global_std.
        tickers (Optional[List[str]]): List of stock tickers to include.
        features (Optional[List[str]]): List of feature column names.
        global_mean (Optional[np.ndarray]): Pre-calculated global mean (from training set).
        global_std (Optional[np.ndarray]): Pre-calculated global standard deviation (from training set).
        shuffle (bool): Whether to shuffle the data in the DataLoader. Should be True for training.
        mode (str): Mode of operation ('sliding_window' or 'full_day').
        interpolate_max_missing (int): Maximum number of consecutive NaNs to interpolate.

    Returns:
        Tuple[StockDataset, Optional[DataLoader]]: The created dataset and DataLoader (or None if dataset is empty).
    """
    # Ensure file_paths is a list for consistent processing
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    print(f"\nCreating DataLoader for {len(file_paths)} file(s)...")
    print(f"Tickers: {'All' if tickers is None else tickers}")
    print(f"Scale: {scale}")

    # Removed logic relying on 'config' object. Parameters are now passed directly.

    dataset = StockDataset(
        file_paths=file_paths,
        tickers=tickers,
        seq_len=seq_len,
        pred_len=pred_len,
        scale=scale,
        features=features,
        global_mean=global_mean,
        global_std=global_std,
        mode=mode,
        interpolate_max_missing=interpolate_max_missing
    )

    # Apply max_samples limit if specified
    if max_samples is not None and len(dataset) > max_samples:
        print(f"Limiting dataset from {len(dataset)} to {max_samples} samples for testing/debugging")
        # Truncate the all_sequences list
        dataset.all_sequences = dataset.all_sequences[:max_samples]
        dataset.total_sequences = len(dataset.all_sequences)

    # Check if dataset creation was successful
    if len(dataset) == 0:
         warnings.warn("DataLoader creation skipped because the dataset is empty.")
         return dataset, None # Return dataset and None for dataloader

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle, # Use the passed shuffle argument
        num_workers=0, # Consider increasing num_workers if I/O is a bottleneck
        drop_last=True # Keep drop_last=True for consistent batch sizes during training
    )

    return dataset, dataloader


if __name__ == "__main__":
    # Example usage - Needs update to reflect new normalization flow
    print("\n--- Running Example Usage ---")
    config = StockPredictionConfig()

    # 1. Calculate global stats from training files
    train_files = config.train_files # Get training files from config
    features = config.features
    mean, std = calculate_global_stats(train_files, features)

    if mean is not None and std is not None:
        # 2. Create dataloader for a specific file (e.g., first test file) using global stats
        test_file_path = config.test_files[0] if config.test_files else None
        if test_file_path:
            dataset, dataloader = create_dataloader(
                file_paths=[test_file_path], # Pass as list
                batch_size=config.batch_size,
                seq_len=config.seq_len,
                pred_len=config.pred_len,
                scale=config.scale,
                features=features,
                global_mean=mean,
                global_std=std,
                shuffle=False, # No need to shuffle test data
                mode=config.mode,
                interpolate_max_missing=config.interpolate_max_missing
            )

            if dataloader:
                # Print dataset info
                print(f"Dataset size: {len(dataset)}")

                # Get a batch
                for batch_x, batch_y, batch_x_mark, batch_y_mark in dataloader:
                    print(f"Batch shapes:")
                    print(f"Input (x): {batch_x.shape}")
                    print(f"Target (y): {batch_y.shape}")
                    break # Only process one batch for example
