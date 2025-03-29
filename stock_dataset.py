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
    """Custom dataset for stock market data with memory-efficient loading and global normalization"""

    # Type hints for attributes
    seq_len: int
    pred_len: int
    scale: bool
    features: List[str]
    mean_: Optional[np.ndarray]
    std_: Optional[np.ndarray]
    all_sequences: List[Tuple[np.ndarray, pd.Timestamp]] # List of (sequence_data, start_timestamp)
    total_sequences: int

    def __init__(self,
                 file_paths: Union[str, List[str]],
                 tickers: Optional[List[str]] = None,
                 seq_len: int = 60,
                 pred_len: int = 30,
                 scale: bool = True,
                 features: Optional[List[str]] = None,
                 global_mean: Optional[np.ndarray] = None,
                 global_std: Optional[np.ndarray] = None):
        """
        Initializes the StockDataset.

        Args:
            file_paths (Union[str, List[str]]): Path(s) to the CSV file(s) for this dataset split.
            tickers (list, optional): List of stock tickers to include.
            seq_len (int): Input sequence length.
            pred_len (int): Prediction sequence length.
            scale (bool): Whether to apply standardization using global stats.
            features (list): List of features to use.
            global_mean (np.ndarray, optional): Global mean calculated from the training set.
            global_std (np.ndarray, optional): Global standard deviation calculated from the training set.
        """
        print(f"StockDataset.__init__ - Received tickers: {tickers}")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        self.mean_ = global_mean # Store global stats
        self.std_ = global_std   # Store global stats
        self.features = features or ['volume', 'close', 'transactions']

        if self.scale and (self.mean_ is None or self.std_ is None):
             warnings.warn("Scaling is enabled, but global_mean or global_std were not provided. Data will not be scaled.")
             self.scale = False # Disable scaling if stats are missing
        elif self.scale:
             print("Global mean/std provided. Scaling enabled.")
             # Ensure stats have correct shape
             expected_shape = (len(self.features),)
             if self.mean_.shape != expected_shape or self.std_.shape != expected_shape:
                 raise ValueError(f"global_mean/std shape mismatch. Expected {expected_shape}, got mean: {self.mean_.shape}, std: {self.std_.shape}")


        # Convert single file path to list
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        print(f"\nProcessing {len(file_paths)} data file(s) for dataset...")
        # self.data_chunks = [] # Old structure - replaced by all_sequences
        self.all_sequences = [] # Store all sequences directly
        # total_sequences = 0 # Not needed here

        for file_path in file_paths:
            print(f"\nProcessing {os.path.basename(file_path)}:")

            # Read and process file in chunks
            try:
                # Make sure window_start is parsed as datetime
                chunks = pd.read_csv(file_path, chunksize=10000, parse_dates=['window_start'])
            except FileNotFoundError:
                warnings.warn(f"File not found: {file_path}. Skipping.")
                continue
            except ValueError as e:
                 warnings.warn(f"Error parsing dates in {file_path}: {e}. Skipping.")
                 continue
            except Exception as e: # Catch other potential pd.read_csv errors
                 warnings.warn(f"Error reading {file_path}: {e}. Skipping.")
                 continue


            file_data = {}  # {ticker: DataFrame} for this file

            for chunk in tqdm(chunks, desc=f"Reading {os.path.basename(file_path)}"):
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


                # Check for sufficient length AFTER combining chunks
                if len(ticker_data) >= (self.seq_len + self.pred_len):
                    # Extract feature data
                    try:
                        feature_data = ticker_data[self.features].values.astype(np.float32) # Ensure float32
                    except KeyError as e:
                         warnings.warn(f"Missing feature {e} for ticker {ticker}. Skipping.")
                         continue

                    # mean = feature_data.mean(axis=0) # REMOVED per-file calculation
                    # std = feature_data.std(axis=0)   # REMOVED per-file calculation

                    # Extract timestamps corresponding to the feature data
                    timestamps = ticker_data['window_start'].values

                    # Create sequences for this ticker
                    sequences_for_ticker = [] # Now stores tuples (sequence_data, start_timestamp)
                    nan_skipped_count = 0
                    for i in range(len(ticker_data) - self.seq_len - self.pred_len + 1):
                        seq_data = feature_data[i : i + self.seq_len + self.pred_len]
                        seq_start_time = timestamps[i] # Get the timestamp for the start of the sequence

                        # Check for NaNs in sequence before scaling
                        if np.isnan(seq_data).any():
                            nan_skipped_count += 1
                            nan_skipped_count += 1
                            continue # Skip sequences with NaNs

                        if self.scale:
                             # Use global stats for normalization
                            seq_data = (seq_data - self.mean_) / (self.std_ + 1e-7) # Apply scaling to seq_data

                        # Append tuple of (data, start_timestamp)
                        sequences_for_ticker.append((seq_data, seq_start_time))

                    if nan_skipped_count > 0:
                         warnings.warn(f"Skipped {nan_skipped_count} sequences containing NaNs for ticker {ticker}.")

                    if sequences_for_ticker:
                        self.all_sequences.extend(sequences_for_ticker) # Add sequences to the main list
                        valid_tickers += 1
                        file_sequences += len(sequences_for_ticker)
                # else:
                #     print(f"Skipping ticker {ticker} due to insufficient length: {len(ticker_data)}")


            print(f"Valid tickers processed in file: {valid_tickers}")
            print(f"Sequences added from file: {file_sequences}")
            # total_sequences += file_sequences # Not needed here

        self.total_sequences = len(self.all_sequences) # Update total count based on list length
        print(f"\nTotal sequences loaded for this dataset split: {self.total_sequences}")
        if self.total_sequences == 0:
            warnings.warn("Dataset created with 0 sequences. Check file paths, ticker lists, and sequence length requirements.")


    def __len__(self) -> int:
        return self.total_sequences # Use the length of the sequence list

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx >= self.total_sequences:
            raise IndexError("Index out of range")

        # Retrieve the pre-processed sequence data and its start timestamp
        sequence_data, start_timestamp = self.all_sequences[idx] # Unpack tuple

        # Split into x and y data
        x_data = sequence_data[:self.seq_len]
        y_data = sequence_data[self.seq_len:]

        # --- Create Time Features ---
        # Generate time index for the input sequence (x)
        # Assuming 'min' frequency based on config.freq (might need to pass freq)
        try:
            x_time_index = pd.date_range(start=start_timestamp, periods=self.seq_len, freq='min')

            # Generate time index for the target sequence (y)
            # Start time is the timestamp after the last input timestamp
            y_start_timestamp = x_time_index[-1] + pd.Timedelta(minutes=1)
            y_time_index = pd.date_range(start=y_start_timestamp, periods=self.pred_len, freq='min')
        except Exception as e:
            # Handle potential errors with timestamp/date_range (e.g., invalid start_timestamp)
            warnings.warn(f"Error generating time index for sequence {idx}: {e}. Returning zero time features.")
            x_mark = np.zeros((self.seq_len, 5), dtype=np.float32) # 5 features: min, hr_sin, hr_cos, dow_sin, dow_cos
            y_mark = np.zeros((self.pred_len, 5), dtype=np.float32)
            return (
                torch.from_numpy(x_data),
                torch.from_numpy(x_mark),
                torch.from_numpy(y_data),
                torch.from_numpy(y_mark)
            )
        x_mark = get_time_features(x_time_index)
        y_mark = get_time_features(y_time_index)
        # --- End Time Features ---

        # DataLoader handles batch dimension. Return tensors directly.
        return (
            torch.from_numpy(x_data),
            torch.from_numpy(x_mark),
            torch.from_numpy(y_data),
            torch.from_numpy(y_mark)
    )

    # Keep get_last_timestamp and denormalize for now, but denormalize needs update
    # TODO: Implement logic to actually store and return the last timestamp if needed.
    def get_last_timestamp(self) -> Optional[pd.Timestamp]:
        """Return the last timestamp encountered during processing (Not currently implemented)."""
        # return self.last_timestamp # Note: self.last_timestamp is not currently set anywhere
        warnings.warn("get_last_timestamp is not fully implemented.")
        return None

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
                     shuffle: bool = True) -> Tuple[StockDataset, Optional[DataLoader]]:
    """
    Creates a StockDataset and DataLoader for the given file paths, applying global normalization if specified.

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
        file_paths=file_paths, # Use the passed file_paths list
        tickers=tickers,
        seq_len=seq_len,
        pred_len=pred_len,
        scale=scale,
        features=features,
        global_mean=global_mean, # Pass global stats
        global_std=global_std    # Pass global stats
    )

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
                shuffle=False # No need to shuffle test data
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