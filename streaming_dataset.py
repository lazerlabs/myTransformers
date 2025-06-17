"""
Streaming Dataset Manager for Large-Scale Stock Data Processing

This module implements a streaming approach to data loading that allows training
to begin immediately while files are processed in the background.

Key Features:
- Processes files in configurable chunks
- Starts training with initial chunk while processing continues
- Thread-safe dataset expansion during training
- Memory-efficient processing with configurable limits
"""

import os
import threading
import time
import queue
from typing import List, Optional, Tuple, Union, Iterator
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from file_utils import find_csv_files
from stock_dataset import StockDataset, calculate_global_stats


@dataclass
class StreamingConfig:
    """Configuration for streaming data processing"""
    initial_chunk_size: int = 20  # Number of files to process in initial chunk
    streaming_chunk_size: int = 10  # Number of files to process in each streaming chunk
    max_concurrent_files: int = 4  # Maximum files to process concurrently
    max_memory_chunks: int = 50  # Maximum number of chunk datasets to keep in memory
    processing_timeout: float = 300.0  # Timeout for processing a single chunk (seconds)
    enable_background_processing: bool = True  # Whether to enable background processing
    safe_mode: bool = False  # If True, prevent dataset expansion during epoch iteration


class ChunkDataset:
    """Wrapper for a dataset chunk with metadata"""
    def __init__(self, dataset: StockDataset, files: List[str], chunk_id: int):
        self.dataset = dataset
        self.files = files
        self.chunk_id = chunk_id
        self.created_at = time.time()
        self.size = len(dataset)
    
    def __len__(self):
        return len(self.dataset)


class StreamingStockDataset(Dataset):
    """
    A streaming dataset that processes files in chunks and allows dynamic expansion.
    
    This dataset starts with a small initial chunk and continues processing files
    in the background, making them available to the training loop as they're ready.
    """
    
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
                 interpolate_max_missing: int = 3,
                 streaming_config: Optional[StreamingConfig] = None):
        
        self.streaming_config = streaming_config or StreamingConfig()
        
        # Store dataset parameters
        self.tickers = tickers
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        self.features = features
        self.global_mean = global_mean
        self.global_std = global_std
        self.mode = mode
        self.interpolate_max_missing = interpolate_max_missing
        
        # Expand file paths
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        expanded_file_paths = []
        for path in file_paths:
            found_files = find_csv_files(path)
            expanded_file_paths.extend(found_files)
        
        self.all_file_paths = expanded_file_paths
        print(f"StreamingStockDataset: Found {len(self.all_file_paths)} total files to process")
        
        if not self.all_file_paths:
            raise ValueError("No CSV files found in the provided paths")
        
        # Threading and state management
        self._lock = threading.RLock()
        self._chunk_datasets: List[ChunkDataset] = []
        self._processing_queue = queue.Queue()
        self._processed_files = set()
        self._processing_complete = False
        self._background_thread = None
        self._current_total_size = 0
        self._epoch_in_progress = False  # Track if we're in the middle of an epoch
        
        # Populate processing queue
        self._create_file_chunks()
        
        # Process initial chunk synchronously
        print(f"Processing initial chunk of {self.streaming_config.initial_chunk_size} files...")
        self._process_initial_chunk()
        
        # Start background processing if enabled
        if self.streaming_config.enable_background_processing and not self._processing_complete:
            self._start_background_processing()
    
    def _create_file_chunks(self):
        """Create chunks of files for processing"""
        # Initial chunk
        initial_files = self.all_file_paths[:self.streaming_config.initial_chunk_size]
        if initial_files:
            self._processing_queue.put(('initial', initial_files))
        
        # Remaining chunks
        remaining_files = self.all_file_paths[self.streaming_config.initial_chunk_size:]
        chunk_size = self.streaming_config.streaming_chunk_size
        
        for i in range(0, len(remaining_files), chunk_size):
            chunk_files = remaining_files[i:i + chunk_size]
            chunk_id = f"chunk_{i // chunk_size + 1}"
            self._processing_queue.put((chunk_id, chunk_files))
    
    def _process_initial_chunk(self):
        """Process the initial chunk synchronously"""
        try:
            chunk_type, files = self._processing_queue.get_nowait()
            if chunk_type == 'initial':
                chunk_dataset = self._create_chunk_dataset(files, 0)
                if chunk_dataset and len(chunk_dataset) > 0:
                    with self._lock:
                        self._chunk_datasets.append(chunk_dataset)
                        self._processed_files.update(files)
                        self._current_total_size += len(chunk_dataset)
                    print(f"✅ Initial chunk processed: {len(chunk_dataset)} sequences from {len(files)} files")
                else:
                    print("⚠️ Initial chunk produced no valid sequences")
        except queue.Empty:
            print("⚠️ No initial chunk to process")
        
        if self._processing_queue.empty():
            self._processing_complete = True
    
    def _start_background_processing(self):
        """Start background thread for processing remaining chunks"""
        print(f"🔄 Starting background processing for {self._processing_queue.qsize()} remaining chunks...")
        self._background_thread = threading.Thread(
            target=self._background_worker,
            daemon=True,
            name="StreamingDatasetWorker"
        )
        self._background_thread.start()
    
    def _background_worker(self):
        """Background worker that processes file chunks"""
        chunk_counter = 1
        
        while not self._processing_queue.empty():
            try:
                chunk_type, files = self._processing_queue.get(timeout=1.0)
                
                start_time = time.time()
                
                # Process with clear background labeling
                chunk_dataset = self._create_chunk_dataset(files, chunk_counter, quiet=True)
                
                if chunk_dataset and len(chunk_dataset) > 0:
                    # In safe mode, wait for epoch to complete before adding data
                    if self.streaming_config.safe_mode:
                        while self._epoch_in_progress:
                            time.sleep(0.5)  # Wait for epoch to complete
                    
                    with self._lock:
                        # In safe mode or during active epochs, wait before memory management
                        if self.streaming_config.safe_mode and self._epoch_in_progress:
                            # Don't modify chunks during active epoch iteration
                            print(f"⏸️ [Background] Waiting for epoch to complete before adding chunk {chunk_counter}")
                            continue
                        
                        # Memory management: remove oldest chunks if we exceed limit
                        removed_count = 0
                        while (len(self._chunk_datasets) >= self.streaming_config.max_memory_chunks):
                            removed_chunk = self._chunk_datasets.pop(0)
                            self._current_total_size -= removed_chunk.size
                            removed_count += 1
                        
                        if removed_count > 0:
                            print(f"🗑️ [Background] Removed {removed_count} old chunks to make space")
                        
                        self._chunk_datasets.append(chunk_dataset)
                        self._processed_files.update(files)
                        self._current_total_size += len(chunk_dataset)
                    
                    processing_time = time.time() - start_time
                    # Clear completion message with background prefix
                    print(f"✅ [Background] Chunk {chunk_counter}: +{len(chunk_dataset):,} sequences ({processing_time:.1f}s) → Total: {self._current_total_size:,}")
                else:
                    print(f"⚠️ [Background] Chunk {chunk_counter}: No valid sequences from {len(files)} files")
                
                chunk_counter += 1
                
                # Small delay to reduce CPU contention with main training
                time.sleep(0.2)
                
            except queue.Empty:
                break
            except Exception as e:
                print(f"❌ [Background] Chunk {chunk_counter} error: {e}")
                chunk_counter += 1
                continue
        
        with self._lock:
            self._processing_complete = True
        print(f"🎉 [Background] Processing completed! Final dataset size: {self._current_total_size:,} sequences from {len(self._processed_files)} files")
    
    def _create_chunk_dataset(self, files: List[str], chunk_id: int, quiet: bool = False) -> Optional[ChunkDataset]:
        """Create a dataset from a chunk of files"""
        try:
            if quiet:
                # Use a safer approach - just prefix background messages
                print(f"🔄 [Background] Processing chunk {chunk_id} with {len(files)} files...")
            
            dataset = StockDataset(
                file_paths=files,
                tickers=self.tickers,
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                scale=self.scale,
                features=self.features,
                global_mean=self.global_mean,
                global_std=self.global_std,
                mode=self.mode,
                interpolate_max_missing=self.interpolate_max_missing
            )
            
            return ChunkDataset(dataset, files, chunk_id)
            
        except Exception as e:
            if not quiet:
                print(f"❌ Error creating chunk dataset for chunk {chunk_id}: {e}")
            return None
    
    def __len__(self) -> int:
        """Return current total size (may grow during training)"""
        with self._lock:
            return self._current_total_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item by index, searching through chunk datasets"""
        with self._lock:
            # Handle out-of-range access gracefully by wrapping around
            # This prevents crashes when chunks are swapped during training
            if len(self._chunk_datasets) == 0:
                raise IndexError("No chunks available in dataset")
            
            # Calculate actual available size
            actual_size = sum(len(chunk) for chunk in self._chunk_datasets)
            if actual_size == 0:
                raise IndexError("Dataset has no sequences available")
            
            # Wrap index if it's out of range (handles chunk swapping gracefully)
            if idx >= actual_size:
                print(f"⚠️ Index {idx} out of range (size: {actual_size}), wrapping to avoid crash")
                idx = idx % actual_size
            
            current_idx = 0
            for chunk in self._chunk_datasets:
                if idx < current_idx + len(chunk):
                    local_idx = idx - current_idx
                    return chunk.dataset[local_idx]
                current_idx += len(chunk)
            
        # This should never happen with the modulo fix above, but keep as failsafe
        raise IndexError(f"Index {idx} out of range for dataset of size {actual_size}")
    
    def get_processing_status(self) -> dict:
        """Get current processing status"""
        with self._lock:
            return {
                'total_files': len(self.all_file_paths),
                'processed_files': len(self._processed_files),
                'remaining_files': len(self.all_file_paths) - len(self._processed_files),
                'chunks_loaded': len(self._chunk_datasets),
                'total_sequences': self._current_total_size,
                'processing_complete': self._processing_complete,
                'background_active': self._background_thread is not None and self._background_thread.is_alive()
            }
    
    def wait_for_initial_data(self, min_sequences: int = 1000, timeout: float = 300.0) -> bool:
        """Wait for minimum amount of data to be available"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if len(self) >= min_sequences:
                return True
            time.sleep(1.0)
        return False
    
    def start_epoch(self):
        """Mark that an epoch is starting (for safe mode)"""
        if self.streaming_config.safe_mode:
            self._epoch_in_progress = True
    
    def end_epoch(self):
        """Mark that an epoch has ended (for safe mode)"""
        if self.streaming_config.safe_mode:
            self._epoch_in_progress = False
    
    def cleanup(self):
        """Clean up background resources"""
        if self._background_thread and self._background_thread.is_alive():
            print("🛑 Stopping background processing...")
            # Note: We use daemon threads, so they'll stop when main thread exits
        
        with self._lock:
            self._chunk_datasets.clear()
            self._current_total_size = 0
    
    def get_denormalization_stats(self):
        """Get denormalization stats from the first available chunk"""
        with self._lock:
            if self._chunk_datasets:
                return self._chunk_datasets[0].dataset.mean_, self._chunk_datasets[0].dataset.std_
        return None, None
    
    def denormalize(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Denormalize data using stats from first chunk"""
        with self._lock:
            if self._chunk_datasets:
                return self._chunk_datasets[0].dataset.denormalize(data)
        
        # Fallback: return data as-is if no chunks available
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        return data


class StreamingDataLoader:
    """
    A wrapper around DataLoader that handles streaming datasets and 
    provides utilities for monitoring processing progress.
    """
    
    def __init__(self, 
                 streaming_dataset: StreamingStockDataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 0):
        
        self.streaming_dataset = streaming_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # Wait for initial data
        if not self.streaming_dataset.wait_for_initial_data(min_sequences=batch_size):
            raise RuntimeError("Failed to get initial data within timeout")
        
        self._create_dataloader()
    
    def _create_dataloader(self):
        """Create the underlying DataLoader"""
        use_drop_last = len(self.streaming_dataset) >= self.batch_size
        
        self.dataloader = DataLoader(
            self.streaming_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=use_drop_last
        )
    
    def refresh_dataloader(self):
        """Refresh the DataLoader to handle dynamic dataset size changes"""
        old_size = len(self.streaming_dataset)
        self._create_dataloader()
        new_size = len(self.streaming_dataset)
        if new_size != old_size:
            print(f"🔄 DataLoader refreshed: {old_size:,} → {new_size:,} sequences")
    
    def __iter__(self):
        """Iterate through the dataloader"""
        # Mark epoch start for safe mode
        self.streaming_dataset.start_epoch()
        try:
            return iter(self.dataloader)
        except Exception:
            # Make sure to end epoch even if iteration fails
            self.streaming_dataset.end_epoch()
            raise
    
    def end_epoch(self):
        """Call this at the end of each epoch to allow dataset expansion"""
        self.streaming_dataset.end_epoch()
    
    def __len__(self):
        """Return length of current dataloader"""
        return len(self.dataloader)
    
    def get_status(self) -> dict:
        """Get processing status"""
        return self.streaming_dataset.get_processing_status()
    
    def print_status(self):
        """Print current processing status"""
        status = self.get_status()
        progress_pct = (status['processed_files'] / status['total_files']) * 100
        print(f"📊 Streaming Status: {status['processed_files']}/{status['total_files']} files processed "
              f"({progress_pct:.0f}%), {status['total_sequences']:,} sequences available, "
              f"Background: {'Active' if status['background_active'] else 'Complete'}")


def create_streaming_dataloader(
    file_paths: Union[str, List[str]],
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
    streaming_config: Optional[StreamingConfig] = None
) -> Tuple[StreamingStockDataset, StreamingDataLoader]:
    """
    Create a streaming dataset and dataloader.
    
    Returns:
        Tuple of (StreamingStockDataset, StreamingDataLoader)
    """
    
    print(f"🚀 Creating streaming dataloader...")
    
    streaming_dataset = StreamingStockDataset(
        file_paths=file_paths,
        tickers=tickers,
        seq_len=seq_len,
        pred_len=pred_len,
        scale=scale,
        features=features,
        global_mean=global_mean,
        global_std=global_std,
        mode=mode,
        interpolate_max_missing=interpolate_max_missing,
        streaming_config=streaming_config
    )
    
    streaming_dataloader = StreamingDataLoader(
        streaming_dataset=streaming_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # Keep at 0 for thread safety with background processing
    )
    
    print(f"✅ Streaming dataloader created with {len(streaming_dataset)} initial sequences")
    
    return streaming_dataset, streaming_dataloader
