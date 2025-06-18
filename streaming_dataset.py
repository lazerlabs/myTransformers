"""
Interleaved Streaming Dataset Manager for Large-Scale Stock Data Processing

This module implements a simple interleaved approach to data loading that allows training
to begin immediately while files are processed in chunks during training.

Key Features:
- Processes files in configurable chunks during training
- Starts training with initial chunk
- Expands dataset between training cycles  
- No background threads, no epoch dependencies
- Simple and predictable behavior
"""

import os
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from file_utils import find_csv_files
from stock_dataset import StockDataset


class InterleaveStreamingStockDataset(Dataset):
    """
    Interleaved streaming dataset that processes files during training.
    
    Simple approach:
    1. Start with initial chunk of files
    2. During training, when we reach end of current data, process next chunk
    3. Expand dataset dynamically and continue training
    4. No background threads, no epoch dependencies
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
                 chunk_size: int = 3):
        
        # Store all parameters
        self.tickers = tickers
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        self.features = features
        self.global_mean = global_mean
        self.global_std = global_std
        self.mode = mode
        self.interpolate_max_missing = interpolate_max_missing
        self.chunk_size = chunk_size
        
        # Process file paths
        if isinstance(file_paths, str):
            self.all_file_paths = [file_paths]
        else:
            self.all_file_paths = file_paths
            
        if not self.all_file_paths:
            raise ValueError("No files provided for streaming dataset")
            
        print(f"📁 InterleaveStreamingDataset: Found {len(self.all_file_paths)} total files")
        print(f"📊 Will process {self.chunk_size} files at a time during training")
        
        # Initialize state
        self.processed_files = 0
        self.current_datasets = []  # List of processed datasets
        self.total_sequences = 0
        
        # Process initial chunk
        print(f"🚀 Processing initial chunk of {min(self.chunk_size, len(self.all_file_paths))} files...")
        self._process_next_chunk()
        
    def _process_next_chunk(self) -> bool:
        """Process the next chunk of files and add to dataset"""
        if self.processed_files >= len(self.all_file_paths):
            return False  # No more files to process
            
        # Determine files for this chunk  
        start_idx = self.processed_files
        end_idx = min(start_idx + self.chunk_size, len(self.all_file_paths))
        chunk_files = self.all_file_paths[start_idx:end_idx]
        
        print(f"📂 Processing chunk {len(self.current_datasets) + 1}: files {start_idx + 1}-{end_idx} ({len(chunk_files)} files)")
        
        try:
            # Create dataset for this chunk
            chunk_dataset = StockDataset(
                file_paths=chunk_files,
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
            
            # Add to our collection
            self.current_datasets.append(chunk_dataset)
            self.processed_files += len(chunk_files)
            self.total_sequences += len(chunk_dataset)
            
            print(f"✅ Chunk processed: +{len(chunk_dataset)} sequences → Total: {self.total_sequences} sequences")
            print(f"📈 Progress: {self.processed_files}/{len(self.all_file_paths)} files processed")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing chunk: {e}")
            return False
            
    def __len__(self) -> int:
        """Return current total sequences"""
        return self.total_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item by index, searching through datasets"""
        if idx >= self.total_sequences:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_sequences}")
        
        # Find which dataset contains this index
        current_idx = 0
        for dataset in self.current_datasets:
            if idx < current_idx + len(dataset):
                local_idx = idx - current_idx
                return dataset[local_idx]
            current_idx += len(dataset)
            
        raise IndexError(f"Index {idx} not found in datasets")
    
    def can_process_more(self) -> bool:
        """Check if there are more files to process"""
        return self.processed_files < len(self.all_file_paths)
        
    def get_status(self) -> dict:
        """Get current processing status"""
        return {
            'total_files': len(self.all_file_paths),
            'processed_files': self.processed_files,
            'remaining_files': len(self.all_file_paths) - self.processed_files,
            'total_sequences': self.total_sequences,
            'chunks_processed': len(self.current_datasets),
            'processing_complete': self.processed_files >= len(self.all_file_paths)
        }
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up interleaved streaming resources...")
        self.current_datasets.clear()
        self.total_sequences = 0


class InterleaveStreamingDataLoader:
    """
    DataLoader that implements proper interleaved processing:
    1. Read N files
    2. Train on ALL batches from current dataset
    3. Read next N files  
    4. Train on ALL batches from expanded dataset
    5. Repeat until all files processed
    """
    
    def __init__(self, 
                 streaming_dataset: InterleaveStreamingStockDataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 0):
        
        self.streaming_dataset = streaming_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self._create_dataloader()
        
    def _create_dataloader(self):
        """Create the underlying DataLoader"""
        use_drop_last = len(self.streaming_dataset) >= self.batch_size
        
        self.dataloader: DataLoader = DataLoader(
            self.streaming_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=use_drop_last
        )
    
    def __iter__(self):
        """
        Iterate through current dataset only.
        After this iterator is exhausted, the training loop can call expand_dataset_if_needed()
        to check if more data should be processed.
        """
        current_dataset_size = len(self.streaming_dataset)
        print(f"🔄 Training on current dataset: {current_dataset_size:,} sequences ({len(self.dataloader)} batches)")
        
        # Train on ALL batches from current dataset
        batch_count = 0
        for batch in self.dataloader:
            yield batch
            batch_count += 1
        
        print(f"✅ Completed training on {batch_count} batches from {current_dataset_size:,} sequences")
    
    def expand_dataset_if_needed(self) -> bool:
        """
        Expand dataset with next chunk if more files are available.
        Returns True if dataset was expanded, False if no more files or expansion failed.
        """
        if self.streaming_dataset.can_process_more():
            old_size = len(self.streaming_dataset)
            if self.streaming_dataset._process_next_chunk():
                new_size = len(self.streaming_dataset)
                print(f"📂 Dataset expanded: {old_size:,} → {new_size:,} sequences")
                self._create_dataloader()  # Create new dataloader for expanded dataset
                print(f"🔄 Next training cycle will use expanded dataset ({len(self.dataloader)} batches)")
                return True
            else:
                print("❌ Failed to process next chunk")
                return False
        else:
            print("✅ All files processed! No more data to load.")
            return False
    
    def __len__(self):
        """Return current length"""
        return len(self.dataloader)
    
    def get_status(self) -> dict:
        """Get streaming status"""
        status = self.streaming_dataset.get_status()
        status['current_dataloader_length'] = len(self.dataloader)
        return status
    
    def print_status(self):
        """Print current processing status"""
        status = self.get_status()
        progress_pct = (status['processed_files'] / status['total_files']) * 100
        sequences_count = status.get('total_sequences', 0)
        print(f"📊 Streaming Status: {status['processed_files']}/{status['total_files']} files processed "
              f"({progress_pct:.0f}%), {sequences_count:,} sequences available, "
              f"Background: {'Inactive' if status['processing_complete'] else 'Active'}")


def create_interleave_streaming_dataloader(
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
    chunk_size: int = 3
):
    """
    Create an interleaved streaming dataset and dataloader.
    
    Args:
        chunk_size: Number of files to process in each chunk during training
    
    Returns:
        Tuple of (InterleaveStreamingStockDataset, InterleaveStreamingDataLoader)
    """
    
    print(f"🚀 Creating interleaved streaming dataloader (chunk_size={chunk_size})...")
    
    # Create streaming dataset
    streaming_dataset = InterleaveStreamingStockDataset(
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
        chunk_size=chunk_size
    )
    
    # Create streaming dataloader
    streaming_dataloader = InterleaveStreamingDataLoader(
        streaming_dataset=streaming_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # Keep at 0 for simplicity
    )
    
    print(f"✅ Interleaved streaming dataloader created with {len(streaming_dataset)} initial sequences")
    
    return streaming_dataset, streaming_dataloader
