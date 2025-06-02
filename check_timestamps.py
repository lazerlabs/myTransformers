#!/usr/bin/env python3
"""
Script to verify that timestamp reconstruction is working correctly.
"""

import numpy as np
import pandas as pd
from utils.visualization import StockVisualizer
from configs import StockPredictionConfig
from stock_dataset import StockDataset

def test_timestamp_conversion():
    """Test direct timestamp conversion from nanoseconds."""
    print("Testing Direct Timestamp Conversion")
    print("=" * 50)
    
    # Test with some example nanosecond timestamps
    example_timestamps = [
        1730448000000000000,  # Should be Nov 1, 2024 8:00 AM EST
        1732539600000000000,  # From your dataset sample
        1732540200000000000,  # From your dataset sample
    ]
    
    for i, ts_ns in enumerate(example_timestamps):
        print(f"\nTimestamp {i+1}:")
        print(f"  Nanoseconds: {ts_ns}")
        
        # Convert to seconds
        ts_sec = ts_ns // 1000000000
        print(f"  Seconds: {ts_sec}")
        
        # Convert to datetime (already in EST)
        dt = pd.to_datetime(ts_sec, unit='s')
        print(f"  Datetime (EST): {dt}")
        print(f"  Formatted: {dt.strftime('%Y-%m-%d %H:%M:%S')} EST")

def test_dataset_timestamps():
    """Test timestamp reconstruction with actual dataset."""
    print("\n\nTesting with Actual Dataset")
    print("=" * 50)
    
    # Create actual dataset to get real timestamps
    config = StockPredictionConfig()
    
    try:
        # Create dataset with actual data
        dataset = StockDataset(
            file_paths=config.test_files[:1],  # Use first test file
            seq_len=60,
            pred_len=15,
            scale=False,  # Don't scale for this test
            tickers=['AAPL'],  # Just one ticker for testing
            features=['volume', 'close', 'transactions'],
            mode='sliding_window'  # Use sliding_window for this test
        )
        
        print(f"Dataset loaded with {len(dataset.all_sequences)} sequences")
        
        # Test timestamp reconstruction with real dataset
        visualizer = StockVisualizer()
        visualizer._current_dataset = dataset
        
        for i in range(3):
            print(f"\nTest {i+1} with real dataset:")
            
            # Create mock time features (not used in new implementation)
            hist_time_marks = np.random.rand(60, 5)
            
            # Reconstruct timestamps
            timestamps = visualizer._reconstruct_timestamps(hist_time_marks, 60, 15)
            
            print(f"  Start Time: {timestamps['start_time']}")
            print(f"  Base Date: {timestamps['base_date']}")
            print(f"  Source: {timestamps['timestamp_source']}")
            
            if timestamps['start_time']:
                start_time_obj = pd.to_datetime(timestamps['start_time'])
                hour = start_time_obj.hour
                minute = start_time_obj.minute
                
                print(f"  Hour: {hour}, Minute: {minute}")
                print(f"  Market hours (9 AM - 8 PM): {9 <= hour <= 20}")
                
    except Exception as e:
        print(f"Error creating dataset: {e}")
        print("Falling back to mock test...")
        test_timestamp_reconstruction()

def test_timestamp_reconstruction():
    """Fallback test with mock dataset."""
    print("Testing with Mock Dataset")
    print("=" * 30)
    
    # Create visualizer
    visualizer = StockVisualizer()
    
    # Create mock dataset for date extraction
    class MockDataset:
        def __init__(self):
            self.data_dir = "dataset"
    
    mock_dataset = MockDataset()
    
    # Test multiple timestamp reconstructions
    for i in range(3):
        print(f"\nTest {i+1}:")
        
        # Create mock time features (random realistic values)
        seq_len = 60
        pred_len = 15
        
        # Create time features array
        hist_time_marks = np.random.rand(seq_len, 5)
        
        # Store mock dataset for date extraction
        visualizer._current_dataset = mock_dataset
        
        # Reconstruct timestamps
        timestamps = visualizer._reconstruct_timestamps(hist_time_marks, seq_len, pred_len)
        
        print(f"  Start Time: {timestamps['start_time']}")
        print(f"  Source: {timestamps['timestamp_source']}")

if __name__ == "__main__":
    test_timestamp_conversion()
    test_dataset_timestamps() 
