import torch
from configs import StockPredictionConfig
from exp_stock_forecasting import Exp_Stock_Forecast
from utils.loss import get_loss_function
import numpy as np
import json
import os
from tqdm import tqdm

def save_embeddings(embeddings, file_path):
    """Save embeddings to a JSON file in a readable format"""
    # Convert embeddings to numpy and then to list for JSON serialization
    embeddings_dict = {
        'shape': list(embeddings.shape),
        'data': embeddings.cpu().numpy().tolist()
    }
    
    with open(file_path, 'w') as f:
        json.dump(embeddings_dict, f, indent=2)
    print(f"Embeddings saved to {file_path}")

# Removed unused train_model function

def test_training():
    # Modified config for quick test
    config = StockPredictionConfig()
    
    # Override specific parameters for testing
    config.seq_len = 60     # 1 hour of minute-by-minute data
    config.pred_len = 15    # Changed from 60 to 15 to match README
    # config.label_len = 30   # Removed - label_len is no longer used
    config.batch_size = 32
    config.d_model = 512    # Changed from 256 to 512 to match README
    config.n_heads = 8      # Changed from 16 to 8 to match README
    config.e_layers = 4
    config.dropout = 0.2
    config.test_size = 5

    # --- Settings for Quick Test Run ---
    config.train_epochs = 1            # Run only 1 epoch
    config.max_train_iterations = 10    # Run only 1000 batches within that epoch
    print(f"--- QUICK TEST RUN: Limiting to {config.train_epochs} epoch(s) and {config.max_train_iterations} iterations ---")
    # --- End Quick Test Run Settings ---

    # Add loss function configuration
    config.loss_type = "directional"
    config.loss_kwargs = {
        "base_loss": "mae",
        "direction_weight": 0.3
    }
    
    # Automatically select best available device
    if torch.cuda.is_available():
        config.use_gpu = True
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        config.use_gpu = True
        print("Using Apple Silicon MPS")
    else:
        config.use_gpu = False
        print("Using CPU")
    
    # Create and run experiment
    exp = Exp_Stock_Forecast(config)
    
    # Training setting name
    setting = 'test_run_stock_prediction'
    
    try:
        print("Starting test training...")
        print(f"Using {config.seq_len} minutes of data to predict next {config.pred_len} minutes")
        
        # Get first batch of data
        train_data, train_loader = exp._get_data(flag='train')
        
        # Debug print
        print(f"\nDataset info:")
        print(f"Total sequences: {len(train_data)}")
        print(f"Batch size: {config.batch_size}")
        
        # Get first batch with error checking
        batch_iterator = iter(train_loader)
        try:
            batch_x, batch_x_mark, batch_y, batch_y_mark = next(batch_iterator)
            print(f"Batch shapes:")
            print(f"batch_x: {batch_x.shape}")
            print(f"batch_x_mark: {batch_x_mark.shape}")
        except StopIteration:
            raise RuntimeError("Could not get first batch - dataloader is empty")
        
        if batch_x.numel() == 0:
            raise RuntimeError("Got empty batch")
            
        # Move to device
        batch_x = batch_x.float().to(exp.device)
        batch_x_mark = batch_x_mark.float().to(exp.device)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = exp.model.get_embeddings(batch_x[0:1], batch_x_mark[0:1])
            
            # Create directory if it doesn't exist
            os.makedirs(config.embeddings_dir, exist_ok=True)
            
            # Save embeddings
            save_embeddings(embeddings, os.path.join(config.embeddings_dir, 'stock_embeddings.json'))
        
        # Continue with training
        model = exp.train(setting)
        
        print("\nRunning test prediction...")
        exp.test(setting, test=1)
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        raise e

if __name__ == "__main__":
    test_training() 