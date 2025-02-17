import torch
from configs import StockPredictionConfig
from exp_stock_forecasting import Exp_Stock_Forecast
from utils.loss import get_loss_function
import numpy as np
import json
import os

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

def test_training():
    # Modified config for quick test
    config = StockPredictionConfig()  # Remove train_size=1 to use all files
    
    # Override specific parameters for testing
    config.seq_len = 60     # 1 hour of minute-by-minute data
    config.pred_len = 60    # Predict next hour (changed from 120 to match data loader)
    config.label_len = 60   # Set label length to match sequence length for teacher forcing
    config.train_epochs = 5  # Train for more epochs
    config.batch_size = 32   # Increased batch size since we have shorter sequences
    config.d_model = 256     # Increase model capacity
    config.n_heads = 16      # More attention heads for finer-grained patterns
    config.e_layers = 4      # More layers
    config.dropout = 0.2     # Increase dropout to prevent over-smoothing
    config.test_size = 1     # Use 1 file for testing
    
    # Add loss function configuration
    config.loss_type = "directional"  # or "mse", "squared_mae", etc.
    config.loss_kwargs = {
        "base_loss": "squared_mae",
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
        
        # Get first batch of data for AAPL embeddings
        train_data, train_loader = exp._get_data(flag='train')
        
        # Get first batch
        batch_x, batch_x_mark, batch_y, batch_y_mark = next(iter(train_loader))
        
        # Move to device
        batch_x = batch_x.float().to(exp.device)
        batch_x_mark = batch_x_mark.float().to(exp.device)
        
        # Get embeddings for AAPL (assuming it's the first stock)
        with torch.no_grad():
            # Get embeddings
            embeddings = exp.model.get_embeddings(batch_x[0:1], batch_x_mark[0:1])
            
            # Create directory if it doesn't exist
            os.makedirs(config.embeddings_dir, exist_ok=True)
            
            # Save embeddings
            save_embeddings(embeddings, os.path.join(config.embeddings_dir, 'aapl_embeddings.json'))
        
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