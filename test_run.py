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

def train_model(model, train_loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for i, batch_data in enumerate(pbar):
            # Properly unpack the batch data
            batch_x, batch_x_mark, batch_y, batch_y_mark = batch_data
            
            # Move data to the appropriate device
            batch_x = batch_x.float().to(model.device)
            batch_x_mark = batch_x_mark.float().to(model.device)
            batch_y = batch_y.float().to(model.device)
            batch_y_mark = batch_y_mark.float().to(model.device)
            
            optimizer.zero_grad()
            outputs = model(batch_x, batch_x_mark, batch_x, batch_x_mark)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            # Update running loss
            running_loss += loss.item()
            
            # Update progress bar with current loss
            pbar.set_postfix({
                'samples': (i + 1) * batch_x.size(0),
                'loss': f'{running_loss/(i+1):.4f}'
            })
        
        # Print epoch summary
        avg_loss = running_loss / len(train_loader)
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Average Loss: {avg_loss:.4f}')
        print(f'Total Samples: {len(train_loader.dataset)}')

def test_training():
    # Modified config for quick test
    config = StockPredictionConfig()
    
    # Override specific parameters for testing
    config.seq_len = 60     # 1 hour of minute-by-minute data
    config.pred_len = 15    # Changed from 60 to 15 to match README
    config.label_len = 30   # Changed from 60 to 30 to match README
    config.batch_size = 32   
    config.d_model = 512    # Changed from 256 to 512 to match README
    config.n_heads = 8      # Changed from 16 to 8 to match README
    config.e_layers = 4     
    config.dropout = 0.2    
    config.test_size = 1    
    
    # Add loss function configuration
    config.loss_type = "directional"
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