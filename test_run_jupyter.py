# %% [markdown]
# # iTransformer Stock Prediction Test
# This notebook demonstrates the training and testing of the iTransformer model for stock prediction.

# %% [markdown]
# ## Imports and Setup

# %%
import torch
import numpy as np
import json
import os
from IPython.display import display
import matplotlib.pyplot as plt

from configs import StockPredictionConfig
from exp_stock_forecasting import Exp_Stock_Forecast
from utils.loss import get_loss_function
from utils.visualization import StockVisualizer

# Enable inline plotting
%matplotlib inline
plt.style.use('seaborn')

# %% [markdown]
# ## Configuration

# %%
def setup_config():
    config = StockPredictionConfig()
    
    # Model configuration
    config.seq_len = 60      # 1 hour of minute-by-minute data
    config.pred_len = 60     # Predict next hour
    config.label_len = 60    # Set label length to match sequence length
    config.train_epochs = 5  # Number of training epochs
    config.batch_size = 32   # Batch size
    config.d_model = 256     # Model dimension
    config.n_heads = 16      # Number of attention heads
    config.e_layers = 4      # Number of encoder layers
    config.dropout = 0.2     # Dropout rate
    config.test_size = 1     # Number of test files
    
    # Loss configuration
    config.loss_type = "directional"
    config.loss_kwargs = {
        "base_loss": "squared_mae",
        "direction_weight": 0.3
    }
    
    # Device configuration
    if torch.cuda.is_available():
        config.use_gpu = True
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        config.use_gpu = True
        print("Using Apple Silicon MPS")
    else:
        config.use_gpu = False
        print("Using CPU")
    
    return config

# %% [markdown]
# ## Model Training and Visualization

# %%
def save_embeddings(embeddings, file_path):
    """Save embeddings to a JSON file"""
    embeddings_dict = {
        'shape': list(embeddings.shape),
        'data': embeddings.cpu().numpy().tolist()
    }
    
    with open(file_path, 'w') as f:
        json.dump(embeddings_dict, f, indent=2)
    print(f"Embeddings saved to {file_path}")

# %%
def train_and_visualize():
    # Setup
    config = setup_config()
    exp = Exp_Stock_Forecast(config)
    setting = 'jupyter_test_stock_prediction'
    visualizer = StockVisualizer()
    
    try:
        print("Starting training...")
        print(f"Using {config.seq_len} minutes of data to predict next {config.pred_len} minutes")
        
        # Get first batch for embeddings
        train_data, train_loader = exp._get_data(flag='train')
        batch_x, batch_x_mark, batch_y, batch_y_mark = next(iter(train_loader))
        
        # Get embeddings
        batch_x = batch_x.float().to(exp.device)
        batch_x_mark = batch_x_mark.float().to(exp.device)
        
        with torch.no_grad():
            embeddings = exp.model.get_embeddings(batch_x[0:1], batch_x_mark[0:1])
            os.makedirs(config.embeddings_dir, exist_ok=True)
            save_embeddings(embeddings, os.path.join(config.embeddings_dir, 'aapl_embeddings.json'))
        
        # Train model
        model = exp.train(setting)
        
        # Display training metrics plot
        plt.figure(figsize=(12, 6))
        img = plt.imread('./figures/training_metrics.png')
        plt.imshow(img)
        plt.axis('off')
        plt.title('Training Metrics')
        plt.show()
        
        # Display learning rate plot
        plt.figure(figsize=(10, 4))
        img = plt.imread('./figures/learning_rate.png')
        plt.imshow(img)
        plt.axis('off')
        plt.title('Learning Rate Schedule')
        plt.show()
        
        print("\nRunning test prediction...")
        exp.test(setting, test=1)
        
        # Display prediction plots
        plt.figure(figsize=(15, 10))
        img = plt.imread('./figures/all_stocks_predictions_close.png')
        plt.imshow(img)
        plt.axis('off')
        plt.title('Stock Price Predictions')
        plt.show()
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        raise e

# %% [markdown]
# ## Run the Test

# %%
if __name__ == "__main__":
    train_and_visualize() 