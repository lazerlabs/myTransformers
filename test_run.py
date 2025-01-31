import torch
from configs import StockPredictionConfig
from exp_stock_forecasting import Exp_Stock_Forecast

def test_training():
    # Modified config for quick test
    config = StockPredictionConfig(train_size=1)
    
    # Override specific parameters for testing
    config.seq_len = 60     # 1 hour of minute-by-minute data
    config.pred_len = 60    # Predict next hour (changed from 120 to match data loader)
    config.label_len = 60   # Set label length to match sequence length for teacher forcing
    config.train_epochs = 5  # Train for more epochs
    config.batch_size = 32   # Increased batch size since we have shorter sequences
    config.d_model = 128     # Increased model capacity
    config.n_heads = 8
    config.e_layers = 3      # More layers for better feature extraction
    config.test_size = 1     # Use 1 file for testing
    
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
        model = exp.train(setting)
        
        print("\nRunning test prediction...")
        exp.test(setting, test=1)
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        raise e

if __name__ == "__main__":
    test_training() 