import os
import torch
import random
import numpy as np
from configs import StockPredictionConfig
from exp_stock_forecasting import Exp_Stock_Forecast
from utils.loss import get_loss_function

def setup_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # Set random seed
    setup_seed(2024)
    
    # Load config
    config = StockPredictionConfig()
    
    # Modify loss configuration to encourage more dynamic predictions
    config.loss_type = "directional"
    config.loss_kwargs = {
        "base_loss": "squared_mae",
        "direction_weight": 0.8  # Increase from 0.3 to give more weight to directional changes
    }
    
    # Create experiment setting name (Removed label_len 'll{}')
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_dm{}_nh{}_el{}_df{}_eb{}_{}_{}_{}'.format(
        config.model,
        config.train_data_path.split('/')[-1].replace('.csv', ''), # Note: train_data_path only gives first file
        config.features,
        config.enc_in,
        config.seq_len,
        # config.label_len, # Removed
        config.pred_len,
        config.d_model,
        config.n_heads,
        config.e_layers,
        config.d_ff,
        config.embed,
        config.activation,
        config.output_attention,
        config.loss_type  # Add loss type to setting name
    )

    # Create experiment
    exp = Exp_Stock_Forecast(config)
    
    # Training
    print('>>>>>>>Start Training>>>>>>>>>>>>>>>>>>>>>>>>>>')
    try:
        exp.train(setting)
    except KeyboardInterrupt:
        print('\n>>>>>>>Early Stopping Due to KeyboardInterrupt<<<<<<<<<<<<<<<')
        
    # Testing
    print('>>>>>>>Testing<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test(setting, test=1)
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 