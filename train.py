import os
import torch
import random
import numpy as np
from configs import StockPredictionConfig
from exp_stock_forecasting import Exp_Stock_Forecast

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
    
    # Create experiment setting name
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_df{}_eb{}_{}_{}'.format(
        config.model,
        config.train_data_path.split('/')[-1].replace('.csv', ''),
        config.features,
        config.enc_in,
        config.seq_len,
        config.label_len,
        config.pred_len,
        config.d_model,
        config.n_heads,
        config.e_layers,
        config.d_ff,
        config.embed,
        config.activation,
        config.output_attention
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