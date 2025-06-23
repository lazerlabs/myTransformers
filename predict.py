import os
import sys
import click
import torch
import random
import numpy as np
import json
import glob
import re
from datetime import datetime
from configs import StockPredictionConfig, get_config_defaults
from exp_stock_forecasting import Exp_Stock_Forecasting
from utils.loss import get_loss_function

# Get config defaults for use in click decorators
_config_defaults = get_config_defaults()

def setup_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def find_latest_checkpoint(base_checkpoints_dir, setting):
    """Find the latest checkpoint for a given setting"""
    if not os.path.exists(base_checkpoints_dir):
        return None, None
    
    # Look for any existing run directories
    existing_runs = [d for d in os.listdir(base_checkpoints_dir) 
                   if os.path.isdir(os.path.join(base_checkpoints_dir, d))]
    
    # Find the most recent checkpoint with compatible settings
    latest_checkpoint = None
    latest_iteration = -1
    latest_run_info = None
    
    for run_id in existing_runs:
        run_checkpoint_dir = os.path.join(base_checkpoints_dir, run_id, setting)
        if os.path.exists(run_checkpoint_dir):
            # Check for main checkpoint first
            main_checkpoint = os.path.join(run_checkpoint_dir, 'checkpoint.pth')
            if os.path.exists(main_checkpoint):
                latest_checkpoint = main_checkpoint
                latest_run_info = {'run_id': run_id, 'type': 'main', 'iteration': 'final'}
                break  # Main checkpoint is the best choice
            
            # Otherwise look for iteration checkpoints
            checkpoint_pattern = os.path.join(run_checkpoint_dir, 'checkpoint_iter_*.pth')
            checkpoint_files = glob.glob(checkpoint_pattern)
            
            for checkpoint_file in checkpoint_files:
                match = re.search(r'checkpoint_iter_(\d+)\.pth', checkpoint_file)
                if match:
                    iteration = int(match.group(1))
                    if iteration > latest_iteration:
                        latest_iteration = iteration
                        latest_checkpoint = checkpoint_file
                        latest_run_info = {'run_id': run_id, 'type': 'iteration', 'iteration': iteration}
    
    if latest_checkpoint and latest_run_info:
        return latest_checkpoint, latest_run_info
    return None, None

@click.command()
# Key parameters - most common ones first
@click.option('--checkpoint-path', type=str, help='Explicit path to checkpoint file to load')
@click.option('--auto-find-checkpoint', is_flag=True, help='Automatically find the latest checkpoint')
@click.option('--run-validation/--run-test', default=True, help='Run validation (default) or test dataset')
@click.option('--model', type=click.Choice(['iTransformer', 'iInformer', 'iReformer', 'iFlowformer', 'iFlashformer', 'Transformer', 'Informer', 'Reformer', 'Flowformer', 'Flashformer']), default=_config_defaults['model'], show_default=True, help='Model name')

# Data Parameters
@click.option('--data-dir', multiple=True, type=str, default=[_config_defaults['data_dir']], show_default=True, help='Directory(ies) containing data files')
@click.option('--stocks', type=str, default=None, help='Comma-separated list of stock tickers')
@click.option('--features', type=str, default=','.join(_config_defaults['features']), show_default=True, help='Comma-separated list of features')
@click.option('--val-stocks', type=str, default=','.join(_config_defaults['val_stocks']), show_default=True, help='Validation stock tickers')

# Model Architecture
@click.option('--seq-len', type=int, default=_config_defaults['seq_len'], show_default=True, help='Input sequence length')
@click.option('--pred-len', type=int, default=_config_defaults['pred_len'], show_default=True, help='Prediction sequence length')
@click.option('--d-model', type=int, default=_config_defaults['d_model'], show_default=True, help='Model dimension')
@click.option('--n-heads', type=int, default=_config_defaults['n_heads'], show_default=True, help='Number of attention heads')
@click.option('--e-layers', type=int, default=_config_defaults['e_layers'], show_default=True, help='Number of encoder layers')

# Data Processing
@click.option('--scale/--no-scale', default=_config_defaults['scale'], help='Whether to scale the data')
@click.option('--use-returns/--no-use-returns', default=_config_defaults['use_returns'], help='Use returns (rate of change) instead of normalization - MUST match training')

# Other common parameters
@click.option('--mode', type=click.Choice(['sliding_window', 'full_day']), default=_config_defaults['mode'], show_default=True, help='Dataset mode')
@click.option('--batch-size', type=int, default=_config_defaults['batch_size'], show_default=True, help='Batch size')
@click.option('--use-gpu/--no-use-gpu', default=_config_defaults['use_gpu'], help='Whether to use GPU')
@click.option('--checkpoints-dir', type=str, default=_config_defaults['checkpoints_dir'], show_default=True, help='Checkpoints directory')
@click.option('--seed', type=int, default=2024, show_default=True, help='Random seed')
def main(**kwargs):
    """Run predictions using a trained model checkpoint."""
    
    setup_seed(kwargs['seed'])
    print("🔮 Stock Forecasting Model Prediction")
    print("=" * 50)
    
    # Load base config and apply overrides
    config = StockPredictionConfig(data_dir=kwargs.get('data_dir', [_config_defaults['data_dir']]))
    
    # Apply CLI overrides to config
    for key, value in kwargs.items():
        if key in ['checkpoint_path', 'auto_find_checkpoint', 'run_validation', 'seed']:
            continue
        config_key = key.replace('-', '_')
        
        if config_key in ['stocks', 'features', 'val_stocks']:
            if isinstance(value, str) and value:
                setattr(config, config_key, [x.strip() for x in value.split(',') if x.strip()])
            elif value is None and config_key == 'stocks':
                pass
        elif hasattr(config, config_key):
            setattr(config, config_key, value)
    
    # Update paths and dimensions
    if kwargs.get('data_dir'):
        config.data_dir = list(kwargs['data_dir'])
        config._initialize_file_paths()
    
    if kwargs.get('features'):
        config._update_model_dimensions()

    # Create experiment setting string
    features_str = '_'.join(config.features) if isinstance(config.features, list) else str(config.features)
    setting = '{}_data{}_{}_{}_ft{}_sl{}_pl{}_dm{}_nh{}_el{}_df{}_eb{}_{}_{}_{}'.format(
        config.model, len(config.train_files), config.mode, features_str,
        config.enc_in, config.seq_len, config.pred_len, config.d_model,
        config.n_heads, config.e_layers, config.d_ff, config.embed,
        config.activation, config.output_attention, config.loss_type
    )
    
    print(f"📊 Model Configuration: {setting}")
    
    # Handle checkpoint loading
    checkpoint_path = kwargs.get('checkpoint_path')
    
    if not checkpoint_path and kwargs.get('auto_find_checkpoint'):
        print("🔍 Searching for latest checkpoint...")
        checkpoint_path, run_info = find_latest_checkpoint(config.checkpoints_dir, setting)
        if checkpoint_path and run_info:
            print(f"✅ Found checkpoint: {checkpoint_path}")
            print(f"📁 Run ID: {run_info['run_id']}, Type: {run_info['type']}, Iteration: {run_info['iteration']}")
        else:
            print(f"❌ No compatible checkpoints found for setting: {setting}")
            sys.exit(1)
    
    if not checkpoint_path:
        print("❌ No checkpoint specified! Use --checkpoint-path or --auto-find-checkpoint")
        sys.exit(1)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"📥 Loading checkpoint: {checkpoint_path}")
    
    # Create experiment and load checkpoint
    try:
        exp = Exp_Stock_Forecasting(config)
        
        checkpoint = torch.load(checkpoint_path, map_location=exp.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            exp.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded training checkpoint (epoch: {checkpoint.get('epoch', 'N/A')}, iter: {checkpoint.get('global_iter', 'N/A')})")
        else:
            exp.model.load_state_dict(checkpoint)
            print("✅ Loaded model state dict")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Run predictions
    test_flag = 0 if kwargs['run_validation'] else 1
    data_type = "validation" if kwargs['run_validation'] else "test"
    
    print(f"\n🔮 Running {data_type} predictions...")
    print("=" * 50)
    
    result = exp.test(setting, test=test_flag)
    
    if result is not None:
        if test_flag == 0:
            print(f"✅ Validation Loss: {result:.6f}")
        else:
            if isinstance(result, tuple) and len(result) == 5:
                mae, mse, rmse, mape, mspe = result
                print(f"✅ Test Metrics - MAE: {mae:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}")
    
    print(f"🎯 Prediction complete! Check figures: {config.figures_dir}")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 
