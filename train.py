import os
import sys
import click
import torch
import random
import numpy as np
import json
import subprocess
import threading
import time
import webbrowser
import glob
import re
from datetime import datetime
from configs import StockPredictionConfig, get_config_defaults
from exp_stock_forecasting import Exp_Stock_Forecasting
from utils.loss import get_loss_function
from torch.utils.tensorboard import SummaryWriter

# Get config defaults for use in click decorators
_config_defaults = get_config_defaults()

def setup_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_embeddings(embeddings, file_path):
    """Save embeddings to a JSON file in a readable format"""
    embeddings_dict = {
        'shape': list(embeddings.shape),
        'data': embeddings.cpu().numpy().tolist()
    }
    with open(file_path, 'w') as f:
        json.dump(embeddings_dict, f, indent=2)
    print(f"Embeddings saved to {file_path}")

def start_tensorboard(log_dir, port=6006, host='0.0.0.0'):
    """Start TensorBoard in a separate process"""
    try:
        # Check if TensorBoard is already running on this port
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex((host if host != '0.0.0.0' else 'localhost', port))
        sock.close()
        
        if result == 0:
            print(f"TensorBoard is already running on port {port}")
            return None
        
        # Start TensorBoard
        cmd = [
            'tensorboard', 
            f'--logdir={log_dir}',
            f'--port={port}',
            f'--host={host}',
            '--reload_interval=30'
        ]
        
        print(f"Starting TensorBoard: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give it a moment to start
        time.sleep(2)
        
        # Check if it started successfully
        if process.poll() is None:  # Still running
            print(f"✅ TensorBoard started successfully!")
            print(f"📊 Open http://localhost:{port} in your browser to monitor training")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start TensorBoard:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting TensorBoard: {e}")
        return None

def open_tensorboard_browser(port=6006, delay=3):
    """Open TensorBoard in browser after a delay"""
    def delayed_open():
        time.sleep(delay)
        try:
            webbrowser.open(f'http://localhost:{port}')
            print(f"🌐 Opened TensorBoard in your default browser")
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
    
    thread = threading.Thread(target=delayed_open, daemon=True)
    thread.start()

@click.command()
# Data Parameters
@click.option('--data-dir', multiple=True, type=str, default=[_config_defaults['data_dir']], show_default=True, help='Directory(ies) containing data files (can be used multiple times)')
@click.option('--stocks', type=str, default=None, help='Comma-separated list of stock tickers (e.g. AAPL,MSFT) - defaults to all stocks if not specified')
@click.option('--features', type=str, default=','.join(_config_defaults['features']), show_default=True, help='Comma-separated list of features (e.g. volume,close,transactions)')
@click.option('--train-size', type=int, default=_config_defaults['train_size'], show_default=True, help='Number of files to use for training')
@click.option('--test-size', type=int, default=_config_defaults['test_size'], show_default=True, help='Number of files to use for testing')
@click.option('--val-size', type=int, default=_config_defaults['val_size'], show_default=True, help='Number of files to use for validation')
@click.option('--val-stocks', type=str, default=','.join(_config_defaults['val_stocks']), show_default=True, help='Comma-separated list of validation stock tickers')

# Sequence Parameters
@click.option('--seq-len', type=int, default=_config_defaults['seq_len'], show_default=True, help='Input sequence length')
@click.option('--pred-len', type=int, default=_config_defaults['pred_len'], show_default=True, help='Prediction sequence length')
@click.option('--label-len', type=int, default=_config_defaults['label_len'], show_default=True, help='Label length for teacher forcing')
@click.option('--scale/--no-scale', default=_config_defaults['scale'], help='Whether to scale the data')

# Dataset Mode Parameters
@click.option('--mode', type=click.Choice(['sliding_window', 'full_day']), default=_config_defaults['mode'], show_default=True, help='Dataset mode: sliding_window or full_day')
@click.option('--interpolate-max-missing', type=int, default=_config_defaults['interpolate_max_missing'], show_default=True, help='Maximum consecutive NaNs to interpolate')
@click.option('--max-seq-len', type=int, default=_config_defaults['max_seq_len'], show_default=True, help='Maximum sequence length for embedding layer (for full_day mode)')

# Model Parameters
@click.option(
    '--model',
    type=click.Choice([
        'iTransformer', 'iInformer', 'iReformer', 'iFlowformer', 'iFlashformer',
        'Transformer', 'Informer', 'Reformer', 'Flowformer', 'Flashformer'
    ]),
    default=_config_defaults['model'],
    show_default=True,
    help='Model name (choose from iTransformer, iInformer, iReformer, iFlowformer, iFlashformer, Transformer, Informer, Reformer, Flowformer, Flashformer)'
)
@click.option('--d-model', type=int, default=_config_defaults['d_model'], show_default=True, help='Model dimension')
@click.option('--n-heads', type=int, default=_config_defaults['n_heads'], show_default=True, help='Number of attention heads')
@click.option('--e-layers', type=int, default=_config_defaults['e_layers'], show_default=True, help='Number of encoder layers')
@click.option('--d-ff', type=int, default=_config_defaults['d_ff'], show_default=True, help='Feed-forward dimension')
@click.option('--dropout', type=float, default=_config_defaults['dropout'], show_default=True, help='Dropout rate')
@click.option('--embed', type=str, default=_config_defaults['embed'], show_default=True, help='Embedding type')
@click.option('--activation', type=str, default=_config_defaults['activation'], show_default=True, help='Activation function')
@click.option('--output-attention/--no-output-attention', default=_config_defaults['output_attention'], help='Whether to output attention weights')
@click.option('--use-norm/--no-use-norm', default=_config_defaults['use_norm'], help='Whether to use normalization')

# Inference Parameters
@click.option('--temperature', type=float, default=_config_defaults['temperature'], show_default=True, help='Temperature for inference sampling')

# Training Parameters
@click.option('--batch-size', type=int, default=_config_defaults['batch_size'], show_default=True, help='Batch size')
@click.option('--learning-rate', type=float, default=_config_defaults['learning_rate'], show_default=True, help='Learning rate')
@click.option('--train-epochs', type=int, default=_config_defaults['train_epochs'], show_default=True, help='Number of training epochs')
@click.option('--patience', type=int, default=_config_defaults['patience'], show_default=True, help='Early stopping patience')
@click.option('--max-train-iterations', type=int, default=_config_defaults['max_train_iterations'], show_default=True, help='Maximum iterations per epoch')

# Checkpoint Parameters
@click.option('--save-checkpoint-every-n-iterations', type=int, default=_config_defaults['save_checkpoint_every_n_iterations'], show_default=True, help='Save checkpoint every N iterations')
@click.option('--save-checkpoint-every-n-epochs', type=int, default=_config_defaults['save_checkpoint_every_n_epochs'], show_default=True, help='Save checkpoint every N epochs')

# Loss Parameters
@click.option('--loss-type', type=str, default=_config_defaults['loss_type'], show_default=True, help='Loss function type')
@click.option('--loss-kwargs', type=str, default=json.dumps(_config_defaults['loss_kwargs']), show_default=True, help='Loss function kwargs as JSON string')

# Learning Rate Scheduler Parameters
@click.option('--lr-scheduler', type=str, default=_config_defaults['lr_scheduler'], show_default=True, help='Learning rate scheduler type')
@click.option('--lr-decay-factor', type=float, default=_config_defaults['lr_decay_factor'], show_default=True, help='Learning rate decay factor')
@click.option('--lr-patience', type=int, default=_config_defaults['lr_patience'], show_default=True, help='Learning rate scheduler patience')
@click.option('--min-lr', type=float, default=_config_defaults['min_lr'], show_default=True, help='Minimum learning rate')
@click.option('--warmup-epochs', type=int, default=_config_defaults['warmup_epochs'], show_default=True, help='Number of warmup epochs')

# Device Parameters
@click.option('--use-gpu/--no-use-gpu', default=_config_defaults['use_gpu'], help='Whether to use GPU')
@click.option('--use-multi-gpu/--no-use-multi-gpu', default=_config_defaults['use_multi_gpu'], help='Whether to use multiple GPUs')
@click.option('--gpu', type=int, default=_config_defaults['gpu'], show_default=True, help='GPU device ID')
@click.option('--device-ids', type=str, default=None, help='Comma-separated GPU device IDs (defaults to config default)')

# Directory Parameters - show config defaults in help
@click.option('--checkpoints-dir', type=str, default=_config_defaults['checkpoints_dir'], show_default=True, help='Checkpoints directory')
@click.option('--logs-dir', type=str, default=_config_defaults['logs_dir'], show_default=True, help='Logs directory')
@click.option('--figures-dir', type=str, default=_config_defaults['figures_dir'], show_default=True, help='Figures directory')
@click.option('--embeddings-dir', type=str, default=_config_defaults['embeddings_dir'], show_default=True, help='Embeddings directory')

# Model Specific Parameters
@click.option('--factor', type=int, default=_config_defaults['factor'], show_default=True, help='Probsparse attention factor')
@click.option('--enc-in', type=int, default=_config_defaults['enc_in'], show_default=True, help='Number of input features')
@click.option('--freq', type=str, default=_config_defaults['freq'], show_default=True, help='Time feature encoding frequency')

# Test Parameters
@click.option('--test-interval', type=int, default=_config_defaults['test_interval'], show_default=True, help='Test every N epochs during training')
@click.option('--test-iteration-interval', type=int, default=_config_defaults['test_iteration_interval'], show_default=True, help='Test every N iterations during training')

# Logging Parameters
@click.option('--log-every-n-iterations', type=int, default=_config_defaults['log_every_n_iterations'], show_default=True, help='Log detailed metrics every N iterations')
@click.option('--save-iteration-metrics/--no-save-iteration-metrics', default=_config_defaults['save_iteration_metrics'], help='Save iteration-level metrics for visualization')

# Streaming Parameters
@click.option('--enable-streaming', type=click.Choice(['auto', 'on', 'off']), default='auto', show_default=True, help='Streaming mode: auto (detect based on file count), on (force enable), off (force disable)')
@click.option('--streaming-threshold', type=int, default=_config_defaults['streaming_threshold'], show_default=True, help='File count threshold for auto-enabling streaming')
@click.option('--streaming-chunk-size', type=int, default=_config_defaults['streaming_chunk_size'], show_default=True, help='Number of files to process in each chunk during training')

# File Order Parameters
@click.option('--randomize-train-files/--no-randomize-train-files', default=_config_defaults['randomize_train_files'], help='Randomize training file order to prevent learning day-to-day connections')

# TensorBoard Parameters - CLI-only options with their own defaults
@click.option('--auto-start-tensorboard/--no-auto-start-tensorboard', default=True, help='Automatically start TensorBoard server')
@click.option('--tensorboard-host', type=str, default='0.0.0.0', show_default=True, help='Host for TensorBoard server')
@click.option('--tensorboard-port', type=int, default=6006, show_default=True, help='Port for TensorBoard server')
@click.option('--open-browser/--no-open-browser', default=False, help='Automatically open TensorBoard in browser')

# Special Options - CLI-only options with their own defaults
@click.option('--resume-checkpoint', type=str, help='Path to checkpoint to resume training from')
@click.option('--auto-resume', is_flag=True, help='Automatically resume from latest checkpoint if available')
@click.option('--run-name', type=str, help='Custom name prefix for this run (will be added to timestamp)')
@click.option('--quick-test', is_flag=True, help='Run a quick test (1 epoch, 10 iterations, minimal data)')
@click.option('--extract-embeddings-only', is_flag=True, help='Only extract and save embeddings from the first batch, then exit')
@click.option('--seed', type=int, default=2024, show_default=True, help='Random seed')
def main(**kwargs):
    """Train or test the stock forecasting model. All config parameters can be set via CLI. CLI args override configs.py defaults."""
    
    # Set up seed
    setup_seed(kwargs['seed'])
    
    # Load base config - temporarily skip file initialization if data_dir will be overridden
    if kwargs.get('data_dir') is not None:
        # Create config without triggering file initialization
        config = StockPredictionConfig(data_dir=kwargs['data_dir'])
    else:
        config = StockPredictionConfig()
    
    # Apply quick test overrides first if specified
    if kwargs['quick_test']:
        config.seq_len = 60
        config.pred_len = 15
        config.batch_size = 32
        config.d_model = 512
        config.n_heads = 8
        config.e_layers = 4
        config.dropout = 0.2
        config.test_size = 5
        config.train_epochs = 1
        config.max_train_iterations = 10
        config.loss_type = "directional"
        config.loss_kwargs = {"base_loss": "mae", "direction_weight": 0.3}
        # Set checkpoint saving to happen more frequently for quick tests
        config.save_checkpoint_every_n_iterations = 5  # Save every 5 iterations
        config.val_size = 1  # Ensure we have validation data
    
    # Override config with provided CLI arguments (CLI now has config defaults, so this is clean)
    for key, value in kwargs.items():
        # Skip CLI-only arguments that don't have config counterparts
        if key in ['auto_start_tensorboard', 'tensorboard_host', 'tensorboard_port', 'open_browser', 'resume_checkpoint', 'auto_resume', 'run_name', 'quick_test', 'extract_embeddings_only', 'seed']:
            continue
            
        # Convert kebab-case to snake_case
        config_key = key.replace('-', '_')
        
        # Handle special conversions
        if config_key in ['stocks', 'features', 'val_stocks', 'device_ids']:
            # Convert comma-separated string to list
            if isinstance(value, str) and value:
                setattr(config, config_key, [x.strip() for x in value.split(',') if x.strip()])
            elif value is None and config_key == 'stocks':
                # stocks=None means use all stocks (keep config default)
                pass
            else:
                setattr(config, config_key, value)
        elif config_key == 'loss_kwargs':
            # Parse JSON string for loss_kwargs
            if isinstance(value, str):
                setattr(config, config_key, json.loads(value))
            else:
                setattr(config, config_key, value)
        elif config_key == 'enable_streaming':
            # Handle streaming mode setting
            if value == 'auto':
                setattr(config, config_key, None)  # None means auto-detect
            elif value == 'on':
                setattr(config, config_key, True)
            elif value == 'off':
                setattr(config, config_key, False)
        elif hasattr(config, config_key):
            # Direct assignment for other valid config fields
            setattr(config, config_key, value)
        else:
            print(f"Warning: Unknown config field '{config_key}' from CLI argument '--{key}'")
    
    # Reinitialize file paths if data_dir was changed via CLI
    if kwargs.get('data_dir') is not None:
        # Convert tuple to list if multiple directories provided
        data_dirs = list(kwargs['data_dir']) if kwargs['data_dir'] else [_config_defaults['data_dir']]
        config.data_dir = data_dirs
        config._initialize_file_paths()
    
    # Update model dimensions if features were changed via CLI
    if kwargs.get('features') is not None:
        config._update_model_dimensions()

    # Create experiment
    try:
        exp = Exp_Stock_Forecasting(config)
    except ValueError as e:
        # Handle specific model configuration errors gracefully
        error_msg = str(e)
        if "does not support variable sequence lengths in full_day mode" in error_msg:
            print("❌ Model Configuration Error:")
            print(f"   {config.model} does not support variable sequence lengths in full_day mode.")
            print("   💡 Suggestions:")
            print("      1. Use the local iTransformer model (recommended for market data)")
            print("      2. Switch to sliding_window mode: --mode sliding_window")
            print("      3. Use a fixed sequence length configuration")
            sys.exit(1)
        elif "is a classic (time-based) transformer" in error_msg:
            print("❌ Model Configuration Error:")
            print(f"   {config.model} is a classic (time-based) transformer.")
            print("   💡 For main experiments, only inverted models are allowed.")
            print("      To run classic models for ablation, set allow_classic_models=True")
            sys.exit(1)
        else:
            # Re-raise other ValueErrors with the traceback
            raise
    except Exception as e:
        # Re-raise other exceptions with their tracebacks
        raise
    # Create proper experiment setting string (fix: remove filename dependency)
    features_str = '_'.join(config.features) if isinstance(config.features, list) else str(config.features)
    setting = '{}_data{}_{}_{}_ft{}_sl{}_pl{}_dm{}_nh{}_el{}_df{}_eb{}_{}_{}_{}'.format(
        config.model,
        len(config.train_files),  # Use number of files instead of filename
        config.mode,
        features_str,
        config.enc_in,
        config.seq_len,
        config.pred_len,
        config.d_model,
        config.n_heads,
        config.e_layers,
        config.d_ff,
        config.embed,
        config.activation,
        config.output_attention,
        config.loss_type
    )
    
    # Handle resume checkpoint logic BEFORE creating new directories
    resume_checkpoint_path = kwargs.get('resume_checkpoint')
    
    # Determine if we should resume and extract run_id from checkpoint path
    resume_run_id = None
    if resume_checkpoint_path:
        # Extract run_id from the checkpoint path
        # Expected format: ./checkpoints/RUN_ID/setting/checkpoint_iter_*.pth
        checkpoint_parts = os.path.normpath(resume_checkpoint_path).split(os.sep)
        try:
            # Find the checkpoints directory in the path
            checkpoints_idx = None
            for i, part in enumerate(checkpoint_parts):
                if part == 'checkpoints':
                    checkpoints_idx = i
                    break
            
            if checkpoints_idx is not None and checkpoints_idx + 1 < len(checkpoint_parts):
                resume_run_id = checkpoint_parts[checkpoints_idx + 1]
                print(f"🔄 Resuming from checkpoint: {resume_checkpoint_path}")
                print(f"📁 Extracted run ID: {resume_run_id}")
            else:
                print(f"⚠️  Warning: Could not extract run ID from checkpoint path: {resume_checkpoint_path}")
                print("⚠️  Will create new directories and checkpoint path may become invalid")
        except Exception as e:
            print(f"⚠️  Warning: Error parsing checkpoint path: {e}")
            print("⚠️  Will create new directories and checkpoint path may become invalid")
    elif kwargs.get('auto_resume'):
        # Look for existing checkpoints in base directory structure
        base_checkpoints_dir = config.checkpoints_dir
        # Look for any existing run directories
        if os.path.exists(base_checkpoints_dir):
            existing_runs = [d for d in os.listdir(base_checkpoints_dir) 
                           if os.path.isdir(os.path.join(base_checkpoints_dir, d))]
            
            # Find the most recent run with compatible settings
            latest_checkpoint = None
            latest_iteration = -1
            latest_run_id = None
            
            for run_id in existing_runs:
                run_checkpoint_dir = os.path.join(base_checkpoints_dir, run_id, setting)
                if os.path.exists(run_checkpoint_dir):
                    checkpoint_pattern = os.path.join(run_checkpoint_dir, 'checkpoint_iter_*.pth')
                    checkpoint_files = glob.glob(checkpoint_pattern)
                    
                    for checkpoint_file in checkpoint_files:
                        match = re.search(r'checkpoint_iter_(\d+)\.pth', checkpoint_file)
                        if match:
                            iteration = int(match.group(1))
                            if iteration > latest_iteration:
                                latest_iteration = iteration
                                latest_checkpoint = checkpoint_file
                                latest_run_id = run_id
            
            if latest_checkpoint:
                resume_checkpoint_path = latest_checkpoint
                resume_run_id = latest_run_id
                print(f"🔄 Auto-resume enabled: Found latest checkpoint at iteration {latest_iteration}")
                print(f"📁 Resuming from: {resume_checkpoint_path}")
                print(f"📁 Resume run ID: {resume_run_id}")
            else:
                print("🆕 Auto-resume enabled but no compatible checkpoints found - starting fresh training")
    
    # Create or reuse run directories
    if resume_run_id:
        # Reuse existing run directories
        run_id = resume_run_id
        config.logs_dir = os.path.join(config.logs_dir, run_id)
        config.checkpoints_dir = os.path.join(config.checkpoints_dir, run_id)
        config.figures_dir = os.path.join(config.figures_dir, run_id)
        config.embeddings_dir = os.path.join(config.embeddings_dir, run_id)
        
        print(f"🔖 Resuming Run ID: {run_id}")
        print(f"📁 Using existing directories:")
    else:
        # Create new unique run directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name_prefix = kwargs.get('run_name', '')
        if run_name_prefix:
            run_id = f"{run_name_prefix}_{timestamp}_{config.model}_sl{config.seq_len}_pl{config.pred_len}"
        else:
            run_id = f"{timestamp}_{config.model}_sl{config.seq_len}_pl{config.pred_len}"
        
        config.logs_dir = os.path.join(config.logs_dir, run_id)
        config.checkpoints_dir = os.path.join(config.checkpoints_dir, run_id)
        config.figures_dir = os.path.join(config.figures_dir, run_id)
        config.embeddings_dir = os.path.join(config.embeddings_dir, run_id)
        
        print(f"🔖 New Run ID: {run_id}")
        print(f"📁 Creating new directories:")
    
    # Create all directories (existing ones won't be affected)
    for directory in [config.logs_dir, config.checkpoints_dir, config.figures_dir, config.embeddings_dir]:
        os.makedirs(directory, exist_ok=True)
    
    print(f"   📊 Logs: {config.logs_dir}")
    print(f"   💾 Checkpoints: {config.checkpoints_dir}")
    print(f"   📈 Figures: {config.figures_dir}")
    print(f"   🧠 Embeddings: {config.embeddings_dir}")

    # Note: Embedding extraction moved to the train() method to avoid duplicate dataset creation
    if kwargs['extract_embeddings_only']:
        print("Extract-embeddings-only mode enabled - will extract embeddings during training setup.")
        print("Note: This requires at least one training iteration to generate embeddings.")

    # TensorBoard setup
    tensorboard_process = None
    if kwargs['auto_start_tensorboard']:
        tensorboard_process = start_tensorboard(
            log_dir=config.logs_dir,
            port=kwargs['tensorboard_port'],
            host=kwargs['tensorboard_host']
        )
        
        # Auto-open browser if requested
        if kwargs['open_browser'] and tensorboard_process:
            open_tensorboard_browser(port=kwargs['tensorboard_port'], delay=3)
        
        if tensorboard_process:
            print(f"📊 TensorBoard URL: http://localhost:{kwargs['tensorboard_port']}")
    else:
        print(f"TensorBoard auto-start disabled. To monitor training manually run:")
        print(f"tensorboard --logdir={config.logs_dir} --host={kwargs['tensorboard_host']} --port={kwargs['tensorboard_port']}")

    # TensorBoard logging setup
    writer = SummaryWriter(log_dir=config.logs_dir)

    # Training
    print('>>>>>>>Start Training>>>>>>>>>>>>>>>>>>>>>>>>>>')
    try:
        model = exp.train(
            setting,
            writer=writer,
            resume_checkpoint=resume_checkpoint_path
        )
    except KeyboardInterrupt:
        print('\n>>>>>>>Early Stopping Due to KeyboardInterrupt<<<<<<<<<<<<<<<')
    finally:
        # Clean up TensorBoard process
        if tensorboard_process and tensorboard_process.poll() is None:
            print("\n🔄 Keeping TensorBoard running for result viewing...")
            print(f"📊 View results at: http://localhost:{kwargs['tensorboard_port']}")
            print(f"📁 Logs for this run: {config.logs_dir}")
            print("💡 To stop TensorBoard later, run: pkill -f tensorboard")

    writer.close()

    # Testing
    print('>>>>>>>Testing<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test(setting, test=1)

    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
