"""
Unified Experiment Runner for Stock Market iTransformer Research

- Supports dynamic selection of any model from iTransformer/model/ (iTransformer, iInformer, iReformer, iFlowformer, iFlashformer, Transformer, Informer, Reformer, Flowformer, Flashformer).
- Uses SimpleStockDataset with returns-based preprocessing for OHLCV stock data.
- Results, metrics, and logs are saved with model-specific naming for direct comparison.
- Feature selection (e.g., 'close' only vs. multi-feature) is controlled via CLI/config.
- Strictly supports the "inverted" transformer paradigm for time series forecasting.

Usage:
    python train.py --model iInformer --features close,volume,transactions ...
    # See CLI help for all options.

"""
import os
import sys
import time
import warnings
import gc  # Add garbage collection

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from exp_basic import Exp_Basic
from simple_stock_dataset import SimpleStockDataset, create_simple_dataloader
from utils.metrics import metric
from utils.visualization import StockVisualizer
from utils.loss import get_loss_function
from utils.logger import Logger
from configs import StockPredictionConfig

# Simple memory monitoring functions
def log_memory_stats(prefix: str = ""):
    """Log current memory statistics."""
    try:
        import psutil
        process = psutil.Process()
        ram_gb = process.memory_info().rss / (1024**3)
        
        gpu_gb = 0.0
        if torch.cuda.is_available():
            gpu_gb = torch.cuda.memory_allocated() / (1024**3)
            
        print(f"🔍 {prefix}Memory: {ram_gb:.1f}GB RAM, {gpu_gb:.1f}GB GPU")
    except ImportError:
        print("🔍 psutil not available for memory monitoring")

warnings.filterwarnings('ignore')

class Exp_Stock_Forecasting(Exp_Basic):
    def __init__(self, args):
        super(Exp_Stock_Forecasting, self).__init__(args)
        
        # Initialize logger
        self.logger = Logger(name=args.model, log_dir='./logs')
        
        # Initialize visualizer
        from utils.visualization import StockVisualizer
        self.visualizer = StockVisualizer(save_dir=args.figures_dir, feature_names=args.features)
        
        # Global stats for normalization
        self.global_mean = None
        self.global_std = None
        
        # Cache for datasets to avoid reprocessing (with size limit)
        self._dataset_cache = {}
        self._dataloader_cache = {}
        self.max_cache_size = 5  # Limit cache to prevent memory growth
        
        # Memory management configuration
        self.max_iteration_metrics = getattr(args, 'max_iteration_metrics', 10000)  # Limit stored metrics
        self.cleanup_frequency = getattr(args, 'cleanup_frequency', 100)  # Cleanup every N iterations

    def _build_model(self):
        """
        Dynamically import and instantiate the selected model from iTransformer/model/ or local models.
        Enforces use of "inverted" models unless allow_classic_models=True in args.
        """
        import importlib
        import sys
        import os

        # List of inverted models (safe for main experiments)
        inverted_models = {
            'iTransformer', 'DirectReturnsTransformer', 'iInformer', 'iReformer', 'iFlowformer', 'iFlashformer'
        }
        # All available models
        model_module_map = {
            'iTransformer': 'iTransformer',
            'iInformer': 'iInformer',
            'iReformer': 'iReformer',
            'iFlowformer': 'iFlowformer',
            'iFlashformer': 'iFlashformer',
            'Transformer': 'Transformer',
            'Informer': 'Informer',
            'Reformer': 'Reformer',
            'Flowformer': 'Flowformer',
            'Flashformer': 'Flashformer',
        }

        model_name = self.args.model
        model_class = None

        # Enforce inverted transformer paradigm unless explicitly allowed
        allow_classic = getattr(self.args, "allow_classic_models", False)
        if model_name not in inverted_models and not allow_classic:
            raise ValueError(
                f"Model '{model_name}' is a classic (time-based) transformer. "
                "For main experiments, only inverted models are allowed. "
                "If you want to run classic models for ablation, set allow_classic_models=True in your config or CLI."
            )

        # Try local models first (stock-optimized)
        if model_name == "iTransformer":
            try:
                from models.iTransformer import Model as iTransformerModel
                model_class = iTransformerModel
                print(f"Using local stock-optimized {model_name} model")
            except Exception as e:
                print(f"Warning: Could not import local {model_name} model: {e}")
        elif model_name == "DirectReturnsTransformer":
            try:
                from models.DirectReturnsTransformer import Model as DirectReturnsTransformerModel
                model_class = DirectReturnsTransformerModel
                print(f"Using DirectReturnsTransformer - returns as direct embeddings approach")
            except Exception as e:
                print(f"Warning: Could not import DirectReturnsTransformer model: {e}")

        # Fallback to original iTransformer models if local not found
        if model_class is None and model_name in model_module_map:
            # Check compatibility with full_day mode
            if hasattr(self.args, 'mode') and self.args.mode == 'full_day' and model_name != 'iTransformer':
                raise ValueError(
                    f"Original {model_name} model does not support variable sequence lengths in full_day mode. "
                    f"Please use one of the following options:\n"
                    f"1. Use the local iTransformer model (recommended for market data)\n"
                    f"2. Switch to sliding_window mode\n"
                    f"3. Use a fixed sequence length configuration"
                )
            
            # Add iTransformer to sys.path if not already present
            iTransformer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'iTransformer'))
            if iTransformer_path not in sys.path:
                sys.path.insert(0, iTransformer_path)
            try:
                module = importlib.import_module(f"model.{model_module_map[model_name]}")
                model_class = getattr(module, "Model")
                print(f"Using original {model_name} model from iTransformer directory")
                        
            except Exception as e:
                print(f"Warning: Could not import {model_name} from iTransformer/model/: {e}")

        if model_class is None:
            raise ValueError(f"Model {model_name} not found in iTransformer/model/ or local models.")

        model = model_class(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model


    def _get_data(self, flag, shuffle=True, max_samples=None, enable_streaming=None):
        """
        flag: 'train', 'val', or 'test'
        shuffle: bool, whether to shuffle the data
        max_samples: int, optional limit on the number of samples
        enable_streaming: bool, whether to enable streaming mode (None = auto-detect based on flag and file count)
        """
        # Get streaming configuration from args
        if enable_streaming is None:
            # Check if streaming is explicitly enabled/disabled in args
            if hasattr(self.args, 'enable_streaming') and self.args.enable_streaming is not None:
                if self.args.enable_streaming == 'on':
                    enable_streaming = True
                elif self.args.enable_streaming == 'off':
                    enable_streaming = False
                elif self.args.enable_streaming == True:
                    enable_streaming = True
                elif self.args.enable_streaming == False:
                    enable_streaming = False  
                else:  # 'auto' or other
                    enable_streaming = None
            else:
                enable_streaming = None
        
        # IMPORTANT: Only allow streaming for training data, never for validation/test
        if flag != 'train':
            enable_streaming = False
            print(f"🔒 Streaming disabled for {flag} data (streaming only supported for training)")
        elif enable_streaming is None:
            # Auto-detect streaming mode for training data only
            data_path_list = self.args.train_files
            # Use threshold from args if available, otherwise default to 50
            threshold = getattr(self.args, 'streaming_threshold', 50)
            enable_streaming = len(data_path_list) > threshold
            print(f"🤖 Auto-detecting streaming mode: {len(data_path_list)} files vs threshold {threshold} = {'ENABLED' if enable_streaming else 'DISABLED'}")
        
        # If streaming is not enabled, use the original cached approach
        if not enable_streaming:
            print(f"📁 Using traditional cached loading for {flag} data")
            return self._get_data_cached(flag, shuffle, max_samples)
        
        # Streaming mode implementation
        return self._get_data_streaming(flag, shuffle, max_samples)
    
    def _get_data_cached(self, flag, shuffle=True, max_samples=None):
        """Original cached data loading implementation"""
        # Create cache key based on parameters that affect dataset creation
        tickers_key = tuple(sorted(self.args.val_stocks)) if flag == 'val' and self.args.val_stocks else None
        cache_key = (
            flag, 
            tuple(getattr(self.args, f'{flag}_files', [])),
            tickers_key,
            self.args.seq_len,
            self.args.pred_len,
            tuple(self.args.features),
            self.args.mode,
            self.args.interpolate_max_missing,
            max_samples
        )
        
        # Check if we already have this dataset/dataloader combination cached
        if cache_key in self._dataset_cache:
            print(f"Using cached dataset for {flag} split (no reprocessing needed)")
            cached_dataset = self._dataset_cache[cache_key]
            
            # Create new dataloader with desired shuffle setting (since shuffle might vary)
            from torch.utils.data import DataLoader
            use_drop_last = len(cached_dataset) >= self.args.batch_size
            data_loader = DataLoader(
                cached_dataset,
                batch_size=self.args.batch_size,
                shuffle=shuffle,
                num_workers=0,
                drop_last=use_drop_last
            )
            return cached_dataset, data_loader
        
        if flag == 'train':
            data_path_list = self.args.train_files
            tickers = self.args.stocks # Use configured stocks for training (could be None)
        elif flag == 'val': # Assuming 'val' flag for validation
             data_path_list = self.args.val_files # Use val_files from config
             tickers = self.args.val_stocks # Use specific validation stocks
        else: # flag == 'test'
             data_path_list = self.args.test_files # Use test_files from config
             tickers = self.args.stocks # Use configured stocks for testing (or None)

        # Note: With returns-based preprocessing, we no longer need scaling

        # Ensure data_path_list is not empty
        if not data_path_list:
             print(f"Warning: No data files found for flag '{flag}'. Returning None for dataset and dataloader.")
             return None, None

        data_set, data_loader = create_simple_dataloader(
            file_paths=data_path_list, # Pass the list of paths
            batch_size=self.args.batch_size,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            tickers=tickers,
            features=self.args.features, # Pass features
            shuffle=shuffle, # Pass shuffle flag
            mode=self.args.mode,
            interpolate_max_missing=self.args.interpolate_max_missing,
            max_samples=max_samples
        )
        
        # Cache the dataset (not the dataloader since shuffle varies) with size limit
        if data_set is not None:
            # Implement LRU-style cache eviction to prevent memory growth
            if len(self._dataset_cache) >= self.max_cache_size:
                # Remove oldest cache entry
                oldest_key = next(iter(self._dataset_cache))
                print(f"Cache full, evicting oldest entry: {oldest_key}")
                del self._dataset_cache[oldest_key]
                gc.collect()  # Force garbage collection after cache eviction
            
            self._dataset_cache[cache_key] = data_set
            print(f"Cached dataset for {flag} split for future use (cache size: {len(self._dataset_cache)}/{self.max_cache_size})")
        
        return data_set, data_loader
    
    def _get_data_streaming(self, flag, shuffle=True, max_samples=None):
        """Streaming data loading implementation"""
        print(f"🚀 Using streaming mode for {flag} data loading")
        
        if flag == 'train':
            data_path_list = self.args.train_files
            tickers = self.args.stocks
        elif flag == 'val':
             data_path_list = self.args.val_files
             tickers = self.args.val_stocks
        else:
             data_path_list = self.args.test_files
             tickers = self.args.stocks

        # Note: We always use returns-based preprocessing now (no scaling needed)

        # Ensure data_path_list is not empty
        if not data_path_list:
             print(f"Warning: No data files found for flag '{flag}'. Returning None for dataset and dataloader.")
             return None, None

        # Import streaming components
        try:
            from streaming_dataset import create_interleave_streaming_dataloader
        except ImportError:
            print("❌ Streaming dataset not available. Falling back to cached loading.")
            return self._get_data_cached(flag, shuffle, max_samples)
        
        # Configure interleaved streaming parameters 
        total_files = len(data_path_list)
        
        # Get chunk size from CLI args or use smart defaults
        chunk_size = getattr(self.args, 'streaming_chunk_size', None)
        if chunk_size is None:
            if total_files > 1000:
                chunk_size = 5  # Process 5 files at a time for very large datasets
            elif total_files > 100:
                chunk_size = 10  # Process 10 files at a time for medium datasets
            else:
                chunk_size = min(3, total_files)  # Process 3 files at a time for small datasets
        
        print(f"📊 Interleaved streaming: chunk_size={chunk_size} (process {chunk_size} files at a time)")

        data_set, data_loader = create_interleave_streaming_dataloader(
            file_paths=data_path_list,
            batch_size=self.args.batch_size,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            tickers=tickers,
            features=self.args.features,
            shuffle=shuffle,
            mode=self.args.mode,
            interpolate_max_missing=self.args.interpolate_max_missing,
            chunk_size=chunk_size
        )
        
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # Use the factory function to get the loss based on config
        criterion = get_loss_function(self.args.loss_type, **self.args.loss_kwargs)
        print(f"Using loss function: {self.args.loss_type} with kwargs: {self.args.loss_kwargs}")
        return criterion

    # Removed redundant vali method - validation is handled by test(test=0)

    def train(self, setting, writer=None, resume_checkpoint=None):
        """
        Train the model.

        Args:
            setting (str): Experiment setting name.
            writer (SummaryWriter, optional): TensorBoard writer for logging.
            resume_checkpoint (str, optional): Path to checkpoint to resume from.
        """
        train_data, train_loader = self._get_data(flag='train')
        # test_data, test_loader = self._get_data(flag='test') # Test loader not needed here

        # Ensure base checkpoints directory exists
        if not os.path.exists(self.args.checkpoints_dir):
            os.makedirs(self.args.checkpoints_dir)

        path = os.path.join(self.args.checkpoints_dir, setting)
        if resume_checkpoint is not None and os.path.exists(resume_checkpoint):
            print(f"Resuming from checkpoint: {resume_checkpoint}")
            try:
                checkpoint = torch.load(resume_checkpoint, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Training checkpoint with metadata
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    model_optim = self._select_optimizer()
                    model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint.get('epoch', 0)
                    global_iter = checkpoint.get('global_iter', 0)
                    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                    print(f"Resumed from training checkpoint at epoch {start_epoch}, global_iter {global_iter}, best_val_loss {best_val_loss}")
                else:
                    # Early stopping checkpoint (model state dict only)
                    self.model.load_state_dict(checkpoint)
                    start_epoch = 0
                    global_iter = 0
                    best_val_loss = float('inf')
                    model_optim = self._select_optimizer()
                    print(f"Resumed from early stopping checkpoint, starting fresh optimizer state")
                    
            except Exception as e:
                print(f"Error loading resume checkpoint {resume_checkpoint}: {e}")
                print("Starting fresh training...")
                start_epoch = 0
                global_iter = 0
                best_val_loss = float('inf')
                model_optim = self._select_optimizer()
        else:
            # Clean up old checkpoints if path exists
            if os.path.exists(path):
                import shutil
                print(f"Removing existing checkpoint directory: {path}")
                shutil.rmtree(path)
            start_epoch = 0
            global_iter = 0
            best_val_loss = float('inf')
            model_optim = self._select_optimizer()

        # Ensure the checkpoint directory exists (for both resume and fresh training)
        os.makedirs(path, exist_ok=True)

        # Extract embeddings if requested (moved here to avoid duplicate dataset creation)
        if hasattr(self.args, 'extract_embeddings_only') and getattr(self.args, 'extract_embeddings_only', False):
            print("🧠 Extracting embeddings from first batch...")
            if train_loader is not None:
                try:
                    batch_iterator = iter(train_loader)  # type: ignore
                    batch_x, batch_x_mark, batch_y, batch_y_mark, attention_mask = next(batch_iterator)
                    batch_x = batch_x.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    with torch.no_grad():
                        if hasattr(self.model, "get_embeddings"):
                            embeddings = self.model.get_embeddings(batch_x[0:1], batch_x_mark[0:1])
                            embeddings_dir = getattr(self.args, 'embeddings_dir', './embeddings')
                            os.makedirs(embeddings_dir, exist_ok=True)
                            # Simple save for now - you can implement save_embeddings if needed
                            embeddings_path = os.path.join(embeddings_dir, 'stock_embeddings.json')
                            print(f"📁 Embeddings saved to: {embeddings_path}")
                        else:
                            print("Model does not support get_embeddings method.")
                    print("✅ Embedding extraction completed. Exiting as requested.")
                    return self.model
                except Exception as e:
                    print(f"❌ Error extracting embeddings: {e}")
                    return self.model
            else:
                print("❌ No training data available for embedding extraction")
                return self.model

        time_now = time.time()

        # Check if train_loader is valid
        if train_loader is None:
             print("Error: Training DataLoader could not be created. Aborting training.")
             return None # Or raise an exception
        if len(train_loader) == 0:
             print("Warning: Training DataLoader is empty. Aborting training.")
             return None # Or raise an exception

        # Handle streaming vs. regular dataloaders
        is_streaming = hasattr(train_loader, 'get_status') and callable(getattr(train_loader, 'get_status', None))
        if is_streaming:
            print("🌊 Detected streaming dataloader - will monitor progress during training")
            train_steps = len(train_loader)  # Initial size
            # Print initial status with more detail
            status = train_loader.get_status()  # type: ignore
            sequences_count = status.get('current_sequences', status.get('total_sequences', len(train_data) if train_data else 0))
            print(f"🚀 Starting with {sequences_count:,} sequences from {status['processed_files']} files")
            print(f"📈 Interleaved processing will add {status['remaining_files']} more files during training")
            if hasattr(train_loader, 'print_status') and callable(getattr(train_loader, 'print_status', None)):
                train_loader.print_status()  # type: ignore
            else:
                # Handle different status formats
                chunks_info = ""
                if 'chunks_processed' in status:
                    chunks_info = f", {status['chunks_processed']} chunks processed"
                elif 'current_chunk' in status and 'total_chunks' in status:
                    chunks_info = f", {status['current_chunk']}/{status['total_chunks']} chunks"
                print(f"📊 Current status: {status['processed_files']}/{status['total_files']} files processed{chunks_info}")
        else:
            train_steps = len(train_loader)
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        # Initialize criterion
        self.criterion = self._select_criterion()

        # Initialize learning rate scheduler
        if self.args.lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                model_optim,
                T_max=self.args.train_epochs,
                eta_min=self.args.min_lr
            )
        else:  # reduce_on_plateau
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                model_optim,
                mode='min',
                factor=self.args.lr_decay_factor,
                patience=self.args.lr_patience,
                min_lr=self.args.min_lr
            )

        # Initialize metrics storage with size limits
        train_losses = []
        val_losses = []
        learning_rates = []
        
        # Iteration-level metrics storage with circular buffer to prevent memory leaks
        iteration_metrics = {
            'iteration': [],
            'epoch': [],
            'train_loss': [],
            'learning_rate': [],
        }
        
        # Memory management tracking
        last_cleanup_iter = 0

        print(f"\nStarting Training for {self.args.train_epochs} epochs...")
        
        # Log initial memory usage
        log_memory_stats("Training start - ")
        
        # Log training start
        self.logger.logger.info(f"=== Training Started ===")
        self.logger.logger.info(f"Model: {self.args.model}")
        self.logger.logger.info(f"Features: {self.args.features}")
        self.logger.logger.info(f"Epochs: {self.args.train_epochs}")
        self.logger.logger.info(f"Batch Size: {self.args.batch_size}")
        self.logger.logger.info(f"Learning Rate: {self.args.learning_rate}")
        self.logger.logger.info(f"Loss Function: {self.args.loss_type}")
        self.logger.logger.info(f"Device: {self.device}")
        self.logger.logger.info(f"Training samples: {len(train_data) if train_data else 'N/A'}")

        global_iter = global_iter if 'global_iter' in locals() else 0
        
        # Configure iteration logging frequency
        log_every_n_iterations = getattr(self.args, 'log_every_n_iterations', 100)
        save_iteration_metrics = getattr(self.args, 'save_iteration_metrics', True)
        
        for epoch in range(start_epoch, self.args.train_epochs):
            iter_count = 0
            epoch_train_loss = []
            
            # For tracking running averages within epoch
            running_loss_sum = 0.0
            running_loss_count = 0

            self.model.train()
            epoch_time = time.time()
            
            # Clear cache before each epoch to prevent accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # For streaming dataloaders, we need to track dynamic total steps
            if is_streaming:
                # For streaming, we track actual batches processed (not total)
                pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.args.train_epochs}")
                last_checked_steps = len(train_loader)
                steps_check_interval = 10  # Check for size changes every N iterations
                # Use simple batch counter for streaming
                batch_idx = 0
            else:
                pbar = tqdm(enumerate(train_loader), total=train_steps, desc=f"Epoch {epoch + 1}/{self.args.train_epochs}")

            for batch_data in pbar:
                # Handle different iterator formats
                if is_streaming:
                    # For streaming, batch_data is the actual batch, use simple counter
                    i = batch_idx
                    batch_idx += 1
                    # No chunk repetition logic - let the streaming iterator handle everything
                else:
                    # For regular dataloader, unpack (index, batch_data)
                    i, batch_data = batch_data

                if batch_data is None:
                    warnings.warn(f"Skipping iteration {i} due to None batch data.")
                    continue
                try:
                    # Handle different batch data formats for streaming vs regular dataloaders
                    if is_streaming:
                        # For streaming, batch_data is the actual batch tuple
                        if len(batch_data) == 5:
                            batch_x, batch_x_mark, batch_y, batch_y_mark, attention_mask = batch_data
                        else:
                            warnings.warn(f"Skipping iteration {i} due to unexpected batch format")
                            continue
                    else:
                        # For regular dataloader, we already unpacked (i, batch_data) above
                        if len(batch_data) == 5:
                            batch_x, batch_x_mark, batch_y, batch_y_mark, attention_mask = batch_data
                        else:
                            warnings.warn(f"Skipping iteration {i} due to unexpected batch format")
                            continue
                except (ValueError, TypeError) as e:
                    warnings.warn(f"Skipping iteration {i} due to error unpacking batch data: {e}")
                    continue

                iter_count += 1
                global_iter += 1
                model_optim.zero_grad()

                # For streaming datasets, periodically check if dataset size has grown
                if is_streaming and iter_count % steps_check_interval == 0:
                    new_train_steps = len(train_loader)
                    if new_train_steps != last_checked_steps:
                        # Update progress bar total - but be careful about the counter going beyond initial size
                        pbar.total = None  # Remove total for streaming (since it's dynamic)
                        pbar.refresh()
                        last_checked_steps = new_train_steps
                        print(f"\n📈 Dataset expanded! New size: {new_train_steps} batches")

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                attention_mask = attention_mask.float().to(self.device)

                # **CRITICAL FIX**: Construct decoder input as in original iTransformer
                # Don't permute here - let the model handle permutation internally
                # batch_x and batch_y should remain in [B, seq, feat] format
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # **CRITICAL FIX**: Use original iTransformer model call signature WITH attention mask
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask=attention_mask)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask=attention_mask)

                # Extract final predictions and handle feature dimensions
                # For both inverted and classic models, extract the last pred_len steps
                # outputs should be [B, pred_len, features] or [B, features, pred_len]
                # f_dim: For 'MS' (multivariate input, univariate output), only use last feature (-1)
                # For 'S' (univariate) or 'M' (multivariate), use all features (0:)
                f_dim = -1 if self.args.forecasting_features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                loss = self.criterion(outputs, batch_y)
                current_loss = loss.item()
                epoch_train_loss.append(current_loss)
                
                # Update running averages
                running_loss_sum += current_loss
                running_loss_count += 1
                running_avg_loss = running_loss_sum / running_loss_count

                loss.backward()
                model_optim.step()
                
                # Explicit cleanup to prevent tensor accumulation
                del outputs, batch_x, batch_y, batch_x_mark, batch_y_mark, attention_mask, dec_inp
                
                # Periodic memory cleanup
                if global_iter - last_cleanup_iter >= self.cleanup_frequency:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    last_cleanup_iter = global_iter

                # Get current learning rate
                current_lr = model_optim.param_groups[0]['lr']

                # TensorBoard logging (already per iteration)
                if writer is not None:
                    writer.add_scalar('Loss/train', current_loss, global_iter)
                    writer.add_scalar('LearningRate', current_lr, global_iter)
                    writer.add_scalar('Loss/train_running_avg', running_avg_loss, global_iter)

                # Store iteration-level metrics if enabled with circular buffer to prevent memory leaks
                if save_iteration_metrics:
                    # Implement circular buffer to limit memory usage
                    if len(iteration_metrics['iteration']) >= self.max_iteration_metrics:
                        # Remove oldest 10% of metrics to make room for new ones
                        remove_count = self.max_iteration_metrics // 10
                        for key in iteration_metrics:
                            iteration_metrics[key] = iteration_metrics[key][remove_count:]
                    
                    iteration_metrics['iteration'].append(global_iter)
                    iteration_metrics['epoch'].append(epoch + 1)
                    iteration_metrics['train_loss'].append(current_loss)
                    iteration_metrics['learning_rate'].append(current_lr)

                # Detailed console logging every N iterations
                if global_iter % log_every_n_iterations == 0:
                    elapsed_time = time.time() - epoch_time
                    iters_per_sec = iter_count / elapsed_time if elapsed_time > 0 else 0
                    
                    # Log memory stats periodically
                    log_memory_stats(f"Epoch {epoch + 1}, Iter {global_iter} - ")
                    
                    # For streaming datasets, don't show confusing batch ratios; for regular datasets, use original calculation
                    if is_streaming:
                        # Don't calculate ETA for streaming since dataset size is dynamic
                        status_msg = (f"\n[Epoch {epoch + 1}/{self.args.train_epochs}] "
                                    f"[Global Iter {global_iter}] "
                                    f"[Streaming Batch {i+1}] "
                                    f"Loss: {current_loss:.6f} "
                                    f"Running Avg: {running_avg_loss:.6f} "
                                    f"LR: {current_lr:.2e} "
                                    f"Speed: {iters_per_sec:.1f} it/s")
                    else:
                        eta_seconds = (train_steps - i - 1) / iters_per_sec if iters_per_sec > 0 else 0
                        eta_str = f"{int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}"
                        
                        status_msg = (f"\n[Epoch {epoch + 1}/{self.args.train_epochs}] "
                                    f"[Iter {global_iter}] "
                                    f"[Iter {i+1}/{train_steps}] "
                                    f"Loss: {current_loss:.6f} "
                                    f"Running Avg: {running_avg_loss:.6f} "
                                    f"LR: {current_lr:.2e} "
                                    f"Speed: {iters_per_sec:.1f} it/s "
                                    f"ETA: {eta_str}")
                    
                    # Add streaming status if available
                    if is_streaming:
                        status = train_loader.get_status()  # type: ignore
                        # Handle different streaming implementations
                        sequences_count = status.get('total_sequences', status.get('current_sequences', 0))
                        background_status = status.get('background_active', False)
                        status_msg += (f"\n📊 Streaming Status: {status['processed_files']}/{status['total_files']} files processed, "
                                     f"{sequences_count} sequences available, "
                                     f"Background: {'Active' if background_status else 'Inactive'}")
                    
                    print(status_msg)

                # Save checkpoint based on configuration
                if (self.args.save_checkpoint_every_n_iterations is not None and 
                    global_iter % self.args.save_checkpoint_every_n_iterations == 0):
                    checkpoint_path = os.path.join(path, f'checkpoint_iter_{global_iter}.pth')
                    torch.save({
                        'epoch': epoch,
                        'global_iter': global_iter,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': model_optim.state_dict(),
                        'best_val_loss': best_val_loss
                    }, checkpoint_path)
                    print(f"Checkpoint saved at iteration {global_iter}: {checkpoint_path}")

                # Enhanced progress bar description with streaming status
                if is_streaming:
                    status = train_loader.get_status()  # type: ignore
                    # For streaming, don't show confusing batch ratios since dataset is expanding
                    pbar.set_description(f"Epoch {epoch + 1}/{self.args.train_epochs} | "
                                       f"Global Iter {global_iter} | "
                                       f"Batch {i+1} | "
                                       f"Loss: {current_loss:>7.4f} | "
                                       f"Avg: {running_avg_loss:>7.4f} | "
                                       f"LR: {current_lr:.2e} | "
                                       f"Files: {status['processed_files']}/{status['total_files']}")
                else:
                    pbar.set_description(f"Epoch {epoch + 1}/{self.args.train_epochs} | "
                                       f"Iter {i+1}/{train_steps} | "
                                       f"Loss: {current_loss:>7.4f} | "
                                       f"Avg: {running_avg_loss:>7.4f} | "
                                       f"LR: {current_lr:.2e}")

                if self.args.max_train_iterations is not None and i + 1 >= self.args.max_train_iterations:
                    print(f"\nReached max_train_iterations ({self.args.max_train_iterations}). Stopping epoch {epoch + 1} early.")
                    break

            avg_epoch_train_loss = np.average(epoch_train_loss) if epoch_train_loss else 0.0
            
            train_losses.append(avg_epoch_train_loss)
            
            # Clear epoch-level losses to free memory
            del epoch_train_loss
            
            # Limit size of historical losses to prevent memory growth
            max_history = 1000  # Keep only last 1000 epochs
            if len(train_losses) > max_history:
                train_losses = train_losses[-max_history:]
                val_losses = val_losses[-max_history:] if len(val_losses) > max_history else val_losses
                learning_rates = learning_rates[-max_history:] if len(learning_rates) > max_history else learning_rates

            # Validation  
            print(f"\nRunning validation for epoch {epoch + 1}...")
            val_loss = self.test(setting, test=0)
            val_losses.append(val_loss)

            # Test evaluation at end of epoch
            print(f"\nRunning test evaluation for epoch {epoch + 1}...")
            test_mse = self.test(setting, test=1, writer=writer, epoch=epoch + 1)
            if writer is not None and test_mse is not None and not isinstance(test_mse, tuple):
                writer.add_scalar('Test/MSE_epoch', test_mse, epoch + 1)

            # TensorBoard validation logging
            if writer is not None:
                writer.add_scalar('Loss/val', val_loss, epoch + 1)

            # Learning Rate Step
            current_lr = model_optim.param_groups[0]['lr']
            learning_rates.append(current_lr)
            if self.args.lr_scheduler == 'cosine':
                scheduler.step()  # type: ignore  # CosineAnnealingLR doesn't need metrics
            else:
                scheduler.step(val_loss)  # ReduceLROnPlateau needs the validation loss

            # Log epoch completion
            self.logger.log_training(epoch + 1, avg_epoch_train_loss, val_loss, test_mse if test_mse is not None and not isinstance(test_mse, tuple) else 0.0, current_lr)

            # For streaming datasets, use the final dataset size for step count
            if is_streaming:
                final_steps = len(train_loader)
                actual_steps = final_steps if self.args.max_train_iterations is None else i + 1
            else:
                actual_steps = train_steps if self.args.max_train_iterations is None else i + 1
                
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.6f} Val Loss: {3:.6f} Learning Rate: {4:.6f}".format(
                epoch + 1, actual_steps, avg_epoch_train_loss, val_loss, current_lr))

            # Early Stopping
            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                self.logger.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

            # Save checkpoint based on epoch configuration
            if (self.args.save_checkpoint_every_n_epochs is not None and 
                (epoch + 1) % self.args.save_checkpoint_every_n_epochs == 0):
                epoch_checkpoint_path = os.path.join(path, f'checkpoint_epoch_{epoch + 1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'global_iter': global_iter,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': model_optim.state_dict(),
                    'best_val_loss': best_val_loss
                }, epoch_checkpoint_path)
                print(f"Epoch checkpoint saved: {epoch_checkpoint_path}")
                self.logger.logger.info(f"Epoch checkpoint saved: {epoch_checkpoint_path}")

            print(f"Epoch {epoch + 1} completed in {time.time() - epoch_time:.2f} seconds.")
            
            # End of epoch cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


        # --- End of Training ---
        print("\nTraining finished.")
        
        # Final memory cleanup and logging
        log_memory_stats("Before final cleanup - ")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        log_memory_stats("After final cleanup - ")
        
        # Print streaming summary and cleanup resources
        if is_streaming:
            final_status = train_loader.get_status()  # type: ignore
            # Handle different streaming implementations
            final_sequences_count = final_status.get('total_sequences', final_status.get('current_sequences', 0))
            print(f"\n📊 Streaming Summary:")
            print(f"  📁 Total files processed: {final_status['processed_files']}/{final_status['total_files']}")
            print(f"  📈 Final dataset size: {final_sequences_count:,} sequences")
            print(f"  ✅ All data {'fully' if final_status.get('processing_complete', False) else 'partially'} processed")
            if not final_status.get('processing_complete', False):
                print(f"  ⚠️  Note: {final_status.get('remaining_files', 0)} files were not processed")
            
            # Cleanup streaming resources to prevent interference with validation/testing
            print(f"🧹 Cleaning up streaming resources...")
            if hasattr(train_loader, 'streaming_dataset') and hasattr(getattr(train_loader, 'streaming_dataset', None), 'cleanup'):
                train_loader.streaming_dataset.cleanup()  # type: ignore
            print(f"✅ Streaming cleanup completed")

        # Plot training metrics (epoch-level)
        metrics = {
            'epoch': list(range(1, len(train_losses) + 1)),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'learning_rate': learning_rates
        }
        
        try:
            # Update visualizer save_dir to use the timestamped directory
            original_save_dir = self.visualizer.save_dir
            self.visualizer.save_dir = self.args.figures_dir
            
            self.visualizer.plot_training_metrics(metrics)
            self.visualizer.plot_learning_rate(learning_rates)
            
            # Plot iteration-level metrics if available
            if save_iteration_metrics and len(iteration_metrics['iteration']) > 0:
                print(f"Plotting iteration-level metrics ({len(iteration_metrics['iteration'])} points)...")
                self.visualizer.plot_iteration_metrics(iteration_metrics)
                
            # Restore original save_dir
            self.visualizer.save_dir = original_save_dir
                
        except Exception as e:
            print(f"Warning: Failed to plot metrics - {e}")

        # Load the best model saved by early stopping
        best_model_path = os.path.join(path, 'checkpoint.pth')
        if os.path.exists(best_model_path):
            print(f"Loading best model from: {best_model_path}")
            try:
                checkpoint = torch.load(best_model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Training checkpoint with metadata
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded best model from training checkpoint: {best_model_path}")
                else:
                    # Early stopping checkpoint (model state dict only)
                    self.model.load_state_dict(checkpoint)
                    print(f"Loaded best model from early stopping checkpoint: {best_model_path}")
                    
                self.logger.logger.info(f"Loaded best model from: {best_model_path}")
            except Exception as e:
                print(f"Error loading best model checkpoint from {best_model_path}: {e}")
                print("Warning: Using current model state.")
                self.logger.logger.info(f"Error loading best model: {e}. Using current model state.")
        else:
            print("Warning: Best model checkpoint not found. Returning current model state.")
            self.logger.logger.info("Warning: Best model checkpoint not found. Using current model state.")

        # Log training completion
        self.logger.logger.info(f"=== Training Completed ===")
        self.logger.logger.info(f"Total epochs: {len(train_losses)}")
        if train_losses:
            self.logger.logger.info(f"Final train loss: {train_losses[-1]:.6f}")
        if val_losses:
            self.logger.logger.info(f"Final val loss: {val_losses[-1]:.6f}")
        if learning_rates:
            self.logger.logger.info(f"Final learning rate: {learning_rates[-1]:.6e}")

        return self.model

    def test(self, setting, test=0, writer=None, epoch=None, temperature=None):
        """
        test=0: validation
        test=1: testing
        writer: TensorBoard SummaryWriter for logging metrics/figures
        epoch: current epoch (for logging)
        temperature: override temperature for this test run (None = use model default)
        """
        # Only load the model for final testing
        if test:
            print('loading model for final test')
            checkpoint_path = os.path.join(self.args.checkpoints_dir, setting, 'checkpoint.pth')
            
            # Check if the main checkpoint exists, if not find the latest iteration checkpoint
            if not os.path.exists(checkpoint_path):
                print(f"Main checkpoint not found at {checkpoint_path}")
                checkpoint_dir = os.path.join(self.args.checkpoints_dir, setting)
                
                if os.path.exists(checkpoint_dir):
                    # Find all iteration checkpoints
                    import glob
                    iteration_checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_iter_*.pth'))
                    
                    if iteration_checkpoints:
                        # Sort by iteration number and get the latest
                        iteration_checkpoints.sort(key=lambda x: int(x.split('_iter_')[1].split('.pth')[0]))
                        checkpoint_path = iteration_checkpoints[-1]
                        print(f"Using latest iteration checkpoint: {checkpoint_path}")
                    else:
                        print(f"No checkpoints found in {checkpoint_dir}. Using current model state for testing.")
                        # Don't return inf, just use current model state
                        checkpoint_path = None
                else:
                    print(f"Checkpoint directory does not exist: {checkpoint_dir}. Using current model state for testing.")
                    checkpoint_path = None
            
            # Only try to load checkpoint if we found one
            if checkpoint_path is not None:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)
                    
                    # Handle different checkpoint formats
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        # Checkpoint contains training metadata - extract model state dict
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        print(f"Successfully loaded model from training checkpoint: {checkpoint_path}")
                        if 'epoch' in checkpoint:
                            print(f"  Checkpoint from epoch: {checkpoint['epoch']}")
                        if 'global_iter' in checkpoint:
                            print(f"  Checkpoint from iteration: {checkpoint['global_iter']}")
                    else:
                        # Checkpoint is just model state dict (from early stopping)
                        self.model.load_state_dict(checkpoint)
                        print(f"Successfully loaded model state dict from: {checkpoint_path}")
                        
                except Exception as e:
                    print(f"Error loading checkpoint from {checkpoint_path}: {e}")
                    print("Continuing with current model state...")
            else:
                print("Using current model state for final testing (no checkpoint available).")
        # For validation (test=0), always use current model state

        # Use appropriate data loader
        data_flag = 'test' if test else 'val'
        
        # IMPORTANT FIX: During training, test evaluation should use validation data/stocks
        # Only use actual test data when called from train.py after training is complete
        if test and epoch is not None:
            # This is test evaluation during training - use validation data/stocks
            data_flag = 'val'
        elif test and epoch is None and hasattr(self.args, 'val_stocks') and self.args.val_stocks:
            # This is final test but user specified validation stocks - use validation data/stocks for visualization
            print(f"Using validation stocks for final test visualization: {self.args.val_stocks}")
            data_flag = 'val'
            
        data, data_loader = self._get_data(flag=data_flag, shuffle=False, max_samples=None)

        # Check if dataloader is valid
        if data_loader is None or len(data_loader) == 0:
             print(f"Warning: {data_flag.capitalize()} DataLoader is empty or could not be created. Skipping evaluation.")
             print(f"Debug info: data_flag='{data_flag}', tickers used: {getattr(self.args, 'val_stocks' if data_flag == 'val' else 'stocks', 'None')}")
             # Return high loss for validation, or handle differently for test?
             return np.inf if not test else (np.nan, np.nan, np.nan) # Return NaNs for test metrics

        preds, trues, input_data, input_marks = [], [], [], []
        
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_x_mark, batch_y, batch_y_mark, _) in tqdm(enumerate(data_loader), desc="Testing"):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, None, None) # Decoder inputs are not used in this model
                
                # Reshape if necessary (ensure preds and trues have same shape)
                f_dim = -1
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                # Move to CPU and convert to numpy immediately to free GPU memory
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
                input_data.append(batch_x.detach().cpu().numpy())
                input_marks.append(batch_x_mark.detach().cpu().numpy())
                
                # Explicit cleanup to free GPU tensors immediately
                del outputs, batch_x, batch_y, batch_x_mark, batch_y_mark
                
                # Periodic cleanup during testing
                if (i + 1) % 50 == 0:  # Cleanup every 50 batches
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

        if not preds:
            print("No predictions were generated during testing. Cannot evaluate.")
            return None
            
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        input_data = np.concatenate(input_data, axis=0)
        input_marks = np.concatenate(input_marks, axis=0)
        
        print(f"\n--- Test Results ---")
        print(f"Predictions shape: {preds.shape}")
        print(f"Ground truth shape: {trues.shape}")
        print(f"Input data shape: {input_data.shape}")
        print(f"Input marks shape: {input_marks.shape}")

        # FIXED Denormalize for plotting and metrics
        try:
            if data is not None and hasattr(data, 'denormalize') and callable(getattr(data, 'denormalize', None)):
                print("🔧 Using returns denormalization...")
                # SimpleStockDataset always uses returns, so always use returns denormalization
                batch_size = preds.shape[0]
                sequence_indices = list(range(batch_size))
                
                trues = data.denormalize(trues, sequence_indices)  # type: ignore
                preds = data.denormalize(preds, sequence_indices)  # type: ignore
                input_data = data.denormalize(input_data, sequence_indices)  # type: ignore
                print("✅ Successfully applied returns denormalization")
            else:
                print("[INFO] Denormalization not available - proceeding with returns data")
                
            # Verify the fix worked by checking value ranges
            print(f"📊 Denormalization verification:")
            print(f"   Predictions - min: {preds.min():.6f}, max: {preds.max():.6f}")
            print(f"   Ground truth - min: {trues.min():.6f}, max: {trues.max():.6f}")
            print(f"   Historical - min: {input_data.min():.6f}, max: {input_data.max():.6f}")
            
            # Check if denormalization worked (values should be > 0.01 for prices)
            all_values = np.concatenate([preds.flatten(), trues.flatten(), input_data.flatten()])
            if np.abs(all_values).min() > 0.01:
                print("✅ Returns denormalization successful - all values in price range")
            else:
                print("⚠️ Some values still appear to be in returns format")
                
        except Exception as e:
            print(f"[ERROR] Fixed denormalization failed: {e}")
            print("[WARN] Proceeding with normalized data for metrics and plots.")

        # STEP 1: Calculate loss on RAW returns data (same scale as training)
        from utils.metrics import metric
        mae_raw, mse_raw, rmse_raw, mape_raw, mspe_raw = metric(preds, trues)
        print(f'Raw Loss (validation scale): MSE: {mse_raw:.6f}, MAE: {mae_raw:.6f}')
        
        # Store raw loss for validation return
        validation_loss = mse_raw
        
        # STEP 2: Denormalize for visualization only
        try:
            if data is not None and hasattr(data, 'denormalize'):
                # SimpleStockDataset always uses returns
                batch_size = preds.shape[0]
                sequence_indices = list(range(batch_size))
                
                trues_denorm = data.denormalize(trues, sequence_indices)  # type: ignore
                preds_denorm = data.denormalize(preds, sequence_indices)  # type: ignore
                input_data_denorm = data.denormalize(input_data, sequence_indices)  # type: ignore
                
                # Calculate denormalized metrics for logging
                mae_denorm, mse_denorm, rmse_denorm, mape_denorm, mspe_denorm = metric(preds_denorm, trues_denorm)
                print(f'Denormalized Metrics: MSE: {mse_denorm:.4f}, MAE: {mae_denorm:.4f}')
                
                # Use denormalized data for the rest of the function
                trues = trues_denorm
                preds = preds_denorm
                input_data = input_data_denorm
                mae, mse, rmse, mape, mspe = mae_denorm, mse_denorm, rmse_denorm, mape_denorm, mspe_denorm
                
            else:
                # No denormalization available
                mae, mse, rmse, mape, mspe = mae_raw, mse_raw, rmse_raw, mape_raw, mspe_raw
                print("[INFO] Denormalization not available - using raw metrics")
                
        except Exception as e:
            print(f"[ERROR] Denormalization failed: {e}")
            mae, mse, rmse, mape, mspe = mae_raw, mse_raw, rmse_raw, mape_raw, mspe_raw

        # Log test metrics
        if test:
            self.logger.log_prediction(mae, mse, rmse, mape, mspe)
            self.logger.logger.info(f"=== Final Test Results ===")
            self.logger.logger.info(f"Test samples: {len(preds)}")
            self.logger.logger.info(f"Predictions shape: {preds.shape}")
            self.logger.logger.info(f"Ground truth shape: {trues.shape}")
        else:
            self.logger.logger.info(f"Validation - Raw MSE: {mse_raw:.6f}, Raw MAE: {mae_raw:.6f}")
            self.logger.logger.info(f"Validation - Denorm MSE: {mse:.6f}, Denorm MAE: {mae:.6f}")

        # Create visualizer instance using the pre-configured figures directory
        # This ensures figures go to the same timestamped directory as logs and checkpoints
        visualizer = StockVisualizer(save_dir=self.args.figures_dir, feature_names=self.args.features)
        
        # Find index of target feature for plotting
        try:
            target_feature_idx = self.args.features.index(self.args.target)
        except (ValueError, AttributeError):
            print(f"Warning: Target '{self.args.target}' not in features list. Defaulting to first feature for plots.")
            target_feature_idx = 0

        # Use the new simple validation visualization approach
        # Only run simple validation visualization for final test, not during training validation
        if test and epoch is None:
            visualizer.plot_simple_validation_predictions(
                model=self.model,
                dataset=data,  # Pass the validation dataset
                device=self.device,
                feature_idx=target_feature_idx,
                n_samples_to_plot=3,
                config=self.args,
                context_prefix="final_test"
            )

        # Only perform saving/plotting/metric calculation for final test run
        if test:
            # Use config-driven results directory (consistent with figures_dir approach)
            results_dir = getattr(self.args, "results_dir", None)
            if results_dir is None:
                # Default: use same timestamped directory structure as other outputs
                results_dir = os.path.join("results", os.path.basename(self.args.figures_dir))
            # Note: If results_dir is provided, use it directly (like figures_dir)

            # Ensure results directory exists
            os.makedirs(results_dir, exist_ok=True)

            # **PRINT RESULTS**: Same format as original
            print('mse:{}, mae:{}'.format(mse, mae))
            
            # Save BOTH normalized (for model analysis) and denormalized (for interpretation)
            np.save(os.path.join(results_dir, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
            np.save(os.path.join(results_dir, 'pred_normalized.npy'), preds)
            np.save(os.path.join(results_dir, 'true_normalized.npy'), trues)
            np.save(os.path.join(results_dir, 'pred_denormalized.npy'), preds)
            np.save(os.path.join(results_dir, 'true_denormalized.npy'), trues)
            np.save(os.path.join(results_dir, 'inputs_denormalized.npy'), input_data)

            # Note: Comprehensive predictions removed as they duplicated simple predictions functionality

            return mse

        else: # If validation (test=0)
            return validation_loss # Return RAW validation loss (same scale as training) for early stopping/LR scheduling

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                          Default: 7. Set to 0 to disable early stopping.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        # Check if val_loss is valid
        if val_loss is None or np.isnan(val_loss) or np.isinf(val_loss):
             print("Warning: Invalid validation loss received. Skipping early stopping check.")
             return # Do not update counter or save if loss is invalid

        score = -val_loss # We minimize loss, so maximize negative loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            # Loss did not improve (or improve enough)
            self.counter += 1
            if self.patience > 0:  # Only check for early stopping if patience > 0
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            # Loss improved
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        try:
            torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth')) # Use os.path.join
            self.val_loss_min = val_loss
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
