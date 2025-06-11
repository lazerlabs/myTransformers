"""
Unified Experiment Runner for Stock Market iTransformer Research

- Supports dynamic selection of any model from iTransformer/model/ (iTransformer, iInformer, iReformer, iFlowformer, iFlashformer, Transformer, Informer, Reformer, Flowformer, Flashformer).
- Uses StockDataset as the unified data loader for OHLCV stock data.
- Results, metrics, and logs are saved with model-specific naming for direct comparison.
- Feature selection (e.g., 'close' only vs. multi-feature) is controlled via CLI/config.
- Strictly supports the "inverted" transformer paradigm for time series forecasting.

Usage:
    python train.py --model iInformer --features close,volume,transactions ...
    # See CLI help for all options.

"""
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from tqdm import tqdm
from stock_dataset import create_dataloader
from utils.logger import Logger
from utils.visualization import StockVisualizer
from data_provider.data_loader import create_data_loaders
from configs import StockPredictionConfig
from utils.loss import get_loss_function # Added import
warnings.filterwarnings('ignore')

class Exp_Stock_Forecast():
    def __init__(self, args: StockPredictionConfig):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        # Initialize logger and visualizer
        log_dir = args.logs_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.logger = Logger(f"{args.model}_stock_prediction", log_dir=log_dir)
        self.visualizer = StockVisualizer(save_dir=args.figures_dir)

        # Create data loaders, store datasets, and store global stats
        (
            self.train_dataset, self.train_loader,
            self.test_dataset, self.test_loader,
            self.global_mean, self.global_std # Store stats
        ) = create_data_loaders(args)
        # TODO: Add handling for validation dataset/loader if implemented in create_data_loaders

        # Initialize criterion here so it's available for all methods
        self.criterion = self._select_criterion()


    def _acquire_device(self):
        if self.args.use_gpu:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                print('Use GPU:', device)
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
                print('Use MPS:', device)
            else:
                device = torch.device('cpu')
                print('No GPU/MPS available, use CPU instead')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

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
            'iTransformer', 'iInformer', 'iReformer', 'iFlowformer', 'iFlashformer'
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


    def _get_data(self, flag):
        """
        flag: 'train', 'val', or 'test'
        """
        if flag == 'train':
            data_path_list = self.args.train_files
            tickers = self.args.stocks # Use configured stocks for training (could be None)
            shuffle_data = True
        elif flag == 'val': # Assuming 'val' flag for validation
             data_path_list = self.args.val_files # Use val_files from config
             tickers = self.args.val_stocks # Use specific validation stocks
             shuffle_data = False
             # data_path = self.args.val_data_path # Assuming val_data_path exists in config - Use val_files instead
        else: # flag == 'test'
             data_path_list = self.args.test_files # Use test_files from config
             tickers = self.args.stocks # Use configured stocks for testing (or None)
             shuffle_data = False
             # data_path is already set correctly for test

        # Determine if scaling should be used (only if enabled AND stats are valid)
        scale_data = self.args.scale and self.global_mean is not None and self.global_std is not None

        # Ensure data_path_list is not empty
        if not data_path_list:
             print(f"Warning: No data files found for flag '{flag}'. Returning None for dataset and dataloader.")
             return None, None

        data_set, data_loader = create_dataloader(
            file_paths=data_path_list, # Pass the list of paths
            batch_size=self.args.batch_size,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            scale=scale_data, # Use determined scale flag
            tickers=tickers,
            features=self.args.features, # Pass features
            global_mean=self.global_mean, # Pass stored global stats
            global_std=self.global_std,   # Pass stored global stats
            shuffle=shuffle_data, # Pass shuffle flag
            mode=self.args.mode,
            interpolate_max_missing=self.args.interpolate_max_missing
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
            checkpoint = torch.load(resume_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            model_optim = self._select_optimizer()
            model_optim.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            global_iter = checkpoint.get('global_iter', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resumed at epoch {start_epoch}, global_iter {global_iter}, best_val_loss {best_val_loss}")
        else:
            # Clean up old checkpoints if path exists
            if os.path.exists(path):
                import shutil
                print(f"Removing existing checkpoint directory: {path}")
                shutil.rmtree(path)
            os.makedirs(path)
            start_epoch = 0
            global_iter = 0
            best_val_loss = float('inf')
            model_optim = self._select_optimizer()

        time_now = time.time()

        # Check if train_loader is valid
        if train_loader is None:
             print("Error: Training DataLoader could not be created. Aborting training.")
             return None # Or raise an exception
        if len(train_loader) == 0:
             print("Warning: Training DataLoader is empty. Aborting training.")
             return None # Or raise an exception

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

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

        # Initialize metrics storage
        train_losses = []
        val_losses = []
        learning_rates = []
        
        # Iteration-level metrics storage
        iteration_metrics = {
            'iteration': [],
            'epoch': [],
            'train_loss': [],
            'learning_rate': [],
        }

        print(f"\nStarting Training for {self.args.train_epochs} epochs...")

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
            pbar = tqdm(enumerate(train_loader), total=train_steps, desc=f"Epoch {epoch + 1}/{self.args.train_epochs}")

            for i, batch_data in pbar:
                if batch_data is None:
                    warnings.warn(f"Skipping iteration {i} due to None batch data.")
                    continue
                try:
                    batch_x, batch_x_mark, batch_y, batch_y_mark, attention_mask = batch_data
                except ValueError as e:
                    warnings.warn(f"Skipping iteration {i} due to error unpacking batch data: {e}")
                    continue

                iter_count += 1
                global_iter += 1
                model_optim.zero_grad()

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

                # Get current learning rate
                current_lr = model_optim.param_groups[0]['lr']

                # TensorBoard logging (already per iteration)
                if writer is not None:
                    writer.add_scalar('Loss/train', current_loss, global_iter)
                    writer.add_scalar('LearningRate', current_lr, global_iter)
                    writer.add_scalar('Loss/train_running_avg', running_avg_loss, global_iter)

                # Store iteration-level metrics if enabled
                if save_iteration_metrics:
                    iteration_metrics['iteration'].append(global_iter)
                    iteration_metrics['epoch'].append(epoch + 1)
                    iteration_metrics['train_loss'].append(current_loss)
                    iteration_metrics['learning_rate'].append(current_lr)

                # Detailed console logging every N iterations
                if global_iter % log_every_n_iterations == 0:
                    elapsed_time = time.time() - epoch_time
                    iters_per_sec = iter_count / elapsed_time if elapsed_time > 0 else 0
                    eta_seconds = (train_steps - i - 1) / iters_per_sec if iters_per_sec > 0 else 0
                    eta_str = f"{int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}"
                    
                    print(f"\n[Epoch {epoch + 1}/{self.args.train_epochs}] "
                          f"[Iter {global_iter}] "
                          f"[Iter {i+1}/{train_steps}] "
                          f"Loss: {current_loss:.6f} "
                          f"Running Avg: {running_avg_loss:.6f} "
                          f"LR: {current_lr:.2e} "
                          f"Speed: {iters_per_sec:.1f} it/s "
                          f"ETA: {eta_str}")

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

                # Enhanced progress bar description
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

            # Validation
            print(f"\nRunning validation for epoch {epoch + 1}...")
            val_loss = self.test(setting, test=0)
            val_losses.append(val_loss)

            # Test evaluation at end of epoch
            print(f"\nRunning test evaluation for epoch {epoch + 1}...")
            test_mse = self.test(setting, test=1, writer=writer, epoch=epoch + 1)
            if writer is not None:
                writer.add_scalar('Test/MSE_epoch', test_mse, epoch + 1)

            # TensorBoard validation logging
            if writer is not None:
                writer.add_scalar('Loss/val', val_loss, epoch + 1)

            # Learning Rate Step
            current_lr = model_optim.param_groups[0]['lr']
            learning_rates.append(current_lr)
            if self.args.lr_scheduler == 'cosine':
                scheduler.step()
            else:
                scheduler.step(val_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.6f} Val Loss: {3:.6f} Learning Rate: {4:.6f}".format(
                epoch + 1, train_steps if self.args.max_train_iterations is None else i + 1, avg_epoch_train_loss, val_loss, current_lr))

            # Early Stopping
            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
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
                print(f"Epoch checkpoint saved at epoch {epoch + 1}: {epoch_checkpoint_path}")

            print(f"Epoch {epoch + 1} completed in {time.time() - epoch_time:.2f} seconds.")


        # --- End of Training ---
        print("\nTraining finished.")

        # Plot training metrics (epoch-level)
        metrics = {
            'epoch': list(range(1, len(train_losses) + 1)),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'learning_rate': learning_rates
        }
        
        try:
            self.visualizer.plot_training_metrics(metrics)
            self.visualizer.plot_learning_rate(learning_rates)
            
            # Plot iteration-level metrics if available
            if save_iteration_metrics and len(iteration_metrics['iteration']) > 0:
                print(f"Plotting iteration-level metrics ({len(iteration_metrics['iteration'])} points)...")
                self.visualizer.plot_iteration_metrics(iteration_metrics)
                
        except Exception as e:
            print(f"Warning: Failed to plot metrics - {e}")

        # Load the best model saved by early stopping
        best_model_path = os.path.join(path, 'checkpoint.pth')
        if os.path.exists(best_model_path):
             print(f"Loading best model from: {best_model_path}")
             self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        else:
             print("Warning: Best model checkpoint not found. Returning current model state.")

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
                    self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                    print(f"Successfully loaded checkpoint from: {checkpoint_path}")
                except Exception as e:
                    print(f"Error loading checkpoint from {checkpoint_path}: {e}")
                    print("Continuing with current model state...")
            else:
                print("Using current model state for final testing (no checkpoint available).")
        # For validation (test=0), always use current model state

        # Use appropriate data loader
        data_flag = 'test' if test else 'val'
        data, data_loader = self._get_data(flag=data_flag)

        # Check if dataloader is valid
        if data_loader is None or len(data_loader) == 0:
             print(f"Warning: {data_flag.capitalize()} DataLoader is empty or could not be created. Skipping evaluation.")
             # Return high loss for validation, or handle differently for test?
             return np.inf if not test else (np.nan, np.nan, np.nan) # Return NaNs for test metrics

        preds = []
        trues = []
        inputs = []  # Store input sequences for visualization
        input_marks = []  # Store input timestamps
        total_loss = [] # For calculating average validation loss

        self.model.eval()
        test_batches_processed = 0
        max_test_batches = 10 # Limit to 10 batches for quick test=1 evaluation

        # Use config-driven results directory for visualization
        results_dir = getattr(self.args, "results_dir", None)
        if results_dir is None:
            results_dir = os.path.join("results", setting)
        else:
            results_dir = os.path.join(results_dir, setting)
        
        # Create test_results folder for visualization (similar to original)
        folder_path = os.path.join('./test_results/', setting)
        if test and not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with torch.no_grad():
            pbar_test = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Evaluating ({data_flag})")
            for i, batch_data in pbar_test:
                 # Limit batches processed during final testing (test=1) for speed in test_run.py
                 if test == 1 and test_batches_processed >= max_test_batches:
                      print(f"\nLimiting test evaluation to {max_test_batches} batches for speed.")
                      break

                 # Check if batch_data is None
                 if batch_data is None:
                      warnings.warn(f"Skipping iteration {i} in evaluation due to None batch data.")
                      continue
                 # Unpack batch data
                 try:
                      batch_x, batch_x_mark, batch_y, batch_y_mark, attention_mask = batch_data
                 except ValueError as e:
                      warnings.warn(f"Skipping iteration {i} in evaluation due to error unpacking batch data: {e}")
                      continue # Skip this batch

                 batch_x = batch_x.float().to(self.device)
                 batch_y = batch_y.float().to(self.device)
                 batch_x_mark = batch_x_mark.float().to(self.device)
                 batch_y_mark = batch_y_mark.float().to(self.device)
                 attention_mask = attention_mask.float().to(self.device)

                 # **CRITICAL FIX**: Construct decoder input as in original iTransformer
                 # Don't permute here - let the model handle permutation internally
                 dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                 dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                 # **CRITICAL FIX**: Use original iTransformer model call signature WITH attention mask
                 if self.args.output_attention:
                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask=attention_mask)[0]
                 else:
                     outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask=attention_mask)

                 # Extract final predictions and handle feature dimensions
                 f_dim = -1 if self.args.forecasting_features == 'MS' else 0
                 outputs = outputs[:, -self.args.pred_len:, f_dim:]
                 batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                 # **DENORMALIZATION FIX**: Create separate copies for metrics vs visualization
                 # Store ORIGINAL normalized data for metrics calculation (this is correct)
                 pred_normalized = outputs.detach().cpu().numpy()
                 true_normalized = batch_y.detach().cpu().numpy()
                 input_seq_normalized = batch_x.detach().cpu().numpy()
                 input_time = batch_x_mark.detach().cpu().numpy()

                 # Calculate loss using the instance criterion (on normalized data - correct!)
                 loss = self.criterion(outputs, batch_y)
                 total_loss.append(loss.item())

                 # Create DENORMALIZED copies for visualization only
                 if test and data.scale and self.args.inverse:
                     # Denormalize for visualization (create separate copies)
                     pred_denorm = pred_normalized.copy()
                     true_denorm = true_normalized.copy()
                     input_seq_denorm = input_seq_normalized.copy()
                     
                     for batch_idx in range(pred_denorm.shape[0]):
                         pred_denorm[batch_idx] = data.denormalize(pred_denorm[batch_idx])
                         true_denorm[batch_idx] = data.denormalize(true_denorm[batch_idx])
                         input_seq_denorm[batch_idx] = data.denormalize(input_seq_denorm[batch_idx])
                 else:
                     # If no denormalization, use normalized data for visualization too
                     pred_denorm = pred_normalized
                     true_denorm = true_normalized 
                     input_seq_denorm = input_seq_normalized

                 # Store NORMALIZED data for final metrics calculation (correct approach)
                 preds.append(pred_normalized)
                 trues.append(true_normalized)
                 inputs.append(input_seq_normalized)
                 input_marks.append(input_time)
                 test_batches_processed += 1

                 # **FIXED VISUALIZATION**: Use denormalized data and CLOSE PRICE (feature index 1)
                 if test and i % 20 == 0:
                     from utils.tools import visual
                     
                     # FIXED: Dynamically determine close price feature index
                     # Prefer dataset feature ordering (handles default feature list)
                     if hasattr(data, 'features') and 'close' in data.features:
                         close_feature_idx = data.features.index('close')
                     else:
                         # Fall back to args.features or first feature
                         if getattr(self.args, 'features', None) and 'close' in self.args.features:
                             close_feature_idx = self.args.features.index('close')
                         else:
                             close_feature_idx = 0  # default
                     
                     # Use DENORMALIZED data for visualization
                     input_sample = input_seq_denorm[0, :, close_feature_idx]
                     true_sample = true_denorm[0, :, close_feature_idx]  
                     pred_sample = pred_denorm[0, :, close_feature_idx]
                     
                     # Concatenate historical + future (like original)
                     gt = np.concatenate((input_sample, true_sample), axis=0)
                     pd = np.concatenate((input_sample, pred_sample), axis=0)
                     
                     # Save visualization
                     visual_path = os.path.join(folder_path, f'{i}.pdf')
                     visual(gt, pd, visual_path)

        # Calculate average loss for validation
        avg_loss = np.average(total_loss) if total_loss else np.inf

        # Only perform saving/plotting/metric calculation for final test run
        if test:
            # Concatenate NORMALIZED data for metrics (this is correct!)
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            inputs = np.concatenate(inputs, axis=0)
            input_marks = np.concatenate(input_marks, axis=0)

            print('test shape:', preds.shape, trues.shape)
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
            print('test shape:', preds.shape, trues.shape)

            # **CALCULATE METRICS ON NORMALIZED DATA** (this is the correct approach!)
            from utils.metrics import metric
            mae, mse, rmse, mape, mspe = metric(preds, trues)

            # Create denormalized copies for visualization and saving
            if data.scale and self.args.inverse:
                preds_denorm = preds.copy()
                trues_denorm = trues.copy()
                inputs_denorm = inputs.copy()
                
                # Denormalize all samples
                for sample_idx in range(preds_denorm.shape[0]):
                    preds_denorm[sample_idx] = data.denormalize(preds_denorm[sample_idx])
                    trues_denorm[sample_idx] = data.denormalize(trues_denorm[sample_idx])
                    inputs_denorm[sample_idx] = data.denormalize(inputs_denorm[sample_idx])
            else:
                preds_denorm = preds
                trues_denorm = trues
                inputs_denorm = inputs

            # Use config-driven results directory
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            # **PRINT RESULTS**: Same format as original
            print('mse:{}, mae:{}'.format(mse, mae))
            
            # Save BOTH normalized (for model analysis) and denormalized (for interpretation)
            np.save(os.path.join(results_dir, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
            np.save(os.path.join(results_dir, 'pred_normalized.npy'), preds)
            np.save(os.path.join(results_dir, 'true_normalized.npy'), trues)
            np.save(os.path.join(results_dir, 'pred_denormalized.npy'), preds_denorm)
            np.save(os.path.join(results_dir, 'true_denormalized.npy'), trues_denorm)
            np.save(os.path.join(results_dir, 'inputs_denormalized.npy'), inputs_denorm)

            # **FIXED COMPREHENSIVE VISUALIZATION**: Use denormalized data and close price
            if hasattr(self, 'visualizer'):
                try:
                    if len(inputs_denorm) > 0 and len(preds_denorm) > 0:
                        vis_dir = os.path.join(results_dir, 'visualizations')
                        if not os.path.exists(vis_dir):
                            os.makedirs(vis_dir)
                        
                        # FIXED: Dynamically determine close price feature index
                        # Prefer dataset feature ordering (handles default feature list)
                        if hasattr(data, 'features') and 'close' in data.features:
                            close_feature_idx = data.features.index('close')
                        else:
                            # Fall back to args.features or first feature
                            if getattr(self.args, 'features', None) and 'close' in self.args.features:
                                close_feature_idx = self.args.features.index('close')
                            else:
                                close_feature_idx = 0  # default
                        
                        # Use DENORMALIZED data and close price
                        fig = self.visualizer.plot_comprehensive_predictions(
                            historical_data=inputs_denorm, 
                            historical_marks=input_marks,
                            true_values=trues_denorm, 
                            predictions=preds_denorm, 
                            dataset=data, 
                            feature_idx=close_feature_idx,
                            n_samples_to_plot=3,
                            return_fig=True
                        )
                        if writer is not None and epoch is not None and fig is not None:
                            writer.add_figure('Test/Predictions', fig, epoch)
                except Exception as e:
                    print(f"Warning: Failed during comprehensive visualization - {e}")

            return mse

        else: # If validation (test=0)
            return avg_loss # Return average validation loss for early stopping/LR scheduling

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
