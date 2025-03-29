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
        # Import model here to avoid circular imports
        from models.iTransformer import Model as iTransformerModel

        model_dict = {
            'iTransformer': iTransformerModel,
        }

        if self.args.model not in model_dict:
            raise ValueError(f"Model {self.args.model} not found. Available models: {list(model_dict.keys())}")

        model = model_dict[self.args.model](self.args).float()

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
            shuffle=shuffle_data # Pass shuffle flag
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

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        # test_data, test_loader = self._get_data(flag='test') # Test loader not needed here

        # Ensure base checkpoints directory exists
        if not os.path.exists(self.args.checkpoints_dir):
            os.makedirs(self.args.checkpoints_dir)

        path = os.path.join(self.args.checkpoints_dir, setting)
        # Clean up old checkpoints if path exists
        if os.path.exists(path):
            import shutil
            print(f"Removing existing checkpoint directory: {path}")
            shutil.rmtree(path)
        os.makedirs(path)

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

        model_optim = self._select_optimizer() # Use the method
        self.criterion = self._select_criterion() # Get and store the configured criterion

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
        # test_losses = [] # Removed as intermediate testing is removed
        learning_rates = []

        print(f"\nStarting Training for {self.args.train_epochs} epochs...")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            epoch_train_loss = [] # Use a different name to avoid confusion with outer scope train_loss list

            self.model.train()
            epoch_time = time.time()
            # Use tqdm for progress bar
            pbar = tqdm(enumerate(train_loader), total=train_steps, desc=f"Epoch {epoch + 1}/{self.args.train_epochs}")

            for i, batch_data in pbar:
                # Check if batch_data is None (can happen if dataloader failed)
                if batch_data is None:
                     warnings.warn(f"Skipping iteration {i} due to None batch data.")
                     continue

                # Unpack batch data
                try:
                     batch_x, batch_x_mark, batch_y, batch_y_mark = batch_data
                except ValueError as e:
                     warnings.warn(f"Skipping iteration {i} due to error unpacking batch data: {e}")
                     continue # Skip this batch


                iter_count += 1
                model_optim.zero_grad()

                # Move data to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark is not used by the encoder-only model's forward pass

                # Forward pass
                model_output = self.model(batch_x, batch_x_mark)

                # Handle model output - extract predictions if model returns tuple
                outputs = model_output[0] if isinstance(model_output, tuple) else model_output

                # Calculate loss
                loss = self.criterion(outputs, batch_y)
                epoch_train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

                # Update progress bar description
                pbar.set_description(f"Epoch {epoch + 1}/{self.args.train_epochs} | Iter {i+1}/{train_steps} | Loss: {loss.item():.6f}")

                # Check for max_train_iterations limit within the loop
                if self.args.max_train_iterations is not None and i + 1 >= self.args.max_train_iterations:
                    print(f"\nReached max_train_iterations ({self.args.max_train_iterations}). Stopping epoch {epoch + 1} early.")
                    break # Exit the inner batch loop

            # --- End of Epoch ---
            # Calculate average loss, handle case where loop might not have run (e.g., max_iter=0)
            avg_epoch_train_loss = np.average(epoch_train_loss) if epoch_train_loss else 0.0
            train_losses.append(avg_epoch_train_loss)

            # Validation
            print(f"\nRunning validation for epoch {epoch + 1}...")
            val_loss = self.test(setting, test=0) # Validation loss calculation remains
            val_losses.append(val_loss)

            # Learning Rate Step
            current_lr = model_optim.param_groups[0]['lr']
            learning_rates.append(current_lr)
            if self.args.lr_scheduler == 'cosine':
                 scheduler.step()
            else: # reduce_on_plateau
                 scheduler.step(val_loss)


            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.6f} Val Loss: {3:.6f} Learning Rate: {4:.6f}".format(
                epoch + 1, train_steps if self.args.max_train_iterations is None else i + 1, avg_epoch_train_loss, val_loss, current_lr))

            # Early Stopping
            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break # Exit the outer epoch loop

            print(f"Epoch {epoch + 1} completed in {time.time() - epoch_time:.2f} seconds.")


        # --- End of Training ---
        print("\nTraining finished.")

        # Plot training metrics
        metrics = {
            'epoch': list(range(1, len(train_losses) + 1)),
            'train_loss': train_losses,
            'val_loss': val_losses,
            # 'test_loss': test_losses, # Removed
            'learning_rate': learning_rates
        }
        try:
            self.visualizer.plot_training_metrics(metrics)
            self.visualizer.plot_learning_rate(learning_rates)
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

    def test(self, setting, test=0):
        """
        test=0: validation
        test=1: testing
        """
        # Only load the model for final testing
        if test:
            print('loading model for final test')
            checkpoint_path = os.path.join(self.args.checkpoints_dir, setting, 'checkpoint.pth')
            if not os.path.exists(checkpoint_path):
                 print(f"Error: Checkpoint file not found at {checkpoint_path}. Cannot run final test.")
                 return np.inf # Return infinity for validation loss if checkpoint missing
            try:
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            except Exception as e:
                 print(f"Error loading checkpoint from {checkpoint_path}: {e}")
                 return np.inf

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
        total_loss = [] # For calculating average validation loss

        self.model.eval()
        test_batches_processed = 0
        max_test_batches = 10 # Limit to 10 batches for quick test=1 evaluation

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
                      batch_x, batch_x_mark, batch_y, batch_y_mark = batch_data
                 except ValueError as e:
                      warnings.warn(f"Skipping iteration {i} in evaluation due to error unpacking batch data: {e}")
                      continue # Skip this batch

                 batch_x = batch_x.float().to(self.device)
                 batch_y = batch_y.float().to(self.device)
                 batch_x_mark = batch_x_mark.float().to(self.device)
                 # batch_y_mark is not used

                 # Forward pass and handle tuple output
                 model_output = self.model(batch_x, batch_x_mark)
                 outputs = model_output[0] if isinstance(model_output, tuple) else model_output

                 # Calculate loss using the instance criterion
                 loss = self.criterion(outputs, batch_y)
                 total_loss.append(loss.item())

                 # Store predictions and true values (needed for final test metrics/plots)
                 pred = outputs.detach().cpu().numpy()
                 true = batch_y.detach().cpu().numpy()

                 # if test:  # Only print shapes during testing - removed for less verbose output
                 #     print(f"Batch shapes - pred: {pred.shape}, true: {true.shape}")

                 preds.append(pred)
                 trues.append(true)
                 test_batches_processed += 1 # Increment counter

        # Calculate average loss for validation
        avg_loss = np.average(total_loss) if total_loss else np.inf

        # Only perform saving/plotting/metric calculation for final test run
        if test:
            # Concatenate along the batch dimension
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)

            print('Final shapes - preds:', preds.shape, 'trues:', trues.shape)

            # Calculate metrics
            mae = np.mean(np.abs(preds - trues))
            mse = np.mean((preds - trues) ** 2)
            rmse = np.sqrt(mse)

            # result save
            folder_path = os.path.join('./results/', setting) # Use os.path.join
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            print(f'Test Results - MSE:{mse:.6f}, MAE:{mae:.6f}, RMSE:{rmse:.6f}')

            np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse]))
            np.save(os.path.join(folder_path, 'pred.npy'), preds)
            np.save(os.path.join(folder_path, 'true.npy'), trues)

            # Log final metrics
            self.logger.log_prediction(mae, mse, rmse)

            # Get test dataset for denormalization
            test_dataset = data # data holds the dataset instance returned by _get_data

            # Visualize predictions for all stocks (if dataset exists)
            if test_dataset and hasattr(test_dataset, 'denormalize'): # Check if dataset and denormalize exist
                 try:
                      # Visualize predictions for all stocks (e.g., close price feature_idx=1)
                      self.visualizer.plot_all_stocks_predictions(trues, preds, test_dataset, feature_idx=1)

                      # Individual feature plots if needed
                      # for feature_idx in range(len(self.visualizer.feature_names)):
                      #      self.visualizer.plot_all_stocks_predictions(trues, preds, test_dataset, feature_idx=feature_idx)
                 except Exception as e:
                      print(f"Warning: Failed during visualization - {e}")
            else:
                 print("Warning: Skipping visualization as test dataset or denormalize method is unavailable.")

            return mse # Return final test MSE

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