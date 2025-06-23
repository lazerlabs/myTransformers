"""
Simplified Experiment Runner for Returns-Only Stock Forecasting

This is a clean, simple version that:
- Always uses returns-based preprocessing 
- No normalization options or global statistics
- Much simpler API and cleaner code
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import warnings

from exp_basic import Exp_Basic
from utils.metrics import metric
from utils.loss import get_loss_function

# Use the existing stock_dataset but with simplified calls
from stock_dataset import create_dataloader


class SimpleStockExperiment(Exp_Basic):
    """Simplified stock forecasting experiment using returns-only preprocessing."""
    
    def __init__(self, args):
        super(SimpleStockExperiment, self).__init__(args)

    def _get_data(self, flag, shuffle=True, max_samples=None):
        """Get data using simplified approach - always returns-based."""
        
        if flag == 'train':
            data_path_list = self.args.train_files
            tickers = self.args.stocks
        elif flag == 'val':
            data_path_list = self.args.val_files
            tickers = self.args.val_stocks
        else:  # test
            data_path_list = self.args.test_files
            tickers = self.args.stocks

        if not data_path_list:
            print(f"No data files found for {flag}")
            return None, None

        # Use simplified dataloader creation - no scale parameters
        data_set, data_loader = create_dataloader(
            file_paths=data_path_list,
            batch_size=self.args.batch_size,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            tickers=tickers,
            features=self.args.features,
            shuffle=shuffle,
            mode=self.args.mode,
            interpolate_max_missing=self.args.interpolate_max_missing,
            max_samples=max_samples
        )
        
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return get_loss_function(self.args.loss_type, **self.args.loss_kwargs)

    def train(self, setting, writer=None, resume_checkpoint=None):
        """Simplified training loop."""
        train_data, train_loader = self._get_data(flag='train')
        
        if train_loader is None or len(train_loader) == 0:
            print("Error: No training data available")
            return None

        # Setup
        path = os.path.join(self.args.checkpoints_dir, setting)
        os.makedirs(path, exist_ok=True)
        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        print(f"Starting training for {self.args.train_epochs} epochs...")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_losses = []
            
            for i, (batch_x, batch_x_mark, batch_y, batch_y_mark, attention_mask) in enumerate(tqdm(train_loader)):
                model_optim.zero_grad()
                
                # Move to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                attention_mask = attention_mask.float().to(self.device)

                # Create decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Forward pass
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask=attention_mask)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask=attention_mask)

                # Extract predictions
                f_dim = -1 if self.args.forecasting_features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # Compute loss
                loss = criterion(outputs, batch_y)
                epoch_losses.append(loss.item())
                
                # Backward pass
                loss.backward()
                model_optim.step()

                # Log to TensorBoard if available
                if writer is not None:
                    global_iter = epoch * len(train_loader) + i
                    writer.add_scalar('Loss/train_iter', loss.item(), global_iter)

                # Early stopping within epoch if requested
                if self.args.max_train_iterations is not None and i >= self.args.max_train_iterations:
                    break

            # Epoch summary
            avg_train_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.6f}")
            
            # Validation
            val_loss = self.test(setting, test=0)  # validation
            print(f"Epoch {epoch + 1}: Val Loss = {val_loss:.6f}")
            
            if writer is not None:
                writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
                writer.add_scalar('Loss/val_epoch', val_loss, epoch)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), os.path.join(path, 'best_model.pth'))
                print(f"New best model saved with val loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                
            if patience_counter >= self.args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        best_model_path = os.path.join(path, 'best_model.pth')
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            print("Loaded best model for final evaluation")

        return self.model

    def test(self, setting, test=0, writer=None, epoch=None):
        """Simplified testing."""
        # Load model for final test
        if test:
            checkpoint_path = os.path.join(self.args.checkpoints_dir, setting, 'best_model.pth')
            if os.path.exists(checkpoint_path):
                self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
                print(f"Loaded model from {checkpoint_path}")

        # Get data
        data_flag = 'test' if test else 'val'
        data, data_loader = self._get_data(flag=data_flag, shuffle=False)

        if data_loader is None or len(data_loader) == 0:
            print(f"No {data_flag} data available")
            return float('inf') if not test else (np.nan, np.nan, np.nan)

        # Evaluate
        preds, trues = [], []
        self.model.eval()
        
        with torch.no_grad():
            for batch_x, batch_x_mark, batch_y, batch_y_mark, attention_mask in tqdm(data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                attention_mask = attention_mask.float().to(self.device)

                # Forward pass
                outputs = self.model(batch_x, batch_x_mark, None, None, mask=attention_mask)
                
                # Extract predictions
                f_dim = -1 if self.args.forecasting_features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        if not preds:
            print("No predictions generated")
            return float('inf') if not test else (np.nan, np.nan, np.nan)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        print(f"Evaluation shapes - Preds: {preds.shape}, Trues: {trues.shape}")

        # Calculate loss on returns scale (training scale)
        mae_raw, mse_raw, rmse_raw, mape_raw, mspe_raw = metric(preds, trues)
        
        if test:
            # For final test, also try denormalization if available
            try:
                if data is not None and hasattr(data, 'denormalize'):
                    batch_size = preds.shape[0]
                    sequence_indices = list(range(batch_size))
                    
                    preds_denorm = data.denormalize(preds, sequence_indices)
                    trues_denorm = data.denormalize(trues, sequence_indices)
                    
                    mae_denorm, mse_denorm, rmse_denorm, mape_denorm, mspe_denorm = metric(preds_denorm, trues_denorm)
                    
                    print(f"Raw metrics (returns scale): MSE={mse_raw:.6f}, MAE={mae_raw:.6f}")
                    print(f"Denormalized metrics (price scale): MSE={mse_denorm:.6f}, MAE={mae_denorm:.6f}")
                    
                    # Save results
                    results_dir = os.path.join("results", os.path.basename(self.args.checkpoints_dir))
                    os.makedirs(results_dir, exist_ok=True)
                    
                    np.save(os.path.join(results_dir, 'metrics_raw.npy'), np.array([mae_raw, mse_raw, rmse_raw, mape_raw, mspe_raw]))
                    np.save(os.path.join(results_dir, 'metrics_denorm.npy'), np.array([mae_denorm, mse_denorm, rmse_denorm, mape_denorm, mspe_denorm]))
                    np.save(os.path.join(results_dir, 'predictions_raw.npy'), preds)
                    np.save(os.path.join(results_dir, 'predictions_denorm.npy'), preds_denorm)
                    np.save(os.path.join(results_dir, 'ground_truth_raw.npy'), trues)
                    np.save(os.path.join(results_dir, 'ground_truth_denorm.npy'), trues_denorm)
                    
                    return mse_denorm  # Return denormalized MSE for final test
                else:
                    print("Denormalization not available")
                    
            except Exception as e:
                print(f"Denormalization failed: {e}")
                
            print(f"Final test metrics: MSE={mse_raw:.6f}, MAE={mae_raw:.6f}")
            return mse_raw
        else:
            # For validation, return raw loss (same scale as training)
            return mse_raw 
