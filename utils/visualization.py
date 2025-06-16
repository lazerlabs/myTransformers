import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from configs import StockPredictionConfig
import glob

class StockVisualizer:
    def __init__(self, save_dir='./figures/', feature_names=None):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        # Use provided feature names or default fallback
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            # Default feature names - avoid creating new config instance
            self.feature_names = ['close', 'volume', 'transactions']
        
    def plot_training_metrics(self, metrics, title='Training Metrics'):
        """Plot training, validation and test losses"""
        print("[DEBUG] plot_training_metrics called")
        print(f"[DEBUG] metrics keys: {list(metrics.keys())}")
        for k, v in metrics.items():
            print(f"[DEBUG] {k}: length={len(v)} sample={v[:5] if hasattr(v, '__getitem__') else v}")
        plt.figure(figsize=(12, 6))
        epochs = metrics['epoch']  # Use actual epoch numbers

        # Plot training loss
        if len(epochs) == 1:
            plt.scatter(epochs, metrics['train_loss'], label='Train Loss', color='blue')
            plt.text(epochs[0], metrics['train_loss'][0], "Only one epoch run", fontsize=10, color='blue')
        else:
            plt.plot(epochs, metrics['train_loss'], label='Train Loss', color='blue')

        # Plot validation loss if available
        if 'val_loss' in metrics and any(v is not None for v in metrics['val_loss']):
            if len(epochs) == 1:
                plt.scatter(epochs, metrics['val_loss'], label='Validation Loss', color='green')
                plt.text(epochs[0], metrics['val_loss'][0], "Only one epoch run", fontsize=10, color='green')
            else:
                plt.plot(epochs, metrics['val_loss'], label='Validation Loss', color='green')

        # Plot test loss if available
        if 'test_loss' in metrics and any(v is not None for v in metrics['test_loss']):
            if len(epochs) == 1:
                plt.scatter(epochs, metrics['test_loss'], label='Test Loss', color='red')
                plt.text(epochs[0], metrics['test_loss'][0], "Only one epoch run", fontsize=10, color='red')
            else:
                plt.plot(epochs, metrics['test_loss'], label='Test Loss', color='red')

        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'training_metrics.png'))
        plt.close()
        
    def plot_learning_rate(self, lr_history):
        """Plot learning rate changes"""
        print("[DEBUG] plot_learning_rate called")
        print(f"[DEBUG] lr_history length: {len(lr_history)} sample: {lr_history[:5] if hasattr(lr_history, '__getitem__') else lr_history}")
        plt.figure(figsize=(10, 4))
        epochs = range(1, len(lr_history) + 1)
        if len(lr_history) == 1:
            plt.scatter(epochs, lr_history, label='Learning Rate', color='purple')
            plt.text(1, lr_history[0], "Only one epoch run", fontsize=10, color='purple')
        else:
            plt.plot(epochs, lr_history, label='Learning Rate', color='purple')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'learning_rate.png'))
        plt.close()
        
    def plot_iteration_metrics(self, iteration_metrics):
        """Plot iteration-level training metrics"""
        print("[DEBUG] plot_iteration_metrics called")
        print(f"[DEBUG] iteration_metrics keys: {list(iteration_metrics.keys())}")
        
        if not iteration_metrics['iteration']:
            print("Warning: No iteration metrics to plot")
            return
            
        iterations = iteration_metrics['iteration']
        train_losses = iteration_metrics['train_loss']
        learning_rates = iteration_metrics['learning_rate']
        epochs = iteration_metrics['epoch']
        
        print(f"[DEBUG] plotting {len(iterations)} iterations")
        
        # Create subplots for loss and learning rate
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot training loss per iteration
        ax1.plot(iterations, train_losses, alpha=0.7, linewidth=0.5, color='blue', label='Training Loss')
        
        # Add running average for smoother visualization
        window_size = max(1, len(train_losses) // 100)  # Smooth over 1% of data
        if len(train_losses) >= window_size:
            import pandas as pd
            smooth_loss = pd.Series(train_losses).rolling(window=window_size, min_periods=1).mean()
            ax1.plot(iterations, smooth_loss, color='red', linewidth=2, label=f'Running Average (window={window_size})')
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss per Iteration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add epoch boundaries as vertical lines
        unique_epochs = sorted(set(epochs))
        epoch_boundaries = []
        for epoch in unique_epochs[1:]:  # Skip first epoch
            first_iter_of_epoch = next((i for i, e in enumerate(epochs) if e == epoch), None)
            if first_iter_of_epoch is not None:
                epoch_boundaries.append(iterations[first_iter_of_epoch])
        
        for boundary in epoch_boundaries:
            ax1.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
        
        # Plot learning rate per iteration
        ax2.plot(iterations, learning_rates, color='purple', linewidth=1, label='Learning Rate')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate per Iteration')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add epoch boundaries to learning rate plot too
        for boundary in epoch_boundaries:
            ax2.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'iteration_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Iteration-level metrics plot saved to {os.path.join(self.save_dir, 'iteration_metrics.png')}")
        
        # Also create separate plots for better detail
        self._plot_iteration_loss_detail(iterations, train_losses, epochs)
        self._plot_iteration_lr_detail(iterations, learning_rates, epochs)
    
    def _plot_iteration_loss_detail(self, iterations, train_losses, epochs):
        """Plot detailed training loss with better visualization"""
        plt.figure(figsize=(15, 6))
        
        # Color by epoch for better visualization
        unique_epochs = sorted(set(epochs))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_epochs)))
        epoch_color_map = {epoch: color for epoch, color in zip(unique_epochs, colors)}
        
        # Plot points colored by epoch
        for epoch in unique_epochs:
            epoch_mask = [e == epoch for e in epochs]
            epoch_iters = [iterations[i] for i in range(len(iterations)) if epoch_mask[i]]
            epoch_losses = [train_losses[i] for i in range(len(train_losses)) if epoch_mask[i]]
            plt.scatter(epoch_iters, epoch_losses, c=[epoch_color_map[epoch]], 
                       s=1, alpha=0.6, label=f'Epoch {epoch}')
        
        plt.xlabel('Iteration')
        plt.ylabel('Training Loss')
        plt.title('Training Loss per Iteration (Colored by Epoch)')
        plt.grid(True, alpha=0.3)
        
        # Only show legend if not too many epochs
        if len(unique_epochs) <= 10:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'iteration_loss_detail.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_iteration_lr_detail(self, iterations, learning_rates, epochs):
        """Plot detailed learning rate changes"""
        plt.figure(figsize=(15, 6))
        
        plt.plot(iterations, learning_rates, color='purple', linewidth=1.5)
        plt.xlabel('Iteration')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule per Iteration')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Add epoch boundaries
        unique_epochs = sorted(set(epochs))
        for epoch in unique_epochs[1:]:
            first_iter_of_epoch = next((i for i, e in enumerate(epochs) if e == epoch), None)
            if first_iter_of_epoch is not None:
                boundary = iterations[first_iter_of_epoch]
                plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.5)
                plt.text(boundary, max(learning_rates)*0.5, f'Epoch {epoch}', 
                        rotation=90, verticalalignment='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'iteration_lr_detail.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_prediction(self, true_values, predictions, feature_idx=0, stock_idx=0, n_samples=100):
        """
        Plot true vs predicted values for a specific feature and stock
        
        Args:
            true_values: shape [batch, stocks, pred_len, features]
            predictions: shape [batch, stocks, pred_len, features]
            feature_idx: which feature to plot (0: volume, 1: close, 2: transactions)
            stock_idx: which stock to plot (0-4 for our 5 stocks)
            n_samples: number of points to plot
        """
        plt.figure(figsize=(15, 6))
        
        # Flatten batch dimension and select specific stock and feature
        true_flat = true_values.reshape(-1, true_values.shape[2], true_values.shape[3])
        pred_flat = predictions.reshape(-1, predictions.shape[2], predictions.shape[3])
        
        # Get data for specific stock
        true_stock = true_flat[stock_idx::true_values.shape[1]]  # Skip other stocks
        pred_stock = pred_flat[stock_idx::predictions.shape[1]]
        
        # Plot last n_samples points
        x = np.arange(min(n_samples, len(true_stock)))
        plt.plot(x, true_stock[-n_samples:, 0, feature_idx], 
                label='True', marker='o', markersize=2)
        plt.plot(x, pred_stock[-n_samples:, 0, feature_idx], 
                label='Predicted', marker='o', markersize=2)
        
        plt.title(f'True vs Predicted Values ({self.feature_names[feature_idx]})')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, f'prediction_stock_{stock_idx}_feature_{feature_idx}.png'))
        plt.close()
        
    def plot_attention_weights(self, attention_weights, feature_names=None):
        """Plot attention weight matrix"""
        if feature_names is None:
            feature_names = self.feature_names
        
        # Average attention weights across batches and heads if necessary
        if len(attention_weights.shape) > 2:
            attention_weights = attention_weights.mean(axis=(0, 1))
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights, 
                    xticklabels=feature_names,
                    yticklabels=feature_names,
                    cmap='viridis',
                    annot=True,
                    fmt='.2f')
        plt.title('Attention Weights Between Features')
        plt.savefig(os.path.join(self.save_dir, 'attention_weights.png'))
        plt.close()
        
    def plot_feature_importance(self, predictions, true_values, feature_names=None):
        """Plot feature-wise prediction errors"""
        if feature_names is None:
            feature_names = self.feature_names
        
        # Reshape to combine batch and stock dimensions
        true_flat = true_values.reshape(-1, true_values.shape[-2], true_values.shape[-1])
        pred_flat = predictions.reshape(-1, predictions.shape[-2], predictions.shape[-1])
        
        # Calculate MSE per feature
        mse_per_feature = np.mean((pred_flat - true_flat) ** 2, axis=(0, 1))
        
        # Create DataFrame for seaborn
        feature_data = []
        for fname, mse in zip(feature_names, mse_per_feature):
            feature_data.append({'Feature': fname, 'MSE': mse})
        
        df = pd.DataFrame(feature_data)
        
        plt.figure(figsize=(10, 5))
        sns.barplot(data=df, x='Feature', y='MSE')
        plt.title('MSE per Feature')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'feature_importance.png'))
        plt.close()

    def plot_all_stocks_predictions(self, true_values, predictions, dataset, feature_idx=1, n_samples_to_plot=1, return_fig=False, config=None):
        """
        Legacy method - redirects to comprehensive plotting.
        For backward compatibility, this attempts to work with limited data.
        """
        print("Warning: Using legacy plot_all_stocks_predictions method.")
        print("For best results, use plot_comprehensive_predictions with historical data.")
        
        # Since we don't have historical data, create dummy historical data for visualization
        batch_size, pred_len, features = true_values.shape
        seq_len = 60  # Default sequence length
        
        # Create dummy historical data (this won't be accurate, just for visualization structure)
        dummy_historical = np.random.randn(batch_size, seq_len, features) * 0.1
        dummy_hist_marks = np.random.randn(batch_size, seq_len, 5)  # 5 time features
        
        # Use the comprehensive method with dummy data
        return self.plot_comprehensive_predictions(
            historical_data=dummy_historical,
            historical_marks=dummy_hist_marks,
            true_values=true_values,
            predictions=predictions,
            dataset=dataset,
            feature_idx=feature_idx,
            n_samples_to_plot=n_samples_to_plot,
            return_fig=return_fig,
            config=config
        )

    def plot_comprehensive_predictions(self, historical_data, historical_marks, true_values, predictions, dataset, feature_idx=1, n_samples_to_plot=3, return_fig=False, config=None):
        """
        Plot comprehensive predictions showing:
        1. Historical input sequence (60 points)
        2. True future values (15 points) 
        3. Predicted future values (15 points)
        With proper timestamps and ticker information.

        Args:
            historical_data (np.ndarray): Historical input sequences. Shape: [total_sequences, seq_len, features]
            historical_marks (np.ndarray): Historical timestamps. Shape: [total_sequences, seq_len, time_features]
            true_values (np.ndarray): Ground truth future values. Shape: [total_sequences, pred_len, features]
            predictions (np.ndarray): Predicted future values. Shape: [total_sequences, pred_len, features]
            dataset (StockDataset): The dataset instance (used for denormalization).
            feature_idx (int): Index of the feature to plot (e.g., 1 for 'close').
            n_samples_to_plot (int): Default number of sample sequences to plot.
            return_fig (bool): Whether to return the figure object.
            config (StockPredictionConfig): Config object to check if val_stocks were explicitly specified.
        """
        print("\n--- Starting Comprehensive Visualization ---")
        print(f"[DEBUG] historical_data shape: {historical_data.shape}")
        print(f"[DEBUG] historical_marks shape: {historical_marks.shape}")
        print(f"[DEBUG] true_values shape: {true_values.shape}")
        print(f"[DEBUG] predictions shape: {predictions.shape}")
        
        if historical_data.shape[0] == 0 or true_values.shape[0] == 0 or predictions.shape[0] == 0:
            print("Warning: Cannot plot predictions, data arrays are empty.")
            return None
            
        if not hasattr(dataset, 'denormalize'):
            print("Warning: Dataset object does not have a 'denormalize' method. Cannot plot.")
            return None
            
        if not hasattr(dataset, 'features'):
            raise ValueError("Dataset object does not expose a 'features' attribute; cannot map feature indices correctly.")
        # Sync visualizer feature list with the dataset actually used for this plot
        self.feature_names = list(dataset.features)

        # If the caller didn't pass a valid index, fall back to the first feature
        if feature_idx >= len(self.feature_names):
            print(f"[WARN] Requested feature_idx={feature_idx} but dataset only has {len(self.feature_names)} feature(s). Falling back to index 0.")
            feature_idx = 0

        num_total_sequences = historical_data.shape[0]
        
        # NEW: Dynamically determine number of plots based on validation stocks
        # If config is provided and val_stocks were explicitly specified (non-default), plot all of them
        if config is not None and hasattr(config, 'val_stocks') and config.val_stocks:
            # Check if we have ticker information available to identify unique stocks
            try:
                unique_tickers = set()
                for seq_idx in range(min(100, num_total_sequences)):  # Sample first 100 to avoid overhead
                    try:
                        if hasattr(dataset, 'mode') and dataset.mode == 'full_day':
                            if len(dataset.all_sequences[seq_idx]) >= 4:
                                _, _, ticker_name, _ = dataset.all_sequences[seq_idx]
                            else:
                                _, _, ticker_name = dataset.all_sequences[seq_idx][:3]
                        else:
                            _, _, ticker_name = dataset.all_sequences[seq_idx]
                        unique_tickers.add(ticker_name)
                    except:
                        continue
                
                # If we found unique tickers and they match validation stocks, use all of them
                if unique_tickers:
                    val_stocks_set = set(config.val_stocks)
                    tickers_in_data = unique_tickers.intersection(val_stocks_set)
                    if tickers_in_data:
                        num_plots = len(tickers_in_data)
                        print(f"[INFO] Found validation stocks {sorted(tickers_in_data)} in data. Plotting all {num_plots} stocks.")
                    else:
                        num_plots = min(n_samples_to_plot, num_total_sequences)
                        print(f"[INFO] Validation stocks specified but not found in current dataset. Using default {num_plots} samples.")
                else:
                    num_plots = min(n_samples_to_plot, num_total_sequences)
                    print(f"[INFO] Could not determine tickers from dataset. Using default {num_plots} samples.")
            except Exception as e:
                print(f"[WARN] Error determining stocks from dataset: {e}. Using default number of samples.")
                num_plots = min(n_samples_to_plot, num_total_sequences)
        else:
            # Default behavior: use the provided n_samples_to_plot
            num_plots = min(n_samples_to_plot, num_total_sequences)
            
        if num_plots == 0:
            print("No samples to plot.")
            return None

        print(f"Plotting {num_plots} comprehensive sequences for feature '{self.feature_names[feature_idx]}'")

        # New: container to collect data for optional CSV export
        plot_data_records = []  # Each record is a dict for DataFrame rows

        # Create subplot grid
        fig, axes = plt.subplots(num_plots, 1, figsize=(20, 6 * num_plots), squeeze=False)

        # Intelligent sample selection to ensure stock diversity
        sample_indices = self._select_diverse_samples(dataset, num_total_sequences, num_plots)

        for plot_idx, seq_idx in enumerate(sample_indices):
            ax = axes[plot_idx, 0]

            # Get time marks for timestamp reconstruction
            hist_time_marks = historical_marks[seq_idx, :, :]    # [seq_len, time_features]
            
            # Determine the real (unpadded) sequence length using time marks
            if hist_time_marks.ndim == 2 and hist_time_marks.shape[1] > 0:
                # Rows that are entirely zero correspond to padding that we added in the dataloader
                real_seq_mask = ~np.all(hist_time_marks == 0, axis=1)
                real_seq_len = int(real_seq_mask.sum())
            else:
                real_seq_len = hist_time_marks.shape[0]
            if real_seq_len == 0:
                print(f"[WARN] Sequence {seq_idx} appears to be completely padded. Skipping plot.")
                continue

            # Slice to the real part only (discard padding before denormalisation)
            hist_full_features = historical_data[seq_idx, :real_seq_len, :]  # [real_seq_len, all_features]
            true_full_features = true_values[seq_idx, :, :]                  # [pred_len, all_features]
            pred_full_features = predictions[seq_idx, :, :]                  # [pred_len, all_features]

            # Update variables for downstream use
            seq_len = real_seq_len
            pred_len = true_full_features.shape[0]

            # Denormalise
            try:
                # Data is already denormalized from the test() function
                hist_denorm_full = hist_full_features
                true_denorm_full = true_full_features
                pred_denorm_full = pred_full_features
                
                hist_denorm = hist_denorm_full[:, feature_idx]
                true_denorm = true_denorm_full[:, feature_idx]
                pred_denorm = pred_denorm_full[:, feature_idx]
            except Exception as e:
                print(f"Warning: Failed to denormalize data for plot {plot_idx}: {e}. Using normalized data.")
                hist_denorm = hist_full_features[:, feature_idx]
                true_denorm = true_full_features[:, feature_idx]
                pred_denorm = pred_full_features[:, feature_idx]

            # Get Ticker and Timestamp info with improved error handling
            try:
                # Fix the get_sequence_info method to handle both modes
                if hasattr(dataset, 'mode') and dataset.mode == 'full_day':
                    # For full_day mode: (sequence_data, start_timestamp, ticker, input_length)
                    if len(dataset.all_sequences[seq_idx]) >= 4:
                        _, start_timestamp, ticker_name, _ = dataset.all_sequences[seq_idx]
                    else:
                        # Fallback for unexpected structure
                        _, start_timestamp, ticker_name = dataset.all_sequences[seq_idx][:3]
                else:
                    # For sliding_window mode: (sequence_data, start_timestamp, ticker)
                    _, start_timestamp, ticker_name = dataset.all_sequences[seq_idx]
                
                # Ensure timestamp is a usable pandas object for formatting
                if not isinstance(start_timestamp, pd.Timestamp):
                    start_timestamp = pd.to_datetime(start_timestamp)
                start_time_str = start_timestamp.strftime('%Y-%m-%d %H:%M')
                
                # Create a more informative reference
                end_timestamp = start_timestamp + pd.Timedelta(minutes=seq_len-1)
                end_time_str = end_timestamp.strftime('%H:%M')
                time_reference = f"{start_time_str} to {end_time_str}"
                
                timestamp_available = True
            except Exception as e:
                print(f"Warning: Could not retrieve valid sequence info for index {seq_idx}. Error: {e}")
                # Fallback if get_sequence_info fails or returns invalid data
                ticker_name = f"Unknown_Ticker"
                time_reference = f"Sequence #{seq_idx}"
                timestamp_available = False
            
            # Create continuous x-axis (historical + future)
            x_hist = np.arange(seq_len)
            x_future = np.arange(seq_len, seq_len + pred_len)
            
            # Plot historical data
            ax.plot(x_hist, hist_denorm, 
                   label=f'Historical Data ({seq_len} minutes)', 
                   color='#1f77b4', 
                   linewidth=2, 
                   alpha=0.8)
            
            # Plot true future values
            ax.plot(x_future, true_denorm, 
                   label='True Future', 
                   color='#2ca02c', 
                   linewidth=2.5, 
                   marker='o', 
                   markersize=5,
                   alpha=0.9)
            
            # Plot predicted future values
            ax.plot(x_future, pred_denorm, 
                   label='Predicted Future', 
                   color='#d62728', 
                   linewidth=2.5, 
                   marker='s', 
                   markersize=5,
                   linestyle='--',
                   alpha=0.9)
            
            # Add vertical line to separate historical from future
            ax.axvline(x=seq_len-0.5, color='gray', linestyle=':', alpha=0.7, linewidth=2)
            ax.text(seq_len-0.5, ax.get_ylim()[1]*0.95, 'Prediction\nStarts Here', 
                   rotation=0, horizontalalignment='center', verticalalignment='top', 
                   fontsize=10, alpha=0.7, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Calculate and display metrics for this sequence
            mse = np.mean((pred_denorm - true_denorm) ** 2)
            mae = np.mean(np.abs(pred_denorm - true_denorm))
            mape = np.mean(np.abs((pred_denorm - true_denorm) / (true_denorm + 1e-8))) * 100
            
            # Improved title with better ticker display and reference info
            title_text = (f'Stock: {ticker_name} | {self.feature_names[feature_idx].title()} Price\n'
                           f'MSE: {mse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}%')
            if timestamp_available:
                title_text += f'\nData Reference: {time_reference} | Dataset Index: {seq_idx}'
                title_text += f'\nReal Data Start: {start_timestamp.strftime("%Y-%m-%d %H:%M:%S")} (check original CSV at this timestamp)'
            else:
                title_text += f'\nDataset Index: {seq_idx} (for data verification)'
            ax.set_title(title_text, fontsize=13, fontweight='bold')
            
            ax.set_xlabel('Time Steps (Minutes)', fontsize=12)
            ax.set_ylabel(f'{self.feature_names[feature_idx].title()} Price', fontsize=12)
            
            # Position legend to avoid overlap with performance indicator
            ax.legend(fontsize=11, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Improved x-axis labeling with time information
            self._set_time_axis_labels(ax, seq_len, pred_len)
            
            # Add some styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Add performance indicator colors - positioned to avoid legend overlap
            if mape < 5:
                performance_color = '#2ca02c'  # Green for good
                performance_text = 'Good'
            elif mape < 10:
                performance_color = '#ff7f0e'  # Orange for fair
                performance_text = 'Fair'
            else:
                performance_color = '#d62728'  # Red for poor
                performance_text = 'Poor'
                
            # Position performance indicator on the left side to avoid legend overlap
            ax.text(0.02, 0.02, f'Performance: {performance_text}', 
                   transform=ax.transAxes, fontsize=10, fontweight='bold',
                   verticalalignment='bottom', color=performance_color,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=performance_color, alpha=0.2))

            # -------------------------------
            # NEW: collect data for CSV export
            # -------------------------------
            # Get timestamp string for CSV
            timestamp_str = start_timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp_available else "N/A"
            
            # Historical data
            for step_idx, val in enumerate(hist_denorm):
                plot_data_records.append({
                    'sequence_idx': int(seq_idx),
                    'ticker': str(ticker_name),
                    'start_timestamp': timestamp_str,
                    'data_type': 'historical',
                    'time_step': int(step_idx),
                    'value': float(val)
                })
            # True future values
            for fut_idx, val in enumerate(true_denorm):
                plot_data_records.append({
                    'sequence_idx': int(seq_idx),
                    'ticker': str(ticker_name),
                    'start_timestamp': timestamp_str,
                    'data_type': 'true',
                    'time_step': int(seq_len + fut_idx),
                    'value': float(val)
                })
            # Predicted future values
            for fut_idx, val in enumerate(pred_denorm):
                plot_data_records.append({
                    'sequence_idx': int(seq_idx),
                    'ticker': str(ticker_name),
                    'start_timestamp': timestamp_str,
                    'data_type': 'predicted',
                    'time_step': int(seq_len + fut_idx),
                    'value': float(val)
                })

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.suptitle(f'Stock Prediction Analysis: {self.feature_names[feature_idx].title()} Price Forecasting', 
                    fontsize=16, fontweight='bold')
        
        try:
            base_name = f'comprehensive_predictions_{self.feature_names[feature_idx]}'
            png_path = os.path.join(self.save_dir, base_name + '.png')
            pdf_path = os.path.join(self.save_dir, base_name + '.pdf')
            csv_path = os.path.join(self.save_dir, base_name + '.csv')

            # Save PNG
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            # Save PDF
            plt.savefig(pdf_path, dpi=300, bbox_inches='tight', format='pdf')
            print(f"Saved comprehensive prediction plot to {png_path} and {pdf_path}")

            # Save CSV with plotted data
            if len(plot_data_records) > 0:
                try:
                    df_plot = pd.DataFrame(plot_data_records)
                    df_plot.to_csv(csv_path, index=False)
                    print(f"Saved plotted data to {csv_path}")
                except Exception as csv_err:
                    print(f"Error saving CSV data: {csv_err}")
        except Exception as e:
            print(f"Error saving plot or data: {e}")
        
        if return_fig:
            return fig
        plt.close(fig)
        return None
        
    def _select_diverse_samples(self, dataset, num_total_sequences, num_plots):
        """
        Intelligently select sample indices to ensure stock diversity.
        
        Strategy:
        1. Try to select one sample from each unique stock
        2. If more samples needed, fill randomly from remaining sequences
        3. If fewer unique stocks than requested samples, use random sampling
        4. When num_plots equals the number of unique tickers, prioritize even distribution
        
        Args:
            dataset: The dataset object containing sequence information
            num_total_sequences: Total number of sequences available
            num_plots: Number of samples to select
            
        Returns:
            np.ndarray: Selected sample indices
        """
        try:
            # Build a mapping of ticker -> list of sequence indices
            ticker_to_indices = {}
            
            for seq_idx in range(num_total_sequences):
                try:
                    # Extract ticker information from sequence
                    if hasattr(dataset, 'mode') and dataset.mode == 'full_day':
                        if len(dataset.all_sequences[seq_idx]) >= 4:
                            _, _, ticker_name, _ = dataset.all_sequences[seq_idx]
                        else:
                            _, _, ticker_name = dataset.all_sequences[seq_idx][:3]
                    else:
                        _, _, ticker_name = dataset.all_sequences[seq_idx]
                    
                    if ticker_name not in ticker_to_indices:
                        ticker_to_indices[ticker_name] = []
                    ticker_to_indices[ticker_name].append(seq_idx)
                    
                except Exception as e:
                    # If we can't get ticker info, skip this sequence for diversity selection
                    continue
            
            unique_tickers = list(ticker_to_indices.keys())
            print(f"[DEBUG] Found {len(unique_tickers)} unique tickers: {unique_tickers}")
            print(f"[DEBUG] Sequences per ticker: {[(ticker, len(indices)) for ticker, indices in ticker_to_indices.items()]}")
            
            selected_indices = []
            
            if len(unique_tickers) == 0:
                # Fallback: no ticker info available, use random sampling
                print("[DEBUG] No ticker information available, using random sampling")
                return np.random.choice(num_total_sequences, num_plots, replace=False)
            
            # Phase 1: Select samples from each unique ticker
            # If num_plots >= number of unique tickers, select at least one from each ticker
            if num_plots >= len(unique_tickers):
                # Select one sample from each ticker first
                for ticker in unique_tickers:
                    ticker_indices = ticker_to_indices[ticker]
                    selected_idx = np.random.choice(ticker_indices)
                    selected_indices.append(selected_idx)
                    print(f"[DEBUG] Selected sequence {selected_idx} for ticker {ticker}")
                
                # If we need more samples, distribute them evenly across tickers
                remaining_plots = num_plots - len(unique_tickers)
                if remaining_plots > 0:
                    # Calculate how many additional samples per ticker
                    additional_per_ticker = remaining_plots // len(unique_tickers)
                    extra_samples = remaining_plots % len(unique_tickers)
                    
                    for i, ticker in enumerate(unique_tickers):
                        ticker_indices = ticker_to_indices[ticker]
                        # Remove already selected indices for this ticker
                        available_indices = [idx for idx in ticker_indices if idx not in selected_indices]
                        
                        if available_indices:
                            # Add extra sample to first few tickers if remainder exists
                            samples_to_add = additional_per_ticker + (1 if i < extra_samples else 0)
                            samples_to_add = min(samples_to_add, len(available_indices))
                            
                            if samples_to_add > 0:
                                additional_selected = np.random.choice(available_indices, samples_to_add, replace=False)
                                selected_indices.extend(additional_selected)
                                print(f"[DEBUG] Added {samples_to_add} additional samples for ticker {ticker}")
                        
            else:
                # num_plots < number of unique tickers, select from first num_plots tickers
                tickers_to_sample = unique_tickers[:num_plots]
                
                for ticker in tickers_to_sample:
                    ticker_indices = ticker_to_indices[ticker]
                    selected_idx = np.random.choice(ticker_indices)
                    selected_indices.append(selected_idx)
                    print(f"[DEBUG] Selected sequence {selected_idx} for ticker {ticker}")
            
            # Phase 2: Fill any remaining slots with random samples if needed
            if len(selected_indices) < num_plots:
                remaining_indices = [i for i in range(num_total_sequences) if i not in selected_indices]
                
                if len(remaining_indices) > 0:
                    additional_needed = num_plots - len(selected_indices)
                    additional_samples = np.random.choice(
                        remaining_indices, 
                        min(additional_needed, len(remaining_indices)), 
                        replace=False
                    )
                    selected_indices.extend(additional_samples)
                    print(f"[DEBUG] Added {len(additional_samples)} additional random samples")
            
            print(f"[DEBUG] Final selected indices: {selected_indices}")
            return np.array(selected_indices)
            
        except Exception as e:
            print(f"[DEBUG] Error in diverse sampling: {e}, falling back to random sampling")
            # Fallback to random sampling if anything goes wrong
            return np.random.choice(num_total_sequences, num_plots, replace=False)

    def _set_time_axis_labels(self, ax, seq_len, pred_len):
        """Set informative time-based axis labels."""
        try:
            # Create tick positions
            hist_ticks = list(range(0, seq_len, max(1, seq_len // 6)))  # ~6 ticks for historical
            future_ticks = list(range(seq_len, seq_len + pred_len, max(1, pred_len // 3)))  # ~3 ticks for future
            all_ticks = hist_ticks + future_ticks
            
            # Create labels
            tick_labels = []
            for pos in all_ticks:
                if pos < seq_len:
                    minutes_ago = seq_len - pos
                    tick_labels.append(f'-{minutes_ago}m')
                else:
                    minutes_ahead = pos - seq_len + 1
                    tick_labels.append(f'+{minutes_ahead}m')
            
            ax.set_xticks(all_ticks)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
            
        except Exception as e:
            print(f"Warning: Could not set time axis labels: {e}")
            # Fallback to simple labels
            ax.set_xticks(range(0, seq_len + pred_len, 10))
            ax.set_xticklabels([f'T{i}' for i in range(0, seq_len + pred_len, 10)])
