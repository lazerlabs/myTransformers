import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from configs import StockPredictionConfig

class StockVisualizer:
    def __init__(self, save_dir='./figures/'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        # Updated feature names to match dataset order
        self.feature_names = StockPredictionConfig().features
        
    def plot_training_metrics(self, metrics, title='Training Metrics'):
        """Plot training, validation and test losses"""
        plt.figure(figsize=(12, 6))
        epochs = metrics['epoch']  # Use actual epoch numbers
        
        # Plot training loss
        plt.plot(epochs, metrics['train_loss'], label='Train Loss', color='blue')
        
        # Plot validation loss if available
        if 'val_loss' in metrics and any(v is not None for v in metrics['val_loss']):
            plt.plot(epochs, metrics['val_loss'], label='Validation Loss', color='green')
        
        # Plot test loss if available
        if 'test_loss' in metrics and any(v is not None for v in metrics['test_loss']):
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
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(lr_history) + 1), lr_history)  # Start from epoch 1
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, 'learning_rate.png'))
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
        
    def plot_all_stocks_predictions(self, true_values, predictions, dataset, feature_idx=1, n_samples=100):
        """
        Plot predictions for all stocks for a specific feature
        
        Args:
            true_values: shape [batch, stocks, pred_len, features]
            predictions: shape [batch, stocks, pred_len, features]
            dataset: StockDataset instance for denormalization
            feature_idx: which feature to plot (0: volume, 1: close, 2: transactions)
            n_samples: number of points to plot
        """
        stocks = ['AAPL', 'MSFT', 'JPM', 'JNJ', 'AXP']
        num_stocks = len(stocks)
        
        # Create subplot grid
        fig, axes = plt.subplots(num_stocks, 1, figsize=(15, 5*num_stocks))
        
        # Calculate indices for the last sequence
        total_len = dataset.seq_len + dataset.pred_len
        start_idx = max(0, len(dataset.timestamps) - total_len)  # Ensure we don't go negative
        
        # Get timestamps for the entire sequence
        timestamps = dataset.timestamps[start_idx:start_idx + total_len]
        
        # Add title with time range
        start_time = timestamps[0]
        end_time = timestamps[-1]
        fig.suptitle(f'Predictions for {self.feature_names[feature_idx]}\nTime Range: {start_time} to {end_time}', fontsize=16)
        
        # For each stock
        for idx, (stock, ax) in enumerate(zip(stocks, axes)):
            # Get input sequence and true future values
            input_seq = dataset.data[idx, start_idx:start_idx + dataset.seq_len, feature_idx]
            true_future = dataset.data[idx, (start_idx + dataset.seq_len):(start_idx + total_len), feature_idx]
            
            # Get the predictions for future values
            pred_future = predictions[0, idx, :, feature_idx]  # Take first batch for visualization
            
            # Combine sequences
            true_seq = np.concatenate([input_seq, true_future])
            pred_seq = np.concatenate([input_seq, pred_future])
            
            # Denormalize if needed
            true_seq = dataset.denormalize(true_seq, idx)
            pred_seq = dataset.denormalize(pred_seq, idx)
            
            # Plot with timestamps
            ax.plot(timestamps, true_seq, 
                    label='True', marker='o', markersize=2)
            ax.plot(timestamps, pred_seq, 
                    label='Predicted', marker='o', markersize=2)
            
            # Add vertical line to separate input from prediction
            split_time = timestamps[dataset.seq_len - 1]
            ax.axvline(x=split_time, color='r', linestyle='--', alpha=0.5)
            ax.text(split_time, ax.get_ylim()[0], 'Prediction Start', 
                   rotation=90, verticalalignment='bottom')
            
            ax.set_title(f'{stock}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'all_stocks_predictions_{self.feature_names[feature_idx]}.png'))
        plt.close() 