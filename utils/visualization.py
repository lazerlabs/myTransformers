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

    def plot_all_stocks_predictions(self, true_values, predictions, dataset, feature_idx=1, n_samples_to_plot=1):
        """
        Plot predictions vs true values for a specific feature across multiple stocks
        from a few sample sequences.

        Args:
            true_values (np.ndarray): Ground truth values. Shape: [total_sequences, pred_len, features]
            predictions (np.ndarray): Predicted values. Shape: [total_sequences, pred_len, features]
            dataset (StockDataset): The dataset instance (used for denormalization).
            feature_idx (int): Index of the feature to plot (e.g., 1 for 'close').
            n_samples_to_plot (int): Number of sample sequences to plot from the results.
        """
        print("\n--- Starting Visualization ---")
        if true_values.shape[0] == 0 or predictions.shape[0] == 0:
             print("Warning: Cannot plot predictions, true_values or predictions array is empty.")
             return
        if not hasattr(dataset, 'denormalize'):
             print("Warning: Dataset object does not have a 'denormalize' method. Cannot plot.")
             return
        if feature_idx >= len(self.feature_names):
             print(f"Warning: feature_idx {feature_idx} is out of bounds for features {self.feature_names}. Skipping plot.")
             return

        # Determine number of stocks based on how data was likely structured before concatenation
        # This is an assumption - might need adjustment if batch/stock structure changes
        # A better approach would be to pass stock identifiers alongside predictions/trues
        num_total_sequences = true_values.shape[0]
        # Heuristic: Assume data was [batch, stocks, pred, feats] -> [batch*stocks, pred, feats]
        # Let's just plot N samples without assuming stock structure for now.
        num_plots = min(n_samples_to_plot, num_total_sequences)
        if num_plots == 0:
             print("No samples to plot.")
             return

        print(f"Plotting {num_plots} sample sequences for feature '{self.feature_names[feature_idx]}'")

        # Create subplot grid
        # Adjust layout dynamically if needed, for now just plot vertically
        fig, axes = plt.subplots(num_plots, 1, figsize=(15, 5 * num_plots), squeeze=False) # Ensure axes is always 2D

        # Select random samples to plot
        sample_indices = np.random.choice(num_total_sequences, num_plots, replace=False)

        for plot_idx, seq_idx in enumerate(sample_indices):
            ax = axes[plot_idx, 0] # Access subplot correctly

            # Get true and predicted sequences for the chosen sample and feature
            true_seq = true_values[seq_idx, :, feature_idx]
            pred_seq = predictions[seq_idx, :, feature_idx]

            # Denormalize using the dataset's global stats
            # Note: We only denormalize the specific feature being plotted
            # Create dummy arrays with correct feature dimension for denormalize method
            num_features = len(self.feature_names)
            pred_len = true_seq.shape[0]

            true_full_features = np.zeros((pred_len, num_features))
            pred_full_features = np.zeros((pred_len, num_features))
            true_full_features[:, feature_idx] = true_seq
            pred_full_features[:, feature_idx] = pred_seq

            try:
                true_denorm_full = dataset.denormalize(true_full_features)
                pred_denorm_full = dataset.denormalize(pred_full_features)
                # Extract the denormalized feature we care about
                true_denorm = true_denorm_full[:, feature_idx]
                pred_denorm = pred_denorm_full[:, feature_idx]
            except Exception as e:
                 print(f"Warning: Failed to denormalize data for plot {plot_idx} (seq_idx {seq_idx}): {e}. Plotting normalized data.")
                 true_denorm = true_seq
                 pred_denorm = pred_seq


            # --- Plotting ---
            # Use a simple numerical index for x-axis as timestamps are not stored per sequence
            time_steps = np.arange(len(true_denorm))
            x_label = "Time Step in Prediction Window"

            ax.plot(time_steps, true_denorm, label='True', marker='o', markersize=3, linestyle='-')
            ax.plot(time_steps, pred_denorm, label='Predicted', marker='x', markersize=4, linestyle='--')

            ax.set_title(f'Sample Sequence {seq_idx} - Feature: {self.feature_names[feature_idx]}')
            ax.set_xlabel(x_label)
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)

            # Optional: Rotate x-axis labels if using timestamps later
            # plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        fig.suptitle('Sample Predictions vs True Values', fontsize=16)
        try:
            save_path = os.path.join(self.save_dir, f'sample_predictions_{self.feature_names[feature_idx]}.png')
            plt.savefig(save_path)
            print(f"Saved sample prediction plot to {save_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        plt.close(fig)