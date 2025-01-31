import logging
import os
import time
from datetime import datetime

class Logger:
    def __init__(self, name):
        self.name = name
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'learning_rate': [],
            'mae': None,
            'mse': None,
            'rmse': None
        }
        
        # Create figures directory if it doesn't exist
        if not os.path.exists('figures'):
            os.makedirs('figures')
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{name}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(name)
    
    def log_training(self, epoch, train_loss, val_loss, test_loss, lr):
        """Log training metrics for one epoch"""
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_loss'].append(val_loss)
        self.metrics['test_loss'].append(test_loss)
        self.metrics['learning_rate'].append(lr)
        
        self.logger.info(
            f"Epoch: {epoch}, Train Loss: {train_loss:.6f}, "
            f"Val Loss: {val_loss:.6f}, Test Loss: {test_loss:.6f}, "
            f"Learning Rate: {lr:.6f}"
        )
    
    def log_prediction(self, mae, mse, rmse):
        """Log final prediction metrics"""
        self.metrics['mae'] = mae
        self.metrics['mse'] = mse
        self.metrics['rmse'] = rmse
        
        self.logger.info(
            f"Final Metrics - MAE: {mae:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}"
        ) 