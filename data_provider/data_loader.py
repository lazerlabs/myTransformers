# Remove old StockDataset import since we deleted that file
# from stock_dataset import StockDataset, create_dataloader, calculate_global_stats # Added calculate_global_stats import
import numpy as np # Import numpy for stats handling
# Import the new simplified dataset
from simple_stock_dataset import SimpleStockDataset, create_simple_dataloader

def create_data_loaders(config):
    """
    Create train and test data loaders using SimpleStockDataset with returns-based preprocessing.
    No global statistics needed since returns are naturally normalized.
    """
    print("Creating data loaders with returns-based preprocessing...")
    
    # Use configured stocks if provided, else None (all)
    train_tickers = config.stocks
    test_tickers = config.stocks

    # Create Training DataLoader
    train_dataset, train_dataloader = create_simple_dataloader(
        file_paths=config.train_files,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        tickers=train_tickers,
        features=config.features,
        shuffle=True,  # Shuffle training data
        mode=config.mode,
        interpolate_max_missing=config.interpolate_max_missing,
        max_samples=getattr(config, 'max_train_samples', None)
    )

    # Create Test DataLoader
    test_dataset, test_dataloader = create_simple_dataloader(
        file_paths=config.test_files,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        tickers=test_tickers,
        features=config.features,
        shuffle=False,  # Do not shuffle test data
        mode=config.mode,
        interpolate_max_missing=config.interpolate_max_missing,
        max_samples=getattr(config, 'max_test_samples', None)
    )

    # Create Validation DataLoader if validation files are specified
    val_dataset, val_dataloader = None, None
    if hasattr(config, 'val_files') and config.val_files:
        val_tickers = getattr(config, 'val_stocks', config.stocks)
        val_dataset, val_dataloader = create_simple_dataloader(
            file_paths=config.val_files,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            pred_len=config.pred_len,
            tickers=val_tickers,
            features=config.features,
            shuffle=False,  # Do not shuffle validation data
            mode=config.mode,
            interpolate_max_missing=config.interpolate_max_missing,
            max_samples=getattr(config, 'max_val_samples', None)
        )

    # Return datasets and dataloaders (no global stats needed)
    return train_dataset, train_dataloader, test_dataset, test_dataloader, val_dataset, val_dataloader
