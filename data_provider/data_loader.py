from stock_dataset import StockDataset, create_dataloader, calculate_global_stats # Added calculate_global_stats import
import numpy as np # Import numpy for stats handling

def create_data_loaders(config):
    """
    Create train and test data loaders with global normalization based on the training set.
    """
    # 1. Calculate global statistics from the training set
    # Use all tickers from training files for stats calculation unless config specifies otherwise
    train_tickers = config.stocks # Use configured stocks if provided, else None (all)
    global_mean, global_std = calculate_global_stats(
        file_paths=config.train_files,
        features=config.features,
        tickers=train_tickers
    )

    # Handle case where stats could not be calculated
    if global_mean is None or global_std is None:
        print("Warning: Global stats calculation failed. Disabling scaling.")
        scale_data = False
    else:
        scale_data = config.scale # Use scaling setting from config if stats are available

    # 2. Create Training DataLoader
    # Use all tickers for training unless specified in config
    train_dataset, train_dataloader = create_dataloader(
        file_paths=config.train_files,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        scale=scale_data,
        tickers=train_tickers, # Use same tickers as stats calculation
        features=config.features,
        global_mean=global_mean,
        global_std=global_std,
        shuffle=True # Shuffle training data
    )

    # 3. Create Test DataLoader
    # Typically evaluate on all tickers present in test files, unless config specifies otherwise
    test_tickers = config.stocks # Or potentially None if you want all tickers in test files
    test_dataset, test_dataloader = create_dataloader(
        file_paths=config.test_files,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        scale=scale_data,
        tickers=test_tickers,
        features=config.features,
        global_mean=global_mean,
        global_std=global_std,
        shuffle=False # Do not shuffle test data
    )

    # TODO: Consider adding a validation dataloader here using config.val_files

    # Return datasets, dataloaders, and the calculated global stats
    return train_dataset, train_dataloader, test_dataset, test_dataloader, global_mean, global_std