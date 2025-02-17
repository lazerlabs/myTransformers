from stock_dataset import StockDataset, create_dataloader

def create_data_loaders(config):
    """Create train/val/test data loaders"""
    train_loader = create_dataloader(
        file_path=config.train_files,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        scale=True,
        tickers=None,
        features=config.features,
        config=config
    )
    
    test_loader = create_dataloader(
        file_path=config.test_files,
        batch_size=config.batch_size,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        scale=True,
        tickers=None,
        features=config.features,
        config=config
    )
    
    return train_loader[1], test_loader[1] # Return just the dataloaders 