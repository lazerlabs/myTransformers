from dataclasses import dataclass, field
import torch
import os
import glob
import random
from typing import List, Optional

@dataclass
class StockPredictionConfig:
    # Data Parameters
    data_dir: str = "/Users/spider/dev/mscAI/CS5004/dataset"
    stocks: Optional[List[str]] = None  # If None, will use all stocks in CSV
    default_stocks: List[str] = field(
        default_factory=lambda: ['AAPL', 'MSFT', 'JPM', 'JNJ', 'AXP']
    )
    features: List[str] = field(
        default_factory=lambda: ['volume', 'close', 'transactions']
    )
    train_size: Optional[int] = None  # Number of files to use for training (None means use all remaining files)
    test_size: int = 1   # Number of files to use for testing
    val_size: int = 1    # Number of files to use for validation
    
    # Sequence Parameters
    seq_len: int = 60  # input sequence length (1 hour of minute data)
    pred_len: int = 30  # prediction sequence length (30 minutes ahead)
    label_len: int = 30  # length of labels for teacher forcing
    scale: bool = True  # whether to scale data
    
    # Model Parameters
    model: str = 'iTransformer'  # model name
    d_model: int = 512  # dimension of model
    n_heads: int = 8  # number of heads in multi-head attention
    e_layers: int = 2  # number of encoder layers
    d_ff: int = 2048  # dimension of fcn in transformer
    dropout: float = 0.1  # dropout rate
    embed: str = 'fixed'  # embedding type
    activation: str = 'gelu'  # activation function
    output_attention: bool = False  # whether to output attention weights
    use_norm: bool = True  # whether to use data normalization
    
    # Training Parameters
    batch_size: int = 32
    learning_rate: float = 1e-3  # Increased initial learning rate
    train_epochs: int = 10
    patience: int = 3  # early stopping patience
    
    # Learning Rate Scheduler Parameters
    lr_scheduler: str = 'cosine'  # Type of scheduler: 'cosine' or 'reduce_on_plateau'
    lr_decay_factor: float = 0.1  # Factor to reduce learning rate by
    lr_patience: int = 2  # Epochs to wait before reducing LR
    min_lr: float = 1e-5  # Minimum learning rate
    warmup_epochs: int = 1  # Number of epochs for warmup
    
    # Device Parameters
    use_gpu: bool = True  # This will now include both CUDA and MPS
    use_multi_gpu: bool = False
    gpu: int = 0
    device_ids: Optional[List[int]] = field(default=None)
    
    # Data Paths
    checkpoints: str = "./checkpoints/"
    
    # Model Specific
    factor: int = 5  # probsparse attn factor
    enc_in: int = 3  # number of input features (Volume, Close, Transactions)
    freq: str = 'min'  # time feature encoding frequency 
    
    # Runtime storage for file paths
    available_files: List[str] = field(default_factory=list)
    train_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Get all available CSV files
        csv_files = sorted(glob.glob(os.path.join(self.data_dir, "*.csv")))
        random.shuffle(csv_files)
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")
        
        # Split into train, validation, and test
        total_files = len(csv_files)
        if total_files < (self.test_size + self.val_size + 1):
            raise ValueError(f"Not enough data files for split. Need at least {self.test_size + self.val_size + 1} files, but only found {total_files}")
        
        # Calculate available files for training
        available_train_files = total_files - (self.test_size + self.val_size)
        
        # If train_size is specified, use minimum of available and requested
        if self.train_size is not None:
            if self.train_size <= 0:
                raise ValueError(f"train_size must be positive, got {self.train_size}")
            train_size = min(self.train_size, available_train_files)
            print(f"Using {train_size} files for training (limited by train_size={self.train_size})")
        else:
            train_size = available_train_files
            print(f"Using all {train_size} available files for training")
        
        # Split the files
        self.test_files = csv_files[-self.test_size:]  # Last N files for testing
        self.val_files = csv_files[-(self.test_size + self.val_size):-self.test_size]  # Files before test set for validation
        self.train_files = csv_files[-(self.test_size + self.val_size + train_size):-(self.test_size + self.val_size)]  # Limited training files
        
        print(f"Data split - Train: {len(self.train_files)} files, Validation: {len(self.val_files)} files, Test: {len(self.test_files)} files")
        
        # Use default stocks if none specified
        if self.stocks is None:
            self.stocks = self.default_stocks.copy()  # Make a copy to be safe
            
        # Auto-detect best available device
        if self.use_gpu:
            if torch.cuda.is_available():
                print("CUDA GPU available")
            elif torch.backends.mps.is_available():
                print("Apple Silicon MPS available")
            else:
                print("No GPU available, will use CPU")
                self.use_gpu = False
    
    @property
    def train_data_path(self) -> str:
        return self.train_files[0]  # Return first training file for now
    
    @property
    def val_data_path(self) -> str:
        return self.val_files[0]  # Return first validation file
    
    @property
    def test_data_path(self) -> str:
        return self.test_files[0]  # Return first test file 