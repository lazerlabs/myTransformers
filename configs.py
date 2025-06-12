from dataclasses import dataclass, field
import torch
import os
import glob
import random
import pandas as pd
from typing import List, Optional

# Get workspace root (parent directory of myTransformer)
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASEDIR = os.path.join(WORKSPACE_ROOT, "myTransformers")

def get_config_defaults():
    """Get config defaults without file path initialization for use in CLI decorators."""
    # Create a temporary config instance with file init disabled
    import dataclasses
    
    # Get the default values from the dataclass fields
    defaults = {}
    for field in dataclasses.fields(StockPredictionConfig):
        if field.default != dataclasses.MISSING:
            defaults[field.name] = field.default
        elif field.default_factory != dataclasses.MISSING:
            defaults[field.name] = field.default_factory()
    
    return defaults

@dataclass
class StockPredictionConfig:
    # Data Parameters
    data_dir: str = "dataset"  # Use local dataset directory
    stocks: Optional[List[str]] = None  # If None, will use all stocks in CSV
    features: List[str] = field(
        default_factory=lambda: ['close', 'volume', 'transactions']
    )
    target: str = 'close'  # Primary target feature for prediction and visualization
    train_size: Optional[int] = 1  # Number of files to use for training (None means use all remaining files)
    test_size: int = 2   # Number of files to use for testing
    val_size: int = 2    # Number of files to use for validation
    val_stocks: List[str] = field(
        default_factory=lambda: ['AAPL', 'MSFT', 'JPM', 'JNJ', 'AXP']
    )
    
    # Sequence Parameters
    seq_len: int = 60      # 1 hour lookback
    pred_len: int = 15     # Predict next 15 minutes
    label_len: int = 30    # Label length for teacher forcing
    scale: bool = True
    inverse: bool = True   # Whether to denormalize predictions for metrics and visualization
    
    # Dataset Mode Parameters
    mode: str = 'full_day'  # 'sliding_window' or 'full_day'
    interpolate_max_missing: int = 3
    max_seq_len: int = 2000  # Maximum sequence length for embedding layer (for full_day mode)
    
    # Model Parameters
    model: str = 'iTransformer'
    d_model: int = 512     # Dimension of model
    n_heads: int = 8       # Number of attention heads
    e_layers: int = 4      # Increased number of encoder layers
    d_ff: int = 2048      # Dimension of FCN
    dropout: float = 0.2   # Increased dropout
    embed: str = 'fixed'
    activation: str = 'gelu'
    output_attention: bool = False
    use_norm: bool = True
    allow_classic_models: bool = False  # Allow classic (non-inverted) models for comparison
    
    # Inference Parameters
    temperature: float = 0.0  # Temperature for inference sampling (0.0 = deterministic, >0 = stochastic)
    
    # Training Parameters
    batch_size: int = 64
    learning_rate: float = 5e-4    # Reduced learning rate
    train_epochs: int = 20         # Number of training epochs
    patience: int = 5              # Early stopping patience
    max_train_iterations: Optional[int] = None # Limit iterations per epoch (for testing)

    # Checkpoint Saving Parameters
    save_checkpoint_every_n_iterations: Optional[int] = 1000  # Save checkpoint every N iterations (None to disable)
    save_checkpoint_every_n_epochs: Optional[int] = None      # Save checkpoint every N epochs (None to disable)
    
    # Loss Function Parameters
    loss_type: str = "adaptive"    # Use our new adaptive loss
    loss_kwargs: dict = field(default_factory=lambda: {
        "alpha": 0.3,              # Weight for relative error component
        "beta": 2.0                # Exponential scaling for MSE
    })
    
    # Learning Rate Scheduler Parameters
    lr_scheduler: str = 'cosine'
    lr_decay_factor: float = 0.1
    lr_patience: int = 3           # Adjusted LR patience
    min_lr: float = 1e-5
    warmup_epochs: int = 2         # Increased warmup period
    
    # Device Parameters
    use_gpu: bool = True  # This will now include both CUDA and MPS
    use_multi_gpu: bool = False
    gpu: int = 0
    device_ids: Optional[List[int]] = field(default=None)
    
    # Data Paths
    checkpoints_dir: str = os.path.join(BASEDIR, "checkpoints/")
    logs_dir: str = os.path.join(BASEDIR, "logs/")
    figures_dir: str = os.path.join(BASEDIR, "figures/")
    embeddings_dir: str = os.path.join(BASEDIR, "embeddings/")  # Add embeddings directory
    
    # Model Specific
    factor: int = 5  # probsparse attn factor
    enc_in: int = 3  # number of input features (Volume, Close, Transactions)
    dec_in: int = 3  # decoder input size (same as enc_in for iTransformer)
    c_out: int = 3   # output size (same as enc_in for iTransformer)
    freq: str = 'min'  # time feature encoding frequency
    forecasting_features: str = 'M'  # forecasting task: 'M'=multivariate predict multivariate, 'S'=univariate predict univariate, 'MS'=multivariate predict univariate
    class_strategy: str = 'projection'  # classification strategy (not used in regression, but needed for compatibility) 
    
    # Runtime storage for file paths
    available_files: List[str] = field(default_factory=list)
    train_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    val_files: List[str] = field(default_factory=list)  # Add missing val_files field
    
    # How often to run test predictions during training (0 to disable)
    test_interval: int = 0  # Will test every 1 epochs
    test_iteration_interval: int = 5000  # Will also test every 5000 iterations (0 to disable)
    
    # Logging Parameters
    log_every_n_iterations: int = 100  # Log detailed metrics every N iterations
    save_iteration_metrics: bool = True  # Save iteration-level metrics for visualization
    
    # Debug/Testing Parameters
    max_train_samples: Optional[int] = None  # Limit training samples for testing (None = unlimited)
    max_test_samples: Optional[int] = None   # Limit test samples for testing (None = unlimited)
    
    def __post_init__(self):
        # Update model parameters based on number of features
        self._update_model_dimensions()
        
        # Initialize file paths - this can be called again after CLI overrides
        self._initialize_file_paths()
        
        # Handle stock selection
        if self.stocks is None or len(self.stocks) == 0:  # Handle both None and empty list
            # Don't set self.stocks to the list of all tickers
            # Just leave it as None to indicate we want all stocks
            print(f"\nUsing all available stocks")
        else:
            self.stocks = self.stocks.copy()  # Make a copy to be safe
            
        # Auto-detect best available device
        if self.use_gpu:
            if torch.cuda.is_available():
                print("NVIDIA CUDA GPU available")
            elif torch.backends.mps.is_available():
                print("Apple Silicon MPS available")
            else:
                print("No GPU available, will use CPU")
                self.use_gpu = False
        else:
            print("GPU usage is disabled, will use CPU")
        print(f"Config stocks type: {type(self.stocks)}")
        print(f"Config stocks value: {self.stocks}")
    
    def _update_model_dimensions(self):
        """Update model dimensions based on number of features"""
        num_features = len(self.features)
        self.enc_in = num_features
        self.dec_in = num_features  
        self.c_out = num_features
        print(f"Updated model dimensions for {num_features} features: enc_in={self.enc_in}, dec_in={self.dec_in}, c_out={self.c_out}")
    
    def _initialize_file_paths(self):
        """Initialize file paths based on current data_dir. Can be called after CLI overrides."""
        # Get all available CSV files and ensure they are sorted chronologically
        csv_files = sorted(glob.glob(os.path.join(self.data_dir, "*.csv")))
        # random.shuffle(csv_files) # Removed shuffle to maintain chronological order
        print(f"Found {len(csv_files)} data files, sorted chronologically.")
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
    
    def update_features(self, new_features: List[str]):
        """Update features and corresponding model dimensions. Call this after CLI overrides."""
        self.features = new_features
        self._update_model_dimensions()
    
    @property
    def train_data_path(self) -> str:
        return self.train_files[0]  # Return first training file for now
    
    @property
    def val_data_path(self) -> str:
        return self.val_files[0]  # Return first validation file
    
    @property
    def test_data_path(self) -> str:
        return self.test_files[0]  # Return first test file 
