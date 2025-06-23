
This directory contains tools for running predictions using trained model checkpoints **without interfering with ongoing training sessions**. Perfect for monitoring training progress and evaluating model performance while training is still running.

## 🚀 Quick Start

### Option 1: Command Line Script

```bash
# Auto-find and use the latest checkpoint
python predict.py --auto-find-checkpoint

# Use a specific checkpoint
python predict.py --checkpoint-path "./checkpoints/20241210_143022_iTransformer_sl240_pl60/iTransformer_data100_sliding_window_volume_close_transactions_ft3_sl240_pl60_dm512_nh8_el4_df2048_ebfixed_gelu_True_directional/checkpoint_iter_500.pth"

# Run test dataset instead of validation
python predict.py --auto-find-checkpoint --run-test
```

### Option 2: Jupyter Notebook

1. **Convert the Python file to notebook:**
   ```bash
   # If you have jupytext installed
   jupytext --to notebook predict_notebook.py
   
   # Or manually import predict_notebook.py into Jupyter
   ```

2. **Or use the Python file directly:**
   ```bash
   # Run as a Python script with cell-like execution
   python predict_notebook.py
   ```

3. **Configure and run:**
   - Edit the configuration section in the notebook
   - Run all cells to get predictions and visualizations

## 🛠️ Features

### Both Tools Support:
- ✅ **Auto-discovery of latest checkpoints** - no need to specify paths
- ✅ **Manual checkpoint specification** - for specific model versions
- ✅ **Validation and test dataset evaluation** - choose your evaluation mode
- ✅ **All model types** - iTransformer, iInformer, iReformer, etc.
- ✅ **Automatic visualization** - plots saved to figures directory
- ✅ **No training interference** - safe to run while training is ongoing
- ✅ **Flexible configuration** - match any training setup

### Script-Specific Features:
- Command-line interface with all training parameters
- Quick one-line execution
- Ideal for automated evaluation workflows

### Notebook-Specific Features:
- Interactive configuration
- Step-by-step execution
- Rich visualizations and analysis
- Parameter exploration
- Detailed model information display

## 📋 Command Line Options

### Essential Options
```bash
--checkpoint-path PATH          # Explicit checkpoint file path
--auto-find-checkpoint         # Auto-discover latest checkpoint
--run-validation/--run-test    # Choose evaluation dataset (default: validation)
```

### Model Configuration (should match training)
```bash
--model MODEL                  # Model type (iTransformer, iInformer, etc.)
--seq-len INT                  # Input sequence length
--pred-len INT                 # Prediction sequence length
--d-model INT                  # Model dimension
--n-heads INT                  # Number of attention heads
--e-layers INT                 # Number of encoder layers
```

### Data Configuration
```bash
--data-dir PATH                # Data directory (can be used multiple times)
--stocks LIST                  # Comma-separated stock tickers
--features LIST                # Comma-separated features
--val-stocks LIST              # Validation stock tickers
--mode MODE                    # Dataset mode (sliding_window/full_day)
--scale/--no-scale             # Use data scaling (MUST match training)
--use-returns/--no-use-returns # Use returns preprocessing (MUST match training)
```

### System Configuration
```bash
--batch-size INT               # Batch size for inference
--use-gpu/--no-use-gpu        # GPU usage
--checkpoints-dir PATH         # Checkpoints directory
--seed INT                     # Random seed
```

## 📊 Usage Examples

### 1. Monitor Training Progress
```bash
# Check latest validation performance every hour
python predict.py --auto-find-checkpoint --run-validation

# Compare with test performance
python predict.py --auto-find-checkpoint --run-test
```

### 2. Evaluate Specific Checkpoints
```bash
# Test a specific iteration checkpoint
python predict.py --checkpoint-path "./checkpoints/run_20241210/setting/checkpoint_iter_1000.pth"

# Test the final model
python predict.py --checkpoint-path "./checkpoints/run_20241210/setting/checkpoint.pth"
```

### 3. Custom Configuration
```bash
# Use different validation stocks
python predict.py --auto-find-checkpoint --val-stocks "AAPL,MSFT,GOOGL"

# Use different features
python predict.py --auto-find-checkpoint --features "volume,close,high,low"

# Use different model parameters (must match training)
python predict.py --auto-find-checkpoint --seq-len 480 --pred-len 120 --d-model 1024

# IMPORTANT: Match data preprocessing used during training
python predict.py --auto-find-checkpoint --use-returns --no-scale  # For models trained with returns
python predict.py --auto-find-checkpoint --no-use-returns --scale  # For models trained with scaling
```

## 🔍 Checkpoint Discovery

The auto-discovery feature searches for checkpoints in this order:

1. **Main checkpoint** (`checkpoint.pth`) - Final model after training completion
2. **Latest iteration checkpoint** (`checkpoint_iter_*.pth`) - Most recent training iteration
3. **Run directory matching** - Finds the most recent run with compatible settings

### Directory Structure Expected:
```
checkpoints/
├── 20241210_143022_iTransformer_sl240_pl60/          # Run directory
│   └── iTransformer_data100_sliding_window_.../       # Setting directory
│       ├── checkpoint.pth                            # Final checkpoint
│       ├── checkpoint_iter_100.pth                   # Iteration checkpoints
│       ├── checkpoint_iter_200.pth
│       └── ...
└── 20241211_090000_iTransformer_sl240_pl60/          # Another run
    └── ...
```

## 🎯 Typical Workflows

### During Training
1. **Start training** with `train.py`
2. **Monitor progress** by running prediction script every few hours
3. **Check for overfitting** by comparing validation vs training metrics
4. **Make early stopping decisions** based on validation loss trends

### After Training
1. **Final evaluation** on test dataset
2. **Compare different checkpoints** to find the best model
3. **Analyze model performance** across different stocks/features
4. **Generate publication-ready figures** from the notebook

### Model Comparison
1. **Compare different architectures** by running predictions on each
2. **Ablation studies** by testing different feature combinations
3. **Hyperparameter analysis** by evaluating different model configurations

## 🚨 Troubleshooting

### Common Issues

**"No checkpoint found"**
- Check that `--checkpoints-dir` points to the correct directory
- Verify that the model configuration matches your training setup
- Ensure the training has saved at least one checkpoint

**"Configuration mismatch"**
- Make sure model parameters (`--seq-len`, `--pred-len`, etc.) match training
- Check that `--features` match what was used during training
- Verify `--mode` (sliding_window/full_day) matches training configuration
- **CRITICAL**: Ensure `--use-returns` and `--scale` match training preprocessing

**"GPU memory error"**
- Reduce `--batch-size` for inference
- Use `--no-use-gpu` to run on CPU
- Close other GPU-intensive processes

**"Data files not found"**
- Check `--data-dir` path is correct
- Verify that validation/test files exist in the data directory
- Ensure `--val-stocks` refers to stocks that have data files

### Debug Tips

1. **Check the experiment setting string** - it should match between training and prediction
2. **Verify file paths** - use absolute paths if relative paths cause issues
3. **Test with a simple configuration** - start with default parameters
4. **Check logs** - training logs contain the exact configuration used

## 📈 Output

### Script Output
- **Metrics**: Validation loss or test metrics (MAE, MSE, RMSE, MAPE, MSPE)
- **Model info**: Architecture details and parameter counts
- **Checkpoint info**: Which checkpoint was loaded and its training progress
- **File locations**: Where figures and results are saved

### Generated Files
- **Figures**: Prediction plots, attention maps, and performance charts
- **Logs**: Evaluation logs with detailed metrics
- **Visualizations**: Automatically generated charts in the figures directory

## 🔗 Integration

### With Training Scripts
- **Safe concurrent execution** - no file conflicts with ongoing training
- **Shared configuration** - uses the same config system as `train.py`
- **Compatible outputs** - figures and logs use the same format

### With Analysis Workflows
- **Scriptable** - easy to integrate into automated evaluation pipelines
- **Jupyter-compatible** - notebook version for interactive analysis
- **Export-friendly** - results can be easily exported for reports

## 💡 Pro Tips

1. **Set up aliases** for common prediction commands
2. **Use the notebook** for exploratory analysis and the script for automation
3. **Monitor multiple runs** by running prediction on different checkpoint directories
4. **Create comparison scripts** that run predictions on multiple models
5. **Use validation predictions** during training and test predictions for final evaluation
6. **Save important checkpoints** manually to prevent overwriting during long training runs

---

🎯 **Remember**: These tools are designed to work alongside your training process, giving you insights into model performance without interrupting the training workflow! 
# Stock Forecasting Model Prediction Tools
 