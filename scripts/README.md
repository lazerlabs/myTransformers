# Experiment Scripts

Clean, organized scripts for running systematic transformer experiments on stock market data.

## 📁 Directory Structure

```
scripts/
├── single_feature/          # Close price only experiments
│   ├── iTransformer.sh      # Inverted transformer
│   ├── iInformer.sh         # Inverted informer (efficient)
│   └── Transformer.sh       # Classic transformer (baseline)
├── multi_feature/           # Multi-feature experiments
│   ├── iTransformer.sh      # Inverted transformer
│   ├── iInformer.sh         # Inverted informer
│   └── Transformer.sh       # Classic transformer
├── comparative/
│   └── full_comparison.sh   # Comprehensive comparison study
└── README.md               # This file
```

## 🚀 Quick Start

### Individual Experiments

```bash
# Single feature experiments (close price only)
bash scripts/single_feature/iTransformer.sh
bash scripts/single_feature/iInformer.sh
bash scripts/single_feature/Transformer.sh

# Multi-feature experiments (close + volume + transactions)
bash scripts/multi_feature/iTransformer.sh
bash scripts/multi_feature/iInformer.sh
bash scripts/multi_feature/Transformer.sh
```

### Comprehensive Comparison

```bash
# Run all experiments systematically
bash scripts/comparative/full_comparison.sh
```

## 🎯 Experiment Design

### Single Feature Experiments
- **Purpose**: Test core model capabilities on univariate prediction
- **Features**: Close price only
- **Expected**: Baseline performance comparison

### Multi-Feature Experiments  
- **Purpose**: Test multivariate correlation modeling
- **Features**: Close price + Volume + Transactions
- **Expected**: iTransformer advantage due to feature-wise attention

### Comprehensive Comparison
- **Purpose**: Systematic evaluation across all models and feature sets
- **Runtime**: ~4-6 hours
- **Output**: Complete performance matrix and analysis

## 📊 Expected Results

Based on the iTransformer paper:

| Experiment | Expected Winner | Reason |
|------------|----------------|--------|
| **Single Feature** | iTransformer ≈ Transformer | Limited multivariate advantage |
| **Multi-Feature** | **iTransformer** | Superior feature correlation modeling |
| **Efficiency** | **iInformer** | ProbSparse attention optimization |

## 📈 Results Analysis

After running experiments, check:

1. **TensorBoard**: `tensorboard --logdir logs/`
2. **Figures**: `figures/` directory for prediction plots
3. **Checkpoints**: `checkpoints/` for model weights
4. **Logs**: `logs/` for detailed training metrics

## ⚙️ Configuration

All scripts use standard configuration:
- **Sequence**: 60 minutes → 15 minutes prediction
- **Model Size**: 512 dimensions, 8 heads, 4 layers
- **Training**: 20 epochs, batch size 32
- **Learning Rate**: 0.0005

## 🔧 Customization

To modify experiments, edit the script parameters:

```bash
# Example: Longer sequences
--seq_len 120 \
--pred_len 30 \
--label_len 60

# Example: Larger model
--d_model 1024 \
--n_heads 16 \
--d_ff 4096
```

## 📝 Notes

- All scripts use `run.py` for compatibility with original iTransformer interface
- Results are automatically saved with timestamps
- GPU usage is set to device 0 (modify `CUDA_VISIBLE_DEVICES` if needed)
- Scripts include progress indicators and completion summaries 
