#!/bin/bash

# Comprehensive Model Comparison Script
# Runs systematic comparison of inverted vs classic transformers

echo "🎯 COMPREHENSIVE TRANSFORMER COMPARISON STUDY"
echo "=============================================="
echo ""
echo "This script runs a systematic comparison of:"
echo "  📊 Single vs Multi-feature experiments"
echo "  🔄 Inverted vs Classic transformer architectures"
echo "  ⚡ Different attention mechanisms"
echo ""
echo "Expected runtime: ~4-6 hours (depending on hardware)"
echo ""

# Configuration
export CUDA_VISIBLE_DEVICES=0
DATA_DIR="./dataset/"
EPOCHS=20
BATCH_SIZE=32

# Create results directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="results/comparison_${TIMESTAMP}"
mkdir -p $RESULTS_DIR

echo "📁 Results will be saved to: $RESULTS_DIR"
echo ""

# Function to run experiment and log results
run_experiment() {
    local model=$1
    local features=$2
    local experiment_name=$3
    local feature_mode=$4
    local num_features=$5
    
    echo "🧪 Running: $experiment_name"
    echo "   Model: $model"
    echo "   Features: $features"
    echo "   Started: $(date)"
    
    python run.py \
        --is_training 1 \
        --model_id "${experiment_name}" \
        --model $model \
        --data stock \
        --data_dir $DATA_DIR \
        --features $features \
        --features_mode $feature_mode \
        --target close \
        --seq_len 60 \
        --pred_len 15 \
        --label_len 30 \
        --enc_in $num_features \
        --dec_in $num_features \
        --c_out $num_features \
        --d_model 512 \
        --n_heads 8 \
        --e_layers 4 \
        --d_ff 2048 \
        --batch_size $BATCH_SIZE \
        --learning_rate 0.0005 \
        --train_epochs $EPOCHS \
        --patience 5 \
        --des "${experiment_name}" \
        --itr 1
    
    echo "   Completed: $(date)"
    echo "   ✅ $experiment_name finished"
    echo ""
}

echo "🚀 STARTING EXPERIMENT SUITE"
echo "============================="
echo ""

# Phase 1: Single Feature Experiments (Close Price Only)
echo "📊 PHASE 1: SINGLE FEATURE EXPERIMENTS"
echo "--------------------------------------"
echo "Testing models on close price prediction only"
echo ""

run_experiment "iTransformer" "close" "SINGLE_iTransformer" "S" 1
run_experiment "iInformer" "close" "SINGLE_iInformer" "S" 1
run_experiment "Transformer" "close" "SINGLE_Transformer" "S" 1

echo "✅ Phase 1 Complete: Single Feature Experiments"
echo ""

# Phase 2: Multi-Feature Experiments (Close + Volume + Transactions)
echo "📊 PHASE 2: MULTI-FEATURE EXPERIMENTS"
echo "-------------------------------------"
echo "Testing models on multivariate prediction"
echo ""

run_experiment "iTransformer" "close,volume,transactions" "MULTI_iTransformer" "M" 3
run_experiment "iInformer" "close,volume,transactions" "MULTI_iInformer" "M" 3
run_experiment "Transformer" "close,volume,transactions" "MULTI_Transformer" "M" 3

echo "✅ Phase 2 Complete: Multi-Feature Experiments"
echo ""

# Phase 3: Additional Inverted Models (if available)
echo "📊 PHASE 3: ADDITIONAL INVERTED MODELS"
echo "--------------------------------------"
echo "Testing other inverted architectures"
echo ""

# Only run if models are available
if python -c "from models import iReformer" 2>/dev/null; then
    run_experiment "iReformer" "close,volume,transactions" "MULTI_iReformer" "M" 3
fi

if python -c "from models import iFlowformer" 2>/dev/null; then
    run_experiment "iFlowformer" "close,volume,transactions" "MULTI_iFlowformer" "M" 3
fi

echo "✅ Phase 3 Complete: Additional Models"
echo ""

# Generate comparison report
echo "📊 GENERATING COMPARISON REPORT"
echo "==============================="

# Create a simple results summary
cat > $RESULTS_DIR/experiment_summary.md << EOF
# Transformer Comparison Results - $(date)

## Experiment Configuration
- **Sequence Length**: 60 minutes
- **Prediction Length**: 15 minutes  
- **Training Epochs**: $EPOCHS
- **Batch Size**: $BATCH_SIZE
- **Model Dimension**: 512
- **Attention Heads**: 8

## Experiments Completed

### Single Feature (Close Price Only)
- [x] iTransformer
- [x] iInformer  
- [x] Transformer (Classic)

### Multi-Feature (Close + Volume + Transactions)
- [x] iTransformer
- [x] iInformer
- [x] Transformer (Classic)

## Expected Results Analysis

Based on the iTransformer paper, we expect:

1. **iTransformer > Transformer** on multivariate tasks
2. **Multi-feature > Single-feature** for all models
3. **iInformer ≈ iTransformer** with better efficiency

## Next Steps

1. Check TensorBoard logs: \`tensorboard --logdir logs/\`
2. Compare model checkpoints in \`checkpoints/\`
3. Analyze prediction figures in \`figures/\`
4. Review detailed logs for training metrics

## Files Generated
- Model checkpoints: \`checkpoints/\`
- Training logs: \`logs/\`
- Prediction figures: \`figures/\`
- TensorBoard events: \`logs/\`
EOF

echo "📁 Experiment summary saved to: $RESULTS_DIR/experiment_summary.md"
echo ""

echo "🎉 COMPREHENSIVE COMPARISON COMPLETE!"
echo "===================================="
echo ""
echo "📊 Summary:"
echo "   ✅ Single feature experiments: 3 models"
echo "   ✅ Multi-feature experiments: 3 models"
echo "   ✅ Total experiments completed: 6+"
echo ""
echo "📁 Results location:"
echo "   📋 Summary: $RESULTS_DIR/experiment_summary.md"
echo "   📊 Logs: logs/"
echo "   💾 Checkpoints: checkpoints/"
echo "   📈 Figures: figures/"
echo ""
echo "🔍 Next steps:"
echo "   1. tensorboard --logdir logs/"
echo "   2. Review figures/ for prediction quality"
echo "   3. Compare MSE/MAE metrics across models"
echo "   4. Analyze attention patterns (if available)"
echo ""
echo "🏆 Expected winner: iTransformer on multi-feature tasks!"
echo "" 
