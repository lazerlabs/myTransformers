#!/bin/bash

# Classic Transformer Multi-Feature Experiment
# Baseline comparison with traditional transformer on multiple features

echo "🚀 Starting Classic Transformer Multi-Feature Experiment"
echo "📊 Features: Close price + Volume + Transactions"
echo "🎯 Model: Transformer (Classic - Multivariate Baseline)"

export CUDA_VISIBLE_DEVICES=0

# Standard configuration
MODEL="Transformer"
FEATURES="close,volume,transactions"
DATA_DIR="./dataset/"

echo "⚙️  Configuration:"
echo "   Model: $MODEL (Classic for comparison)"
echo "   Features: $FEATURES"
echo "   Data: $DATA_DIR"
echo ""

# Experiment: 60 minutes -> 15 minutes prediction
echo "🧪 Running 60min → 15min prediction with multivariate features..."
python run.py \
  --is_training 1 \
  --model_id "MULTI_FEATURES_60_15" \
  --model $MODEL \
  --data stock \
  --data_dir $DATA_DIR \
  --features $FEATURES \
  --features_mode "M" \
  --target close \
  --seq_len 60 \
  --pred_len 15 \
  --label_len 30 \
  --enc_in 3 \
  --dec_in 3 \
  --c_out 3 \
  --d_model 512 \
  --n_heads 8 \
  --e_layers 4 \
  --d_ff 2048 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --train_epochs 20 \
  --patience 5 \
  --des "Multi_Feature_CVT" \
  --itr 1

echo "✅ Classic Transformer Multi-Feature Experiment Complete!"
echo "📁 Results saved in: checkpoints/, logs/, figures/"
echo "📊 Compare with iTransformer to see multivariate correlation modeling benefits"
echo "" 
