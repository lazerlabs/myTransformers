#!/bin/bash

# Classic Transformer Single Feature Experiment (Close Price Only)
# Baseline comparison with traditional transformer architecture

echo "🚀 Starting Classic Transformer Single Feature Experiment"
echo "📊 Feature: Close price only"
echo "🎯 Model: Transformer (Classic - Baseline)"

export CUDA_VISIBLE_DEVICES=0

# Standard configuration
MODEL="Transformer"
FEATURES="close"
DATA_DIR="./dataset/"

echo "⚙️  Configuration:"
echo "   Model: $MODEL (Classic for comparison)"
echo "   Features: $FEATURES"
echo "   Data: $DATA_DIR"
echo ""

# Experiment: 60 minutes -> 15 minutes prediction
echo "🧪 Running 60min → 15min prediction..."
python run.py \
  --is_training 1 \
  --model_id "SINGLE_CLOSE_60_15" \
  --model $MODEL \
  --data stock \
  --data_dir $DATA_DIR \
  --features $FEATURES \
  --features_mode "S" \
  --target close \
  --seq_len 60 \
  --pred_len 15 \
  --label_len 30 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 512 \
  --n_heads 8 \
  --e_layers 4 \
  --d_ff 2048 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --train_epochs 20 \
  --patience 5 \
  --des "Single_Feature_Close" \
  --itr 1

echo "✅ Classic Transformer Single Feature Experiment Complete!"
echo "📁 Results saved in: checkpoints/, logs/, figures/"
echo "📊 Compare with iTransformer results to see inverted architecture benefits"
echo "" 
