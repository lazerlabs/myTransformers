#!/bin/bash

# Loss Function Comparison Experiments
# Optimized for RTX A6000 Ada (48GB VRAM) - can run multiple experiments in parallel

echo "🚀 Starting Loss Function Experiments on RTX A6000 Ada"
echo "📊 Running 5 experiments in parallel..."

# Set common parameters
COMMON_ARGS="--batch-size 64 --learning-rate 5e-4 --train-epochs 20 --seq-len 60 --pred-len 15"

# Create results directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="loss_comparison_${TIMESTAMP}"
mkdir -p results/${RESULTS_DIR}

echo "📁 Results will be saved to: results/${RESULTS_DIR}"

# Function to run experiment and save output
run_experiment() {
    local loss_type=$1
    local loss_kwargs=$2
    local run_name=$3
    local log_file="results/${RESULTS_DIR}/${run_name}.log"
    
    echo "🔄 Starting experiment: ${run_name}"
    
    if [ -n "$loss_kwargs" ]; then
        python train.py ${COMMON_ARGS} \
            --loss-type ${loss_type} \
            --loss-kwargs "${loss_kwargs}" \
            --run-name ${run_name} \
            --figures-dir figures/${RESULTS_DIR} \
            --logs-dir logs/${RESULTS_DIR} \
            --checkpoints-dir checkpoints/${RESULTS_DIR} \
            > ${log_file} 2>&1
    else
        python train.py ${COMMON_ARGS} \
            --loss-type ${loss_type} \
            --run-name ${run_name} \
            --figures-dir figures/${RESULTS_DIR} \
            --logs-dir logs/${RESULTS_DIR} \
            --checkpoints-dir checkpoints/${RESULTS_DIR} \
            > ${log_file} 2>&1
    fi
    
    echo "✅ Completed experiment: ${run_name}"
}

# Start experiments in parallel (background processes)
echo "🎯 Launching parallel experiments..."

# Experiment 1: Pure MAE (baseline)
run_experiment "mae" "" "mae_baseline" &
PID1=$!

# Experiment 2: MAE with direction awareness
run_experiment "directional" '{"base_loss": "mae", "direction_weight": 0.2}' "mae_directional" &
PID2=$!

# Experiment 3: Huber loss with small delta (MAE-like for small errors)
run_experiment "huber" '{"delta": 0.01}' "huber_small_delta" &
PID3=$!

# Experiment 4: Log-cosh (smooth MAE alternative)
run_experiment "log_cosh" "" "log_cosh" &
PID4=$!

# Experiment 5: Adaptive (for comparison)
run_experiment "adaptive" '{"alpha": 0.3, "beta": 2.0}' "adaptive_comparison" &
PID5=$!

# Store PIDs for monitoring
echo "📋 Experiment PIDs: $PID1 $PID2 $PID3 $PID4 $PID5"

# Function to check if process is still running
is_running() {
    kill -0 $1 2>/dev/null
}

# Monitor progress
echo "📊 Monitoring experiment progress..."
echo "💡 You can monitor individual experiments with: tail -f results/${RESULTS_DIR}/<experiment_name>.log"

while true; do
    running_count=0
    
    if is_running $PID1; then ((running_count++)); fi
    if is_running $PID2; then ((running_count++)); fi
    if is_running $PID3; then ((running_count++)); fi
    if is_running $PID4; then ((running_count++)); fi
    if is_running $PID5; then ((running_count++)); fi
    
    if [ $running_count -eq 0 ]; then
        echo "🎉 All experiments completed!"
        break
    fi
    
    echo "⏳ $running_count experiments still running..."
    sleep 60  # Check every minute
done

# Wait for all background processes to complete
wait

echo "📈 Generating comparison report..."

# Create a summary report
cat > results/${RESULTS_DIR}/comparison_summary.md << EOF
# Loss Function Comparison Results

**Experiment Date:** $(date)
**Hardware:** RTX A6000 Ada (48GB VRAM)
**Configuration:** batch_size=64, lr=5e-4, epochs=20

## Experiments Run:

1. **MAE Baseline** - Pure Mean Absolute Error
   - Philosophy: Equal treatment of all errors
   - Log: mae_baseline.log

2. **MAE + Direction** - MAE with directional awareness
   - Combines value accuracy with trend prediction
   - Log: mae_directional.log

3. **Huber (Small Delta)** - MAE-like with outlier robustness
   - Linear for small errors, quadratic for large outliers
   - Log: huber_small_delta.log

4. **Log-Cosh** - Smooth MAE alternative
   - Approximately linear for small errors, differentiable
   - Log: log_cosh.log

5. **Adaptive (Comparison)** - Multi-component loss
   - Contains squared terms (may overstate small changes)
   - Log: adaptive_comparison.log

## Analysis:

Check the individual experiment logs and TensorBoard results in:
- Logs: logs/${RESULTS_DIR}/
- Figures: figures/${RESULTS_DIR}/
- Checkpoints: checkpoints/${RESULTS_DIR}/

## Expected Results:

Based on your preference for not overstating small changes:
- MAE should provide most consistent performance
- MAE + Direction might offer best balance
- Adaptive may show higher training loss due to squared terms
EOF

echo "📊 Comparison summary saved to: results/${RESULTS_DIR}/comparison_summary.md"
echo "🔍 To view results:"
echo "   - Summary: cat results/${RESULTS_DIR}/comparison_summary.md"
echo "   - Individual logs: ls results/${RESULTS_DIR}/*.log"
echo "   - TensorBoard: tensorboard --logdir logs/${RESULTS_DIR}"

echo "✨ Loss function comparison experiment completed!" 
