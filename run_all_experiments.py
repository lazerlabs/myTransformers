#!/usr/bin/env python3
"""
Automated Experiment Runner for Inverted Transformer Comparison

Runs a systematic comparison of inverted vs classic transformers on stock market prediction.
This script automates the core experiments needed to evaluate the iTransformer architecture.
"""

import subprocess
import itertools
import time
from datetime import datetime

# Core models for comparison (focused on key architectures)
MODELS = [
    "iTransformer",    # Core inverted transformer
    "iInformer",       # Efficient inverted transformer
    "Transformer",     # Classic transformer baseline
]

# Feature sets to compare single vs multivariate performance
FEATURE_SETS = [
    ["close"],                           # Single feature baseline
    ["close", "volume", "transactions"]  # Multi-feature (tests inverted advantage)
]

# Configuration
EPOCHS = 20
BATCH_SIZE = 32
DATA_DIR = "./dataset/"

def print_header():
    """Print experiment suite header"""
    print("=" * 80)
    print("🎯 INVERTED TRANSFORMER COMPARISON SUITE")
    print("=" * 80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🧪 Models: {', '.join(MODELS)}")
    print(f"📊 Feature Sets: {len(FEATURE_SETS)} configurations")
    print(f"⚙️  Training: {EPOCHS} epochs, batch size {BATCH_SIZE}")
    print(f"🎲 Total Experiments: {len(MODELS) * len(FEATURE_SETS)}")
    print("=" * 80)
    print()

def run_experiment(model, features, experiment_num, total_experiments):
    """Run a single experiment with comprehensive logging"""
    features_str = ",".join(features)
    num_features = len(features)
    feature_mode = "S" if num_features == 1 else "M"
    
    # Create experiment identifier
    feature_desc = "SINGLE" if num_features == 1 else "MULTI"
    experiment_id = f"{feature_desc}_{model}_{num_features}F"
    
    print(f"🧪 EXPERIMENT {experiment_num}/{total_experiments}")
    print(f"📊 Model: {model}")
    print(f"🔢 Features: {features_str} ({num_features} features)")
    print(f"🆔 ID: {experiment_id}")
    print(f"⏰ Started: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 60)
    
    # Build command
    cmd = [
        "python", "run.py",
        "--is_training", "1",
        "--model_id", experiment_id,
        "--model", model,
        "--data", "stock",
        "--data_dir", DATA_DIR,
        "--features", features_str,
        "--features_mode", feature_mode,
        "--target", "close",
        "--seq_len", "60",
        "--pred_len", "15",
        "--label_len", "30",
        "--enc_in", str(num_features),
        "--dec_in", str(num_features),
        "--c_out", str(num_features),
        "--d_model", "512",
        "--n_heads", "8",
        "--e_layers", "4",
        "--d_ff", "2048",
        "--batch_size", str(BATCH_SIZE),
        "--learning_rate", "0.0001",
        "--train_epochs", str(EPOCHS),
        "--patience", "5",
        "--des", f"{feature_desc}_Feature",
        "--itr", "1"
    ]
    
    start_time = time.time()
    
    try:
        # Run experiment
        result = subprocess.run(cmd, check=True, text=True)
        elapsed = time.time() - start_time
        
        print(f"✅ SUCCESS: {experiment_id}")
        print(f"⏱️  Duration: {elapsed/60:.1f} minutes")
        print(f"📁 Results saved to checkpoints/logs/figures/")
        
        return True, elapsed
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        
        print(f"❌ FAILED: {experiment_id}")
        print(f"💥 Exit Code: {e.returncode}")
        print(f"⏱️  Duration: {elapsed/60:.1f} minutes")
        
        # Print relevant error info (last few lines)
        if e.stderr:
            error_lines = e.stderr.strip().split('\n')[-3:]
            print("🔍 Error details:")
            for line in error_lines:
                print(f"   {line}")
        
        return False, elapsed

def print_summary(results, total_time):
    """Print final experiment summary"""
    successful = sum(1 for success, _ in results if success)
    failed = len(results) - successful
    avg_time = sum(duration for _, duration in results) / len(results) if results else 0
    
    print("=" * 80)
    print("🏁 EXPERIMENT SUITE COMPLETE")
    print("=" * 80)
    print(f"📊 Results Summary:")
    print(f"   ✅ Successful: {successful}/{len(results)} experiments")
    print(f"   ❌ Failed: {failed}/{len(results)} experiments")
    print(f"   📈 Success Rate: {(successful/len(results)*100):.1f}%")
    print(f"   ⏱️  Total Time: {total_time/60:.1f} minutes")
    print(f"   ⚡ Avg Time/Experiment: {avg_time/60:.1f} minutes")
    print()
    
    if successful > 0:
        print("🎯 Next Steps:")
        print("   1. tensorboard --logdir logs/")
        print("   2. Check figures/ for prediction plots")
        print("   3. Compare checkpoints/ for best models")
        print("   4. Run scripts/comparative/full_comparison.sh for detailed analysis")
        print()
        
        print("🔍 Expected Findings:")
        print("   • iTransformer should outperform Transformer on multi-feature tasks")
        print("   • Multi-feature should outperform single-feature for all models")
        print("   • iInformer should offer good efficiency vs accuracy trade-off")
    
    print("=" * 80)

def main():
    """Main experiment runner"""
    print_header()
    
    total_experiments = len(MODELS) * len(FEATURE_SETS)
    results = []
    start_time = time.time()
    
    experiment_num = 1
    
    for model, features in itertools.product(MODELS, FEATURE_SETS):
        success, duration = run_experiment(model, features, experiment_num, total_experiments)
        results.append((success, duration))
        
        experiment_num += 1
        
        # Add spacing between experiments
        if experiment_num <= total_experiments:
            print("=" * 60)
            print()
    
    total_time = time.time() - start_time
    print_summary(results, total_time)

if __name__ == "__main__":
    main()
