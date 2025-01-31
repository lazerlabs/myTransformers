from configs import StockPredictionConfig
from train import main as train_model

def run_sequence_length_experiment():
    """Test different input sequence lengths"""
    sequence_lengths = [30, 60, 120, 240]  # 30min, 1h, 2h, 4h
    
    for seq_len in sequence_lengths:
        config = StockPredictionConfig()
        config.seq_len = seq_len
        config.label_len = seq_len // 2
        print(f"\nRunning experiment with sequence length: {seq_len}")
        train_model(config)

def run_prediction_length_experiment():
    """Test different prediction horizons"""
    pred_lengths = [15, 30, 60, 120]  # 15min, 30min, 1h, 2h
    
    for pred_len in pred_lengths:
        config = StockPredictionConfig()
        config.pred_len = pred_len
        print(f"\nRunning experiment with prediction length: {pred_len}")
        train_model(config)

def run_feature_ablation():
    """Test different feature combinations"""
    feature_sets = [
        ['close'],  # Only close price
        ['close', 'volume'],  # Close price and volume
        ['open', 'high', 'low', 'close'],  # OHLC
        ['open', 'high', 'low', 'close', 'volume']  # OHLCV
    ]
    
    for features in feature_sets:
        config = StockPredictionConfig()
        config.enc_in = len(features)
        print(f"\nRunning experiment with features: {features}")
        train_model(config)

if __name__ == "__main__":
    # Run all experiments
    print("Running sequence length experiments...")
    run_sequence_length_experiment()
    
    print("\nRunning prediction length experiments...")
    run_prediction_length_experiment()
    
    print("\nRunning feature ablation experiments...")
    run_feature_ablation() 