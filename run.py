import argparse
import torch
import random
import numpy as np
import sys
import os
from pathlib import Path

# Import our experiment class
from exp_stock_forecasting import Exp_Stock_Forecast
from configs import StockPredictionConfig

def setup_seed(seed=2023):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def create_config_from_args(args):
    """Convert argparse args to StockPredictionConfig"""
    
    # Handle comma-separated lists
    features = args.features.split(',') if args.features else ['close']
    stocks = args.stocks.split(',') if args.stocks else None
    val_stocks = args.val_stocks.split(',') if args.val_stocks else ['AAPL', 'MSFT', 'JPM', 'JNJ', 'AXP']
    
    # Auto-adjust model dimensions based on actual number of features
    num_features = len(features)
    enc_in = num_features
    dec_in = num_features  
    c_out = num_features
    
    print(f"Auto-adjusting model dimensions: features={features}, enc_in={enc_in}, dec_in={dec_in}, c_out={c_out}")
    
    # Convert device_ids if provided
    device_ids = None
    if args.device_ids:
        device_ids = [int(id_.strip()) for id_ in args.device_ids.split(',')]
    
    # Create config with args
    config = StockPredictionConfig(
        # Data parameters
        data_dir=args.data_dir or "dataset",
        stocks=stocks,
        features=features,
        train_size=args.train_size,
        test_size=args.test_size,
        val_size=args.val_size,
        val_stocks=val_stocks,
        
        # Sequence parameters
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        label_len=args.label_len,
        scale=args.scale,
        inverse=args.inverse,
        
        # Dataset mode parameters
        mode=args.mode,
        interpolate_max_missing=args.interpolate_max_missing,
        max_seq_len=args.max_seq_len,
        
        # Model parameters
        model=args.model,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        embed=args.embed,
        activation=args.activation,
        output_attention=args.output_attention,
        use_norm=args.use_norm,
        
        # Training parameters
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_epochs=args.train_epochs,
        patience=args.patience,
        
        # Loss parameters
        loss_type=args.loss_type,
        
        # Device parameters
        use_gpu=args.use_gpu,
        use_multi_gpu=args.use_multi_gpu,
        gpu=args.gpu,
        device_ids=device_ids,
        
        # Model specific
        factor=args.factor,
        enc_in=enc_in,  # Use auto-adjusted value
        dec_in=dec_in,  # Use auto-adjusted value
        c_out=c_out,    # Use auto-adjusted value
        freq=args.freq,
        forecasting_features=args.features_mode,
        class_strategy=args.class_strategy,
        
        # Allow classic models for comparison
        allow_classic_models=True,  # Enable comparison with classic transformers
        
        # Test parameters
        test_interval=args.test_interval if hasattr(args, 'test_interval') else 0,
        
        # Debug/Testing parameters
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )
    
    return config

if __name__ == '__main__':
    setup_seed()
    
    parser = argparse.ArgumentParser(description='Stock Market iTransformer - Unified Interface')
    
    # Basic config - compatible with original
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        choices=['iTransformer', 'iInformer', 'iReformer', 'iFlowformer', 'iFlashformer',
                                'Transformer', 'Informer', 'Reformer', 'Flowformer', 'Flashformer'],
                        help='model name')
    
    # Data loader - adapted for stock data
    parser.add_argument('--data', type=str, required=True, default='stock', help='dataset type')
    parser.add_argument('--data_dir', type=str, default='dataset', help='root path of the data files')
    parser.add_argument('--stocks', type=str, default=None, help='comma-separated stock tickers (e.g., AAPL,MSFT)')
    parser.add_argument('--features', type=str, default='close,volume,transactions',
                        help='comma-separated features (e.g., close,volume,transactions)')
    parser.add_argument('--features_mode', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='close', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='min',
                        help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    
    # Data split parameters
    parser.add_argument('--train_size', type=int, default=None, help='number of files for training')
    parser.add_argument('--test_size', type=int, default=2, help='number of files for testing')
    parser.add_argument('--val_size', type=int, default=2, help='number of files for validation')
    parser.add_argument('--val_stocks', type=str, default='AAPL,MSFT,JPM,JNJ,AXP', help='validation stocks')
    
    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=60, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=30, help='start token length')
    parser.add_argument('--pred_len', type=int, default=15, help='prediction sequence length')
    
    # Data processing
    parser.add_argument('--scale', type=bool, default=True, help='whether to scale data')
    parser.add_argument('--inverse', type=bool, default=True, help='whether to inverse output data')
    parser.add_argument('--mode', type=str, default='sliding_window', choices=['sliding_window', 'full_day'],
                        help='dataset mode')
    parser.add_argument('--interpolate_max_missing', type=int, default=3, help='max consecutive NaNs to interpolate')
    parser.add_argument('--max_seq_len', type=int, default=2000, help='max sequence length for embedding')
    
    # Model define - compatible with original
    parser.add_argument('--enc_in', type=int, default=3, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=3, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=3, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout')
    parser.add_argument('--embed', type=str, default='fixed',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    
    # Optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss_type', type=str, default='adaptive', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--device_ids', type=str, default=None, help='device ids of multiple gpus')
    
    # iTransformer specific
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--use_norm', type=bool, default=True, help='use norm and denorm')
    parser.add_argument('--temperature', type=float, default=0.0, help='temperature for sampling')
    
    # Additional options
    parser.add_argument('--test_interval', type=int, default=0, help='test every N epochs during training')
    
    # Debug/Testing options
    parser.add_argument('--max_train_samples', type=int, default=None, help='limit training samples for testing (None = unlimited)')
    parser.add_argument('--max_test_samples', type=int, default=None, help='limit test samples for testing (None = unlimited)')
    
    args = parser.parse_args()
    
    # Convert to our config format
    config = create_config_from_args(args)
    
    print('Args in experiment:')
    print(args)
    print('\nConfig created:')
    print(f"Model: {config.model}")
    print(f"Features: {config.features}")
    print(f"Stocks: {config.stocks}")
    print(f"Data dir: {config.data_dir}")
    print(f"Sequence length: {config.seq_len}")
    print(f"Prediction length: {config.pred_len}")
    
    # Create experiment
    exp = Exp_Stock_Forecast(config)
    
    if args.is_training:
        for ii in range(args.itr):
            # Setting record of experiments - compatible with original naming
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features_mode,
                len(config.features),  # number of features
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.des,
                ii)
            
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            
            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                # Add predict method if needed
                pass
            
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features_mode,
            len(config.features),
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.des,
            ii)
        
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
