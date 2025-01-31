import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from stock_dataset import create_dataloader
from utils.logger import Logger
from utils.visualization import StockVisualizer
from data_provider.data_loader import create_data_loaders
from configs import StockPredictionConfig
warnings.filterwarnings('ignore')

class Exp_Stock_Forecast():
    def __init__(self, args: StockPredictionConfig):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
        # Initialize logger and visualizer
        log_dir = './logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.logger = Logger(f"{args.model}_stock_prediction", log_dir=log_dir)
        self.visualizer = StockVisualizer()

        # Create data loaders
        self.train_loader, self.test_loader = create_data_loaders(args)


    def _acquire_device(self):
        if self.args.use_gpu:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                print('Use GPU:', device)
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
                print('Use MPS:', device)
            else:
                device = torch.device('cpu')
                print('No GPU/MPS available, use CPU instead')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        # Import model here to avoid circular imports
        from models.iTransformer import Model as iTransformerModel
        
        model_dict = {
            'iTransformer': iTransformerModel,
        }
        
        if self.args.model not in model_dict:
            raise ValueError(f"Model {self.args.model} not found. Available models: {list(model_dict.keys())}")
        
        model = model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    
    def _get_data(self, flag):
        """
        flag: 'train' or 'test'
        """
        data_path = self.args.train_data_path if flag == 'train' else self.args.test_data_path
        
        data_set, data_loader = create_dataloader(
            file_path=data_path,
            batch_size=self.args.batch_size,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            scale=self.args.scale,
            tickers=['AAPL', 'MSFT', 'JPM', 'JNJ', 'AXP']
        )
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, data_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_mark, batch_y, batch_y_mark) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Prepare decoder input
                dec_inp = torch.zeros_like(batch_y[:, :, :self.args.pred_len, :])
                dec_inp = torch.cat([batch_y[:, :, :self.args.label_len, :], dec_inp], dim=2)

                # Forward pass
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        test_data, test_loader = self._get_data(flag='test')
        
        # Ensure base checkpoints directory exists
        if not os.path.exists(self.args.checkpoints):
            os.makedirs(self.args.checkpoints)
            
        path = os.path.join(self.args.checkpoints, setting)
        # Clean up old checkpoints
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        os.makedirs(path)
        
        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        criterion = nn.MSELoss()  # Define the loss criterion
        
        # Initialize learning rate scheduler
        if self.args.lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                model_optim,
                T_max=self.args.train_epochs,
                eta_min=self.args.min_lr
            )
        else:  # reduce_on_plateau
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                model_optim,
                mode='min',
                factor=self.args.lr_decay_factor,
                patience=self.args.lr_patience,
                min_lr=self.args.min_lr
            )
        
        # Initialize metrics storage
        train_losses = []
        val_losses = []
        test_losses = []
        learning_rates = []
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_x_mark, batch_y, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                # Move data to device
                batch_x = batch_x.float().to(self.device)  # [batch, stocks, seq_len, features]
                batch_y = batch_y.float().to(self.device)  # [batch, stocks, pred_len, features]
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y).float()
                dec_inp = torch.cat([batch_y[:, :, :self.args.label_len, :], dec_inp], dim=2).float().to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # [batch, stocks, pred_len, features]
                
                # Calculate loss
                f_dim = -1 if self.args.features == 'MS' else 0
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()
            
            train_loss = np.average(train_loss)
            
            # Validation only during training
            val_loss = self.test(setting, test=0)
            
            # Update learning rate
            if self.args.lr_scheduler == 'cosine':
                scheduler.step()
            else:
                scheduler.step(val_loss)
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            learning_rates.append(model_optim.param_groups[0]['lr'])
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.6f} Val Loss: {3:.6f} Learning Rate: {4:.6f}".format(
                epoch + 1, train_steps, train_loss, val_loss, model_optim.param_groups[0]['lr']))
            
            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        # Plot training metrics
        metrics = {
            'epoch': list(range(1, len(train_losses) + 1)),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'test_loss': test_losses,
            'learning_rate': learning_rates
        }
        self.visualizer.plot_training_metrics(metrics)
        self.visualizer.plot_learning_rate(learning_rates)
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting, test=0):
        """
        test=0: validation
        test=1: testing
        """
        # Only load the model for final testing
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        # Use appropriate data loader
        data_flag = 'test' if test else 'val'
        data, data_loader = self._get_data(flag=data_flag)

        preds = []
        trues = []
        
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_mark, batch_y, batch_y_mark) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)  # [batch, stocks, seq_len, features]
                batch_y = batch_y.float().to(self.device)  # [batch, stocks, pred_len, features]
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y).float()
                dec_inp = torch.cat([batch_y[:, :, :self.args.label_len, :], dec_inp], dim=2).float().to(self.device)

                # Forward pass
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)  # [batch, stocks, pred_len, features]
                
                # Move to CPU and convert to numpy
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                
                if test:  # Only print shapes during testing
                    print(f"Batch shapes - pred: {pred.shape}, true: {true.shape}")
                
                preds.append(pred)
                trues.append(true)

        # Concatenate along the batch dimension
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        if test:  # Only print shapes during testing
            print('Final shapes - preds:', preds.shape, 'trues:', trues.shape)

        # Calculate metrics
        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)
        rmse = np.sqrt(mse)
        
        if test:  # Only save results and plot during testing
            # result save
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            print(f'mse:{mse}, mae:{mae}')
            
            np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

            # Log final metrics
            self.logger.log_prediction(mae, mse, rmse)
            
            # Get test dataset for denormalization
            test_dataset = data
            
            # Visualize predictions for all stocks
            self.visualizer.plot_all_stocks_predictions(trues, preds, test_dataset, feature_idx=1)  # Plot close price
            
            # Individual feature plots if needed
            for feature_idx in range(len(self.visualizer.feature_names)):
                self.visualizer.plot_all_stocks_predictions(trues, preds, test_dataset, feature_idx=feature_idx)

        return mse  # Return MSE for early stopping and learning rate scheduling

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss 