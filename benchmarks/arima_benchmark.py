# benchmarks/arima_benchmark.py
import os
import sys
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add project root to path to import configs
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from configs import StockPredictionConfig

# Suppress warnings from statsmodels
warnings.filterwarnings("ignore")

def calculate_metrics(y_true, y_pred):
    """Calculates MAE, MSE, RMSE."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def arima_rolling_forecast(series, seq_len, pred_len, order=(5,1,0)):
    """
    Performs rolling ARIMA forecasts.

    Args:
        series (pd.Series): The time series data (e.g., close prices).
        seq_len (int): Length of the history window to use for fitting.
        pred_len (int): Number of steps to forecast ahead.
        order (tuple): The (p, d, q) order for the ARIMA model.

    Returns:
        tuple: (list of true values, list of predicted values)
    """
    history = list(series[0:seq_len])
    predictions = []
    true_values = []

    # print(f"Total steps to predict: {len(series) - seq_len}") # Verbose
    for t in range(len(series) - seq_len - pred_len + 1):
        current_window = series[t : t + seq_len]
        actual_future = series[t + seq_len : t + seq_len + pred_len]

        if len(actual_future) < pred_len:
            # Not enough data left for a full prediction window
            break

        try:
            # Fit ARIMA model on the current window
            # Using enforce_stationarity=False, enforce_invertibility=False
            # can help convergence but might lead to less stable models.
            # Consider trying different orders or auto_arima if performance is poor.
            model = ARIMA(current_window, order=order, enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit()

            # Forecast pred_len steps ahead
            forecast = model_fit.forecast(steps=pred_len)

            predictions.append(forecast)
            true_values.append(actual_future.values) # Append the actual future values

        except Exception as e:
            # Handle potential errors during model fitting (e.g., non-stationarity issues)
            # print(f"Warning: ARIMA failed for step {t}. Error: {e}. Skipping.")
            # Append NaNs or handle differently? For now, skip this step's prediction.
            # To keep arrays aligned, we might need to append NaNs of the correct shape
            predictions.append(np.full(pred_len, np.nan))
            true_values.append(actual_future.values) # Still append true values to keep alignment for metrics

        # Update history for the next iteration (optional, depends if you want pure rolling or expanding window)
        # For pure rolling as described:
        # history.append(series[t + seq_len])
        # history.pop(0)
        # For simplicity here, we just slice the next window `series[t+1 : t+1 + seq_len]` implicitly

    # Filter out steps where prediction failed (NaNs) before metric calculation
    valid_indices = [i for i, p in enumerate(predictions) if not np.isnan(p).any()]
    if not valid_indices:
        # print("Warning: All ARIMA predictions failed for this series.") # Verbose
        return np.array([]), np.array([])

    predictions_valid = np.array([predictions[i] for i in valid_indices])
    true_values_valid = np.array([true_values[i] for i in valid_indices])

    return true_values_valid, predictions_valid


def main():
    print("--- Starting ARIMA Benchmark ---")
    config = StockPredictionConfig()

    # Use the same test files as the main model
    test_files = config.test_files
    seq_len = config.seq_len
    pred_len = config.pred_len
    feature_to_predict = 'close' # Focus on closing price for ARIMA benchmark

    if not test_files:
        print("Error: No test files found in config. Aborting.")
        return

    all_preds = []
    all_trues = []

    for file_path in test_files:
        print(f"\nProcessing test file: {os.path.basename(file_path)}")
        try:
            df = pd.read_csv(file_path, parse_dates=['window_start'])
        except Exception as e:
            print(f"Error reading file {file_path}: {e}. Skipping.")
            continue

        tickers = df['ticker'].unique()
        print(f"Found {len(tickers)} tickers in file.")

        for ticker in tqdm(tickers, desc=f"Tickers in {os.path.basename(file_path)}"):
            ticker_data = df[df['ticker'] == ticker].sort_values('window_start')

            if feature_to_predict not in ticker_data.columns:
                # print(f"Feature '{feature_to_predict}' not found for ticker {ticker}. Skipping.")
                continue

            series = ticker_data[feature_to_predict].reset_index(drop=True)

            if len(series) < seq_len + pred_len:
                # print(f"Not enough data for ticker {ticker} (found {len(series)}, need {seq_len + pred_len}). Skipping.")
                continue

            # Perform rolling forecast for this ticker
            true_vals, pred_vals = arima_rolling_forecast(series, seq_len, pred_len)

            if true_vals.size > 0 and pred_vals.size > 0:
                 all_trues.append(true_vals)
                 all_preds.append(pred_vals)

    if not all_trues or not all_preds:
        print("\nNo valid predictions generated across all test files.")
        return

    # Concatenate results from all tickers and files
    all_trues_np = np.concatenate(all_trues, axis=0)
    all_preds_np = np.concatenate(all_preds, axis=0)

    print(f"\nTotal evaluated prediction steps: {all_trues_np.shape[0]}")
    print(f"Shape of true values: {all_trues_np.shape}") # Should be [N, pred_len]
    print(f"Shape of predictions: {all_preds_np.shape}") # Should be [N, pred_len]

    # Calculate overall metrics
    # Reshape for sklearn metrics if needed (MAE/MSE work element-wise)
    mae, mse, rmse = calculate_metrics(all_trues_np.ravel(), all_preds_np.ravel())

    print("\n--- ARIMA Benchmark Results ---")
    print(f"Feature Predicted: {feature_to_predict}")
    print(f"ARIMA Order: {(5,1,0)}") # TODO: Make order configurable or use auto_arima
    print(f"Sequence Length (History): {seq_len}")
    print(f"Prediction Length (Forecast Horizon): {pred_len}")
    print("-----------------------------")
    print(f"Overall MAE:  {mae:.6f}")
    print(f"Overall MSE:  {mse:.6f}")
    print(f"Overall RMSE: {rmse:.6f}")
    print("-----------------------------")

    # Save metrics
    results_dir = './results/ARIMA_benchmark/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    metrics_path = os.path.join(results_dir, 'metrics.npy')
    np.save(metrics_path, np.array([mae, mse, rmse]))
    print(f"ARIMA benchmark metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()