# Temperature for Inference in Stock Prediction Model

## Overview

Temperature is a parameter that controls the randomness during model inference. It has been added to the stock prediction model to enable:

1. **Deterministic predictions** (temperature = 0.0)
2. **Stochastic predictions** with controlled uncertainty (temperature > 0.0)
3. **Uncertainty quantification** through multiple inference runs

## Configuration

### Default Setting
The temperature defaults to **0.0** (deterministic behavior) in the `StockPredictionConfig` class:

```python
# In configs.py
temperature: float = 0.0  # Temperature for inference sampling (0.0 = deterministic, >0 = stochastic)
```

### Setting Temperature

#### 1. Via Configuration
```python
from configs import StockPredictionConfig

config = StockPredictionConfig()
config.temperature = 1.0  # Set temperature for stochastic inference
```

#### 2. Via Command Line
```bash
python train.py --temperature 0.5
```

#### 3. Dynamically During Inference
```python
# Set temperature on the model
model.set_temperature(1.0)

# Or pass directly to forward()
output = model(input_data, input_marks, temperature=0.5)
```

## How Temperature Works

### Temperature = 0.0 (Default)
- **Deterministic**: Same input always produces the same output
- **No randomness**: Predictions are consistent across multiple runs
- **Best for**: Production inference, benchmarking, reproducible results

### Temperature > 0.0
- **Stochastic**: Same input can produce slightly different outputs
- **Controlled randomness**: Higher temperature = more variation
- **Gaussian noise**: Added to predictions scaled by temperature
- **Best for**: Uncertainty quantification, ensemble-like behavior

### Implementation Details
The temperature is applied in the `apply_temperature_sampling()` method:

```python
def apply_temperature_sampling(self, logits, temperature):
    if temperature <= 0.0:
        return logits  # Deterministic
    
    # Scale logits by temperature
    scaled_logits = logits / temperature
    
    # Add Gaussian noise for uncertainty
    if self.training or temperature > 0.0:
        noise = torch.randn_like(scaled_logits) * temperature * 0.1
        return scaled_logits + noise
    
    return scaled_logits
```

## Usage Examples

### Example 1: Basic Usage
```python
from configs import StockPredictionConfig
from exp_stock_forecasting import Exp_Stock_Forecast

# Create config with temperature
config = StockPredictionConfig()
config.temperature = 0.5

# Initialize experiment
exp = Exp_Stock_Forecast(config)

# Run test with temperature
exp.test(setting="test_run", test=1, temperature=0.5)
```

### Example 2: Uncertainty Quantification
```python
# Run multiple predictions with temperature > 0
predictions = []
for _ in range(10):  # 10 runs
    pred = model(input_data, input_marks, temperature=1.0)
    predictions.append(pred)

# Calculate uncertainty
predictions = torch.stack(predictions)
mean_pred = predictions.mean(dim=0)
std_pred = predictions.std(dim=0)
```

### Example 3: Temperature Sweep
```python
temperatures = [0.0, 0.1, 0.5, 1.0, 2.0]
results = {}

for temp in temperatures:
    result = exp.test(setting="temp_sweep", test=1, temperature=temp)
    results[temp] = result
```

## Testing Temperature Effects

Run the included test script to see temperature effects:

```bash
python test_temperature.py
```

This script will:
1. Test different temperature values (0.0, 0.1, 0.5, 1.0, 2.0)
2. Show variance across multiple runs
3. Generate visualization plots
4. Save results to `temperature_effects.png`

## Recommended Temperature Values

| Temperature | Use Case | Effect |
|-------------|----------|--------|
| 0.0 | Production inference | Deterministic, reproducible |
| 0.1-0.3 | Light uncertainty | Small variation, confidence intervals |
| 0.5-1.0 | Moderate uncertainty | Noticeable variation, ensemble-like |
| 1.0-2.0 | High uncertainty | Large variation, stress testing |
| >2.0 | Extreme cases | Very high variation, not recommended |

## Integration with Existing Code

The temperature functionality is backward compatible:
- **Existing code continues to work** without changes
- **Default behavior unchanged** (temperature = 0.0)
- **Optional parameter** in model forward pass
- **Available in training and testing** scripts

## Performance Considerations

- **No performance impact** when temperature = 0.0
- **Minimal overhead** for temperature > 0.0 (just noise generation)
- **Same memory usage** regardless of temperature setting
- **Deterministic when seeded** even with temperature > 0.0

## Advanced Usage

### Custom Temperature Scheduling
```python
# Different temperature for different prediction horizons
def adaptive_temperature(pred_step, max_steps):
    # Increase uncertainty for longer horizons
    return 0.1 * (pred_step / max_steps)

# Apply during inference
for step in range(pred_len):
    temp = adaptive_temperature(step, pred_len)
    pred = model(input_data, input_marks, temperature=temp)
```

### Temperature in Loss Functions
```python
# You could potentially use temperature in training
# as a form of label smoothing or noise regularization
# (This would require additional implementation)
```

## Troubleshooting

### Common Issues

1. **Too high temperature**: If temperature > 2.0, predictions may become unrealistic
2. **Inconsistent results**: Use `torch.manual_seed()` for reproducible stochastic results
3. **Performance issues**: Temperature should not affect performance significantly

### Debug Temperature
```python
print(f"Model temperature: {model.temperature}")
print(f"Config temperature: {config.temperature}")

# Test with/without temperature
pred_det = model(x, x_mark, temperature=0.0)
pred_stoch = model(x, x_mark, temperature=1.0)
print(f"Deterministic vs Stochastic difference: {torch.abs(pred_det - pred_stoch).mean()}")
```

## Future Enhancements

Potential future improvements:
1. **Learned temperature**: Make temperature a learnable parameter
2. **Feature-specific temperature**: Different temperature per feature
3. **Time-varying temperature**: Temperature that changes across prediction horizon
4. **Temperature scheduling**: Automatic temperature adjustment during training 