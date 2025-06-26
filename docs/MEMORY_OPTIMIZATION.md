# Memory Optimization and Leak Prevention

This document describes the memory optimizations implemented to prevent memory leaks during long training runs (16+ hours).

## 🐛 **Identified Memory Leak Sources**

1. **Iteration-level metrics accumulation** - Storing unlimited metrics in lists
2. **Epoch-level losses accumulation** - Growing loss history without bounds  
3. **Dataset caching without limits** - Unlimited dataset cache growth
4. **Tensor references in progress bars** - TQDM keeping tensor references
5. **Missing explicit cleanup** - No garbage collection or GPU cache cleanup
6. **Accumulated batch data** - Tensors not explicitly deleted after use

## 🔧 **Implemented Fixes**

### 1. **Circular Buffer for Iteration Metrics**
```python
# Before: Unlimited growth
iteration_metrics['iteration'].append(global_iter)

# After: Circular buffer with size limit
if len(iteration_metrics['iteration']) >= self.max_iteration_metrics:
    remove_count = self.max_iteration_metrics // 10
    for key in iteration_metrics:
        iteration_metrics[key] = iteration_metrics[key][remove_count:]
```

### 2. **Explicit Tensor Cleanup in Training Loop**
```python
# After model step, explicitly delete tensors
del outputs, batch_x, batch_y, batch_x_mark, batch_y_mark, attention_mask, dec_inp

# Periodic cleanup
if global_iter - last_cleanup_iter >= self.cleanup_frequency:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    last_cleanup_iter = global_iter
```

### 3. **Limited Dataset Cache with LRU Eviction**
```python
# Implement LRU-style cache eviction
if len(self._dataset_cache) >= self.max_cache_size:
    oldest_key = next(iter(self._dataset_cache))
    del self._dataset_cache[oldest_key]
    gc.collect()
```

### 4. **Bounded Loss History**
```python
# Limit size of historical losses
max_history = 1000  # Keep only last 1000 epochs
if len(train_losses) > max_history:
    train_losses = train_losses[-max_history:]
    val_losses = val_losses[-max_history:]
    learning_rates = learning_rates[-max_history:]
```

### 5. **Memory Cleanup in Test/Validation**
```python
# Explicit cleanup during testing
del outputs, batch_x, batch_y, batch_x_mark, batch_y_mark

# Periodic cleanup every 50 batches
if (i + 1) % 50 == 0:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
```

### 6. **Epoch-level Memory Management**
```python
# Clear epoch losses and run cleanup
del epoch_train_loss

# End of epoch cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
```

## ⚙️ **Configuration Parameters**

New memory management parameters added to config:

```python
# Memory Management Parameters
max_iteration_metrics: int = 10000   # Limit iteration metrics storage
cleanup_frequency: int = 100         # Cleanup every N iterations  
max_cache_size: int = 5             # Limit dataset cache size
```

### CLI Arguments:
```bash
--max-iteration-metrics 10000     # Prevent metrics memory leak
--cleanup-frequency 100           # More frequent cleanup for long runs
--max-cache-size 5               # Smaller cache for memory-constrained runs
```

## 📊 **Memory Monitoring**

Simple memory logging function added:
```python
def log_memory_stats(prefix: str = ""):
    """Log current memory statistics."""
    import psutil
    process = psutil.Process()
    ram_gb = process.memory_info().rss / (1024**3)
    
    gpu_gb = 0.0
    if torch.cuda.is_available():
        gpu_gb = torch.cuda.memory_allocated() / (1024**3)
        
    print(f"🔍 {prefix}Memory: {ram_gb:.1f}GB RAM, {gpu_gb:.1f}GB GPU")
```

Memory is logged at:
- Training start
- Every 100 iterations (configurable)
- After epoch completion
- Training completion

## 🎯 **Expected Impact**

### Before Fixes:
- Memory usage: 12-14 GB after 16+ hours
- Potential unbounded growth over time
- Risk of OOM errors in very long runs

### After Fixes:
- Bounded memory usage with circular buffers
- Explicit cleanup prevents accumulation
- Memory usage should stabilize after initial ramp-up
- Safe for 24+ hour training runs

## 🔍 **Monitoring Memory During Training**

Watch for these patterns:
1. **Normal**: Steady memory usage after initial few epochs
2. **Leak Warning**: Continuous memory growth over hours
3. **Cleanup Working**: Memory drops after cleanup periods

### Sample Output:
```
🔍 Training start - Memory: 2.1GB RAM, 1.2GB GPU
🔍 Epoch 1, Iter 100 - Memory: 3.2GB RAM, 2.1GB GPU
🔍 Epoch 5, Iter 500 - Memory: 3.5GB RAM, 2.1GB GPU  # Stable
🔍 After final cleanup - Memory: 2.8GB RAM, 0.1GB GPU
```

## 🚨 **Warning Signs of Memory Leaks**

If you still see:
1. Continuous RAM growth beyond 8GB
2. GPU memory not releasing between epochs
3. "CUDA out of memory" errors
4. System becoming unresponsive

Then consider:
- Reducing `batch_size`
- Increasing `cleanup_frequency` (e.g., 50)
- Reducing `max_iteration_metrics` (e.g., 5000)
- Adding `--save-iteration-metrics false` to disable detailed metrics

## 🔧 **Additional Optimizations for Extreme Cases**

For very long runs or memory-constrained systems:

```bash
# Conservative memory settings
python train.py \
  --max-iteration-metrics 5000 \
  --cleanup-frequency 50 \
  --max-cache-size 3 \
  --batch-size 32 \
  --save-iteration-metrics false
```

## 📈 **Performance vs Memory Trade-offs**

- **More frequent cleanup**: Lower memory, slightly slower training
- **Smaller metrics storage**: Lower memory, less detailed logging
- **Smaller cache**: Lower memory, potential for more disk I/O
- **Smaller batch size**: Lower memory, potentially slower convergence

Choose settings based on your system constraints and training duration. 
