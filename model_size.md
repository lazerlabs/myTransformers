# Adjusting iTransformer Model Size

This document explains how to modify the hyperparameters in `configs.py` to increase or decrease the size (capacity) of the iTransformer model and provides an approximate formula for estimating the total parameter count.

## Goal

Adjusting model size can be useful for:
*   **Increasing Capacity:** Allowing the model to potentially learn more complex patterns and relationships within the time series data, especially if the current model seems to be underfitting.
*   **Decreasing Capacity:** Reducing computational cost (training time, VRAM usage) or mitigating overfitting if the current model is too large for the dataset.

## Key Hyperparameters for Size Adjustment

Modify these parameters in `configs.py` to change the model size:

1.  **`d_model` (Model Dimension):**
    *   **Description:** The core embedding size used throughout the Transformer layers.
    *   **Effect:** Significantly impacts overall size. Larger `d_model` allows richer representations but increases parameters in embeddings, attention projections, and feed-forward layers (often quadratically).
    *   **Example:** Increase from 512 to 768 or 1024.

2.  **`e_layers` (Number of Encoder Layers):**
    *   **Description:** The depth of the Transformer stack.
    *   **Effect:** Increases size linearly with the number of layers. Deeper models can capture more abstract relationships but may be harder to train.
    *   **Example:** Increase from 4 to 6 or 8.

3.  **`d_ff` (Feed-Forward Dimension):**
    *   **Description:** The hidden dimension within the feed-forward network of each encoder layer. Typically set relative to `d_model` (e.g., `d_ff = 4 * d_model`).
    *   **Effect:** Increases the capacity of the feed-forward component. Adjust proportionally if changing `d_model`.
    *   **Example:** If `d_model`=768, set `d_ff`=3072.

4.  **`n_heads` (Number of Attention Heads):**
    *   **Description:** Number of parallel attention mechanisms. Must divide `d_model`.
    *   **Effect:** Doesn't drastically change total parameters (as dimension per head decreases), but increases computational complexity and allows focusing on different feature relationships simultaneously.
    *   **Example:** Increase from 8 to 16 (if `d_model` is divisible by 16).

**Note:** Parameters like `seq_len` and `pred_len` also affect parameter count (in embedding and projection layers) but are usually determined by the task requirements rather than used primarily for scaling model capacity.

## Estimating Total Parameter Count

The total number of parameters can be approximated by summing the parameters from each major component. Let:
*   `N = enc_in` (Number of input features/variates, typically `len(features)`)
*   `L = seq_len`
*   `P = pred_len`
*   `D = d_model`
*   `H = n_heads`
*   `E = e_layers`
*   `DFF = d_ff`

**Approximate Formula:**

```
Total Params ≈ Params(Embedding) + E * Params(EncoderLayer) + Params(FinalLayerNorm) + Params(Projection)
```

Where:

1.  **`Params(Embedding)`** (using `DataEmbedding_inverted`):
    *   `value_embedding`: `(L * D) + D`
    *   `temporal_embedding`: `(L * D) + D`
    *   `feature_embedding`: `N * D`
    *   `pos_scale`: `1`
    *   `layer_norm`: `2 * D`
    *   *Total Embedding ≈ `2*(L*D + D) + N*D + 1 + 2*D`*

2.  **`Params(EncoderLayer)`**:
    *   `AttentionLayer`: `4 * ((D * D) + D)` (for Q, K, V, O projections)
    *   `Conv1 (FFN)`: `(D * 1 * DFF) + DFF` (kernel size 1)
    *   `Conv2 (FFN)`: `(DFF * 1 * D) + D` (kernel size 1)
    *   `LayerNorm1`: `2 * D`
    *   `LayerNorm2`: `2 * D`
    *   *Total EncoderLayer ≈ `4*(D^2 + D) + (D*DFF + DFF) + (DFF*D + D) + 4*D`*

3.  **`Params(FinalLayerNorm)`**: `2 * D` (The LayerNorm after the last encoder layer)

4.  **`Params(Projection)`**: `(D * P) + P` (Final linear layer)

**Example Calculation (Current Config):**
*   N=3, L=60, P=15, D=512, H=8, E=4, DFF=2048
*   Embedding ≈ 2*(60*512 + 512) + 3*512 + 1 + 2*512 ≈ 62,464 + 1,536 + 1 + 1,024 ≈ 65,025
*   EncoderLayer ≈ 4*(512^2 + 512) + (512*2048 + 2048) + (2048*512 + 512) + 4*512 ≈ 1,050,624 + 1,050,624 + 1,049,088 + 2,048 ≈ 3,152,384
*   FinalLayerNorm ≈ 2 * 512 = 1,024
*   Projection ≈ (512 * 15) + 15 = 7,695
*   Total ≈ 65,025 + 4 * 3,152,384 + 1,024 + 7,695 ≈ **12,683,280**

*(Note: This is an approximation, exact counts might differ slightly due to implementation details.)*

## Memory Requirements

*   **Model Parameters:** The size calculated above determines the memory needed for parameters, gradients, and optimizer states (roughly `Total Params * 4 bytes/param * (1 + 1 + 2 for Adam)`). For ~12.7M params, this is ~200-250MB.
*   **Activations:** This is the main VRAM consumer during training and depends heavily on `batch_size`, `seq_len`, `d_model`, `d_ff`, and `e_layers`. It's harder to estimate precisely without profiling but is generally much larger than the parameter memory.
*   **Data Batch:** Memory for the input/output tensors for each batch.

## Considerations

*   **Overfitting:** Larger models require more data and/or stronger regularization (e.g., increase `dropout` in `configs.py`).
*   **Compute Cost:** Training time per epoch increases with model size.
*   **Experimentation:** Systematically increase parameters and monitor validation loss to find the optimal size for your task and data. Start with `d_model` or `e_layers`.