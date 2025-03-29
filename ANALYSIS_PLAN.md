# iTransformer Implementation Analysis Plan

This document outlines the plan to analyze the provided iTransformer implementation for stock market forecasting.

## Goals

1.  **Correctness:** Verify if the implementation aligns with the iTransformer paper ([arXiv:2310.06625](https://arxiv.org/abs/2310.06625)).
2.  **Parameter Count:** Determine the total number of trainable parameters in the model.
3.  **Improvements:** Suggest enhancements for code clarity, effectiveness, and efficiency.
4.  **Training Window:** Evaluate the suitability of the 60-minute training window.

## Phases

**Phase 1: Understanding & Analysis**

1.  **Review iTransformer Concepts:**
    *   Analyze the core concepts from the iTransformer paper (details pending user input).
    *   Focus on the "inversion" mechanism, embedding strategies, and attention application.
2.  **Code & README Review:**
    *   Analyze provided Python files (`data_provider/data_loader.py`, `layers/*`, `models/iTransformer.py`, `train.py`, `utils/*`).
    *   Analyze pending Python files (`configs.py`, `stock_dataset.py`, `exp_stock_forecasting.py` - content pending user input).
    *   Cross-reference implementation details with `README.md`.
    *   Visualize the architecture using Mermaid:
      ```mermaid
      graph TD
          A[Input: Batch, Stocks, Time, Features] --> B(Reshape: Batch*Stocks, Time, Features);
          B --> C{Normalization};
          C --> D[Embedding: DataEmbedding_inverted];
          D -- Permute --> E(Embedded: Batch*Stocks, Features, d_model);
          E --> F[Encoder: Multi-Head Attention + FF];
          F --> G[Projection: Linear Layer];
          G -- Transpose --> H(Output: Batch*Stocks, pred_len, Features);
          H --> I{De-Normalization};
          I --> J(Reshape: Batch, Stocks, pred_len, Features);

          subgraph Embedding
              D1[Value Embedding]
              D2[Temporal Embedding]
              D3[Feature Embedding]
              D4[Positional Encoding]
              D5[Layer Norm + Dropout]
              D1 & D2 & D3 & D4 --> D5
          end

          subgraph Encoder Layer
              F1[Multi-Head Self-Attention across Features]
              F2[Add & Norm]
              F3[Feed Forward]
              F4[Add & Norm]
              F1 --> F2 --> F3 --> F4
          end
      ```
3.  **Address Question 1 (Correctness):**
    *   Compare code logic (dimension permutation, attention) with the paper's description.
    *   Identify discrepancies or confirmations.
4.  **Address Question 2 (Parameter Count):**
    *   Outline the formula for parameter calculation based on layers and hyperparameters (from `configs.py`).
5.  **Address Question 3 (Improvements):**
    *   Suggest improvements for clarity (type hints, comments), effectiveness (loss function, normalization), efficiency (data loading, potential optimizations), and structure.
6.  **Address Question 4 (Training Window):**
    *   Discuss pros/cons of the 60-minute window.
    *   Recommend experimentation.

**Phase 2: Review & Next Steps**

7.  **Present Findings:** Summarize the analysis once pending information is received and reviewed.
8.  **Plan Review:** Seek user feedback on the findings and analysis.
9.  **Mode Switch:** Suggest switching modes to implement any agreed-upon changes or further steps.

## Dependencies

*   Content/details from the iTransformer paper ([arXiv:2310.06625](https://arxiv.org/abs/2310.06625)).
*   Content of `configs.py`.
*   Content of `stock_dataset.py`.
*   Content of `exp_stock_forecasting.py`.