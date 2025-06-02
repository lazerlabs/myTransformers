import torch
import torch.nn as nn
import math

class DataEmbedding_inverted(nn.Module):
    """
    iTransformer Data Embedding for stock market data with positional encoding
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, max_len=60):
        super(DataEmbedding_inverted, self).__init__()
        # For iTransformer: we transform from seq_len to d_model for each feature
        # So the linear layer should map from sequence_length to d_model dimension
        # But we need to be flexible about sequence lengths
        self.c_in = c_in  # Number of features
        self.d_model = d_model
        
        # Use adaptive pooling or a more flexible approach
        # We'll use a different strategy: project each feature's time series separately
        self.feature_projections = nn.ModuleList([
            nn.Linear(max_len, d_model) for _ in range(c_in)
        ])
        
        # Add positional encoding for features (not timesteps)
        pe = torch.zeros(c_in, d_model)
        position = torch.arange(0, c_in, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Add learnable scale parameter for positional encoding
        self.pos_scale = nn.Parameter(torch.ones(1))
        
        # Add layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        
        # Store max_len for adaptive handling
        self.max_len = max_len

    def forward(self, x, x_mark):
        # iTransformer treats features as tokens, not timesteps
        # Input x shape: [batch, seq_len, features]
        batch_size, seq_len, features = x.shape
        
        # Handle variable sequence lengths
        if seq_len != self.max_len:
            # Use adaptive pooling to handle variable sequence lengths
            x_resampled = torch.zeros(batch_size, self.max_len, features, device=x.device, dtype=x.dtype)
            for i in range(features):
                # Interpolate each feature's time series to max_len
                feature_series = x[:, :, i]  # [batch, seq_len]
                # Use interpolation to resize to max_len
                feature_series = torch.nn.functional.interpolate(
                    feature_series.unsqueeze(1),  # [batch, 1, seq_len]
                    size=self.max_len,
                    mode='linear',
                    align_corners=False
                ).squeeze(1)  # [batch, max_len]
                x_resampled[:, :, i] = feature_series
            x = x_resampled
        
        # Now x has shape [batch, max_len, features]
        # Permute to [batch, features, max_len] for feature-wise processing
        x = x.permute(0, 2, 1)  # [batch, features, max_len]
        
        # Apply feature-wise projections
        feature_embeddings = []
        for i in range(features):
            feature_ts = x[:, i, :]  # [batch, max_len]
            feature_emb = self.feature_projections[i](feature_ts)  # [batch, d_model]
            feature_embeddings.append(feature_emb)
        
        # Stack to get [batch, features, d_model]
        value_embed = torch.stack(feature_embeddings, dim=1)
        
        # Add positional encoding for features
        x = value_embed + self.pos_scale * self.pe[:, :features, :]
        
        # Apply layer normalization and dropout
        x = self.layer_norm(x)
        return self.dropout(x)
