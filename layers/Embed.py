import torch
import torch.nn as nn
import math

class DataEmbedding_inverted(nn.Module):
    """
    Enhanced Data Embedding for stock market data with positional encoding
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, max_len=60):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.temporal_embedding = nn.Linear(c_in, d_model)
        self.feature_embedding = nn.Embedding(3, d_model)  # 3 features: close, volume, transactions
        
        # Add positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Add learnable scale parameter for positional encoding
        self.pos_scale = nn.Parameter(torch.ones(1))
        
        # Add layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # Compute temporal gradients with scale normalization
        temporal_grad = torch.gradient(x, dim=1)[0]
        grad_scale = torch.std(temporal_grad, dim=1, keepdim=True) + 1e-5
        temporal_grad = temporal_grad / grad_scale
        temporal_info = self.temporal_embedding(temporal_grad.permute(0, 2, 1))
        
        # Original value embedding
        x = x.permute(0, 2, 1)
        value_embed = self.value_embedding(x)
        
        # Feature embedding with learned importance
        feature_indices = torch.arange(x.size(1), device=x.device)
        feature_embed = self.feature_embedding(feature_indices)
        
        # Add scaled positional encoding
        x = value_embed + feature_embed.unsqueeze(0) + temporal_info + self.pos_scale * self.pe[:, :x.size(1)]
        
        # Apply layer normalization and dropout
        x = self.layer_norm(x)
        return self.dropout(x) 