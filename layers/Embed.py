import torch
import torch.nn as nn

class DataEmbedding_inverted(nn.Module):
    """
    Data Embedding for stock market data
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.feature_embedding = nn.Embedding(3, d_model)  # 3 features: close, volume, transactions
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x: [Batch, Time, Features] -> [Batch, Features, Time]
        x = x.permute(0, 2, 1)
        
        # Create feature indices tensor
        feature_indices = torch.arange(x.size(1), device=x.device)
        
        # Get feature embeddings
        feature_emb = self.feature_embedding(feature_indices)
        
        # Combine value and feature embeddings
        x = self.value_embedding(x)
        x = x + feature_emb.unsqueeze(0)  # Broadcasting feature embeddings
        
        return self.dropout(x) 