import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

class Model(nn.Module):
    """
    Stock Market Prediction using iTransformer
    Modified from: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # Embedding
        # In stock data, each time step contains multiple features (OHLCV)
        # We treat each feature as a token
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, 
            configs.d_model, 
            configs.embed, 
            configs.freq,
            configs.dropout
        )

        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, 
                            configs.factor, 
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ), 
                        configs.d_model, 
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Projection layer to predict future values
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def get_embeddings(self, x_enc, x_mark_enc):
        """Get embeddings for input data. Handles both 4D [batch, stocks, seq, features] and 3D [batch, seq, features] inputs."""
        # Check input dimensions
        if x_enc.ndim == 4:
            # Original 4D handling
            batch_size, num_stocks, seq_len, features = x_enc.shape
            x_enc_reshaped = x_enc.reshape(-1, seq_len, features)
            x_mark_enc_reshaped = x_mark_enc.reshape(-1, x_mark_enc.shape[2], x_mark_enc.shape[3])
            reshape_back = True
        elif x_enc.ndim == 3:
            # Handle 3D input (e.g., from test_run.py for a single sequence)
            batch_size, seq_len, features = x_enc.shape
            num_stocks = 1 # Assume single stock/sequence
            x_enc_reshaped = x_enc
            x_mark_enc_reshaped = x_mark_enc
            reshape_back = False # No need to reshape back if input was 3D
        else:
            raise ValueError(f"Unsupported input dimension for x_enc: {x_enc.ndim}. Expected 3 or 4.")

        # Apply normalization to the reshaped data
        if self.use_norm:
            means = x_enc_reshaped.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc_reshaped = x_enc_reshaped - means
            stdev = torch.sqrt(torch.var(x_enc_reshaped, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc_reshaped /= stdev
        else:
            # If not using norm, ensure we use the reshaped variable name
             stdev = None # Define stdev as None if not used, for consistency if needed later
             means = None # Define means as None if not used

        # Get embeddings from embedding layer using the potentially reshaped data
        embeddings = self.enc_embedding(x_enc_reshaped, x_mark_enc_reshaped)

        # Reshape back only if input was 4D
        if reshape_back:
            embeddings = embeddings.reshape(batch_size, num_stocks, *embeddings.shape[1:])
        # If input was 3D, embeddings shape is already [batch_size, features, d_model], which is fine.

        return embeddings

    def forecast(self, x_enc, x_mark_enc): # Removed x_dec, x_mark_dec
        # Handle potential 3D input [batch, seq, features] vs 4D [batch, stocks, seq, features]
        if x_enc.ndim == 4:
            # Original 4D handling
            batch_size, num_stocks, seq_len, features = x_enc.shape
            x_enc_reshaped = x_enc.reshape(-1, seq_len, features)
            x_mark_enc_reshaped = x_mark_enc.reshape(-1, x_mark_enc.shape[2], x_mark_enc.shape[3])
            reshape_back = True
        elif x_enc.ndim == 3:
            # Handle 3D input
            batch_size, seq_len, features = x_enc.shape
            num_stocks = 1 # Assume single stock/sequence
            x_enc_reshaped = x_enc
            x_mark_enc_reshaped = x_mark_enc # Assuming x_mark_enc is also 3D [batch, seq, time_features]
            reshape_back = False # No need to reshape back if input was 3D
        else:
            raise ValueError(f"Unsupported input dimension for x_enc in forecast: {x_enc.ndim}. Expected 3 or 4.")


        # Apply normalization to the reshaped data
        if self.use_norm:
            means = x_enc_reshaped.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc_reshaped = x_enc_reshaped - means
            stdev = torch.sqrt(torch.var(x_enc_reshaped, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc_reshaped /= stdev
        else:
             # Define stdev/means as None if not used, for de-normalization step
             stdev = None
             means = None

        # Embedding
        # Input shape: [batch*stocks or batch, seq_len, features]
        # Output shape: [batch*stocks or batch, features, d_model]
        enc_out = self.enc_embedding(x_enc_reshaped, x_mark_enc_reshaped)

        # Encoder
        # Input shape: [batch*stocks or batch, features, d_model]
        # Output shape: [batch*stocks or batch, features, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Projection
        # Input shape: [batch*stocks or batch, features, d_model]
        # Output shape: [batch*stocks or batch, pred_len, features] (after transpose)
        dec_out = self.projection(enc_out).transpose(1, 2) # Transpose to [batch*stocks or batch, pred_len, features]

        # De-Normalization
        if self.use_norm and means is not None and stdev is not None:
             # Ensure means/stdev are correctly shaped for broadcasting
             # Original shape was [batch*stocks or batch, 1, features], need means/stdev[:, 0, :] -> [batch*stocks or batch, features]
             stdev_reshaped = stdev[:, 0, :] # Shape: [batch*stocks or batch, features]
             means_reshaped = means[:, 0, :] # Shape: [batch*stocks or batch, features]
             dec_out = dec_out * stdev_reshaped.unsqueeze(1).repeat(1, self.pred_len, 1)
             dec_out = dec_out + means_reshaped.unsqueeze(1).repeat(1, self.pred_len, 1)

        # Reshape back only if input was 4D
        if reshape_back:
             # Reshape back: [batch*stocks, pred_len, features] -> [batch, stocks, pred_len, features]
             dec_out = dec_out.reshape(batch_size, num_stocks, self.pred_len, features)
        # If input was 3D, dec_out shape is already [batch_size, pred_len, features]

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out

    def forward(self, x_enc, x_mark_enc, mask=None): # Removed x_dec, x_mark_dec
        # Note: The first argument to forecast was already updated in the previous step
        dec_out = self.forecast(x_enc, x_mark_enc) # Removed x_dec, x_mark_dec from call
        return dec_out  # [batch, stocks, pred_len, features]