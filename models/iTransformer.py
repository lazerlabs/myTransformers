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
        """Get embeddings for input data"""
        batch_size, num_stocks, seq_len, features = x_enc.shape
        x_enc = x_enc.reshape(-1, seq_len, features)
        x_mark_enc = x_mark_enc.reshape(-1, x_mark_enc.shape[2], x_mark_enc.shape[3])

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Get embeddings from embedding layer
        embeddings = self.enc_embedding(x_enc, x_mark_enc)
        
        # Reshape back to include stock dimension
        embeddings = embeddings.reshape(batch_size, num_stocks, *embeddings.shape[1:])
        
        return embeddings

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Reshape input: [batch, stocks, seq_len, features] -> [batch*stocks, seq_len, features]
        batch_size, num_stocks, seq_len, features = x_enc.shape
        x_enc = x_enc.reshape(-1, seq_len, features)
        x_mark_enc = x_mark_enc.reshape(-1, x_mark_enc.shape[2], x_mark_enc.shape[3])

        # Remove debug prints
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Embedding
        # [batch*stocks, seq_len, features] -> [batch*stocks, features, d_model]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Projection
        dec_out = self.projection(enc_out).transpose(1, 2)

        # De-Normalization
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # Reshape back: [batch*stocks, pred_len, features] -> [batch, stocks, pred_len, features]
        dec_out = dec_out.reshape(batch_size, num_stocks, dec_out.shape[1], features)

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out  # [batch, stocks, pred_len, features] 