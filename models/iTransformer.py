import sys
import os
# Ensure iTransformer directory is in sys.path for utils imports
#itransformer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'iTransformer'))
#if itransformer_dir not in sys.path:
#    sys.path.insert(0, itransformer_dir)

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
    
    Key differences from original:
    - Supports variable sequence lengths through adaptive embedding
    - Optimized for market data (OHLCV) features
    - Handles both sliding_window and full_day modes
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.temperature = getattr(configs, 'temperature', 0.0)

        # Get the maximum sequence length for initialization
        # For full_day mode, use max_seq_len; for sliding_window, use seq_len
        self.max_seq_len = getattr(configs, 'max_seq_len', 2000) if getattr(configs, 'mode', 'sliding_window') == 'full_day' else configs.seq_len
        self.mode = getattr(configs, 'mode', 'sliding_window')
        
        # Calculate total input features (data features + time features)
        # Time features are added as additional tokens in the embedding layer
        self.n_features = configs.enc_in
        
        # Embedding - the key insight of iTransformer is here
        # We embed from sequence_length to d_model for each feature
        # For variable length support, we'll handle padding/truncation in forward pass
        self.enc_embedding = DataEmbedding_inverted(
            self.max_seq_len,  # c_in = sequence length dimension 
            configs.d_model,   # Output dimension = model dimension
            configs.embed, 
            configs.freq,
            configs.dropout
        )

        # Encoder-only architecture (same as original)
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
        # Projects from d_model to pred_len for each feature
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, attn_mask=None, temperature=None):
        """
        Forecasting function that implements the iTransformer approach.
        
        Args:
            x_enc: [Batch, Time, Features] - Input sequences
            x_mark_enc: [Batch, Time, TimeFeatures] - Time features 
            x_dec: Not used in encoder-only architecture
            x_mark_dec: Not used in encoder-only architecture
            attn_mask: Attention mask for padded sequences
            temperature: Temperature for sampling (if specified)
        
        Returns:
            dec_out: [Batch, PredLen, Features] - Predictions
            attns: Attention weights (if output_attention=True)
        """
        batch_size, seq_len, n_features = x_enc.shape
        
        # Handle variable sequence lengths by padding/truncating to max_seq_len
        if seq_len != self.max_seq_len:
            if seq_len > self.max_seq_len:
                # Truncate if too long
                x_enc = x_enc[:, -self.max_seq_len:, :]
                if x_mark_enc is not None:
                    x_mark_enc = x_mark_enc[:, -self.max_seq_len:, :]
                if attn_mask is not None:
                    attn_mask = attn_mask[:, -self.max_seq_len:]
                seq_len = self.max_seq_len
            else:
                # Pad if too short - use zero padding at the beginning
                pad_length = self.max_seq_len - seq_len
                x_enc = F.pad(x_enc, (0, 0, pad_length, 0), value=0.0)
                if x_mark_enc is not None:
                    x_mark_enc = F.pad(x_mark_enc, (0, 0, pad_length, 0), value=0.0)
                if attn_mask is not None:
                    # For attention mask, pad with False (invalid tokens)
                    attn_mask = F.pad(attn_mask, (pad_length, 0), value=False)
                seq_len = self.max_seq_len

        # Apply normalization (Non-stationary Transformer approach)
        means, stdev = None, None
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # Get dimensions: B L N -> B N L (after permutation in embedding)
        B, L, N = x_enc.shape
        
        # Embedding: B L N -> B N E (key iTransformer transformation)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # Handle attention mask for encoder
        # In iTransformer, attention is over features, not time
        # So we don't need to modify the mask for feature-wise attention
        
        # Encoder processing: B N E -> B N E
        enc_out, attns = self.encoder(enc_out, attn_mask=None)  # No masking in feature space

        # Projection: B N E -> B N S
        dec_out = self.projector(enc_out)
        
        # Permute back to time-first: B N S -> B S N
        dec_out = dec_out.permute(0, 2, 1)
        
        # Only keep the original features (filter out time features if they were added)
        dec_out = dec_out[:, :, :N]

        # Apply temperature sampling if specified
        if temperature is not None and temperature > 0.0:
            dec_out = self.apply_temperature_sampling(dec_out, temperature)

        # De-normalization (if normalization was applied)
        if self.use_norm and means is not None and stdev is not None:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns

    def apply_temperature_sampling(self, logits, temperature):
        """Apply temperature sampling to the model outputs."""
        if temperature <= 0.0:
            return logits
        
        # Apply temperature scaling
        scaled_logits = logits / temperature
        
        # Add Gaussian noise scaled by temperature for regression tasks
        if self.training or temperature > 0.0:
            noise = torch.randn_like(scaled_logits) * temperature * 0.1
            return scaled_logits + noise
        
        return scaled_logits

    def set_temperature(self, temperature):
        """Set the temperature for inference sampling."""
        self.temperature = temperature

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, attn_mask=None, temperature=None):
        """
        Forward pass with compatibility for both original iTransformer signature 
        and custom features.
        
        Args:
            x_enc: Input sequences [Batch, Time, Features]
            x_mark_enc: Time features [Batch, Time, TimeFeatures] 
            x_dec: Decoder input (not used in encoder-only)
            x_mark_dec: Decoder time features (not used)
            mask: Legacy parameter for compatibility
            attn_mask: Attention mask for variable length sequences
            temperature: Temperature for sampling (overrides model default)
        
        Returns:
            Predictions [Batch, PredLen, Features]
        """
        # Use provided temperature or fall back to model default
        temp = temperature if temperature is not None else self.temperature
        
        # Forecast
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, attn_mask, temp)
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
