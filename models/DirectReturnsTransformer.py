import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer

class Model(nn.Module):
    """
    Direct Returns Inverted Transformer for Stock Market Prediction
    
    Key Innovation: Combines two powerful approaches:
    1. Returns-based preprocessing (our contribution)
    2. Inverted transformer architecture (iTransformer approach)
    
    Architecture Flow:
    Returns [B, T, 1] → Permute [B, 1, T] → Embed to [B, 1, d_model] → 
    Attention over Features → Project to [B, 1, pred_len] → Permute [B, pred_len, 1]
    
    Benefits:
    - Memory efficient: O(features²) instead of O(time²) 
    - Semantic alignment: input returns = output returns
    - Perfect for univariate data: 1×1 attention matrix instead of 944×944
    - Maintains temporal information through embedding, not attention
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        self.temperature = getattr(configs, 'temperature', 0.0)
        self.d_model = configs.d_model

        # Get the maximum sequence length for variable length support
        self.max_seq_len = getattr(configs, 'max_seq_len', 2000) if getattr(configs, 'mode', 'sliding_window') == 'full_day' else configs.seq_len
        self.mode = getattr(configs, 'mode', 'sliding_window')
        
        # For univariate stock prediction, we always have 1 feature (close price returns)
        self.n_features = 1
        
        # INVERTED TRANSFORMER ARCHITECTURE:
        # Instead of embedding time steps, we embed the entire time series of each feature
        # Input: [Batch, Features=1, Time] - the whole time series becomes the "token"
        # This is the key insight from iTransformer paper
        
        # Embedding layer: maps from sequence_length to d_model for each feature
        # For univariate data, we have 1 feature, so attention is 1×1 (extremely efficient!)
        self.variate_embedding = nn.Linear(self.max_seq_len, self.d_model)
        
        # Projection layer: maps from d_model to prediction_length for each feature  
        self.projector = nn.Linear(self.d_model, self.pred_len, bias=True)

        # Standard Transformer Encoder (same as iTransformer)
        # But now attention is over FEATURES (1×1) instead of TIME (944×944)
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
                        self.d_model, 
                        configs.n_heads
                    ),
                    self.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, attn_mask=None, temperature=None):
        """
        Inverted transformer forecasting with returns.
        
        Args:
            x_enc: [Batch, Time, 1] - Returns sequences 
            x_mark_enc: [Batch, Time, TimeFeatures] - Time features (not used)
            x_dec: Not used in encoder-only architecture
            x_mark_dec: Not used in encoder-only architecture
            attn_mask: Not needed for feature-wise attention
            temperature: Temperature for sampling
        
        Returns:
            dec_out: [Batch, PredLen, 1] - Predicted returns
            attns: Attention weights (if output_attention=True)
        """
        batch_size, seq_len, n_features = x_enc.shape
        
        # Ensure we're working with univariate data
        if n_features != 1:
            print(f"Warning: Expected 1 feature (close price), got {n_features}. Using first feature only.")
            x_enc = x_enc[:, :, :1]  # Take only first feature (close)
        
        # Handle variable sequence lengths by padding/truncating to max_seq_len
        if seq_len != self.max_seq_len:
            if seq_len > self.max_seq_len:
                # Truncate if too long (keep most recent data)
                x_enc = x_enc[:, -self.max_seq_len:, :]
                seq_len = self.max_seq_len
            else:
                # Pad if too short - use zero padding at the beginning
                pad_length = self.max_seq_len - seq_len
                x_enc = F.pad(x_enc, (0, 0, pad_length, 0), value=0.0)
                seq_len = self.max_seq_len

        # Apply normalization if enabled (Non-stationary Transformer approach)
        means, stdev = None, None
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        # CRITICAL: INVERTED TRANSFORMER PERMUTATION
        # Change from [Batch, Time, Features] to [Batch, Features, Time]
        # This makes each feature's entire time series a "token"
        # x_enc: [Batch, Time=944, Features=1] → [Batch, Features=1, Time=944]
        x = x_enc.permute(0, 2, 1)  # [B, T, F] → [B, F, T]
        
        # Now x: [Batch, Features=1, Time=944]
        # Each "token" is the entire time series of one feature
        
        # Embedding: Map entire time series to d_model
        # x: [Batch, Features=1, Time=944] → [Batch, Features=1, d_model]
        enc_out = self.variate_embedding(x)
        
        # Transformer encoding over FEATURES (not time!)
        # For univariate data: attention is 1×1 instead of 944×944!
        # enc_out: [Batch, Features=1, d_model] → [Batch, Features=1, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)  # No masking needed for features
        
        # Projection: Map from d_model to prediction length
        # enc_out: [Batch, Features=1, d_model] → [Batch, Features=1, pred_len]
        dec_out = self.projector(enc_out)
        
        # Permute back to time-first format
        # dec_out: [Batch, Features=1, pred_len] → [Batch, pred_len, Features=1]
        dec_out = dec_out.permute(0, 2, 1)
        
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
        Forward pass compatible with existing training pipeline.
        
        Args:
            x_enc: Returns sequences [Batch, Time, 1]
            x_mark_enc: Time features (not used in this simplified version)
            x_dec: Decoder input (not used in encoder-only)
            x_mark_dec: Decoder time features (not used)
            mask: Legacy parameter for compatibility
            attn_mask: Not needed for feature-wise attention
            temperature: Temperature for sampling
        
        Returns:
            Predicted returns [Batch, PredLen, 1]
        """
        # Use provided temperature or fall back to model default
        temp = temperature if temperature is not None else self.temperature
        
        # Forecast
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, attn_mask, temp)
        
        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out


class PositionalEncoding(nn.Module):
    """
    Standard positional encoding for transformer.
    Since we're treating returns as direct embeddings, we still need 
    positional information to understand temporal relationships.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [Batch, Seq, d_model]
        Returns:
            x + positional encoding: [Batch, Seq, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)  # type: ignore 
