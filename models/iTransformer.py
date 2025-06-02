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
        self.temperature = getattr(configs, 'temperature', 0.0)  # Default to 0.0 if not present

        # Embedding
        # In stock data, each time step contains multiple features (OHLCV)
        # We treat each feature as a token
        # Use max_seq_len for full_day mode or seq_len for sliding_window mode
        max_len = getattr(configs, 'max_seq_len', 2000) if getattr(configs, 'mode', 'sliding_window') == 'full_day' else configs.seq_len
        self.enc_embedding = DataEmbedding_inverted(
            configs.enc_in,  # Number of input features, not sequence length
            configs.d_model, 
            configs.embed, 
            configs.freq,
            configs.dropout,
            max_len=max_len
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

    def apply_temperature_sampling(self, logits, temperature):
        """
        Apply temperature sampling to the model outputs.
        
        Args:
            logits: Model outputs before final activation
            temperature: Temperature parameter (0.0 = deterministic, >0 = stochastic)
            
        Returns:
            Sampled outputs with temperature applied
        """
        if temperature <= 0.0:
            # Deterministic mode - return logits as-is
            return logits
        
        # Apply temperature scaling
        scaled_logits = logits / temperature
        
        # Add Gaussian noise scaled by temperature for regression tasks
        if self.training or temperature > 0.0:
            noise = torch.randn_like(scaled_logits) * temperature * 0.1
            return scaled_logits + noise
        
        return scaled_logits

    def set_temperature(self, temperature):
        """
        Set the temperature for inference sampling.
        
        Args:
            temperature (float): Temperature value (0.0 = deterministic, >0 = stochastic)
        """
        self.temperature = temperature

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, attn_mask=None, temperature=None):
        # Handle potential 3D input [batch, seq, features] vs 4D [batch, stocks, seq, features]
        if x_enc.ndim == 4:
            # Original 4D handling
            batch_size, num_stocks, seq_len, features = x_enc.shape
            x_enc_reshaped = x_enc.reshape(-1, seq_len, features)
            x_mark_enc_reshaped = x_mark_enc.reshape(-1, x_mark_enc.shape[2], x_mark_enc.shape[3])
            
            # Reshape attention mask if provided
            if attn_mask is not None:
                attn_mask_reshaped = attn_mask.reshape(-1, attn_mask.shape[-1])  # [batch*stocks, seq_len]
            else:
                attn_mask_reshaped = None
            reshape_back = True
        elif x_enc.ndim == 3:
            # Handle 3D input
            batch_size, seq_len, features = x_enc.shape
            num_stocks = 1 # Assume single stock/sequence
            x_enc_reshaped = x_enc
            x_mark_enc_reshaped = x_mark_enc # Assuming x_mark_enc is also 3D [batch, seq, time_features]
            attn_mask_reshaped = attn_mask  # Pass through as-is for 3D input
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

        # Get the actual number of features from the data
        _, _, N = x_enc_reshaped.shape # B L N 
        # B: batch_size; E: d_model; 
        # L: seq_len; S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc_reshaped, x_mark_enc_reshaped) # covariates (e.g timestamp) can be also embedded as tokens
        
        # Create attention mask for the encoder if provided
        # The iTransformer inverts the dimensions, so we need to handle this carefully
        encoder_attn_mask = None
        if attn_mask_reshaped is not None:
            # For iTransformer, we're doing feature-wise attention not time-wise
            # The attention mask [batch, seq_len] indicates which time steps are valid
            # Since we embed across time dimension, we need to convert this to feature space
            # For now, we'll disable masking in feature space as it doesn't directly apply
            # The masking was primarily for handling variable sequence lengths in time dimension
            # which is handled during embedding/preprocessing
            encoder_attn_mask = None  # Disable for feature-wise attention
            
            # Note: If specific feature masking is needed, it would require a different approach
            # than temporal masking used in vanilla transformers
            
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=encoder_attn_mask)

        # B N E -> B N S -> B S N 
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates
        
        # Apply temperature sampling if enabled
        if temperature is None:
            temperature = self.temperature
        dec_out = self.apply_temperature_sampling(dec_out, temperature)

        # De-Normalization
        if self.use_norm and means is not None and stdev is not None:
             # De-Normalization from Non-stationary Transformer
             dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
             dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # Reshape back only if input was 4D
        if reshape_back:
             # Reshape back: [batch*stocks, pred_len, features] -> [batch, stocks, pred_len, features]
             dec_out = dec_out.reshape(batch_size, num_stocks, self.pred_len, features)
        # If input was 3D, dec_out shape is already [batch_size, pred_len, features]

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, attn_mask=None, temperature=None):
        # Note: Added compatibility with original iTransformer signature while keeping custom features
        # x_dec and x_mark_dec are expected by original iTransformer but not used in encoder-only architecture
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, attn_mask=attn_mask, temperature=temperature)
        return dec_out  # [batch, stocks, pred_len, features] or [batch, pred_len, features]
