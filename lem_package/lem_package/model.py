"""
Model definitions for the Large Event Model (LEM).
"""

import torch
import torch.nn as nn

class EventTransformerModel(nn.Module):
    """
    Transformer-based model for event sequence processing.
    """
    def __init__(self, input_dim=7, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # Ensure input has the right shape
        if x.shape[-1] != 7:
            # Pad or truncate to 7 features
            if x.shape[-1] < 7:
                # Pad with zeros
                padding = torch.zeros(*x.shape[:-1], 7 - x.shape[-1], device=x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                # Truncate
                x = x[..., :7]
                
        x = self.input_proj(x)
        embeddings = self.transformer_encoder(x)
        pooled_embedding = embeddings.mean(dim=1)
        return self.output_proj(pooled_embedding)

class EventEmbeddingModel(nn.Module):
    """
    Transformer-based model for generating event embeddings.
    """
    def __init__(self, input_dim, embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.input_proj(x)
        embeddings = self.transformer(x)
        return embeddings.mean(dim=1)

class LSTMEmbeddingModel(nn.Module):
    """
    LSTM-based model for sequence processing.
    """
    def __init__(self, input_dim=8, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # Ensure input has the right shape
        if x.shape[-1] != 8:
            # Pad or truncate to 8 features
            if x.shape[-1] < 8:
                # Pad with zeros
                padding = torch.zeros(*x.shape[:-1], 8 - x.shape[-1], device=x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                # Truncate
                x = x[..., :8]
        
        x = self.encoder(x)
        x, _ = self.lstm(x)
        # Take the last output for classification/embedding
        return x[:, -1, :] 