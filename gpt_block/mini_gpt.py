import torch
import torch.nn as nn

from attention_mechanism.MultiheadAttention import MultiHeadAttention

cfg = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}


class GPTmodel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # tok_embeddings, positional_embedding, dropout
        self.tok_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = cfg["drop_rate"]



    def forward(self, x):
