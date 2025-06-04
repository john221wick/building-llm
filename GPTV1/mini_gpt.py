import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt_block.transformer_block import TransformerBlock, LayerNorm
from gpt_block.utils import calcSize, testFunc, testOurModel
from configuration.config import GPT_CONFIGS

cfg = GPT_CONFIGS["gpt-test"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GPTmodel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )


    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_embedding(in_idx)
        pos_embeds = self.pos_embedding(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds

        x = self.dropout(x)

        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
def model_init(cfg):
    model = GPTmodel(cfg)
    model = model.to(device)
    return model

# model = model_init(cfg=cfg)

# # testing the model
# testFunc(model = model)
# # Calculating the size of the model
# calcSize(model = model)
# # For testing our gpt model
# testOurModel(model = model, context_size = cfg["context_length"])
