GPT_CONFIGS = {
    "gpt-small": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    },
    "gpt-big": {
        "vocab_size": 50257,
        "context_length": 2048,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "drop_rate": 0.1,
        "qkv_bias": True
    },
    "gpt-test": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 2,
        "n_layers": 1,
        "drop_rate": 0.1,
        "qkv_bias": False
    }
}
