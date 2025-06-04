import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt_block.mini_gpt import model_init
from dataloader_v1 import create_dataloader_v1
from GPTV1.utils import (
    calc_loss_batch,
    calc_loss_loader,
    text_to_token_ids,
    token_ids_to_text,
    generate,
    evaluate_model,
    train_model_simple,
    generate_and_print_sample
)
import torch
from configuration.config import GPT_CONFIGS
import tiktoken

# Data loading and splitting
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data', 'the-verdict.txt')
with open(data_path, "r", encoding="utf-8") as f:
    text_data = f.read()
train_ratio = 0.70
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(221)

GPT_CONFIG = GPT_CONFIGS["gpt-test"]
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max=GPT_CONFIG["context_length"],
    stride=GPT_CONFIG["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max=GPT_CONFIG["context_length"],
    stride=GPT_CONFIG["context_length"],
    drop_last=True,
    shuffle=False,
    num_workers=0
)

# Print first batch from val_loader for inspection
data_iter = iter(val_loader)
for idx, i in enumerate(data_iter):
    print(f'{idx}th batch is --------{i}')
    break

# Device setup and model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(221)
# Move model to device
model = model_init(cfg=GPT_CONFIG).to(device)

# Initial loss calculation
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)


# Optimizer setup
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

# Tokenizer setup
start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

# Initial generation before training
token_ids = generate(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer).to(device),
    max_new_tokens=10,
    context_size=GPT_CONFIG["context_length"],
    temperature=0.0,
    top_k=None,
    eos_id=None
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# Training
num_epochs = 1
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

# Final generation after training
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=15,
    context_size=GPT_CONFIG["context_length"],
    top_k=25,
    temperature=1.4
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
