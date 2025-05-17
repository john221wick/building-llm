import torch
from torch.utils.data import Dataset, Dataloader

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max, stride):
        self.input_enc = []
        self.target_enc = []
        token_enc = tokenizer.encode(text)

        for i in range(0, len(token_enc)-max, stride):
            input_chunk = token_enc[i: i+max]
            target_chunk = token_enc[i+1: i+max+1]
            self.input_enc.append(torch.tensor(input_chunk))
            self.target_enc.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_enc)

    def __getitem__(self, idx):
        return self.input_enc[idx], self.target_enc[idx]
