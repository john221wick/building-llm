import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

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

def create_dataloader_v1(text, batch_size = 4, max = 256, stride = 128, shuffle = False, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, max, stride)
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle= shuffle,
        num_workers=num_workers
    )
    return dataloader

with open("../data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(text=raw_text, batch_size=1, max=4, stride=1)
data_iter = iter(dataloader)
for i in range(0, 4):
    batches = next(data_iter)
    print(f'{i}th batch is --------')
    print(batches)
