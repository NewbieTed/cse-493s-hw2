# Note: Some of Andre Kaparthy's work in the GPT from scratch video was used as a reference
# To help when doing this assignment. https://www.youtube.com/watch?v=kCc8FmEb1nY

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import model as m

torch.manual_seed(1)

# Hyperparameters
# ---------------
batch_size = 4 # Num of blocks to process in parallel
block_size = 32 # size of block (number of characters) (aka sequence length)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-3
# dataset = 'input.txt'
dataset = 'shakespeare.txt'
# ---------------

# Data initialization and encoding
with open(dataset, 'r', encoding = 'utf-8') as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# Model and optimizer instantiation
config = m.GPTConfig(block_size=block_size, vocab_size=vocab_size)
model = m.GPT(config)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

def get_batch():
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return x.to(device), y.to(device)

def get_loss(logits, targets):
    B, T, C = logits.shape
    assert B == batch_size and T == block_size
    assert B, T == targets.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)
    loss = F.cross_entropy(logits, targets)
    return loss

def train_loop(num_iters, print_interval):
    for i in range(num_iters):
        idx, targets = get_batch()
        logits = model(idx)
        loss = get_loss(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if i % print_interval == 0:
            print(f"Steps = {i}, loss = {loss.item()}")

