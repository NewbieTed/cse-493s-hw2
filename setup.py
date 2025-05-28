import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import model as m

torch.manual_seed(1)

# Hyper parameters
batch_size = 4 # Num of blocks to process in parallel
block_size = 32 # size of block (number of characters) (aka sequence length)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-3
dataset = 'aaa.txt'
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

def get_batch():
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return x.to(device), y.to(device)
