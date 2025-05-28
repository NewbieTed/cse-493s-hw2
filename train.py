# Note: Some of Andre Kaparthy's work in the GPT from scratch video was used as a reference
# To help when doing this assignment. https://www.youtube.com/watch?v=kCc8FmEb1nY

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import model as m

import setup as s

torch.manual_seed(1)

# Hyperparameters
# ---------------
# TODO: move most (all?) of these hyperparameters inside of model config and access them that way
batch_size = s.batch_size
block_size = s.block_size
learning_rate = s.learning_rate
device = s.device
# ---------------
data = s.data
vocab_size = s.vocab_size

def get_loss(logits, targets):
    B, T, C = logits.shape
    assert B == batch_size and T == block_size
    assert B, T == targets.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)
    loss = F.cross_entropy(logits, targets)
    return loss

def train_step(model, optimizer):
    idx, targets = s.get_batch()
    logits = model(idx)
    loss = get_loss(logits, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss

def train_loop(model, optimizer, num_iters, print_interval):
    for i in range(num_iters):
        loss = train_step(model, optimizer)
        if i % print_interval == 0:
            print(f"Steps = {i}, loss = {loss.item()}")

def save_model(model, name):
    torch.save({
        'model_state_dict' : model.state_dict(),
        'config' : model.config
    }, name)

def main():
    argv = sys.argv
    if len(argv) == 1:
        name = 'model.pt'
    else:
        name = argv[1]

    # Model and optimizer instantiation
    config = m.GPTConfig(block_size=block_size, vocab_size=vocab_size)
    model = m.GPT(config)
    model = model.to(model.config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # This training loop is kinda arbitrary and will not converge on complicated txt files
    loss = train_step(model, optimizer)
    count = 0
    while loss > 1.5:
        loss = train_step(model, optimizer)
        if count % 100 == 0:
            print(f"Steps = {count}, loss = {loss.item()}")
        count += 1
    save_model(model, name)

if __name__ == '__main__':
    main()
