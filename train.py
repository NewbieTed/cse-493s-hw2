# Note: Some of Andre Kaparthy's work in the GPT from scratch video was used as a reference
# To help when doing this assignment. https://www.youtube.com/watch?v=kCc8FmEb1nY

import sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import model as m
import random

import setup as s

torch.manual_seed(1)

# Hyperparameters
# ---------------
# TODO: move most (all?) of these hyperparameters inside of model config and access them that way
batch_size = s.batch_size
block_size = s.block_size
learning_rate = s.learning_rate
device = s.device
layers = s.layers
n_embd = s.n_embd
n_head = s.n_head
# ---------------
# data = s.data
# vocab_size = s.vocab_size

def get_loss(logits, targets):
    B, T, C = logits.shape
    assert B == batch_size and T == block_size
    assert B, T == targets.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)
    loss = F.cross_entropy(logits, targets)
    return loss

def get_batch(data):
    idx = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in idx])
    y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return x.to(device), y.to(device)

def train_step(model, optimizer, data):
    idx, targets = get_batch(data)
    logits = model(idx)
    loss = get_loss(logits, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss

def train_loop(model, optimizer, data, num_iters, print_interval):
    for i in range(num_iters):
        loss = train_step(model, optimizer, data)
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


    primes = ["97", "113"]
    opps = ["add", "sub", "div"]
    for prime in primes:
        for opp in opps:
            dataset = prime + opp + "test.txt"
            with open(dataset, 'r', encoding = 'utf-8') as file:
                text = file.read()

            chars = sorted(list(set(text)))
            vocab_size = len(chars)

            stoi = { ch:i for i,ch in enumerate(chars) }
            itos = { i:ch for i,ch in enumerate(chars) }
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])

            data = torch.tensor(encode(text), dtype=torch.long)

            # The random restarts
            for i in range(3):
                seed = random.randint(0,1000)
                torch.manual_seed(seed)
                lowest_loss = float('inf')

                # gives us training on all of the interested datasets
                log_file = open("Logs/" + prime + opp + str(i) + "training.log", "w")
                stdout = sys.stdout
                sys.stdout = log_file

                print(f"LR = {learning_rate}, batch_size = {batch_size}, block_size = {block_size}\n\n")

                # Model and optimizer instantiation
                config = m.GPTConfig(block_size=block_size, vocab_size=vocab_size, n_layer=layers, n_embd=n_embd, n_head=n_head)
                model = m.GPT(config)
                model = model.to(model.config.device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


                train_loop(model, optimizer, data, 1000, 100)
                loss = train_step(model, optimizer, data)
                if loss.item() < lowest_loss:
                    lowest_loss = loss.item()
                save_model(model, prime + opp + str(i) +  "model.pt")
                log_file.close()
                sys.stdout = stdout
                print("Done training on ", prime + opp + str(i))
                

    # This training loop is kinda arbitrary and will not converge on complicated txt files
    # loss = train_step(model, optimizer)
    # count = 0
    # while loss > 1e-6:
    #     loss = train_step(model, optimizer)
    #     if count % 10 == 0:
    #         print(f"Steps = {count}, loss = {loss.item()}")
    #     count += 1
    # print(f"Final loss: {loss.item()} at step {count}")
    # save_model(model, name)

    # log_file.close()
    # return

if __name__ == '__main__':
    main()
