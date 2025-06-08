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

def get_batch_multi(lines, stoi, encode):
    x = []
    y = []

    i = 0
    while i < batch_size:
        ctr = 0
        context_examples = []
        while ctr < block_size:
            line = random.choice(lines).strip()
            context_examples.append(line + '\n')
            ctr += len(line + '\n')

        # remove the last 2 examples to leave room for the 'real' example
        context_examples = context_examples[:-2]
        # for ex in context_examples:

        # Finding the line we will do prediction on 
        line = random.choice(lines).strip()
        context_part, result_part = line.split('=')
        context = context_part.strip() + ' = '
        result = result_part.strip()
        full_string = context + result + '\n'

        result_start = len(context)
        result_end = len(full_string)

        # Want to sometimes include new line predictions
        cut_pos = random.randint(result_start, result_end - 1)

        x_str = full_string[:cut_pos]
        final_char = full_string[cut_pos]
        # y_str = full_string[1:cut_pos + 1]

        ex_str = ''.join(context_examples)

        full_str = ex_str + x_str
        if (len(ex_str) < block_size):
            pad_len = block_size - len(full_str)
            x_enc = [stoi[' ']] * pad_len + encode(full_str)
            y_enc = x_enc[1:] + [stoi[final_char]]
        else:
            print("Something peculiar happened")
            continue

        # print(full_str)
        # print(x_enc)
        # print(y_enc)

        x.append(torch.tensor(x_enc, dtype=torch.long))
        y.append(torch.tensor(y_enc, dtype=torch.long))

        i += 1

    return torch.stack(x).to(device), torch.stack(y).to(device)
            

# We try only training on things after the = sign
# def get_batch(lines, stoi, encode):
#     x = []
#     y = []

#     for _ in range(batch_size):
#         line = random.choice(lines).strip()
#         assert('=' in line)
#         try:
#             context_part, result_part = line.split('=')
#         except ValueError:
#             print("Something really bad happened")
#             break

#         context = context_part.strip() + ' = '
#         result = result_part.strip()

#         full_string = context + result  # full line, no \n
#         # print(context)
#         # print(result)

#         result_start = len(context)
#         result_end = len(full_string)

#         # Randomly choose a position to end the context (somewhere in the result)
#         cut_pos = random.randint(result_start, result_end - 1)

#         x_str = full_string[:cut_pos]           # everything up to (but not including) the char to predict
#         y_str = full_string[1:cut_pos + 1]      # same string but shifted by 1, target is next char

#         # print(x_str)
#         # print(y_str)

#         # Encode
#         x_enc = encode(x_str)
#         y_enc = encode(y_str)

#         # Pad if shorter than block_size
#         if len(x_enc) < block_size:
#             pad_len = block_size - len(x_enc)
#             x_enc = [stoi[' ']] * pad_len + x_enc
#             y_enc = [stoi[' ']] * pad_len + y_enc
#             # y_enc[pad_len - 1] = x_enc[pad_len]  # Want to make sure y stays 1 ahead of x
#         else:
#             x_enc = x_enc[-block_size:]
#             y_enc = y_enc[-block_size:]

#         # print(x_enc)
#         # print(y_enc)

#         x.append(torch.tensor(x_enc, dtype=torch.long))
#         y.append(torch.tensor(y_enc, dtype=torch.long))

#     return torch.stack(x).to(device), torch.stack(y).to(device)

def train_step(model, optimizer, lines, stoi, encode):
    idx, targets = get_batch_multi(lines, stoi, encode)
    logits = model(idx)
    loss = get_loss(logits, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss

def train_loop(model, optimizer, lines, num_iters, print_interval, stoi, encode):
    for i in range(num_iters):
        loss = train_step(model, optimizer, lines, stoi, encode)
        if i % print_interval == 0:
            print(f"Steps = {i}, loss = {loss.item()}")

def save_model(model, name):
    torch.save({
        'model_state_dict' : model.state_dict(),
        'config' : model.config
    }, name)

def test_on_batch(model, lines, stoi, encode):
    x, y = get_batch_multi(lines, stoi, encode)
    # Assume x is the right block size (it should be)
    # print(x, y)
    # print(len(x[0]))
    assert(len(x[0]) == block_size)
    logits = model(x)
    logits = logits[:, -1, :]
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs,num_samples=1)
    idx_next = idx_next.reshape((idx_next.shape[0]))
    y_pred = y[:, -1]
    return (idx_next == y_pred).sum().item()

def get_data(file_name):
    with open(file_name, 'r', encoding = 'utf-8') as f:
        text = f.read()
        f.seek(0)
        lines = f.readlines()
    return text, lines

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
            dataset = prime + opp + "train.txt"
            text,lines = get_data(dataset)

            chars = sorted(list(set(text)))
            vocab_size = len(chars)

            stoi = { ch:i for i,ch in enumerate(chars) }
            itos = { i:ch for i,ch in enumerate(chars) }
            encode = lambda s: [stoi[c] for c in s]
            decode = lambda l: ''.join([itos[i] for i in l])

            data = torch.tensor(encode(text), dtype=torch.long)

            # get_batch_multi(lines, stoi, encode)
            # continue

            # The random restarts
            for i in range(0,1):
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


                train_loop(model, optimizer, lines, 1000, 100, stoi, encode)
                loss = train_step(model, optimizer, lines, stoi, encode)
                if loss.item() < lowest_loss:
                    lowest_loss = loss.item()
                save_model(model, prime + opp + str(i) +  "model.pt")

                dataset = prime + opp + "val.txt"
                text, lines = get_data(dataset)
                count = 0
                iterations = 10
                for _ in range(iterations):
                    count += test_on_batch(model, lines, stoi, encode)
                print(f"Validation accuracy: {count/(batch_size * iterations)} (got {count} right out of {batch_size * iterations})")

                log_file.close()


                sys.stdout = stdout
                print("Done training on ", prime + opp + str(i))
                # test_on_batch(model, lines, stoi, encode)

if __name__ == '__main__':
    main()
