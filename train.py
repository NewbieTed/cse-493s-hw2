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
import matplotlib.pyplot as plt

torch.manual_seed(1)

# Hyperparameters
# ---------------
batch_size = s.batch_size
block_size = s.block_size
learning_rate = s.learning_rate
device = s.device
layers = s.layers
n_embd = s.n_embd
n_head = s.n_head
# ---------------

diag = False

# Gets the loss on some targets
def get_loss(logits, targets):
    B, T, C = logits.shape
    assert B == batch_size and T == block_size
    assert (B, T) == targets.shape
    # logits = logits.view(B*T, C)
    # targets = targets.view(B*T)
    logits = logits[:, -1, :]
    targets = targets[:, -1]
    loss = F.cross_entropy(logits, targets)
    return loss

# Return a full batch for our data
# Consists of blocks of lines with the result c as the target
def get_batch_multi(lines, stoi, encode):
    x = []
    y = []
    for _ in range(batch_size):
        line = random.choice(lines).strip()
        context_part , result_part = line.split('=')
        context = ' ' + context_part.strip() + ' = '
        result = result_part.strip()

        if diag:
            print(context)
            print(result)

        assert(len(encode(result)) == 1)

        x_tokens = encode(context)
        y_tokens = x_tokens[1:] + encode(result)
        if diag:
            print(x_tokens)
            print(y_tokens)
            print(len(x_tokens))
        assert(len(x_tokens) == block_size)
        assert(len(y_tokens) == block_size)
        x.append(torch.tensor(x_tokens, dtype=torch.long))
        y.append(torch.tensor(y_tokens, dtype=torch.long))

    return torch.stack(x).to(device), torch.stack(y).to(device)

# Do a training step
def train_step(model, optimizer, lines, stoi, encode):
    idx, targets = get_batch_multi(lines, stoi, encode)
    logits = model(idx)
    loss = get_loss(logits, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss

import re

def split_str(s):
    return re.findall(r'\d+|\D', s)

# Returns our vocabulary and necessary information for encoding/decoding
def build_vocab_and_tokenizer(text, max_num):
    # Get all unique tokens (numbers and single characters) from training data
    all_tokens = []
    for line in text.split('\n'):
        if line.strip():
            tokens = split_str(line)
            all_tokens.extend(tokens)
            break
    
    # Separate numbers from non-digit tokens
    non_digit_tokens = set()
    
    for token in all_tokens:
        if not token.isdigit():
            non_digit_tokens.add(token)
    
    # Add all possible numbers (specified by param)
    all_numbers = {str(i) for i in range(max_num)}
    
    # Combine all tokens: all possible numbers + non-digit characters from data
    vocab_tokens = list(set(all_numbers)) + list(set(non_digit_tokens)) + ['\n']
    vocab = sorted(vocab_tokens)
    vocab_size = len(vocab)

    
    # Create mappings
    stoi = {token: i for i, token in enumerate(vocab)}
    itos = {i: token for i, token in enumerate(vocab)}
    
    # Create encode/decode functions
    def encode(s):
        tokens = split_str(s)
        return [stoi[token] for token in tokens]
    
    def decode(token_ids):
        tokens = [itos[i] for i in token_ids]
        return ''.join(tokens)

    # if diag:
    #     print(f"vocab: {vocab}")
    #     print(f"encoded vocab: {encode(vocab)}")
    
    return vocab_size, stoi, itos, encode, decode, non_digit_tokens

def train_loop(model, optimizer, lines, num_iters, print_interval, stoi, encode):
    loss_list = []
    for i in range(num_iters):
        loss = train_step(model, optimizer, lines, stoi, encode)
        if i % print_interval == 0:
            print(f"Steps = {i}, loss = {loss.item()}")
            loss_list.append((i, loss.item()))
    return loss_list

def save_model(model, name):
    torch.save({
        'model_state_dict' : model.state_dict(),
        'config' : model.config
    }, name)

# Return the number of correct predictions on a batch
def test_on_batch(model, lines, stoi, encode):
    x, y = get_batch_multi(lines, stoi, encode)
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

# Contains our training loop across combinations of operations and primes
def main():
    print(torch.cuda.is_available())
    if not torch.cuda.is_available():
        print(torch.cuda.is_available())
        return

    argv = sys.argv
    if len(argv) == 1:
        name = 'model.pt'
    else:
        name = argv[1]


    primes = ["97", "113"]
    opps = ["add", "sub", "div"]
    done = False
    for layer in [1,2]:
        for prime in primes:
            if done:
                break
            for opp in opps[:2]:
                if done:
                    break
                dataset = prime + opp + "train.txt"
                text,lines = get_data(dataset)

                vocab_size, stoi, itos, encode, decode, non_digit_tokens = build_vocab_and_tokenizer(text, int(prime))

                # The random restarts
                for i in range(0,3):
                    seed = random.randint(0,1000)
                    torch.manual_seed(seed)
                    lowest_loss = float('inf')

                    # gives us training on all of the interested datasets
                    model_name = prime + opp + str(i) + "layer" + str(layer)
                    log_file = open("Logs/" + model_name + "training.log", "w")
                    stdout = sys.stdout
                    sys.stdout = log_file

                    print(f"LR = {learning_rate}, batch_size = {batch_size}, block_size = {block_size}\n\n")

                    # Model and optimizer instantiation
                    config = m.GPTConfig(block_size=block_size, vocab_size=vocab_size, n_layer=layer, n_embd=n_embd, n_head=n_head)
                    model = m.GPT(config)
                    model = model.to(model.config.device)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


                    loss_list = train_loop(model, optimizer, lines, 20010, 100, stoi, encode)
                    loss = train_step(model, optimizer, lines, stoi, encode)
                    if loss.item() < lowest_loss:
                        lowest_loss = loss.item()
                    save_model(model, model_name +  "model.pt")

                    dataset = prime + opp + "test.txt"
                    text, lines = get_data(dataset)
                    count = 0
                    iterations = 3
                    for _ in range(iterations):
                        count += test_on_batch(model, lines, stoi, encode)
                    accuracy = count / (batch_size * iterations)
                    print(f"Test accuracy: {count/(batch_size * iterations)} (got {count} right out of {batch_size * iterations})")

                    log_file.close()


                    sys.stdout = stdout
                    print("Done training on ", prime + opp + str(i))

                    # Unpack loss values
                    steps, losses = zip(*loss_list)

                    # Line plot for loss
                    plt.figure(figsize=(10, 5))
                    plt.plot(steps, losses, marker='o', linestyle='-')
                    plt.xlabel("Training Steps")
                    plt.ylabel("Loss")
                    plt.title("Loss Over Time")
                    plt.grid(True)
                    plt.savefig(model_name + "loss_plot.png")
                    plt.show()

                    # Bar plot for final validation accuracy
                    plt.figure(figsize=(5, 5))
                    plt.bar(["Test Accuracy"], [accuracy], color="skyblue")
                    plt.ylim(0, 1)
                    plt.title("Final Test Accuracy")
                    plt.ylabel("Accuracy")
                    plt.savefig(model_name + "test_acc.png")
                    plt.show()
                    # done = True
                    # break

if __name__ == '__main__':
    main()
