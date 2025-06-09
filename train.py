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
batch_size = s.batch_size
block_size = s.block_size
learning_rate = s.learning_rate
device = s.device
layers = s.layers
n_embd = s.n_embd
n_head = s.n_head
# ---------------

diag = False

def get_loss(logits, targets):
    B, T, C = logits.shape
    assert B == batch_size and T == block_size
    assert (B, T) == targets.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)
    # logits = logits[:, -1, :]
    # targets = targets[:, -1]
    loss = F.cross_entropy(logits, targets)
    return loss

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

# def get_batch_multi(lines, stoi, encode):
#     x = []
#     y = []

#     i = 0
#     while i < batch_size:
#         ctr = 0
#         context_examples = []
#         while ctr < block_size:
#             line = random.choice(lines).strip()
#             context_examples.append(line + '\n')
#             ctr += len(encode(line + '\n'))

#         # remove the last 2 examples to leave room for the 'real' example
#         context_examples = context_examples[:-2]

#         # Finding the line we will do prediction on 
#         line = random.choice(lines).strip()
#         context_part, result_part = line.split('=')
#         context = context_part.strip() + ' = '
#         result = result_part.strip()
#         full_string = context + result + '\n'
#         if diag:
#             print(f"context_part: {context_part}, result_part: {result_part}")

#         full_tokens = encode(full_string)
#         context_tokens = encode(context)

#         result_start = len(context_tokens)
#         result_end = len(full_tokens)

#         cut_pos = result_start # result_end -1
#         if diag:
#             print(full_string)
#             print(full_tokens)
#             print(f"Result start: {result_start}, result_end - 1: {result_end - 1}")
#             print(f"Result start token: {full_tokens[result_start]}, result_end - 1 token: {full_tokens[result_end -1]}")
#             print(f"Cut_pos: {cut_pos}")

#         x_tokens = full_tokens[:cut_pos]
#         final_token = full_tokens[cut_pos]

#         if diag:
#             print(f"x_tokens: {x_tokens}, final_token: {final_token}")

#         ex_str = ''.join(context_examples)
#         ex_tokens = encode(ex_str)

#         full_tokens_with_context = ex_tokens + x_tokens
#         if (len(full_tokens_with_context) < block_size):
#             pad_len = block_size - len(full_tokens_with_context)
#         else:
#             pad_len = 0
#             print("Something peculiar happened")
#             # continue
#         x_enc = [stoi[' ']] * pad_len + full_tokens_with_context
#         y_enc = x_enc[1:] + [final_token]

#         if diag:
#             print(f"x_enc: {x_enc}, y_enc = {y_enc}")

#         x.append(torch.tensor(x_enc, dtype=torch.long))
#         y.append(torch.tensor(y_enc, dtype=torch.long))

#         i += 1

    # return torch.stack(x).to(device), torch.stack(y).to(device)
            

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
    done = False
    for prime in primes:
        if done:
            break
        for opp in opps:
            if done:
                break
            dataset = prime + opp + "train.txt"
            text,lines = get_data(dataset)

            vocab_size, stoi, itos, encode, decode, non_digit_tokens = build_vocab_and_tokenizer(text, int(prime))

            # x, y = get_batch_multi(lines, stoi, encode)
            # print(decode(x[0].tolist()))
            # print(decode([y[0][-1].item()]))
            # done = True
            # break

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


                train_loop(model, optimizer, lines, 5000, 100, stoi, encode)
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
                done = True
                break

if __name__ == '__main__':
    main()
