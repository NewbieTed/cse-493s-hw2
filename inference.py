# Note: Some of Andre Kaparthy's work in the GPT from scratch video was used as a reference
# To help when doing this assignment. https://www.youtube.com/watch?v=kCc8FmEb1nY

import sys
import model as m
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import model as m
import setup as s

torch.manual_seed(1)

def load_model(name):
    torch.serialization.add_safe_globals([m.GPTConfig])
    saved_info = torch.load(name)
    model = m.GPT(saved_info['config'])
    model.load_state_dict(saved_info['model_state_dict'])
    return model.to(model.config.device)

def generate(model, idx, max_tokens):
    for _ in range(max_tokens):
        idx_cropped = idx[:, (-1 * model.config.block_size) :]
        logits = model(idx_cropped)
        logits= logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def generate_from_scratch(model, max_tokens):
    context = torch.zeros((1, 1), dtype=torch.long, device=model.config.device)
    result = generate(model, context, max_tokens)[0].tolist()
    return result

    
def main():
    argv = sys.argv
    if len(argv) != 3:
        print("Did not receive command line arguments for model and num tokens to generate")
        return
    model = load_model(argv[1])
    return print(s.decode(generate_from_scratch(model, int(argv[2]))))

if __name__ == '__main__':
    main()
