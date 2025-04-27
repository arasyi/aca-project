#!/usr/bin/env python3

import torch

# Seed to ensure same results each run
torch.manual_seed(1234)

size = 16
runs = 100
device = 'cuda'

for i in range(runs):
    x = torch.rand(size, size, device=device)
    y = torch.rand(size, size, device=device)
    z = x @ y

    print(f"#{ i }. {z.to('cpu') = }")

print('Done')
