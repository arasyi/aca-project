#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Set random seed
torch.manual_seed(0)

# Define a simple linear dataset
class LinearDataset(Dataset):
    def __init__(self, num_samples, input_size):
        self.num_samples = num_samples
        self.input_size = input_size
        self.data = torch.randn(num_samples, input_size)
        self.targets = torch.randn(num_samples, 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Define a simple linear model
class LinearRegression(nn.Module):
    def __init__(self, input_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# Hyperparameters
num_samples = 1000
input_size = 10
batch_size = 32
learning_rate = 0.01
num_epochs = 1

# Create dataset and dataloader
dataset = LinearDataset(num_samples, input_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create model, loss function, and optimizer
model = LinearRegression(input_size).cuda()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')