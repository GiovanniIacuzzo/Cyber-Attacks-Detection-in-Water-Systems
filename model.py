# model.py

import torch
import torch.nn as nn

class DeepClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout, output_size):
        super(DeepClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[2], output_size)
        )

    def forward(self, x):
        return self.model(x)

