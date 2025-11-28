import torch
import torch.nn as nn

class StudentPolicy(nn.Module):
    def __init__(self, state_dim=4, action_dim=5, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
    def forward(self, x):
        return self.net(x)  # logits