import torch
import torch.nn as nn

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(40, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

    def forward(self, x):
        # x: [B,40]
        return self.net(x)
