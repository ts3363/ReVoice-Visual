import torch
import torch.nn as nn

class AudioNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(40, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, 512)

    def forward(self, x):
        # x: [B,T,40]
        x,_ = self.rnn(x)
        return self.fc(x)   # [B,T,512]
