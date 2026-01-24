import torch
import torch.nn as nn

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
        )

        self.lstm = None
        self.classifier = nn.Linear(512, 30)

    def forward(self, x):
        x = self.cnn(x)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # create LSTM dynamically
        if self.lstm is None:
            self.lstm = nn.LSTM(c * f, 256, bidirectional=True, batch_first=True).to(x.device)

        x, _ = self.lstm(x)
        return self.classifier(x)
