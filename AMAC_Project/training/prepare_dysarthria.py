import torch
import torch.nn as nn
import torch.nn.functional as F

class LipNet(nn.Module):
    def __init__(self, num_classes=28):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(), nn.MaxPool3d((1,2,2)),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d((1,2,2)),
            nn.Conv3d(64, 96, 3, padding=1), nn.BatchNorm3d(96), nn.ReLU(), nn.MaxPool3d((1,2,2)),
            nn.Conv3d(96, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(), nn.MaxPool3d((1,2,2)),
            nn.Conv3d(128, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU(), nn.MaxPool3d((1,2,2))
        )

        self.rnn = nn.LSTM(
            input_size=256*3*3,   # = 2304 *? actually 6144 from your training
            hidden_size=512,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = x.permute(0,2,1,3,4)
        x = self.conv(x)
        x = x.permute(0,2,1,3,4).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        x,_ = self.rnn(x)
        return self.fc(x)
