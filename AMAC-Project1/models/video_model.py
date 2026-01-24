import torch
import torch.nn as nn

class LipNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3,5,5), padding=(1,2,2)),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),

            nn.Conv3d(32, 64, kernel_size=(3,5,5), padding=(1,2,2)),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),

            nn.Conv3d(64, 96, kernel_size=(3,3,3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2))
        )

        self.rnn = nn.LSTM(96*8*8, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, 28)

    def forward(self, x):
        x = x.permute(0,2,1,3,4)
        x = self.conv(x)
        b,c,t,h,w = x.size()
        x = x.permute(0,2,1,3,4).contiguous().view(b,t,-1)
        x,_ = self.rnn(x)
        return self.fc(x)
