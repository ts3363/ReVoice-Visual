import torch
import torch.nn as nn

class AMACFusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.video_fc = nn.Linear(512, 256)
        self.audio_fc = nn.Linear(512, 256)

        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 30)
        )

    def forward(self, video_feat, audio_feat):
        v = self.video_fc(video_feat)
        a = self.audio_fc(audio_feat)
        fused = torch.cat([v, a], dim=1)
        return self.fusion(fused)
