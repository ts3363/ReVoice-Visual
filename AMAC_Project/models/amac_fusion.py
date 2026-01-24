import torch
import torch.nn as nn

class AMACFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 70 video + 40 audio = 110
        self.fusion_fc = nn.Sequential(
            nn.Linear(110, 256),
            nn.ReLU(),
            nn.Linear(256, 30)
        )

    def forward(self, v, a):
        fused = torch.cat([v, a], dim=2)  # [B,T,110]
        return self.fusion_fc(fused)
