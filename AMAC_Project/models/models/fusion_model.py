import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, lip_model, audio_model):
        super().__init__()
        self.lip = lip_model
        self.audio = audio_model
        self.attn = nn.MultiheadAttention(512, 4, batch_first=True)
        self.fc = nn.Linear(512, 28)

    def forward(self, video, audio):
        v = self.lip(video)      # [B,T,512]
        a = self.audio(audio)   # [B,T,256]

        fused = torch.cat([v,a], dim=2)
        fused,_ = self.attn(fused, fused, fused)
        return self.fc(fused)
