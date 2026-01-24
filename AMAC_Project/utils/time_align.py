import torch
import torch.nn.functional as F

def align_time(audio_feat, T):
    # audio_feat: [B, Ta, 512]
    audio_feat = audio_feat.permute(0,2,1)          # [B,512,Ta]
    audio_feat = F.interpolate(audio_feat, size=T, mode="linear", align_corners=False)
    return audio_feat.permute(0,2,1)                 # [B,T,512]
