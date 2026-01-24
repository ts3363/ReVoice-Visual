import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from models.video_model import LipNet
from models.audio_model import AudioNet
from models.amac_fusion import AMACFusion

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

v = LipNet().to(DEVICE)
a = AudioNet().to(DEVICE)
f = AMACFusion().to(DEVICE)

fake_v = torch.randn(8,75,1,64,64).to(DEVICE)
fake_a = torch.randn(8,200,40).to(DEVICE)

out = f(v.extract_features(fake_v), a(fake_a))
print("Fusion Output Shape:", out.shape)
