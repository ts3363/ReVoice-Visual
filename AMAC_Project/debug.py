import torch
from models.video_model import LipNet

m = LipNet()
fake = torch.randn(1,75,1,64,64)
f = m.extract_features(fake)
print("FEATURE SHAPE:", f.shape)
