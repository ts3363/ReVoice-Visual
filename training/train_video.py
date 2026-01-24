from training.video_dataset import VideoDataset
from models.video_model import LipEncoder
import torch, torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

ds = VideoDataset()
model = LipEncoder().to(device)
opt = optim.Adam(model.parameters(),1e-4)

print("Training Lip Model...")

for e in range(10):
    tot=0
    for x,y in ds:
        x = x.unsqueeze(0).to(device)
        out = model(x)
        loss = out.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()
    print("Epoch",e+1,"Loss",tot/len(ds))

torch.save(model.state_dict(),"models/video_model.pt")
print("✅ Lip Model Saved")
