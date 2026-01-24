import torch
from models.video_model import LipNet
from utils.lip_dataset import LipDataset

device = "cpu"
model = LipNet().to(device)
model.load_state_dict(torch.load("models/video_model.pt", map_location=device))
model.eval()

dataset = LipDataset()

video, text = dataset[0]
video = video.unsqueeze(0).to(device)

with torch.no_grad():
    out = model(video)
    pred = out.argmax(2)[0]

letters = " abcdefghijklmnopqrstuvwxyz"
decoded = "".join([letters[i] for i in pred if i < len(letters)])

print("REAL TEXT :", text)
print("PREDICTED  :", decoded)
