import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from models.video_model import LipNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PHRASES = [
    "please open the door","thank you very much","good morning doctor","can you hear me",
    "i am feeling better","how are you today","please sit down","take a deep breath",
    "open your mouth","close your eyes","raise your hand","touch your nose",
    "turn your head","smile slowly","say hello","count from one to five",
    "read the sentence","look at the camera","speak clearly","relax and breathe"
]

PHONEME_ANCHORS = {
    "p": ["please","open"],
    "t": ["thank","today"],
    "g": ["good"],
    "b": ["better"],
    "d": ["door","doctor"],
    "h": ["hear"],
    "m": ["me"],
}

# ================= DATASET =================

class PersonalDataset(Dataset):
    def __init__(self):
        self.tf = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        self.items = [(f"data/calibration/lips/cal{i+1}", PHRASES[i]) for i in range(len(PHRASES))]

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        label = label.lower()

        # phoneme anchors
        for ph, words in PHONEME_ANCHORS.items():
            for w in words:
                if w in label:
                    label = ph + " " + label

        frames=[]
        for f in sorted(os.listdir(path))[:75]:
            img = Image.open(os.path.join(path,f)).convert("RGB")
            frames.append(self.tf(img))

        while len(frames)<75: frames.append(frames[-1])
        return torch.stack(frames), label

# ================= TEXT ENCODER =================

def encode(txt):
    seq=[]
    for c in txt:
        if c.isalpha(): seq.append(ord(c)-96)
        elif c==" ": seq.append(27)
    return torch.tensor(seq, dtype=torch.long)

# ================= TRAIN =================

model = LipNet().to(DEVICE)

# 🔥 CRITICAL FIX — DROP OLD CLASSIFIER
ckpt = torch.load("models/video_model.pt", map_location=DEVICE)
ckpt.pop("fc.weight", None)
ckpt.pop("fc.bias", None)
model.load_state_dict(ckpt, strict=False)

model.train()

loader = DataLoader(PersonalDataset(), batch_size=2, shuffle=True)
opt = optim.Adam(model.parameters(), lr=3e-5)
ctc = nn.CTCLoss(blank=0)

print("\n🔥 PERSONAL ADAPTATION TRAINING...")

for epoch in range(40):
    tot=0
    for vids, txts in loader:
        vids = vids.to(DEVICE)

        targets=[encode(t) for t in txts]
        lens=[len(t) for t in targets]

        targets=torch.cat(targets).to(DEVICE)
        lens=torch.tensor(lens).to(DEVICE)

        out=model(vids).permute(1,0,2)
        inp_len=torch.full((out.size(1),), out.size(0), device=DEVICE, dtype=torch.long)

        loss = ctc(out.log_softmax(2), targets, inp_len, lens)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()

    print(f"Epoch {epoch+1} Loss: {round(tot/len(loader),4)}")

torch.save(model.state_dict(), "models/personal_lip_model.pt")
print("✅ PERSONAL MODEL SAVED")
