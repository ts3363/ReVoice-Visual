import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from models.video_model import LipNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PHRASES = [
    "please open the door",
    "thank you very much",
    "good morning doctor",
    "can you hear me",
    "i am feeling better",
    "how are you today",
    "please sit down",
    "take a deep breath",
    "open your mouth",
    "close your eyes",
    "raise your hand",
    "touch your nose",
    "turn your head",
    "smile slowly",
    "say hello",
    "count from one to five",
    "read the sentence",
    "look at the camera",
    "speak clearly",
    "relax and breathe"
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

# -------------------- DATASET --------------------

class PersonalDataset(Dataset):
    def __init__(self):
        self.items = []
        self.tf = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

        for i, p in enumerate(PHRASES):
            self.items.append((f"data/calibration/lips/cal{i+1}", p))

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]

        # 🔥 PHONEME ANCHOR INJECTION
        label = label.lower()
        for ph, words in PHONEME_ANCHORS.items():
            for w in words:
                if w in label:
                    label = ph + " " + label

        imgs=[]
        for f in sorted(os.listdir(path))[:75]:
            img = Image.open(os.path.join(path,f)).convert("RGB")
            imgs.append(self.tf(img))

        while len(imgs) < 75:
            imgs.append(imgs[-1])

        return torch.stack(imgs), label

# -------------------- ENCODER --------------------

def encode(txt):
    seq=[]
    for c in txt:
        if c.isalpha(): seq.append(ord(c)-96)
        elif c==" ": seq.append(27)
    return torch.tensor(seq, dtype=torch.long)

# -------------------- TRAIN --------------------

model = LipNet().to(DEVICE)
model.load_state_dict(torch.load("models/video_model.pt", map_location=DEVICE))
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

    print("Epoch",epoch+1,"Loss:",round(tot/len(loader),4))

torch.save(model.state_dict(), "models/personal_lip_model.pt")
print("✅ PERSONAL MODEL SAVED")
