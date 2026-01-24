import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class LipDataset(Dataset):
    def __init__(self, root="data/GRID/lips", transcript_root="data/GRID/transcripts"):
        self.samples = []

        for f in os.listdir(root):
            lip_path = os.path.join(root, f)
            txt_path = os.path.join(transcript_root, f + ".align")
            if os.path.isdir(lip_path) and os.path.exists(txt_path):
                self.samples.append((lip_path, txt_path))

        print("Loaded lip samples:", len(self.samples))

        self.transform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lip_dir, txt_path = self.samples[idx]

        imgs = []
        for fname in sorted(os.listdir(lip_dir)):
            try:
                img = Image.open(os.path.join(lip_dir, fname)).convert("RGB")
                imgs.append(self.transform(img))
            except:
                continue

        # 🔒 HARD GUARANTEE at least 1 frame
        if len(imgs) == 0:
            imgs = [torch.zeros(1,64,64)]

        # 🔒 FORCE exactly 10 frames
        # Change this
        # if len(imgs) >= 10:
       # 🔒 FORCE exactly 6 frames (fast + stable)
        TARGET_FRAMES = 75
        # ALWAYS force exactly 75 frames  
        if len(imgs) >= 75:
            imgs = imgs[:75]
        else:
            while len(imgs) < 75:
                imgs.append(imgs[-1])


        
        video = torch.stack(imgs)   # [75,1,64,64]

        with open(txt_path, "r", encoding="utf8") as f:
            text = f.read().strip()

        return video, text