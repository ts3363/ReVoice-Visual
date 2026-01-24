import os, torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, root="data/GRID/lips"):
        self.samples = []
        for spk in os.listdir(root):
            for utt in os.listdir(f"{root}/{spk}"):
                frames = sorted(os.listdir(f"{root}/{spk}/{utt}"))
                self.samples.append((f"{root}/{spk}/{utt}", frames,
                                     f"data/GRID/transcripts/{utt}.txt"))

        self.tf = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((112,112)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder, frames, txt = self.samples[idx]
        imgs = []
        for f in frames:
            img = Image.open(f"{folder}/{f}")
            imgs.append(self.tf(img))
        x = torch.stack(imgs)
        y = open(txt).read().strip().lower()
        return x, y
