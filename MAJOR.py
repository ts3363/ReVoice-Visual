import os, numpy as np, torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self):
        self.samples = []
        for d in [ "KaggleDys"]:
            for f in os.listdir(f"data/{d}/features"):
                self.samples.append((f"data/{d}/features/{f}",
                                     f"data/{d}/transcripts/{f.replace('.npy','.txt')}"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(np.load(self.samples[idx][0])).unsqueeze(0).float()
        with open(self.samples[idx][1]) as f:
            y = f.read().lower().strip()
        return x, y

def collate_fn(batch):
    xs, ys = zip(*batch)
    max_len = max(x.shape[-1] for x in xs)
    xs_pad = []
    for x in xs:
        pad = max_len - x.shape[-1]
        xs_pad.append(torch.nn.functional.pad(x, (0, pad)))
    return torch.stack(xs_pad), ys
