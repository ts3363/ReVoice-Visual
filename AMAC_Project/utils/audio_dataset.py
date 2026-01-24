import os, torch, librosa
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, root="data/GRID/audio", sr=16000):
        self.samples=[]
        self.sr=sr
        for f in os.listdir(root):
            if f.endswith(".wav"):
                self.samples.append(os.path.join(root,f))

    def __len__(self): return len(self.samples)

    def __getitem__(self,i):
        y,_ = librosa.load(self.samples[i], sr=self.sr)
        spec = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=80)
        spec = torch.tensor(spec).unsqueeze(0)
        return spec
