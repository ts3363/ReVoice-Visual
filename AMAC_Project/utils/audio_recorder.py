import sounddevice as sd
import numpy as np
import torch
import librosa

def record_audio(seconds=3, sr=16000):
    print("🎤 Recording audio...")

    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1)
    sd.wait()

    audio = audio.squeeze()

    # MFCC features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = torch.tensor(mfcc.T, dtype=torch.float32)

    return mfcc.unsqueeze(0)
