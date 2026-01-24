import librosa
import numpy as np
import os
from tqdm import tqdm

SR = 16000
N_MFCC = 40

def load_audio(path):
    audio, _ = librosa.load(path, sr=SR)
    return audio

def extract_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=128)
    log_mel = librosa.power_to_db(mel)
    return mfcc, log_mel

def preprocess_torgo():
    audio_dir = "data/TORGO/audio"
    save_dir = "data/TORGO/features"
    os.makedirs(save_dir, exist_ok=True)

    for f in tqdm(os.listdir(audio_dir), desc="TORGO"):
        audio = load_audio(os.path.join(audio_dir, f))
        mfcc, mel = extract_features(audio)
        np.save(os.path.join(save_dir, f.replace(".wav",".npy")), np.vstack((mfcc, mel)))

    print("TORGO READY")

def preprocess_kaggledys():
    audio_dir = "data/KaggleDys/audio"
    save_dir = "data/KaggleDys/features"
    os.makedirs(save_dir, exist_ok=True)

    for f in tqdm(os.listdir(audio_dir), desc="KaggleDys"):
        audio = load_audio(os.path.join(audio_dir, f))
        mfcc, mel = extract_features(audio)
        np.save(os.path.join(save_dir, f.replace(".wav",".npy")), np.vstack((mfcc, mel)))

    print("KAGGLE DYS READY")
