import os
import numpy as np
from tqdm import tqdm
from utils.video_processing import extract_lip_frames

def preprocess_torgo_lips():
    video_dir = "data/TORGO/video"
    save_dir = "data/TORGO/lip_features"
    os.makedirs(save_dir, exist_ok=True)

    print("Extracting TORGO lip features...")

    for v in tqdm(os.listdir(video_dir)):
        lips = extract_lip_frames(os.path.join(video_dir, v))
        np.save(os.path.join(save_dir, v.replace(".mp4",".npy")), lips)

    print("✅ TORGO lip preprocessing finished")
