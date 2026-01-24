import kagglehub
import os
import shutil

print("Downloading GRID lip dataset from Kaggle...")
path = kagglehub.dataset_download("jedidiahangekouakou/grid-corpus-dataset-for-training-lipnet")

print("Downloaded at:", path)

video_dst = "data/GRID/video"
text_dst  = "data/GRID/transcripts"

os.makedirs(video_dst, exist_ok=True)
os.makedirs(text_dst, exist_ok=True)

# Walk and move files
for root, dirs, files in os.walk(path):
    for f in files:
        src = os.path.join(root, f)

        if f.endswith(".mpg") or f.endswith(".mp4"):
            shutil.copy(src, os.path.join(video_dst, f))

        if f.endswith(".align") or f.endswith(".txt"):
            shutil.copy(src, os.path.join(text_dst, f))

print("✅ GRID dataset loaded into project")
