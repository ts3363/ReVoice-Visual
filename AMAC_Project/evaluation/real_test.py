import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import os, torch
from models.video_model import LipNet
from utils.lip_dataset import LipDataset
from evaluation.clarity_score import clarity_score
from evaluation.articulation_feedback import articulation_feedback
import torch.nn.functional as F

device = "cpu"
model = LipNet().to(device)
model.load_state_dict(torch.load("models/video_model.pt", map_location=device))
model.eval()

print("\n===== REAL USER TEST RESULTS =====\n")

for file in os.listdir("data/real_tests/videos"):
    if not file.endswith(".mp4"):
        continue

    name = file.replace(".mp4", "")
    gt = open(f"data/real_tests/transcripts/{name}.txt").read().strip()

    os.system(f"python utils/lip_extractor.py data/real_tests/videos/{file}")

    ds = LipDataset()
    vid, _ = ds[0]
    vid = vid.unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(vid)                  # [1,T,C]
        out = out.permute(1, 0, 2)        # [T,1,C]
        probs = F.log_softmax(out, dim=2)
        best = probs.argmax(dim=2).squeeze(1).cpu().numpy()

    # Proper CTC greedy decoder
    decoded = []
    prev = -1
    for b in best:
        if b != prev and b != 0:
            if b == 27:
                decoded.append(" ")
            else:
                decoded.append(chr(b + 96))
        prev = b

    hyp = "".join(decoded)

    print("FILE:", file)
    print("REAL :", gt)
    print("PRED :", hyp)
    conf = out.softmax(2).max(2)[0].mean().item()
    print("CLARITY:", clarity_score(gt, hyp, conf))

    print("FEEDBACK:", articulation_feedback(gt, hyp))
    print("-" * 60)