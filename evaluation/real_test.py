import os, torch
from models.video_model import LipNet
from utils.lip_extractor import extract_lip_frames
from evaluation.clarity_score import clarity_score
from evaluation.articulation_feedback import articulation_feedback

device="cpu"
model = LipNet().to(device)
model.load_state_dict(torch.load("models/video_model.pt", map_location=device))
model.eval()

letters=" abcdefghijklmnopqrstuvwxyz"

print("\n===== REAL USER TEST RESULTS =====\n")

for f in os.listdir("data/real_tests/videos"):
    name=f.split(".")[0]
    video_path="data/real_tests/videos/"+f
    gt=open("data/real_tests/transcripts/"+name+".txt").read().strip()

    frames=extract_lip_frames(video_path)
    frames=frames.unsqueeze(0).to(device)

    with torch.no_grad():
        out=model(frames)
        pred=out.argmax(2)[0]

    hyp="".join([letters[i] for i in pred if i<len(letters)]).replace("  "," ").strip()

    print("FILE:",f)
    print("REAL :",gt)
    print("PRED :",hyp)
    print("CLARITY:",clarity_score(gt,hyp))
    print("FEEDBACK:",articulation_feedback(gt,hyp))
    print("-"*60)
