import torch
from models.video_model import LipNet
from models.amac_fusion import AMACFusion
from utils.audio_recorder import record_audio
from utils.lip_extractor import extract_lips_from_webcam
from utils.ctc_decoder import ctc_decode
from evaluation.clarity_score import clarity_score
from evaluation.articulation_feedback import articulation_feedback


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

video_model = LipNet().to(DEVICE)
video_model.load_state_dict(torch.load("models/personal_lip_model.pt", map_location=DEVICE))
video_model.eval()

fusion_model = AMACFusion().to(DEVICE)
fusion_model.load_state_dict(torch.load("models/personal_fusion_model.pt", map_location=DEVICE))
fusion_model.eval()

print("\n🧠🎤📷 REAL MULTIMODAL DYSARTHRIA THERAPY MODE\n")

while True:
    task = input("Type sentence to speak: ").lower()
    input("Press ENTER and speak...")

    video = extract_lips_from_webcam(3).to(DEVICE)
    audio = record_audio(3).to(DEVICE)

    with torch.no_grad():
        vfeat = video_model.extract_features(video)
        logits = fusion_model(vfeat, audio)[0]

    predicted = ctc_decode(logits)
    clarity = clarity_score(task, predicted, [0.9]*len(predicted))
    feedback = articulation_feedback(task, predicted)

    print("\nYOU SAID:", predicted)
    print("CLARITY:", clarity)
    print("FEEDBACK:", feedback)
    print("-"*60)
