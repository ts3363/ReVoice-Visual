import torch
from models.video_model import LipNet
from models.amac_fusion import AMACFusion
from utils.lip_extractor import extract_lips_from_webcam
from utils.audio_recorder import record_audio
from evaluation.clarity_score import clarity_score
from evaluation.articulation_feedback import articulation_feedback
from utils.ctc_decoder import ctc_beam_decode
from utils.time_align import align_time



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LETTERS = " abcdefghijklmnopqrstuvwxyz"

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
        # Extract video and audio features
        video_feat = video_model(video)         # [T, Dv]
        audio_feat = audio                      # already feature tensor

        audio_feat = align_time(audio_feat, video_feat.shape[1])
        fused_logits = fusion_model(video_feat, audio_feat)[0]

    pred = ctc_beam_decode(fused_logits)
    predicted = "".join([LETTERS[i] for i in pred if i < len(LETTERS)])

    clarity = clarity_score(task, predicted, [0.9] * len(predicted))
    feedback = articulation_feedback(task, predicted)

    print("\nYOU SAID:", predicted)
    print("CLARITY:", clarity)
    print("FEEDBACK:", feedback)
