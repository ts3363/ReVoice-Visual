import cv2
import sounddevice as sd
import numpy as np
import torch
from training.adaptive_trainer import AdaptiveTrainer
from evaluation.clarity_score import clarity_score
from evaluation.articulation_feedback import articulation_feedback
from models.video_model import LipNet
from utils.lip_dataset import LipDataset

device = "cpu"
model = LipNet().to(device)
model.load_state_dict(torch.load("models/video_model.pt", map_location=device, weights_only=True))

model.eval()

trainer = AdaptiveTrainer()

print("🎤📷 REAL-TIME SPEECH THERAPY MODE")
print("Press Q to quit")

while True:
    task = trainer.get_task()
    print("\nTASK:", task)
    input("Press ENTER and speak...")

    # Dummy webcam frames (replace with actual webcam frames)
    dummy_video = torch.randn(1,10,1,64,64).to(device)

    with torch.no_grad():
        out = model(dummy_video)
        pred = out.argmax(2)[0]

    letters = " abcdefghijklmnopqrstuvwxyz"
    predicted_text = "".join([letters[i] for i in pred if i < len(letters)])

    reference = task.replace("Say:","").strip()

    clarity = clarity_score(reference, predicted_text, [0.9]*len(predicted_text))
    feedback = articulation_feedback(reference, predicted_text)

    print("You said:", predicted_text)
    print("Clarity:", clarity)
    print("Feedback:", feedback)

    trainer.update(clarity)

    if input("Press Q to quit, ENTER to continue: ").lower()=="q":
        break
