import torch
from models.video_model import LipNet
from utils.lip_extractor import extract_lips_from_webcam
from evaluation.clarity_score import clarity_score
from evaluation.articulation_feedback import articulation_feedback

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LipNet().to(device)
model.load_state_dict(torch.load("models/personal_lip_model.pt", map_location=device))
model.eval()

TASKS = [
    "please open the door",
    "thank you very much",
    "good morning doctor",
    "can you hear me",
    "i am feeling better today",
    "speak clearly"
]

LETTERS = " abcdefghijklmnopqrstuvwxyz"

print("\n🧠🎤 PERSONAL DY S AR T H R I A THERAPY MODE")
print("Press ENTER to speak | Q to quit\n")

while True:
    task = TASKS[torch.randint(0, len(TASKS), (1,)).item()]
    print("👉 SAY:", task)
    input("Press ENTER to start recording...")

    video = extract_lips_from_webcam(3).to(device)

    with torch.no_grad():
        out = model(video)
        pred = out.argmax(2)[0]

    predicted = "".join([LETTERS[i] for i in pred if i < len(LETTERS)])

    clarity = clarity_score(task, predicted, [0.9]*len(predicted))
    feedback = articulation_feedback(task, predicted)

    print("\nYOU SAID:", predicted)
    print("CLARITY:", clarity)
    print("FEEDBACK:", feedback)
    print("-"*60)

    if input("Press Q to quit, ENTER to continue: ").lower() == "q":
        break
