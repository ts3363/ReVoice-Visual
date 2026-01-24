import cv2, os, time

SENTENCES = [
    "please open the door",
    "thank you very much",
    "good morning doctor",
    "can you hear me",
    "i am feeling better",
    "how are you today",
    "please sit down",
    "take a deep breath",
    "open your mouth",
    "close your eyes",
    "raise your hand",
    "touch your nose",
    "turn your head",
    "smile slowly",
    "say hello",
    "count from one to five",
    "read the sentence",
    "look at the camera",
    "speak clearly",
    "relax and breathe"
]

os.makedirs("data/calibration/videos", exist_ok=True)
os.makedirs("data/calibration/texts", exist_ok=True)

cap = cv2.VideoCapture(0)

for i, text in enumerate(SENTENCES):
    print("\nSAY:", text)
    input("Press ENTER to record 3 seconds...")

    path = f"data/calibration/videos/cal{i+1}.mp4"
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (640,480))

    start = time.time()
    while time.time() - start < 3:
        ret, frame = cap.read()
        if ret:
            writer.write(frame)
            cv2.imshow("Recording", frame)
            if cv2.waitKey(1)==27:
                break

    writer.release()
    open(f"data/calibration/texts/cal{i+1}.txt","w").write(text)

cap.release()
cv2.destroyAllWindows()

print("\n✅ CALIBRATION RECORDING COMPLETE")
