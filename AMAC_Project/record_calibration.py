import cv2, os, time

PHRASES = [
    "please open the door",
    "thank you very much",
    "good morning doctor",
    "can you hear me",
    "i am feeling better",
    "how are you today",
    "i need some water",
    "turn on the light",
    "close the window",
    "i feel very tired",
    "i am not feeling well",
    "help me please",
    "call the doctor",
    "thank you for helping",
    "i am hungry",
    "i am thirsty",
    "please sit down",
    "please stand up",
    "open the book",
    "close the door"
]

os.makedirs("data/calibration/videos", exist_ok=True)
os.makedirs("data/calibration/transcripts", exist_ok=True)

cap = cv2.VideoCapture(0)

for i, text in enumerate(PHRASES):
    name = f"cal{i+1}.mp4"
    print(f"\n👉 SAY SLOWLY: {text}")
    input("Press ENTER to start recording...")

    out = cv2.VideoWriter(
        f"data/calibration/videos/{name}",
        cv2.VideoWriter_fourcc(*'mp4v'),
        25,
        (640,480)
    )

    start = time.time()
    while time.time() - start < 3:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow("Recording", frame)
            if cv2.waitKey(1) == 27:
                break

    out.release()
    cv2.destroyAllWindows()

    with open(f"data/calibration/transcripts/cal{i+1}.txt","w") as f:
        f.write(text)

cap.release()
print("\n✅ Calibration recording complete!")
