import cv2, os, time

PHONEMES = [
    "ba","da","ma","pa","ta",
    "ee","oo","aa",
    "ka","ga",
    "sa","sha","fa"
]

SAVE_DIR = "data/calibration/videos_phonemes"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

print("\n🎯 PHONEME CALIBRATION")
print("Speak SLOWLY and clearly. 3 seconds each.\n")

for i,p in enumerate(PHONEMES):
    print(f"SAY: {p}")
    input("Press ENTER to record...")
    out = cv2.VideoWriter(
        f"{SAVE_DIR}/{p}_{i}.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'),
        25,(640,480)
    )
    for _ in range(75):
        ret,frame = cap.read()
        if ret: out.write(frame)
    out.release()

cap.release()
print("\n✅ PHONEME VIDEO RECORDING COMPLETE")
