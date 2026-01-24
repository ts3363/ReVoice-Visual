import os, cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(base_options=base)
detector = vision.FaceLandmarker.create_from_options(options)

LIPS = list(range(61, 88))

def extract(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mp_image = vision.MPImage(image_format=vision.ImageFormat.SRGB, data=frame)
        res = detector.detect(mp_image)

        if not res.face_landmarks:
            continue

        h, w, _ = frame.shape
        lm = res.face_landmarks[0]
        xs = [int(lm[p].x * w) for p in LIPS]
        ys = [int(lm[p].y * h) for p in LIPS]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)

        mouth = frame[y1:y2, x1:x2]
        if mouth.size:
            cv2.imwrite(f"{out_dir}/{i}.jpg", mouth)
            i += 1

def scan(root):
    if not os.path.exists(root): return
    for p,_,f in os.walk(root):
        for x in f:
            if x.endswith(".mp4"):
                out = "data/dys_lips/" + x.replace(".mp4","")
                extract(os.path.join(p,x), out)

scan("data/TORGO")
scan("data/kaggle_dysarthria")
print("✅ Dysarthric lip frames created")
