# ===============================
# REAL-TIME WEBCAM LIP EXTRACTOR
# ===============================

import cv2
import torch
from torchvision import transforms
import mediapipe as mp

MODEL = "utils/face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
Image = mp.Image
ImageFormat = mp.ImageFormat

_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL),
    num_faces=1
)
_detector = FaceLandmarker.create_from_options(_options)

_LIPS = list(range(61, 88)) + list(range(291, 319))

_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64,64)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

def extract_lips_from_webcam(seconds=3):
    cap = cv2.VideoCapture(0)
    frames = []

    for _ in range(seconds * 25):
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape
        mp_img = Image(image_format=ImageFormat.SRGB, data=frame)
        res = _detector.detect(mp_img)

        if res.face_landmarks:
            lm = res.face_landmarks[0]
            xs = [int(lm[p].x * w) for p in _LIPS]
            ys = [int(lm[p].y * h) for p in _LIPS]

            x1, x2 = max(min(xs)-15, 0), min(max(xs)+15, w)
            y1, y2 = max(min(ys)-15, 0), min(max(ys)+15, h)

            crop = frame[y1:y2, x1:x2]
            if crop.size:
                frames.append(_transform(crop))

    cap.release()

    if len(frames) == 0:
        frames = [torch.zeros(1,64,64)]

    while len(frames) < 75:
        frames.append(frames[-1])

    return torch.stack(frames[:75]).unsqueeze(0)
