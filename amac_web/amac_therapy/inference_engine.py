import sys
import os
import torch
import cv2
import numpy as np
import mediapipe as mp
import torchaudio.transforms as T
import torchaudio

# --- BRIDGE TO YOUR RESEARCH FOLDER ---
RESEARCH_DIR = r"C:\Users\shari\Downloads\AMAC-Project1"
sys.path.append(RESEARCH_DIR)

# Now we can import your model and visual frontend from the other folder
from model import ReVoiceFusion
from visual_front_end import VisualFrontend

class ReVoicePredictor:
    def __init__(self, checkpoint_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize Model with Video Mode Enabled
        self.model = ReVoiceFusion(num_classes=500, use_video=True).to(self.device)
        self.model.eval()

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint, strict=False)
            print(f"[+] Weights loaded from {checkpoint_path}")

        # MediaPipe for real-time lip cropping
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        self.MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88]

    def get_mouth_crop(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            coords = [(int(l.x * w), int(l.y * h)) for i, l in enumerate(landmarks) if i in self.MOUTH_INDICES]
            x_min, y_min = np.min(coords, axis=0)
            x_max, y_max = np.max(coords, axis=0)
            cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
            size = max(x_max - x_min, y_max - y_min) + 20
            y1, y2 = max(0, cy - size//2), min(h, cy + size//2)
            x1, x2 = max(0, cx - size//2), min(w, cx + size//2)
            crop = frame[y1:y2, x1:x2]
            return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (64, 64))
        return None

    def predict(self, video_path):
        # 1. Process Video File into Tensors
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            crop = self.get_mouth_crop(frame)
            if crop is not None: frames.append(crop)
        cap.release()

        if not frames: return "No face detected"

        # [Batch, Time, Channel, H, W]
        video_tensor = torch.FloatTensor(np.array(frames)).unsqueeze(1).unsqueeze(0).to(self.device) / 255.0
        
        # 2. Dummy Audio (For real-time, we'll eventually fuse actual mic input)
        dummy_audio = torch.zeros(1, video_tensor.size(1) * 160, 80).to(self.device)

        with torch.no_grad():
            logits = self.model(dummy_audio, video_tensor)
            # Placeholder: return the index of the highest probability
            predicted_idx = torch.argmax(logits, dim=-1)[0, 0].item()
            return f"Prediction Index: {predicted_idx}"