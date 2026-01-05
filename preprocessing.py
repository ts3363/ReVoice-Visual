import cv2
import mediapipe as mp
import numpy as np
import torch

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

def extract_mouth_features(video_path):
    """
    Reads a video, detects lips, crops them, and returns a tensor.
    Output Shape: (1, Time, 50, 100) -> Grayscale, 50px height, 100px width
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    with mp_face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=1, 
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Get lip coordinates (indices for lips in MediaPipe)
                # Outer lip indices approx: 61, 291, 0, 17 (left, right, top, bottom)
                h, w, _ = image.shape
                
                # Get specific landmarks for mouth area
                # 61: Left corner, 291: Right corner, 0: Upper lip top, 17: Lower lip bottom
                lip_points = [61, 291, 0, 17]
                
                lip_x = [int(landmarks[i].x * w) for i in lip_points]
                lip_y = [int(landmarks[i].y * h) for i in lip_points]
                
                # Bounding box with margin
                margin = 5
                min_x, max_x = min(lip_x) - margin, max(lip_x) + margin
                min_y, max_y = min(lip_y) - margin, max(lip_y) + margin
                
                # Ensure coordinates are within image bounds
                min_x, max_x = max(0, min_x), min(w, max_x)
                min_y, max_y = max(0, min_y), min(h, max_y)
                
                # Check if crop area is valid
                if max_x - min_x > 0 and max_y - min_y > 0:
                    # Crop the mouth area
                    crop = image[min_y:max_y, min_x:max_x]
                    
                    try:
                        # Resize to fixed standard (100w x 50h)
                        crop = cv2.resize(crop, (100, 50))
                        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) # Grayscale
                        frames.append(crop)
                    except Exception as e:
                        pass # Skip bad frames
                    
    cap.release()
    
    if len(frames) == 0:
        return None

    # Convert to Tensor: (Channel=1, Time, Height, Width)
    data = np.array(frames)
    # Normalize pixel values to 0-1 range
    data = torch.tensor(data).float() / 255.0 
    
    # Add Channel dimension (Time, H, W) -> (1, Time, H, W)
    return data.unsqueeze(0)