import torch
import cv2
from preprocessing import extract_mouth_features
from visual_model import VisualEncoder

def run_test():
    print("--- Starting System Check ---")
    
    # 1. Test Video Processing
    video_path = "test.mp4"
    print(f"Reading video from: {video_path}")
    
    # Check if file exists
    import os
    if not os.path.exists(video_path):
        print(f"ERROR: Could not find {video_path}. Please record a video and name it 'test.mp4'")
        return

    # Extract lips
    lip_features = extract_mouth_features(video_path)
    
    if lip_features is None:
        print("ERROR: No face detected in the video. Try recording with better lighting.")
        return
        
    print(f"✅ Preprocessing Success! Output shape: {lip_features.shape}")
    print("(Shape should be: [1, Time_Steps, 50, 100])")

    # 2. Test Model
    print("\nLoading Visual Model...")
    model = VisualEncoder()
    
    # Add batch dimension (Batch=1, Channels, Time, H, W)
    input_tensor = lip_features.unsqueeze(0)
    
    # Run model
    with torch.no_grad():
        prediction = model(input_tensor)
    
    print(f"✅ Model Success! Output shape: {prediction.shape}")
    print(f"(Expected output: [1, {lip_features.shape[1]}, 512])")
    print("--- System Check Complete ---")

if __name__ == "__main__":
    run_test()