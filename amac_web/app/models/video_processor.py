# app/models/video_processor.py
import cv2
import mediapipe as mp
import numpy as np
from .dysarthria_lip_analyzer import lip_analyzer

class VideoProcessor:
    def __init__(self, model_path=None):
        self.lip_analyzer = lip_analyzer
        self.model = None
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load trained video model for dysarthria"""
        try:
            # TODO: Load your actual video model
            self.model = "video_model_placeholder"
            print(f"Video model loaded (placeholder)")
        except Exception as e:
            print(f"Error loading video model: {e}")
    
    def extract_dysarthria_lip_features(self, video_path):
        """Extract dysarthria-specific lip features"""
        return self.lip_analyzer.extract_dysarthria_lip_features(video_path)
    
    def analyze_phoneme_articulation(self, video_path, target_phoneme):
        """Analyze how well a specific phoneme was articulated"""
        features = self.extract_dysarthria_lip_features(video_path)
        if features:
            return self.lip_analyzer.analyze_phoneme_articulation(features, target_phoneme)
        return None
    
    def get_lip_exercises(self, difficulty="beginner"):
        """Get lip exercises for dysarthria therapy"""
        return self.lip_analyzer.create_lip_exercises(difficulty)
    
    def process_realtime_for_dysarthria(self, frame):
        """Process frame for real-time dysarthria feedback"""
        return self.lip_analyzer.process_realtime_frame(frame)
    
    # Keep original method for compatibility
    def extract_landmarks(self, video_path, frame_skip=5):
        """Original method - kept for compatibility"""
        return self.lip_analyzer.extract_dysarthria_lip_features(video_path)
    
    def predict(self, video_landmarks):
        """Make prediction using video model"""
        if self.model and video_landmarks is not None:
            return {"video_score": 0.78, "lip_movement_score": 0.85}
        return {"error": "Model not loaded or no landmarks"}
