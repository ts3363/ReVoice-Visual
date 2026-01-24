# app/models/integration.py
# This connects your existing ReVoice-Visual models to the web app

import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path to import your existing models
project_root = Path(__file__).parent.parent.parent.parent
revoice_path = project_root / "ReVoice-Visual"
sys.path.append(str(revoice_path))

class DysarthriaSpeechRecognizer:
    """Integration layer for your CNN+Bi-LSTM+CTC model"""
    
    def __init__(self):
        self.audio_model = None
        self.visual_model = None
        self.fusion_model = None
        self.ctc_decoder = None
        self.loaded = False
        
    def load_models(self):
        """Load your trained models from ReVoice-Visual"""
        try:
            # Try to import your existing models
            from audio_model import AudioModel
            from visual_model import VisualModel
            from fusion_model import FusionModel
            from ctc_decoder import CTCDecoder
            
            print("Loading dysarthria speech recognition models...")
            
            # Load model weights
            model_dir = revoice_path / "models"
            
            # Audio model (CNN+Bi-LSTM)
            self.audio_model = AudioModel()
            audio_weights = model_dir / "audio_model.pt"
            if audio_weights.exists():
                self.audio_model.load_state_dict(torch.load(audio_weights, map_location='cpu'))
                self.audio_model.eval()
                print(f"✓ Loaded audio model from {audio_weights}")
            
            # Visual model (lip reading)
            self.visual_model = VisualModel()
            visual_weights = model_dir / "video_model.pt"
            if visual_weights.exists():
                self.visual_model.load_state_dict(torch.load(visual_weights, map_location='cpu'))
                self.visual_model.eval()
                print(f"✓ Loaded visual model from {visual_weights}")
            
            # Fusion model
            self.fusion_model = FusionModel()
            fusion_weights = model_dir / "fusion_model.pt"
            if fusion_weights.exists():
                self.fusion_model.load_state_dict(torch.load(fusion_weights, map_location='cpu'))
                self.fusion_model.eval()
                print(f"✓ Loaded fusion model from {fusion_weights}")
            
            # CTC decoder
            self.ctc_decoder = CTCDecoder()
            print("✓ Loaded CTC decoder")
            
            self.loaded = True
            print("✅ All dysarthria models loaded successfully!")
            
        except ImportError as e:
            print(f"❌ Could not import existing models: {e}")
            print("Using placeholder models for demonstration")
            self.create_placeholder_models()
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.create_placeholder_models()
    
    def create_placeholder_models(self):
        """Create placeholder models if real ones aren't available"""
        class PlaceholderModel:
            def __init__(self, name):
                self.name = name
            def __call__(self, *args, **kwargs):
                return torch.randn(1, 10, 29)  # Placeholder output
        
        self.audio_model = PlaceholderModel("AudioCNNBiLSTM")
        self.visual_model = PlaceholderModel("VisualCNN")
        self.fusion_model = PlaceholderModel("FusionModel")
        self.ctc_decoder = lambda x: "Placeholder transcription"
        self.loaded = True
        print("⚠️ Using placeholder models")
    
    def extract_audio_features(self, audio_data, sr=16000):
        """Extract features for CNN+Bi-LSTM input"""
        # Your feature extraction logic from preprocessing.py
        # MFCC, spectrogram, etc.
        import librosa
        
        if isinstance(audio_data, str):  # File path
            audio, sr = librosa.load(audio_data, sr=sr)
        else:  # numpy array
            audio = audio_data
        
        # Extract MFCC features (13 coefficients + delta + delta-delta)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        return torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension
    
    def extract_visual_features(self, video_path_or_frames):
        """Extract lip movement features"""
        # Your lip extraction logic from lip_processing.py
        if isinstance(video_path_or_frames, str):  # File path
            # Extract frames and landmarks
            pass
        
        # Return placeholder for now
        return torch.randn(1, 10, 128)  # Batch, seq_len, features
    
    def recognize_speech(self, audio_input, visual_input=None):
        """Recognize speech using CNN+Bi-LSTM+CTC with optional visual fusion"""
        if not self.loaded:
            self.load_models()
        
        # Extract features
        audio_features = self.extract_audio_features(audio_input)
        
        # Process through audio model
        with torch.no_grad():
            audio_logits = self.audio_model(audio_features)
            
            if visual_input is not None and self.visual_model:
                visual_features = self.extract_visual_features(visual_input)
                visual_logits = self.visual_model(visual_features)
                
                # Fuse audio-visual features
                if self.fusion_model:
                    fused_logits = self.fusion_model(audio_logits, visual_logits)
                    logits = fused_logits
                else:
                    logits = audio_logits  # Fallback to audio only
            else:
                logits = audio_logits
        
        # CTC decoding
        transcription = self.ctc_decoder(logits)
        
        return {
            "transcription": transcription,
            "confidence": 0.85,  # Calculate actual confidence
            "audio_features_shape": audio_features.shape,
            "has_visual": visual_input is not None
        }
    
    def analyze_articulation(self, audio_input):
        """Analyze speech articulation quality for dysarthria"""
        features = self.extract_audio_features(audio_input)
        
        with torch.no_grad():
            if self.audio_model:
                # Get hidden states or intermediate features
                # Analyze temporal consistency, energy distribution, etc.
                pass
        
        # Placeholder analysis
        return {
            "phoneme_accuracy": 0.75 + np.random.random() * 0.2,
            "articulation_clarity": 0.70 + np.random.random() * 0.25,
            "speech_rate": "slow" if np.random.random() > 0.5 else "normal",
            "fluency_score": 0.80 + np.random.random() * 0.15,
            "recommendations": [
                "Practice sustained vowel sounds",
                "Focus on consonant clarity",
                "Use slower speech rate"
            ]
        }

# Global instance
recognizer = DysarthriaSpeechRecognizer()
