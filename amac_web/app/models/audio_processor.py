# app/models/audio_processor.py
import torch
import numpy as np
import librosa
from .integration import recognizer

class AudioProcessor:
    """Processor for dysarthria speech using CNN+Bi-LSTM+CTC"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.recognizer = recognizer
        self.recognizer.load_models()
    
    def extract_dysarthria_features(self, audio_path, sr=16000):
        """Extract features optimized for dysarthria speech"""
        try:
            audio, sr = librosa.load(audio_path, sr=sr)
            
            # Features for dysarthria analysis
            features = {}
            
            # 1. MFCCs (for CNN input)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
            features['mfcc'] = mfcc
            
            # 2. Spectrogram (for visual analysis)
            D = np.abs(librosa.stft(audio))**2
            spectrogram = librosa.amplitude_to_db(D, ref=np.max)
            features['spectrogram'] = spectrogram
            
            # 3. Prosodic features (important for dysarthria)
            features['pitch'] = librosa.yin(audio, fmin=50, fmax=500)
            features['energy'] = librosa.feature.rms(y=audio)
            
            # 4. Temporal features
            features['duration'] = len(audio) / sr
            features['speech_rate'] = self.estimate_speech_rate(audio, sr)
            
            return features
            
        except Exception as e:
            print(f"Error extracting dysarthria features: {e}")
            return None
    
    def estimate_speech_rate(self, audio, sr):
        """Estimate speech rate for dysarthria assessment"""
        # Simple energy-based speech activity detection
        energy = librosa.feature.rms(y=audio)
        threshold = np.mean(energy) * 0.3
        speech_frames = np.sum(energy > threshold)
        speech_seconds = speech_frames * 512 / sr  # hop_length = 512
        
        if speech_seconds > 0:
            return len(audio) / sr / speech_seconds
        return 1.0
    
    def recognize_dysarthria_speech(self, audio_path):
        """Recognize speech from dysarthria individuals"""
        result = self.recognizer.recognize_speech(audio_path)
        
        # Add dysarthria-specific analysis
        analysis = self.recognizer.analyze_articulation(audio_path)
        result.update(analysis)
        
        return result
    
    def assess_articulation(self, audio_path):
        """Assess articulation quality for therapy feedback"""
        features = self.extract_dysarthria_features(audio_path)
        if features is None:
            return None
        
        # Calculate dysarthria metrics
        metrics = {
            "mfcc_variance": float(np.var(features['mfcc'])),  # Articulation variability
            "pitch_stability": float(np.std(features['pitch'][features['pitch'] > 0])),
            "energy_consistency": float(np.std(features['energy'])),
            "speech_rate": float(features['speech_rate'])
        }
        
        # Generate therapy feedback
        feedback = self.generate_therapy_feedback(metrics)
        
        return {
            "metrics": metrics,
            "feedback": feedback,
            "score": self.calculate_articulation_score(metrics)
        }
    
    def generate_therapy_feedback(self, metrics):
        """Generate personalized feedback for dysarthria therapy"""
        feedback = []
        
        if metrics['mfcc_variance'] > 100:  # High variability
            feedback.append("Focus on consistent articulation")
        else:
            feedback.append("Good articulation consistency")
            
        if metrics['pitch_stability'] > 50:  # Unstable pitch
            feedback.append("Practice maintaining steady pitch")
            
        if metrics['speech_rate'] < 0.5:  # Very slow
            feedback.append("Try to increase speech rate slightly")
        elif metrics['speech_rate'] > 2.0:  # Very fast
            feedback.append("Slow down for clearer speech")
            
        if not feedback:
            feedback.append("Good speech production!")
            
        return " ".join(feedback)
    
    def calculate_articulation_score(self, metrics):
        """Calculate overall articulation score (0-100)"""
        score = 100
        
        # Penalize high variability
        if metrics['mfcc_variance'] > 150:
            score -= 20
        elif metrics['mfcc_variance'] > 100:
            score -= 10
            
        # Penalize unstable pitch
        if metrics['pitch_stability'] > 100:
            score -= 15
        elif metrics['pitch_stability'] > 50:
            score -= 5
            
        # Adjust for speech rate
        if metrics['speech_rate'] < 0.3 or metrics['speech_rate'] > 2.5:
            score -= 10
            
        return max(0, min(100, score))
    
    def process_realtime_audio(self, audio_chunk, sr=16000):
        """Process real-time audio chunks for live therapy"""
        # Convert to features
        mfcc = librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=13)
        
        # Simple real-time analysis
        energy = np.mean(np.abs(audio_chunk))
        zero_crossings = np.mean(librosa.zero_crossings(audio_chunk))
        
        return {
            "energy": float(energy),
            "zero_crossings": float(zero_crossings),
            "mfcc_mean": float(np.mean(mfcc)),
            "is_speech": energy > 0.01  # Simple VAD
        }
