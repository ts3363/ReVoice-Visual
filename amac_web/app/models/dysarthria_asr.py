# app/models/dysarthria_asr.py
import torch
import whisper
import numpy as np
from typing import Optional, Tuple, Dict
import torchaudio
import torchaudio.transforms as T
from datetime import datetime
from .adaptive_feedback import feedback_engine, SpeechAnalysis

class DysarthriaASR:
    """Dysarthria-specific ASR with Whisper/DeepSpeech integration"""
    
    def __init__(self, model_type: str = "whisper", model_size: str = "base"):
        self.model_type = model_type
        self.model_size = model_size
        self.model = None
        self.processor = None
        self.sample_rate = 16000
        
        # Load model
        self.load_model()
        
        # Session tracking
        self.current_session = None
        self.real_text_buffer = []
        self.predicted_text_buffer = []
    
    def load_model(self):
        """Load ASR model (Whisper or DeepSpeech)"""
        try:
            if self.model_type.lower() == "whisper":
                print(f"Loading Whisper model ({self.model_size})...")
                self.model = whisper.load_model(self.model_size)
                print("✅ Whisper model loaded")
            elif self.model_type.lower() == "deepspeech":
                print("Loading DeepSpeech model...")
                # You would load DeepSpeech here
                # self.model = load_deepspeech_model()
                self.model = "DeepSpeech_placeholder"
                print("⚠️ DeepSpeech placeholder loaded (install separately)")
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = "placeholder_model"
    
    def transcribe_audio(self, audio_data: np.ndarray, 
                        real_text: Optional[str] = None) -> SpeechAnalysis:
        """Transcribe audio and compare with real text"""
        
        # Get prediction
        predicted_text = self._get_prediction(audio_data)
        
        # Store in buffers
        self.predicted_text_buffer.append(predicted_text)
        if real_text:
            self.real_text_buffer.append(real_text)
        
        # Analyze if we have real text
        if real_text:
            analysis = feedback_engine.compare_real_vs_predicted(
                real_text, predicted_text
            )
            
            # Store in session
            if self.current_session:
                self.current_session['utterances'].append({
                    'timestamp': datetime.now().isoformat(),
                    'real_text': real_text,
                    'predicted_text': predicted_text,
                    'analysis': {
                        'clarity': analysis.clarity_score,
                        'articulation': analysis.articulation_score,
                        'fluency': analysis.fluency_score,
                        'phoneme_accuracy': analysis.phoneme_accuracy
                    }
                })
            
            return analysis
        else:
            # Return analysis with only prediction
            return SpeechAnalysis(
                real_text="",
                predicted_text=predicted_text,
                clarity_score=0.0,
                articulation_score=0.0,
                fluency_score=0.0,
                phoneme_accuracy=0.0,
                feedback_messages=["No reference text for comparison"],
                timestamp=datetime.now()
            )
    
    def _get_prediction(self, audio_data: np.ndarray) -> str:
        """Get prediction from ASR model"""
        if self.model == "placeholder_model":
            # Simulate dysarthria prediction errors
            return self._simulate_dysarthria_prediction()
        
        try:
            if self.model_type.lower() == "whisper":
                return self._whisper_transcribe(audio_data)
            else:
                # DeepSpeech or other model
                return "Placeholder transcription for dysarthria speech"
        except Exception as e:
            print(f"Transcription error: {e}")
            return "Transcription error occurred"
    
    def _whisper_transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe using Whisper"""
        # Ensure audio is float32 and normalized
        audio_float = audio_data.astype(np.float32)
        if audio_float.max() > 0:
            audio_float = audio_float / audio_float.max()
        
        # Transcribe
        result = self.model.transcribe(
            audio_float,
            language='en',
            task='transcribe',
            fp16=torch.cuda.is_available()  # Use FP16 if GPU available
        )
        
        return result['text'].strip()
    
    def _simulate_dysarthria_prediction(self) -> str:
        """Simulate dysarthria ASR predictions (for testing)"""
        # Common dysarthria error patterns
        patterns = [
            "gkzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzkcc",
            "bin blu at b ate now",
            "p t k s m n repeated",
            "sh she sells seashells",
            "hello how are you today",
            "my name is patient with dysarthria",
            "the quick brown fox jumps over"
        ]
        
        # Add some random errors
        import random
        base = random.choice(patterns)
        
        # Add dysarthria-like errors
        errors = random.randint(0, 3)
        for _ in range(errors):
            if random.random() > 0.5:
                # Add repetition
                idx = random.randint(0, len(base)-1)
                base = base[:idx] + base[idx]*2 + base[idx:]
            else:
                # Add substitution
                substitutions = {'b':'p', 'd':'t', 'g':'k', 'v':'f', 'z':'s'}
                for orig, sub in substitutions.items():
                    if orig in base:
                        base = base.replace(orig, sub, 1)
                        break
        
        return base
    
    def start_session(self, user_id: Optional[str] = None):
        """Start a new therapy session"""
        self.current_session = {
            'session_id': f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'user_id': user_id or 'anonymous',
            'start_time': datetime.now().isoformat(),
            'utterances': [],
            'metrics': {
                'total_utterances': 0,
                'avg_clarity': 0.0,
                'avg_articulation': 0.0,
                'improvement_trend': []
            }
        }
        
        # Clear buffers
        self.real_text_buffer = []
        self.predicted_text_buffer = []
        
        print(f"Started session: {self.current_session['session_id']}")
        return self.current_session['session_id']
    
    def end_session(self) -> Dict:
        """End current session and return summary"""
        if not self.current_session:
            return {"error": "No active session"}
        
        session = self.current_session
        session['end_time'] = datetime.now().isoformat()
        
        # Calculate session metrics
        utterances = session['utterances']
        if utterances:
            session['metrics']['total_utterances'] = len(utterances)
            
            # Calculate averages
            clarities = [u['analysis']['clarity'] for u in utterances]
            articulations = [u['analysis']['articulation'] for u in utterances]
            
            session['metrics']['avg_clarity'] = np.mean(clarities) if clarities else 0.0
            session['metrics']['avg_articulation'] = np.mean(articulations) if articulations else 0.0
            
            # Calculate improvement trend
            if len(clarities) > 1:
                first_half = np.mean(clarities[:len(clarities)//2])
                second_half = np.mean(clarities[len(clarities)//2:])
                improvement = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0.0
                session['metrics']['improvement_trend'] = [first_half, second_half, improvement]
        
        # Generate personalized exercises
        session['recommended_exercises'] = feedback_engine.generate_personalized_exercises()
        
        # Reset session
        self.current_session = None
        
        return session
    
    def get_realtime_metrics(self) -> Dict:
        """Get real-time metrics for display"""
        if not self.current_session or not self.current_session['utterances']:
            return {
                'clarity': 0.0,
                'articulation': 0.0,
                'utterance_count': 0,
                'session_duration': 0
            }
        
        utterances = self.current_session['utterances']
        recent = utterances[-5:]  # Last 5 utterances
        
        metrics = {
            'clarity': np.mean([u['analysis']['clarity'] for u in recent]) if recent else 0.0,
            'articulation': np.mean([u['analysis']['articulation'] for u in recent]) if recent else 0.0,
            'utterance_count': len(utterances),
            'session_duration': (datetime.now() - 
                               datetime.fromisoformat(self.current_session['start_time'])).seconds
        }
        
        return metrics
    
    def process_audio_features(self, audio_data: np.ndarray) -> Dict:
        """Process audio for real-time feature extraction"""
        features = feedback_engine.analyze_realtime_audio(audio_data)
        
        # Add dysarthria-specific features
        if len(audio_data) > 0:
            # Calculate jitter (pitch variability)
            if 'pitch' in features and features['pitch'] > 0:
                # Simulate jitter calculation
                features['jitter'] = np.random.random() * 0.1
            
            # Calculate shimmer (amplitude variability)
            features['shimmer'] = np.std(audio_data) / (np.mean(np.abs(audio_data)) + 1e-10)
            
            # Speech rate estimation (simplified)
            features['speech_rate'] = self._estimate_speech_rate(audio_data)
        
        return features
    
    def _estimate_speech_rate(self, audio_data: np.ndarray) -> float:
        """Estimate speech rate from audio"""
        # Simple energy-based VAD
        energy = np.mean(np.abs(audio_data))
        threshold = 0.01
        
        # Count "speech frames"
        frame_size = 160  # 10ms at 16kHz
        speech_frames = 0
        for i in range(0, len(audio_data), frame_size):
            frame = audio_data[i:i+frame_size]
            if len(frame) > 0 and np.mean(np.abs(frame)) > threshold:
                speech_frames += 1
        
        total_frames = len(audio_data) // frame_size
        speech_ratio = speech_frames / total_frames if total_frames > 0 else 0
        
        # Convert to approximate syllables per second
        return speech_ratio * 15  # Rough estimate
    
    def get_session_summary_display(self) -> str:
        """Get formatted session summary for display"""
        if not self.current_session:
            return "No active session"
        
        metrics = self.get_realtime_metrics()
        utterances = self.current_session['utterances']
        
        if not utterances:
            return "Session started. Begin speaking..."
        
        # Get last utterance for display
        last = utterances[-1]
        
        # Format like your example
        output = [
            f"SESSION: {self.current_session['session_id'][-8:]}",
            f"DURATION: {metrics['session_duration']}s | UTTERANCES: {metrics['utterance_count']}",
            "-" * 60,
            f"REAL : {last['real_text']}",
            f"PRED : {last['predicted_text'][:80]}",
            f"CLARITY: {last['analysis']['clarity']*100:.1f}",
            f"ARTICULATION: {last['analysis']['articulation']*100:.1f}",
            f"FEEDBACK: {feedback_engine.format_analysis_display}"
        ]
        
        # Add running averages
        output.extend([
            "-" * 40,
            f"RUNNING AVG - Clarity: {metrics['clarity']*100:.1f}%",
            f"RUNNING AVG - Articulation: {metrics['articulation']*100:.1f}%"
        ])
        
        return "\n".join(output)

# Global ASR instance
asr_engine = DysarthriaASR(model_type="whisper", model_size="base")
