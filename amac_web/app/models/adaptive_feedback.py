# app/models/adaptive_feedback.py
import numpy as np
import librosa
from typing import List, Dict, Tuple
import re
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SpeechAnalysis:
    """Analysis results for dysarthria speech"""
    real_text: str
    predicted_text: str
    clarity_score: float
    articulation_score: float
    fluency_score: float
    phoneme_accuracy: float
    feedback_messages: List[str]
    timestamp: datetime

class AdaptiveFeedbackEngine:
    """Real-time adaptive feedback engine for dysarthria"""
    
    def __init__(self):
        # Phoneme groups for dysarthria analysis
        self.DIFFICULT_PHONEMES = {
            'bilabial': ['p', 'b', 'm'],
            'labiodental': ['f', 'v'],
            'alveolar': ['t', 'd', 'n', 'l'],
            'velar': ['k', 'g'],
            'sibilant': ['s', 'z', 'ʃ', 'ʒ'],
            'affricate': ['tʃ', 'dʒ']
        }
        
        # Feedback templates for different issues
        self.FEEDBACK_TEMPLATES = {
            'slow_speech': [
                "Slow down your speech pace",
                "Take brief pauses between words",
                "Focus on one word at a time"
            ],
            'consonant_clarity': [
                "Over-articulate consonant sounds",
                "Press lips firmly for /p/, /b/, /m/ sounds",
                "Touch tongue to roof for /t/, /d/, /n/ sounds"
            ],
            'vowel_accuracy': [
                "Open mouth wider for clear vowels",
                "Hold vowel sounds longer",
                "Exaggerate lip movements for vowels"
            ],
            'fluency': [
                "Practice smoother transitions between words",
                "Reduce hesitations and filler sounds",
                "Maintain consistent speech rhythm"
            ],
            'volume': [
                "Increase vocal volume",
                "Project your voice forward",
                "Take deeper breaths before speaking"
            ],
            'pitch_variation': [
                "Add more pitch variation",
                "Use rising pitch for questions",
                "Avoid monotone speech"
            ]
        }
        
        # User profile for adaptive learning
        self.user_profile = {
            'common_errors': {},
            'improvement_areas': [],
            'session_history': [],
            'personalized_exercises': []
        }
    
    def analyze_realtime_audio(self, audio_chunk: np.ndarray, sr: int = 16000) -> Dict:
        """Analyze real-time audio chunk for dysarthria features"""
        features = {}
        
        # 1. Energy/Volume analysis
        energy = np.mean(np.abs(audio_chunk))
        features['energy'] = float(energy)
        features['is_speech'] = energy > 0.01
        
        # 2. Zero-crossing rate (roughness)
        zcr = np.mean(librosa.zero_crossings(audio_chunk))
        features['zero_crossing_rate'] = float(zcr)
        
        # 3. Pitch analysis
        if len(audio_chunk) > 2048:
            pitches, magnitudes = librosa.piptrack(y=audio_chunk, sr=sr)
            pitch = np.max(pitches[magnitudes > np.median(magnitudes)])
            features['pitch'] = float(pitch) if not np.isnan(pitch) else 0.0
        
        # 4. Spectral features
        if len(audio_chunk) > 512:
            spectrogram = np.abs(librosa.stft(audio_chunk))
            spectral_centroid = librosa.feature.spectral_centroid(S=spectrogram)
            features['spectral_centroid'] = float(np.mean(spectral_centroid))
        
        return features
    
    def compare_real_vs_predicted(self, real_text: str, predicted_text: str) -> SpeechAnalysis:
        """Compare real and predicted text for dysarthria analysis"""
        
        # Clean texts
        real_clean = self._clean_text(real_text)
        pred_clean = self._clean_text(predicted_text)
        
        # Calculate scores
        clarity = self._calculate_clarity_score(real_clean, pred_clean)
        articulation = self._calculate_articulation_score(real_clean, pred_clean)
        fluency = self._calculate_fluency_score(real_clean, pred_clean)
        phoneme_acc = self._calculate_phoneme_accuracy(real_clean, pred_clean)
        
        # Generate adaptive feedback
        feedback = self._generate_adaptive_feedback(
            real_clean, pred_clean, 
            clarity, articulation, fluency, phoneme_acc
        )
        
        # Update user profile
        self._update_user_profile(real_clean, pred_clean, feedback)
        
        return SpeechAnalysis(
            real_text=real_text,
            predicted_text=predicted_text,
            clarity_score=clarity,
            articulation_score=articulation,
            fluency_score=fluency,
            phoneme_accuracy=phoneme_acc,
            feedback_messages=feedback,
            timestamp=datetime.now()
        )
    
    def _calculate_clarity_score(self, real: str, pred: str) -> float:
        """Calculate speech clarity score (0-1)"""
        if not real or not pred:
            return 0.0
        
        # Word-level accuracy
        real_words = real.split()
        pred_words = pred.split()
        
        if not real_words:
            return 0.0
        
        # Simple word matching
        matches = sum(1 for rw in real_words if any(rw in pw or pw in rw for pw in pred_words))
        word_accuracy = matches / len(real_words)
        
        # Character-level accuracy (for partial matches)
        char_similarity = self._levenshtein_similarity(real, pred)
        
        # Combine scores
        clarity = 0.7 * word_accuracy + 0.3 * char_similarity
        return max(0.0, min(1.0, clarity))
    
    def _calculate_articulation_score(self, real: str, pred: str) -> float:
        """Calculate articulation score based on consonant accuracy"""
        if not real:
            return 0.0
        
        # Extract consonants
        consonants_real = re.findall(r'[bcdfghjklmnpqrstvwxyz]', real.lower())
        consonants_pred = re.findall(r'[bcdfghjklmnpqrstvwxyz]', pred.lower())
        
        if not consonants_real:
            return 1.0  # No consonants to articulate
        
        # Check if consonants are present in prediction
        matches = sum(1 for c in consonants_real if c in consonants_pred)
        articulation_score = matches / len(consonants_real)
        
        return articulation_score
    
    def _calculate_fluency_score(self, real: str, pred: str) -> float:
        """Calculate fluency score based on speech flow"""
        # Analyze word count and pattern
        real_words = real.split()
        pred_words = pred.split()
        
        if len(real_words) < 2:
            return 0.8  # Single word/short phrase
        
        # Check for consistent word order
        fluency = 0.5
        
        # Look for pattern breaks (like repetitions in prediction)
        if len(pred_words) > len(real_words) * 1.5:
            fluency -= 0.3  # Too many repetitions
        
        # Check for hesitations (represented as "zzz" in prediction)
        if 'zzz' in pred.lower() or 'uh' in pred.lower() or 'um' in pred.lower():
            fluency -= 0.2
        
        return max(0.0, min(1.0, fluency))
    
    def _calculate_phoneme_accuracy(self, real: str, pred: str) -> float:
        """Calculate phoneme-level accuracy"""
        # This would use a phoneme alignment tool in production
        # For now, use approximate matching
        
        if not real or not pred:
            return 0.0
        
        # Simple character-level similarity for phonemes
        similarity = self._levenshtein_similarity(real.lower(), pred.lower())
        
        # Penalize for dysarthria-specific errors
        common_errors = self._detect_dysarthria_errors(real, pred)
        penalty = len(common_errors) * 0.1
        
        return max(0.0, similarity - penalty)
    
    def _detect_dysarthria_errors(self, real: str, pred: str) -> List[str]:
        """Detect specific dysarthria-related errors"""
        errors = []
        
        real_lower = real.lower()
        pred_lower = pred.lower()
        
        # Check for consonant weakening
        for cons in ['p', 't', 'k', 's', 'f']:
            if cons in real_lower and cons not in pred_lower:
                errors.append(f'Missing {cons}')
        
        # Check for voicing errors (b->p, d->t, g->k)
        voicing_pairs = [('b', 'p'), ('d', 't'), ('g', 'k'), ('v', 'f'), ('z', 's')]
        for voiced, voiceless in voicing_pairs:
            if voiced in real_lower and voiceless in pred_lower:
                errors.append(f'Voicing error: {voiced}->{voiceless}')
        
        # Check for repetitions (common in dysarthria)
        if re.search(r'(.)\1{2,}', pred_lower):  # Triple repetition
            errors.append('Sound repetition')
        
        return errors
    
    def _generate_adaptive_feedback(self, real: str, pred: str, 
                                  clarity: float, articulation: float, 
                                  fluency: float, phoneme_acc: float) -> List[str]:
        """Generate personalized feedback based on analysis"""
        feedback = []
        
        # Clarity feedback
        if clarity < 0.3:
            feedback.append(self._get_feedback('slow_speech'))
        elif clarity < 0.6:
            feedback.append("Speak more clearly and deliberately")
        
        # Articulation feedback
        if articulation < 0.4:
            feedback.append(self._get_feedback('consonant_clarity'))
            # Specific consonant feedback
            missing = self._identify_missing_consonants(real, pred)
            if missing:
                feedback.append(f"Focus on: {', '.join(missing)}")
        
        # Fluency feedback
        if fluency < 0.5:
            feedback.append(self._get_feedback('fluency'))
        
        # Phoneme accuracy feedback
        if phoneme_acc < 0.5:
            feedback.append("Practice individual sound articulation")
        
        # Energy/volume feedback (would come from audio analysis)
        if len(feedback) < 2:  # Ensure at least 2 feedback items
            if clarity < 0.8:
                feedback.append("Good effort! Practice makes progress")
            else:
                feedback.append("Excellent clarity! Keep practicing")
        
        return feedback[:3]  # Return top 3 feedback items
    
    def _get_feedback(self, category: str) -> str:
        """Get random feedback from category"""
        templates = self.FEEDBACK_TEMPLATES.get(category, ["Keep practicing"])
        return np.random.choice(templates)
    
    def _identify_missing_consonants(self, real: str, pred: str) -> List[str]:
        """Identify which consonants are missing in prediction"""
        real_cons = set(re.findall(r'[bcdfghjklmnpqrstvwxyz]', real.lower()))
        pred_cons = set(re.findall(r'[bcdfghjklmnpqrstvwxyz]', pred.lower()))
        
        missing = real_cons - pred_cons
        return list(missing)[:3]  # Return up to 3 missing consonants
    
    def _update_user_profile(self, real: str, pred: str, feedback: List[str]):
        """Update user profile with latest analysis"""
        # Track common errors
        errors = self._detect_dysarthria_errors(real, pred)
        for error in errors:
            self.user_profile['common_errors'][error] = \
                self.user_profile['common_errors'].get(error, 0) + 1
        
        # Track improvement areas from feedback
        for fb in feedback:
            if fb not in self.user_profile['improvement_areas']:
                self.user_profile['improvement_areas'].append(fb)
        
        # Store session
        self.user_profile['session_history'].append({
            'timestamp': datetime.now().isoformat(),
            'real_text': real,
            'predicted_text': pred,
            'feedback': feedback
        })
        
        # Limit history size
        if len(self.user_profile['session_history']) > 100:
            self.user_profile['session_history'] = self.user_profile['session_history'][-50:]
    
    def generate_personalized_exercises(self) -> List[Dict]:
        """Generate personalized exercises based on user profile"""
        exercises = []
        
        # Get most common errors
        common_errors = sorted(
            self.user_profile['common_errors'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        for error, count in common_errors:
            if 'Missing' in error:
                phoneme = error.split()[-1]
                exercises.append({
                    'type': 'phoneme_repetition',
                    'target': phoneme,
                    'description': f'Repeat /{phoneme}/ sound 10 times clearly',
                    'duration': 60,
                    'difficulty': 'beginner'
                })
            elif 'Voicing error' in error:
                sounds = error.split(': ')[-1].split('->')
                exercises.append({
                    'type': 'minimal_pair',
                    'target': f'{sounds[0]} vs {sounds[1]}',
                    'description': f'Practice "ba" vs "pa", "da" vs "ta"',
                    'duration': 90,
                    'difficulty': 'intermediate'
                })
        
        # Add general exercises if needed
        if len(exercises) < 3:
            exercises.extend([
                {
                    'type': 'sentence_practice',
                    'target': 'General articulation',
                    'description': 'Read "She sells seashells" slowly and clearly',
                    'duration': 120,
                    'difficulty': 'beginner'
                },
                {
                    'type': 'breathing_exercise',
                    'target': 'Speech support',
                    'description': 'Deep breathing before speaking',
                    'duration': 60,
                    'difficulty': 'beginner'
                }
            ])
        
        self.user_profile['personalized_exercises'] = exercises
        return exercises
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove non-alphanumeric except basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text.lower()
    
    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Calculate Levenshtein similarity between two strings"""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0
        
        # Simple implementation
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        
        if len(s2) == 0:
            return 0.0
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        distance = previous_row[-1]
        max_len = max(len(s1), len(s2))
        
        return 1.0 - (distance / max_len)
    
    def format_analysis_display(self, analysis: SpeechAnalysis) -> str:
        """Format analysis for display (like your example)"""
        # Format predicted text (truncate if too long)
        pred_display = analysis.predicted_text
        if len(pred_display) > 80:
            pred_display = pred_display[:77] + "..."
        
        # Format clarity as percentage
        clarity_pct = analysis.clarity_score * 100
        
        # Create formatted output
        lines = [
            f"REAL : {analysis.real_text}",
            f"PRED : {pred_display}",
            f"CLARITY: {clarity_pct:.1f}",
            f"FEEDBACK: {analysis.feedback_messages}",
            "-" * 60
        ]
        
        return "\n".join(lines)

# Global instance
feedback_engine = AdaptiveFeedbackEngine()
