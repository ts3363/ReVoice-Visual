import time
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum

class ExerciseType(Enum):
    PHONEME_PRODUCTION = "phoneme_production"
    WORD_REPETITION = "word_repetition"
    SENTENCE_SPEAKING = "sentence_speaking"
    READING_PASSAGE = "reading_passage"
    CONVERSATION = "conversation"
    BREATHING = "breathing_exercise"

class DifficultyLevel(Enum):
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3

@dataclass
class Exercise:
    id: str
    type: ExerciseType
    difficulty: DifficultyLevel
    target: str  # e.g., "s" sound, "hello", etc.
    duration_seconds: int
    instructions: str
    visual_cue: str = ""  # URL or path to visual aid

@dataclass
class SessionResult:
    clarity_score: float  # 0-100
    articulation_accuracy: Dict[str, float]  # phoneme-specific scores
    pace_score: float  # words per minute, normalized
    volume_consistency: float
    lip_movement_score: float
    feedback_points: List[str]
    suggestions: List[str]
    timestamp: str

class TherapySession:
    def __init__(self, user_id: str, impairment_level: str = "moderate"):
        self.user_id = user_id
        self.impairment_level = impairment_level
        self.session_id = f"{user_id}_{int(time.time())}"
        self.start_time = time.time()
        self.current_exercise_index = 0
        self.exercises: List[Exercise] = []
        self.results: List[SessionResult] = []
        self.is_active = False
        
        # Load baseline for user
        self.baseline_scores = self.load_user_baseline(user_id)
        
        # Initialize exercises based on impairment
        self.generate_personalized_exercises()
    
    def load_user_baseline(self, user_id: str) -> Dict:
        """Load user's baseline performance"""
        # In production, load from database
        return {
            "clarity": 65.0,
            "pace": 120,  # words per minute
            "common_errors": ["s", "r", "th"],
            "strengths": ["vowels", "short_words"]
        }
    
    def generate_personalized_exercises(self):
        """Generate exercises based on user's needs"""
        if self.impairment_level == "severe":
            exercises = [
                Exercise(
                    id="ex1",
                    type=ExerciseType.PHONEME_PRODUCTION,
                    difficulty=DifficultyLevel.BEGINNER,
                    target="ah",  # Open vowel sound
                    duration_seconds=30,
                    instructions="Take a deep breath and say 'ah' slowly and clearly",
                    visual_cue="mouth_open.png"
                ),
                Exercise(
                    id="ex2",
                    type=ExerciseType.WORD_REPETITION,
                    difficulty=DifficultyLevel.BEGINNER,
                    target="mama",
                    duration_seconds=45,
                    instructions="Repeat the word 'mama' 5 times with clear 'm' sounds",
                    visual_cue="lip_closure.png"
                )
            ]
        else:  # Moderate/mild
            exercises = [
                Exercise(
                    id="ex3",
                    type=ExerciseType.SENTENCE_SPEAKING,
                    difficulty=DifficultyLevel.INTERMEDIATE,
                    target="The sun is shining brightly today",
                    duration_seconds=60,
                    instructions="Read this sentence with proper pacing and clear 's' sounds",
                    visual_cue="sentence_visual.png"
                ),
                Exercise(
                    id="ex4",
                    type=ExerciseType.READING_PASSAGE,
                    difficulty=DifficultyLevel.ADVANCED,
                    target="Peter Piper picked a peck of pickled peppers",
                    duration_seconds=90,
                    instructions="Read this tongue twister slowly, focusing on 'p' sounds",
                    visual_cue="tongue_twister.png"
                )
            ]
        
        self.exercises = exercises
    
    def start_session(self):
        """Start a new therapy session"""
        self.is_active = True
        self.start_time = time.time()
        return {
            "session_id": self.session_id,
            "total_exercises": len(self.exercises),
            "estimated_duration": sum(e.duration_seconds for e in self.exercises)
        }
    
    def get_current_exercise(self) -> Dict:
        """Get current exercise details"""
        if self.current_exercise_index < len(self.exercises):
            ex = self.exercises[self.current_exercise_index]
            return {
                "exercise_number": self.current_exercise_index + 1,
                "total_exercises": len(self.exercises),
                "type": ex.type.value,
                "target": ex.target,
                "instructions": ex.instructions,
                "duration": ex.duration_seconds,
                "visual_cue": ex.visual_cue,
                "time_remaining": ex.duration_seconds
            }
        return None
    
    def process_user_response(self, audio_features, video_features, recognized_text: str) -> Dict:
        """Process user's speech attempt for current exercise"""
        current_ex = self.exercises[self.current_exercise_index]
        
        # Calculate various metrics
        clarity = self.calculate_clarity_score(audio_features, current_ex.target)
        articulation = self.analyze_articulation(audio_features, video_features)
        pace = self.calculate_pace(recognized_text, audio_features)
        
        # Compare with target
        accuracy = self.compare_with_target(recognized_text, current_ex.target)
        
        # Generate feedback
        feedback = self.generate_feedback(
            clarity=clarity,
            articulation=articulation,
            accuracy=accuracy,
            pace=pace,
            target=current_ex.target
        )
        
        # Store result
        result = SessionResult(
            clarity_score=clarity,
            articulation_accuracy=articulation,
            pace_score=pace,
            volume_consistency=self.calculate_volume_consistency(audio_features),
            lip_movement_score=self.calculate_lip_movement(video_features),
            feedback_points=feedback["points"],
            suggestions=feedback["suggestions"],
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(result)
        
        # Determine if exercise is complete
        exercise_complete = clarity >= 70 or len(self.results) >= 3  # 3 attempts or good score
        
        return {
            "feedback": feedback,
            "score": clarity,
            "accuracy": accuracy,
            "exercise_complete": exercise_complete,
            "next_step": "next_exercise" if exercise_complete else "repeat"
        }
    
    def calculate_clarity_score(self, audio_features, target: str) -> float:
        """Calculate overall speech clarity score (0-100)"""
        # Implementation using your audio processing
        # This would use features from your audio_model
        score = 75.0  # Placeholder - integrate with your model
        
        # Adjust based on specific phonemes in target
        if "s" in target.lower():
            # Check sibilant quality
            score *= self.analyze_sibilant_quality(audio_features)
        
        return min(100, max(0, score))
    
    def analyze_articulation(self, audio_features, video_features) -> Dict[str, float]:
        """Analyze specific articulation issues"""
        # This should integrate with your phoneme analysis
        return {
            "bilabials": 0.85,  # p, b, m
            "fricatives": 0.65,  # s, z, th
            "vowels": 0.92,
            "nasals": 0.78
        }
    
    def generate_feedback(self, clarity: float, articulation: Dict, 
                         accuracy: float, pace: float, target: str) -> Dict:
        """Generate personalized feedback"""
        feedback_points = []
        suggestions = []
        
        # Clarity feedback
        if clarity < 60:
            feedback_points.append("Speech clarity needs improvement")
            suggestions.append("Try speaking more slowly and with more breath support")
        elif clarity < 80:
            feedback_points.append("Good clarity, some room for improvement")
            suggestions.append("Focus on consonant endings")
        else:
            feedback_points.append("Excellent clarity!")
        
        # Pace feedback
        if pace < 100:  # Too slow
            feedback_points.append("Pace is a bit slow")
            suggestions.append("Try to speak a little faster while maintaining clarity")
        elif pace > 180:  # Too fast
            feedback_points.append("Pace is too fast")
            suggestions.append("Slow down to improve articulation")
        
        # Specific phoneme feedback
        if "s" in target.lower() and articulation.get("fricatives", 1) < 0.7:
            feedback_points.append("'S' sound needs work")
            suggestions.append("Practice hissing like a snake: ssssss")
        
        if any(p in target.lower() for p in ["p", "b", "m"]) and articulation.get("bilabials", 1) < 0.8:
            feedback_points.append("Lip sounds need clearer closure")
            suggestions.append("Press lips together firmly for 'p', 'b', 'm' sounds")
        
        return {
            "points": feedback_points,
            "suggestions": suggestions,
            "encouragement": self.get_encouragement(clarity)
        }
    
    def get_encouragement(self, score: float) -> str:
        """Get motivational message"""
        if score < 50:
            return "Keep going! Every attempt helps strengthen your speech muscles."
        elif score < 70:
            return "Good effort! You're making progress."
        elif score < 85:
            return "Great job! Your clarity is improving."
        else:
            return "Excellent! Your speech is very clear today!"
    
    def next_exercise(self):
        """Move to next exercise"""
        self.current_exercise_index += 1
        return self.get_current_exercise()
    
    def end_session(self) -> Dict:
        """End session and return summary"""
        self.is_active = False
        duration = time.time() - self.start_time
        
        # Calculate session summary
        avg_clarity = np.mean([r.clarity_score for r in self.results]) if self.results else 0
        
        # Compare with baseline
        improvement = avg_clarity - self.baseline_scores["clarity"]
        
        return {
            "session_id": self.session_id,
            "duration_minutes": round(duration / 60, 1),
            "exercises_completed": self.current_exercise_index,
            "average_clarity": round(avg_clarity, 1),
            "baseline_comparison": round(improvement, 1),
            "improvement_percentage": round((improvement / self.baseline_scores["clarity"]) * 100, 1),
            "strengths": self.identify_strengths(),
            "areas_to_improve": self.identify_weaknesses(),
            "recommended_next_session": self.recommend_next_session()
        }