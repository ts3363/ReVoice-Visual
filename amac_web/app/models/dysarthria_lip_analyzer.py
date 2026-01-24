# app/models/dysarthria_lip_analyzer.py
import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional
import json

class DysarthriaLipAnalyzer:
    """Specialized lip movement analysis for dysarthria speech disorders"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Lip landmark indices for dysarthria analysis
        self.LIP_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415]
        self.LIP_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409]
        
        # Phoneme-specific lip shapes for dysarthria training
        self.PHONEME_LIP_SHAPES = {
            # Bilabials: /p/, /b/, /m/
            'bilabial': {'width': 0.8, 'height': 0.4, 'rounding': 0.3},
            # Labiodentals: /f/, /v/
            'labiodental': {'width': 0.6, 'height': 0.2, 'upper_teeth': True},
            # Alveolars: /t/, /d/, /n/, /l/
            'alveolar': {'width': 0.5, 'height': 0.3, 'tongue_tip': True},
            # Velars: /k/, /g/
            'velar': {'width': 0.7, 'height': 0.5, 'back_raised': True},
            # Sibilants: /s/, /z/, /ʃ/, /ʒ/
            'sibilant': {'width': 0.4, 'height': 0.2, 'grooved': True}
        }
    
    def extract_dysarthria_lip_features(self, video_path: str, frame_skip: int = 3):
        """Extract lip movement features specific to dysarthria analysis"""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        lip_features = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                # Process frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    
                    # Extract dysarthria-specific features
                    features = self._analyze_dysarthria_lip_movement(landmarks)
                    lip_features.append(features)
            
            frame_count += 1
        
        cap.release()
        
        if lip_features:
            return self._aggregate_dysarthria_features(lip_features)
        return None
    
    def _analyze_dysarthria_lip_movement(self, landmarks):
        """Analyze lip movement patterns for dysarthria assessment"""
        features = {}
        
        # 1. Lip opening (vertical distance)
        upper_lip = np.mean([self._get_landmark(landmarks, i) for i in [13, 14]], axis=0)
        lower_lip = np.mean([self._get_landmark(landmarks, i) for i in [17, 18]], axis=0)
        features['lip_opening'] = np.linalg.norm(upper_lip - lower_lip)
        
        # 2. Lip stretching (horizontal distance)
        left_corner = self._get_landmark(landmarks, 61)
        right_corner = self._get_landmark(landmarks, 291)
        features['lip_width'] = np.linalg.norm(left_corner - right_corner)
        
        # 3. Lip rounding (for vowels)
        inner_lip_points = [self._get_landmark(landmarks, i) for i in self.LIP_INNER]
        outer_lip_points = [self._get_landmark(landmarks, i) for i in self.LIP_OUTER]
        features['lip_rounding'] = self._calculate_rounding(inner_lip_points)
        
        # 4. Symmetry (important for dysarthria)
        left_side = [self._get_landmark(landmarks, i) for i in [61, 78, 95, 88]]
        right_side = [self._get_landmark(landmarks, i) for i in [291, 308, 415, 324]]
        features['symmetry_score'] = self._calculate_symmetry(left_side, right_side)
        
        # 5. Movement speed (temporal analysis)
        features['movement_variance'] = np.var([p[1] for p in inner_lip_points])  # Vertical variance
        
        # 6. Jaw movement (related to dysarthria)
        chin = self._get_landmark(landmarks, 152)
        nose = self._get_landmark(landmarks, 1)
        features['jaw_opening'] = np.linalg.norm(chin - nose)
        
        return features
    
    def _aggregate_dysarthria_features(self, features_list):
        """Aggregate features across frames for dysarthria assessment"""
        aggregated = {}
        
        for key in features_list[0].keys():
            values = [f[key] for f in features_list]
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)  # Variability is important for dysarthria
            aggregated[f'{key}_range'] = np.max(values) - np.min(values)
        
        # Calculate dysarthria-specific metrics
        aggregated['articulation_consistency'] = 1.0 / (1.0 + aggregated['movement_variance_std'])
        aggregated['symmetry_issue'] = 1.0 - aggregated['symmetry_score_mean']
        aggregated['movement_amplitude'] = aggregated['lip_opening_range']
        
        return aggregated
    
    def analyze_phoneme_articulation(self, lip_features, target_phoneme: str):
        """Analyze how well a specific phoneme was articulated"""
        phoneme_type = self._get_phoneme_type(target_phoneme)
        
        if phoneme_type not in self.PHONEME_LIP_SHAPES:
            return {"error": "Phoneme type not supported"}
        
        target_shape = self.PHONEME_LIP_SHAPES[phoneme_type]
        
        # Calculate accuracy metrics
        accuracy_scores = {}
        
        if 'width' in target_shape:
            actual_width = lip_features.get('lip_width_mean', 0.5)
            target_width = target_shape['width']
            accuracy_scores['width_accuracy'] = 1.0 - abs(actual_width - target_width) / target_width
        
        if 'height' in target_shape:
            actual_height = lip_features.get('lip_opening_mean', 0.3)
            target_height = target_shape['height']
            accuracy_scores['height_accuracy'] = 1.0 - abs(actual_height - target_height) / target_height
        
        # Generate feedback based on accuracy
        feedback = self._generate_phoneme_feedback(accuracy_scores, target_phoneme)
        
        return {
            "phoneme": target_phoneme,
            "type": phoneme_type,
            "accuracy_scores": accuracy_scores,
            "overall_accuracy": np.mean(list(accuracy_scores.values())) if accuracy_scores else 0,
            "feedback": feedback
        }
    
    def _generate_phoneme_feedback(self, accuracy_scores, phoneme):
        """Generate specific feedback for dysarthria articulation"""
        feedback = []
        
        if 'width_accuracy' in accuracy_scores:
            width_acc = accuracy_scores['width_accuracy']
            if width_acc < 0.7:
                if phoneme in ['p', 'b', 'm']:
                    feedback.append(f"For /{phoneme}/, bring lips closer together")
                elif phoneme in ['f', 'v']:
                    feedback.append(f"For /{phoneme}/, touch upper teeth to lower lip")
        
        if 'height_accuracy' in accuracy_scores:
            height_acc = accuracy_scores['height_accuracy']
            if height_acc < 0.7:
                if phoneme in ['a', 'o']:
                    feedback.append(f"For /{phoneme}/, open mouth wider")
                elif phoneme in ['i', 'u']:
                    feedback.append(f"For /{phoneme}/, close lips more")
        
        # Dysarthria-specific recommendations
        if not feedback:
            feedback.append("Good articulation!")
        else:
            feedback.append("Use a mirror to visualize lip movements")
            feedback.append("Practice slowly, then increase speed")
        
        return " ".join(feedback)
    
    def process_realtime_frame(self, frame):
        """Process single frame for real-time dysarthria feedback"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Extract current lip state
            lip_state = self._analyze_dysarthria_lip_movement(landmarks)
            
            # Generate real-time feedback
            feedback = self._generate_realtime_feedback(lip_state)
            
            # Extract lip contour for visualization
            lip_contour = self._extract_lip_contour(landmarks)
            
            return {
                "lip_state": lip_state,
                "feedback": feedback,
                "lip_contour": lip_contour,
                "landmarks_detected": True
            }
        
        return {"landmarks_detected": False}
    
    def _generate_realtime_feedback(self, lip_state):
        """Generate real-time feedback for dysarthria therapy"""
        feedback = []
        
        # Check lip opening
        if lip_state['lip_opening'] < 0.05:
            feedback.append("Open mouth wider for clearer speech")
        elif lip_state['lip_opening'] > 0.15:
            feedback.append("Good mouth opening")
        
        # Check symmetry
        if lip_state['symmetry_score'] < 0.8:
            feedback.append("Focus on symmetrical lip movements")
        
        # Check movement consistency
        if lip_state['movement_variance'] > 0.01:
            feedback.append("Try steadier lip movements")
        
        return feedback[0] if feedback else "Good lip movement"
    
    def _extract_lip_contour(self, landmarks):
        """Extract lip contour points for visualization"""
        contour_points = []
        for idx in self.LIP_OUTER + self.LIP_INNER:
            point = self._get_landmark(landmarks, idx)
            contour_points.append((point[0], point[1]))
        return contour_points
    
    def _get_landmark(self, landmarks, index):
        """Get landmark coordinates"""
        lm = landmarks.landmark[index]
        return np.array([lm.x, lm.y])
    
    def _calculate_rounding(self, points):
        """Calculate how rounded the lips are"""
        centroid = np.mean(points, axis=0)
        distances = [np.linalg.norm(p - centroid) for p in points]
        return np.std(distances)  # Higher std = more rounded
    
    def _calculate_symmetry(self, left_points, right_points):
        """Calculate lip symmetry score"""
        if len(left_points) != len(right_points):
            return 0.5
        
        # Mirror right points
        mirrored_right = [[1.0 - p[0], p[1]] for p in right_points]
        
        # Calculate average distance
        distances = [np.linalg.norm(np.array(lp) - np.array(rp)) 
                    for lp, rp in zip(left_points, mirrored_right)]
        
        avg_distance = np.mean(distances)
        return 1.0 / (1.0 + avg_distance * 10)  # Convert to 0-1 score
    
    def _get_phoneme_type(self, phoneme):
        """Map phoneme to articulation type"""
        phoneme = phoneme.lower()
        
        if phoneme in ['p', 'b', 'm']:
            return 'bilabial'
        elif phoneme in ['f', 'v']:
            return 'labiodental'
        elif phoneme in ['t', 'd', 'n', 'l']:
            return 'alveolar'
        elif phoneme in ['k', 'g']:
            return 'velar'
        elif phoneme in ['s', 'z', 'ʃ', 'ʒ']:
            return 'sibilant'
        else:
            return 'vowel'  # Default for vowels
    
    def create_lip_exercises(self, difficulty_level="beginner"):
        """Create lip exercises for dysarthria therapy"""
        exercises = {
            "beginner": [
                {
                    "name": "Lip Rounding",
                    "description": "Make exaggerated 'oo' sound (like in 'moon')",
                    "duration": 30,
                    "target_phonemes": ["u", "o"],
                    "visual_cue": "Round lips as if holding a straw"
                },
                {
                    "name": "Lip Stretching", 
                    "description": "Smile widely without showing teeth",
                    "duration": 30,
                    "target_phonemes": ["i", "e"],
                    "visual_cue": "Corners of mouth pulled back"
                },
                {
                    "name": "Lip Closure",
                    "description": "Press lips together firmly, then release",
                    "duration": 30,
                    "target_phonemes": ["p", "b", "m"],
                    "visual_cue": "Lips pressed together"
                }
            ],
            "intermediate": [
                {
                    "name": "Lip Trills",
                    "description": "Make motorboat sound with lips",
                    "duration": 45,
                    "target_phonemes": ["br", "pr"],
                    "visual_cue": "Lips vibrating"
                },
                {
                    "name": "Alternating Shapes",
                    "description": "Alternate between 'ee' and 'oo' sounds",
                    "duration": 60,
                    "target_phonemes": ["i", "u"],
                    "visual_cue": "Switch between smile and pucker"
                }
            ],
            "advanced": [
                {
                    "name": "Minimal Pairs",
                    "description": "Practice 'pat' vs 'bat', focusing on lip movement",
                    "duration": 90,
                    "target_phonemes": ["p", "b"],
                    "visual_cue": "Notice lip pressure difference"
                },
                {
                    "name": "Sentence Practice",
                    "description": "Repeat 'Peter Piper picked a peck' with clear lip movements",
                    "duration": 120,
                    "target_phonemes": ["p", "k"],
                    "visual_cue": "Exaggerate lip closure for 'p' sounds"
                }
            ]
        }
        
        return exercises.get(difficulty_level, exercises["beginner"])

# Global instance
lip_analyzer = DysarthriaLipAnalyzer()
