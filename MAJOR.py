import os
import cv2
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, session
from werkzeug.utils import secure_filename
from datetime import datetime
import json
from preprocessing import AudioVideoPreprocessor
from fusion_model import DysarthriaFusionModel
from audio_model import AudioAnalysisModel
from visual_model import VisualAnalysisModel
import torchaudio
import whisper
from pathlib import Path
import tempfile

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit
app.config['TRAINING_DATA'] = 'training_data'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRAINING_DATA'], exist_ok=True)

# Initialize models
class DysarthriaAnalysisSystem:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.preprocessor = AudioVideoPreprocessor()
        self.audio_model = AudioAnalysisModel()
        self.visual_model = VisualAnalysisModel()
        self.fusion_model = DysarthriaFusionModel()
        
        # Load Whisper for transcription
        self.whisper_model = whisper.load_model("base")
        
        # Load models to device
        self.audio_model.to(self.device)
        self.visual_model.to(self.device)
        self.fusion_model.to(self.device)
        
        # Set to evaluation mode
        self.audio_model.eval()
        self.visual_model.eval()
        self.fusion_model.eval()
        
        print("All models loaded successfully")
    
    def analyze_audio_video(self, audio_path, video_path):
        """Main analysis function"""
        try:
            # Preprocess audio and video
            print("Preprocessing audio and video...")
            audio_features = self.preprocessor.extract_audio_features(audio_path)
            visual_features = self.preprocessor.extract_visual_features(video_path)
            
            # Analyze with individual models
            print("Analyzing audio...")
            audio_analysis = self.audio_model.analyze(audio_features)
            
            print("Analyzing visual features...")
            visual_analysis = self.visual_model.analyze(visual_features)
            
            # Fusion analysis
            print("Performing fusion analysis...")
            fusion_result = self.fusion_model.analyze(
                audio_features, 
                visual_features,
                audio_analysis,
                visual_analysis
            )
            
            # Transcribe audio
            print("Transcribing audio...")
            transcription = self.transcribe_audio(audio_path)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(
                audio_analysis, 
                visual_analysis, 
                fusion_result
            )
            
            return {
                'success': True,
                'transcription': transcription,
                'audio_analysis': audio_analysis,
                'visual_analysis': visual_analysis,
                'fusion_analysis': fusion_result,
                'recommendations': recommendations,
                'confidence_score': fusion_result.get('confidence', 0.0)
            }
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper"""
        try:
            result = self.whisper_model.transcribe(audio_path)
            return result['text']
        except Exception as e:
            print(f"Transcription error: {e}")
            return "Transcription failed"
    
    def generate_recommendations(self, audio_analysis, visual_analysis, fusion_analysis):
        """Generate personalized training recommendations"""
        recommendations = []
        
        # Audio-based recommendations
        if audio_analysis.get('clarity_score', 0) < 0.7:
            recommendations.append({
                'type': 'audio',
                'exercise': 'Slow pronunciation practice',
                'description': 'Practice speaking slowly with clear enunciation',
                'duration': '10 minutes daily',
                'difficulty': 'Beginner'
            })
        
        if audio_analysis.get('pitch_variability', 0) < 0.5:
            recommendations.append({
                'type': 'audio',
                'exercise': 'Pitch variation exercises',
                'description': 'Practice varying your pitch while speaking',
                'duration': '5 minutes daily',
                'difficulty': 'Intermediate'
            })
        
        # Visual-based recommendations
        if visual_analysis.get('lip_sync_score', 0) < 0.6:
            recommendations.append({
                'type': 'visual',
                'exercise': 'Lip movement synchronization',
                'description': 'Practice speaking while watching your lip movements in mirror',
                'duration': '15 minutes daily',
                'difficulty': 'Beginner'
            })
        
        if visual_analysis.get('mouth_opening_score', 0) < 0.5:
            recommendations.append({
                'type': 'visual',
                'exercise': 'Mouth opening exercises',
                'description': 'Practice exaggerated mouth movements for better articulation',
                'duration': '8 minutes daily',
                'difficulty': 'Beginner'
            })
        
        # Fusion-based recommendations
        if fusion_analysis.get('sync_confidence', 0) < 0.6:
            recommendations.append({
                'type': 'combined',
                'exercise': 'Audio-visual synchronization',
                'description': 'Record yourself speaking and compare audio with lip movements',
                'duration': '12 minutes daily',
                'difficulty': 'Intermediate'
            })
        
        return recommendations
    
    def create_training_session(self, user_id, analysis_results):
        """Create personalized training session"""
        training_session = {
            'user_id': user_id,
            'session_id': f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'created_at': datetime.now().isoformat(),
            'baseline': analysis_results,
            'exercises': analysis_results.get('recommendations', []),
            'progress': [],
            'current_exercise': 0
        }
        
        # Save training session
        session_file = os.path.join(
            app.config['TRAINING_DATA'], 
            f"{training_session['session_id']}.json"
        )
        
        with open(session_file, 'w') as f:
            json.dump(training_session, f, indent=2)
        
        return training_session

# Initialize the system
analysis_system = DysarthriaAnalysisSystem()

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record_session():
    """Handle recording submission"""
    try:
        # Get files from request
        audio_file = request.files.get('audio')
        video_file = request.files.get('video')
        
        if not audio_file or not video_file:
            return jsonify({'success': False, 'error': 'Both audio and video files are required'})
        
        # Save files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        audio_filename = f"audio_{timestamp}.wav"
        video_filename = f"video_{timestamp}.mp4"
        
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        
        audio_file.save(audio_path)
        video_file.save(video_path)
        
        # Store in session
        session['current_audio'] = audio_path
        session['current_video'] = video_path
        
        return jsonify({
            'success': True,
            'message': 'Files uploaded successfully',
            'audio_path': audio_path,
            'video_path': video_path
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze the recorded session"""
    try:
        audio_path = session.get('current_audio')
        video_path = session.get('current_video')
        
        if not audio_path or not video_path:
            return jsonify({'success': False, 'error': 'No recording found. Please record first.'})
        
        # Perform analysis
        analysis_results = analysis_system.analyze_audio_video(audio_path, video_path)
        
        if analysis_results['success']:
            # Create training session
            user_id = session.get('user_id', 'anonymous')
            training_session = analysis_system.create_training_session(user_id, analysis_results)
            session['current_training'] = training_session
            
            return jsonify({
                'success': True,
                'analysis': analysis_results,
                'training_session': training_session
            })
        else:
            return jsonify({'success': False, 'error': analysis_results.get('error', 'Analysis failed')})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/training/exercises')
def get_exercises():
    """Get training exercises"""
    try:
        training_session = session.get('current_training')
        if not training_session:
            return jsonify({'success': False, 'error': 'No training session found'})
        
        exercises = training_session.get('exercises', [])
        return jsonify({
            'success': True,
            'exercises': exercises,
            'current_exercise': training_session.get('current_exercise', 0)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/training/start', methods=['POST'])
def start_training():
    """Start a training exercise"""
    try:
        data = request.json
        exercise_index = data.get('exercise_index', 0)
        
        training_session = session.get('current_training')
        if not training_session:
            return jsonify({'success': False, 'error': 'No training session found'})
        
        exercises = training_session.get('exercises', [])
        if exercise_index >= len(exercises):
            return jsonify({'success': False, 'error': 'Invalid exercise index'})
        
        # Update current exercise
        training_session['current_exercise'] = exercise_index
        session['current_training'] = training_session
        
        exercise = exercises[exercise_index]
        
        return jsonify({
            'success': True,
            'exercise': exercise,
            'instructions': self._generate_exercise_instructions(exercise)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def _generate_exercise_instructions(self, exercise):
    """Generate detailed instructions for an exercise"""
    exercise_type = exercise.get('type', 'audio')
    
    instructions = {
        'audio': {
            'Slow pronunciation practice': [
                "1. Take a deep breath and relax your facial muscles",
                "2. Choose a simple word or phrase",
                "3. Say it very slowly, exaggerating each syllable",
                "4. Focus on clear enunciation",
                "5. Repeat 10 times, recording each attempt",
                "6. Listen back and compare with the previous attempts"
            ],
            'Pitch variation exercises': [
                "1. Start with a neutral tone",
                "2. Gradually raise your pitch on stressed syllables",
                "3. Practice with sentences that have emotional content",
                "4. Record and analyze your pitch contour",
                "5. Aim for natural variation, not monotone"
            ]
        },
        'visual': {
            'Lip movement synchronization': [
                "1. Stand in front of a mirror",
                "2. Watch your lips as you speak",
                "3. Focus on clear, deliberate movements",
                "4. Practice with vowel sounds first (A, E, I, O, U)",
                "5. Gradually add consonants",
                "6. Record and review your lip movements"
            ],
            'Mouth opening exercises': [
                "1. Practice exaggerated yawning motions",
                "2. Say 'AH' with maximum mouth opening",
                "3. Hold for 3 seconds, then relax",
                "4. Practice with different vowel sounds",
                "5. Focus on consistent opening for similar sounds"
            ]
        },
        'combined': {
            'Audio-visual synchronization': [
                "1. Record yourself saying a simple sentence",
                "2. Watch the recording with sound off first",
                "3. Note when your lips move vs when sound should occur",
                "4. Practice slowing down to match lip movements with sounds",
                "5. Gradually increase speed while maintaining sync"
            ]
        }
    }
    
    return instructions.get(exercise_type, {}).get(exercise.get('exercise', ''), [
        "Follow the general description provided",
        "Practice consistently for best results",
        "Record and review your progress regularly"
    ])

@app.route('/training/complete', methods=['POST'])
def complete_exercise():
    """Mark an exercise as completed and record progress"""
    try:
        data = request.json
        exercise_index = data.get('exercise_index')
        audio_path = data.get('audio_path')
        video_path = data.get('video_path')
        self_assessment = data.get('self_assessment', {})
        
        training_session = session.get('current_training')
        if not training_session:
            return jsonify({'success': False, 'error': 'No training session found'})
        
        # Analyze the practice attempt
        if audio_path and video_path:
            practice_results = analysis_system.analyze_audio_video(audio_path, video_path)
        else:
            practice_results = None
        
        # Record progress
        progress_entry = {
            'exercise_index': exercise_index,
            'timestamp': datetime.now().isoformat(),
            'self_assessment': self_assessment,
            'analysis': practice_results,
            'completed': True
        }
        
        if 'progress' not in training_session:
            training_session['progress'] = []
        
        training_session['progress'].append(progress_entry)
        session['current_training'] = training_session
        
        # Save updated session
        session_file = os.path.join(
            app.config['TRAINING_DATA'], 
            f"{training_session['session_id']}.json"
        )
        with open(session_file, 'w') as f:
            json.dump(training_session, f, indent=2)
        
        # Check if all exercises completed
        exercises = training_session.get('exercises', [])
        progress = training_session.get('progress', [])
        completed_count = len([p for p in progress if p.get('completed', False)])
        
        return jsonify({
            'success': True,
            'progress': progress_entry,
            'completed_count': completed_count,
            'total_exercises': len(exercises),
            'all_completed': completed_count >= len(exercises)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/progress')
def get_progress():
    """Get user's training progress"""
    try:
        training_session = session.get('current_training')
        if not training_session:
            return jsonify({'success': False, 'error': 'No training session found'})
        
        return jsonify({
            'success': True,
            'progress': training_session.get('progress', []),
            'session_info': {
                'session_id': training_session.get('session_id'),
                'created_at': training_session.get('created_at'),
                'exercises_completed': len([p for p in training_session.get('progress', []) if p.get('completed', False)]),
                'total_exercises': len(training_session.get('exercises', []))
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback about the training"""
    try:
        data = request.json
        feedback = {
            'user_id': session.get('user_id', 'anonymous'),
            'timestamp': datetime.now().isoformat(),
            'rating': data.get('rating'),
            'comments': data.get('comments', ''),
            'difficulty': data.get('difficulty'),
            'helpfulness': data.get('helpfulness')
        }
        
        # Save feedback
        feedback_file = os.path.join(
            app.config['TRAINING_DATA'], 
            f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(feedback_file, 'w') as f:
            json.dump(feedback, f, indent=2)
        
        return jsonify({'success': True, 'message': 'Feedback submitted successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)