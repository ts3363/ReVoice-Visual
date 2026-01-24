# app/api/dysarthria_endpoints.py
from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket
from typing import List
import tempfile
import os
import numpy as np
from app.models.audio_processor import AudioProcessor
from app.models.video_processor import VideoProcessor
from datetime import datetime

router = APIRouter(prefix="/dysarthria", tags=["dysarthria"])
audio_processor = AudioProcessor()
video_processor = VideoProcessor()

@router.post("/analyze/audio")
async def analyze_dysarthria_audio(file: UploadFile = File(...)):
    """Analyze dysarthria speech from audio file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Analyze dysarthria speech
        analysis = audio_processor.assess_articulation(tmp_path)
        
        # Also try recognition
        recognition = audio_processor.recognize_dysarthria_speech(tmp_path)
        
        os.unlink(tmp_path)
        
        return {
            "status": "success",
            "analysis": analysis,
            "recognition": recognition,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/audiovisual")
async def analyze_audiovisual(
    audio_file: UploadFile = File(...),
    video_file: UploadFile = File(...)
):
    """Multi-modal analysis for dysarthria"""
    try:
        # Save audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_tmp:
            audio_content = await audio_file.read()
            audio_tmp.write(audio_content)
            audio_path = audio_tmp.name
        
        # Save video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_tmp:
            video_content = await video_file.read()
            video_tmp.write(video_content)
            video_path = video_tmp.name
        
        # Extract features
        audio_features = audio_processor.extract_dysarthria_features(audio_path)
        video_landmarks = video_processor.extract_landmarks(video_path)
        
        # Multi-modal analysis
        audio_analysis = audio_processor.assess_articulation(audio_path)
        
        # Cleanup
        os.unlink(audio_path)
        os.unlink(video_path)
        
        return {
            "status": "success",
            "audio_analysis": audio_analysis,
            "video_landmarks_count": len(video_landmarks) if video_landmarks is not None else 0,
            "modality": "audio-visual",
            "recommended_focus": self.get_recommended_focus(audio_analysis)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/therapy/session")
async def start_therapy_session(session_data: dict):
    """Start a dysarthria therapy session"""
    try:
        # Session configuration
        phonemes = session_data.get("phonemes", ["p", "t", "k", "s", "m", "n"])
        duration = session_data.get("duration", 300)  # 5 minutes
        difficulty = session_data.get("difficulty", "medium")
        
        # Generate therapy exercises
        exercises = self.generate_dysarthria_exercises(phonemes, difficulty)
        
        return {
            "status": "success",
            "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "exercises": exercises,
            "duration_seconds": duration,
            "phonemes_target": phonemes
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/therapy")
async def therapy_websocket(websocket: WebSocket):
    """WebSocket for real-time dysarthria therapy"""
    await websocket.accept()
    try:
        session_data = {
            "start_time": datetime.now(),
            "audio_chunks": [],
            "metrics_history": []
        }
        
        while True:
            # Receive audio data
            data = await websocket.receive_json()
            
            if data["type"] == "audio_chunk":
                # Process audio chunk
                audio_chunk = np.array(data["data"])
                analysis = audio_processor.process_realtime_audio(audio_chunk)
                
                # Store for session analysis
                session_data["audio_chunks"].append(audio_chunk)
                session_data["metrics_history"].append(analysis)
                
                # Send feedback
                feedback = self.generate_realtime_feedback(analysis, session_data["metrics_history"])
                
                await websocket.send_json({
                    "type": "feedback",
                    "analysis": analysis,
                    "feedback": feedback,
                    "timestamp": datetime.now().isoformat()
                })
                
            elif data["type"] == "end_session":
                # Analyze complete session
                session_analysis = self.analyze_session(session_data)
                await websocket.send_json({
                    "type": "session_summary",
                    "analysis": session_analysis
                })
                break
                
    except Exception as e:
        print(f"Therapy WebSocket error: {e}")
    finally:
        await websocket.close()

def get_recommended_focus(analysis):
    """Determine focus area based on analysis"""
    if analysis and "metrics" in analysis:
        metrics = analysis["metrics"]
        
        if metrics.get("mfcc_variance", 0) > 100:
            return "articulation_consistency"
        elif metrics.get("pitch_stability", 0) > 50:
            return "pitch_control"
        elif metrics.get("speech_rate", 1) < 0.5:
            return "speech_rate"
    
    return "general_articulation"

def generate_dysarthria_exercises(phonemes, difficulty):
    """Generate therapy exercises for dysarthria"""
    exercises = []
    
    # Phoneme repetition
    for phoneme in phonemes:
        exercises.append({
            "type": "phoneme_repetition",
            "phoneme": phoneme,
            "instructions": f"Repeat the sound '{phoneme}' clearly 10 times",
            "duration": 30,
            "target": "clear articulation"
        })
    
    # Word practice
    words = ["pat", "bat", "mat", "sat", "cat"]
    exercises.append({
        "type": "word_practice",
        "words": words,
        "instructions": "Pronounce each word slowly and clearly",
        "duration": 60,
        "target": "word clarity"
    })
    
    # Sentence practice
    sentences = [
        "She sells seashells by the seashore",
        "Peter Piper picked a peck of pickled peppers"
    ]
    exercises.append({
        "type": "sentence_practice",
        "sentences": sentences,
        "instructions": "Read each sentence at a comfortable pace",
        "duration": 90,
        "target": "sentence fluency"
    })
    
    return exercises

def generate_realtime_feedback(current_analysis, history):
    """Generate real-time feedback during therapy"""
    if len(history) < 2:
        return "Keep speaking..."
    
    # Analyze trends
    recent_energy = [h["energy"] for h in history[-5:] if "energy" in h]
    if recent_energy:
        avg_energy = np.mean(recent_energy)
        
        if avg_energy < 0.005:
            return "Speak louder"
        elif avg_energy > 0.05:
            return "Good volume!"
    
    return "Good articulation, continue practicing"

def analyze_session(session_data):
    """Analyze complete therapy session"""
    if not session_data["metrics_history"]:
        return {"error": "No data collected"}
    
    metrics = session_data["metrics_history"]
    
    # Calculate session statistics
    energy_values = [m.get("energy", 0) for m in metrics if "energy" in m]
    speech_frames = sum(1 for m in metrics if m.get("is_speech", False))
    
    return {
        "total_duration": len(metrics),
        "speech_percentage": speech_frames / len(metrics) * 100 if metrics else 0,
        "average_energy": np.mean(energy_values) if energy_values else 0,
        "energy_consistency": np.std(energy_values) if energy_values else 0,
        "recommendations": [
            "Practice daily for 15 minutes",
            "Focus on difficult phonemes",
            "Use mirror for visual feedback"
        ]
    }
