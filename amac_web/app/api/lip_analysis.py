# app/api/lip_analysis.py
from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket
from typing import List
import tempfile
import os
import json
from datetime import datetime
from app.models.video_processor import VideoProcessor
import cv2
import base64

router = APIRouter(prefix="/lip", tags=["lip_analysis"])
video_processor = VideoProcessor()

@router.post("/analyze/dysarthria")
async def analyze_dysarthria_lips(
    video_file: UploadFile = File(...),
    target_phoneme: str = "p"
):
    """Analyze lip movements for dysarthria articulation"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            content = await video_file.read()
            tmp.write(content)
            video_path = tmp.name
        
        # Analyze dysarthria lip movements
        features = video_processor.extract_dysarthria_lip_features(video_path)
        
        # Analyze specific phoneme articulation
        phoneme_analysis = video_processor.analyze_phoneme_articulation(
            video_path, target_phoneme
        )
        
        # Get lip exercises
        exercises = video_processor.get_lip_exercises("beginner")
        
        os.unlink(video_path)
        
        return {
            "status": "success",
            "dysarthria_features": features,
            "phoneme_analysis": phoneme_analysis,
            "exercises": exercises[:3],  # First 3 exercises
            "recommendations": self._generate_lip_recommendations(features),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/exercises")
async def get_lip_exercises(
    difficulty: str = "beginner",
    phoneme_focus: str = None
):
    """Get lip exercises for dysarthria therapy"""
    try:
        exercises = video_processor.get_lip_exercises(difficulty)
        
        if phoneme_focus:
            exercises = [e for e in exercises 
                        if phoneme_focus in e.get("target_phonemes", [])]
        
        return {
            "status": "success",
            "difficulty": difficulty,
            "exercises": exercises,
            "count": len(exercises)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/realtime")
async def realtime_lip_analysis(websocket: WebSocket):
    """WebSocket for real-time lip movement analysis"""
    await websocket.accept()
    try:
        cap = cv2.VideoCapture(0)  # Open webcam
        if not cap.isOpened():
            await websocket.send_json({"error": "Could not open camera"})
            return
        
        print("Realtime lip analysis started")
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process for dysarthria analysis
            result = video_processor.process_realtime_for_dysarthria(frame)
            
            if result["landmarks_detected"]:
                # Send analysis results
                await websocket.send_json({
                    "type": "lip_analysis",
                    "lip_state": result["lip_state"],
                    "feedback": result["feedback"],
                    "timestamp": datetime.now().isoformat()
                })
                
                # Send visualization data if requested
                try:
                    data = await websocket.receive_json(timeout=0.1)
                    if data.get("type") == "request_visualization":
                        # Encode frame for visualization
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        await websocket.send_json({
                            "type": "frame",
                            "frame": frame_base64,
                            "lip_contour": result.get("lip_contour", []),
                            "timestamp": datetime.now().isoformat()
                        })
                except:
                    pass  # No visualization request
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": "Face not detected. Please position face in frame.",
                    "timestamp": datetime.now().isoformat()
                })
        
        cap.release()
        
    except Exception as e:
        print(f"Lip analysis WebSocket error: {e}")
    finally:
        await websocket.close()

@router.post("/compare/articulation")
async def compare_articulation(
    video_file1: UploadFile = File(...),
    video_file2: UploadFile = File(...),
    phoneme: str = "p"
):
    """Compare articulation between two attempts"""
    try:
        # Save first video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp1:
            content1 = await video_file1.read()
            tmp1.write(content1)
            path1 = tmp1.name
        
        # Save second video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp2:
            content2 = await video_file2.read()
            tmp2.write(content2)
            path2 = tmp2.name
        
        # Analyze both
        analysis1 = video_processor.analyze_phoneme_articulation(path1, phoneme)
        analysis2 = video_processor.analyze_phoneme_articulation(path2, phoneme)
        
        # Calculate improvement
        improvement = self._calculate_improvement(analysis1, analysis2)
        
        # Cleanup
        os.unlink(path1)
        os.unlink(path2)
        
        return {
            "status": "success",
            "attempt1": analysis1,
            "attempt2": analysis2,
            "improvement": improvement,
            "phoneme": phoneme,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _generate_lip_recommendations(features):
    """Generate personalized lip recommendations for dysarthria"""
    if not features:
        return ["Practice basic lip movements"]
    
    recommendations = []
    
    # Check symmetry
    if features.get('symmetry_issue', 0) > 0.3:
        recommendations.append("Focus on symmetrical lip movements")
        recommendations.append("Practice in front of a mirror")
    
    # Check consistency
    if features.get('movement_variance_std', 0) > 0.02:
        recommendations.append("Work on consistent lip movements")
        recommendations.append("Practice slow, controlled movements")
    
    # Check amplitude
    if features.get('movement_amplitude', 0) < 0.05:
        recommendations.append("Exaggerate lip movements for clarity")
        recommendations.append("Practice wide mouth opening")
    
    if not recommendations:
        recommendations.append("Continue practicing current exercises")
    
    return recommendations

def _calculate_improvement(analysis1, analysis2):
    """Calculate improvement between two attempts"""
    if not analysis1 or not analysis2:
        return {"improvement": 0, "message": "Insufficient data"}
    
    acc1 = analysis1.get("overall_accuracy", 0)
    acc2 = analysis2.get("overall_accuracy", 0)
    
    improvement = ((acc2 - acc1) / acc1 * 100) if acc1 > 0 else 0
    
    if improvement > 10:
        message = "Excellent improvement!"
    elif improvement > 0:
        message = "Good progress"
    else:
        message = "Keep practicing"
    
    return {
        "accuracy_improvement": improvement,
        "message": message,
        "from_accuracy": acc1,
        "to_accuracy": acc2
    }
