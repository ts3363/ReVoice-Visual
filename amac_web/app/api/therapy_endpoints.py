from fastapi import APIRouter, WebSocket, HTTPException
from typing import Dict, List
import json
import asyncio

from app.therapy_session import TherapySession
from app.progress_tracker import ProgressTracker

router = APIRouter()

# Store active sessions
active_sessions: Dict[str, TherapySession] = {}

@router.post("/therapy/start")
async def start_therapy_session(request: Dict):
    """Start a new therapy session"""
    user_id = request.get("user_id")
    impairment_level = request.get("impairment_level", "moderate")
    
    # Create new therapy session
    session = TherapySession(user_id, impairment_level)
    session.start_session()
    
    # Store session
    active_sessions[session.session_id] = session
    
    return {
        "session_id": session.session_id,
        "message": "Therapy session started",
        "total_exercises": len(session.exercises),
        "estimated_duration": sum(e.duration_seconds for e in session.exercises)
    }

@router.get("/therapy/current-exercise")
async def get_current_exercise(session_id: str):
    """Get current exercise for session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    exercise = session.get_current_exercise()
    
    if not exercise:
        return {"completed": True, "message": "All exercises completed"}
    
    return exercise

@router.post("/therapy/process-attempt")
async def process_attempt(request: Dict):
    """Process user's speech attempt"""
    session_id = request.get("session_id")
    audio_data = request.get("audio_data")  # base64 encoded
    
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    # Process audio (simplified - integrate with your model)
    # In reality, you would decode audio and extract features
    
    # Mock processing for demonstration
    import random
    clarity_score = 60 + random.randint(0, 30)
    articulation = {
        "bilabials": random.uniform(0.5, 0.9),
        "fricatives": random.uniform(0.4, 0.8),
        "vowels": random.uniform(0.7, 0.95)
    }
    
    # Process with session
    result = session.process_user_response(
        audio_features={},  # Your actual features
        video_features={},  # Your actual features
        recognized_text=request.get("recognized_text", "")
    )
    
    # Update progress tracker
    tracker = ProgressTracker(session.user_id)
    tracker.add_session({
        "session_id": session_id,
        "average_clarity": clarity_score,
        "exercise_completed": session.current_exercise_index
    })
    
    return result

@router.post("/therapy/next-exercise")
async def next_exercise(request: Dict):
    """Move to next exercise"""
    session_id = request.get("session_id")
    
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    # Move to next exercise
    next_ex = session.next_exercise()
    
    if next_ex is None:
        # Session complete
        summary = session.end_session()
        del active_sessions[session_id]
        
        # Update progress
        tracker = ProgressTracker(session.user_id)
        tracker.add_session(summary)
        
        return {
            "completed": True,
            "summary": summary
        }
    
    return {
        "completed": False,
        "exercise": next_ex
    }

@router.post("/therapy/end-session")
async def end_session(request: Dict):
    """End current session"""
    session_id = request.get("session_id")
    
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    summary = session.end_session()
    
    # Update progress
    tracker = ProgressTracker(session.user_id)
    tracker.add_session(summary)
    
    del active_sessions[session_id]
    
    return summary

@router.get("/user/progress")
async def get_user_progress(user_id: str):
    """Get user's progress overview"""
    tracker = ProgressTracker(user_id)
    progress = tracker.get_progress_summary()
    
    return progress

@router.websocket("/therapy-ws")
async def therapy_websocket(websocket: WebSocket):
    """WebSocket for real-time therapy feedback"""
    await websocket.accept()
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            if data["type"] == "audio_chunk":
                # Process audio chunk in real-time
                # This would use your real-time audio processing
                
                # Send back real-time metrics
                await websocket.send_json({
                    "type": "real_time_metrics",
                    "volume": 65,  # Calculate from audio
                    "pace": 45,    # Calculate from speech rate
                    "clarity": 72  # Real-time clarity estimate
                })
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()