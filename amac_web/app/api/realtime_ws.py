# app/api/realtime_ws.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, List
import numpy as np
import json
import asyncio
from datetime import datetime
import base64
from app.models.dysarthria_asr import asr_engine
from app.models.adaptive_feedback import feedback_engine
import io

router = APIRouter(prefix="/realtime", tags=["realtime"])

# Store active connections
active_connections: List[WebSocket] = []
active_sessions: Dict[str, Dict] = {}

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    session_id = None
    
    try:
        while True:
            # Receive data
            data = await websocket.receive_json()
            
            if data["type"] == "start_session":
                # Start new session
                session_id = data.get("sessionId")
                asr_engine.start_session(session_id)
                active_sessions[session_id] = {
                    "websocket": websocket,
                    "start_time": datetime.now(),
                    "audio_buffer": [],
                    "utterances": []
                }
                
                await manager.send_personal_message({
                    "type": "session_started",
                    "sessionId": session_id,
                    "timestamp": datetime.now().isoformat()
                }, websocket)
            
            elif data["type"] == "audio_chunk":
                # Process audio chunk
                if session_id and session_id in active_sessions:
                    audio_data = np.array(data["audio"], dtype=np.float32)
                    
                    # Store in buffer
                    active_sessions[session_id]["audio_buffer"].extend(audio_data)
                    
                    # Process for real-time features
                    features = asr_engine.process_audio_features(audio_data)
                    
                    # Send real-time feedback
                    feedback = generate_realtime_feedback(features)
                    
                    await manager.send_personal_message({
                        "type": "realtime_feedback",
                        "feedback": feedback,
                        "features": features,
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
            
            elif data["type"] == "end_utterance":
                # Process complete utterance
                if session_id and session_id in active_sessions:
                    session = active_sessions[session_id]
                    
                    if session["audio_buffer"]:
                        # Convert buffer to numpy array
                        audio_array = np.array(session["audio_buffer"], dtype=np.float32)
                        
                        # Get reference text (in real app, would come from exercise prompt)
                        real_text = get_next_exercise_prompt(session_id)
                        
                        # Transcribe and analyze
                        analysis = asr_engine.transcribe_audio(audio_array, real_text)
                        
                        # Format for display
                        display_text = feedback_engine.format_analysis_display(analysis)
                        
                        # Send analysis result
                        await manager.send_personal_message({
                            "type": "analysis_result",
                            "analysis": {
                                "real_text": analysis.real_text,
                                "predicted_text": analysis.predicted_text,
                                "clarity_score": analysis.clarity_score,
                                "articulation_score": analysis.articulation_score,
                                "fluency_score": analysis.fluency_score,
                                "phoneme_accuracy": analysis.phoneme_accuracy,
                                "feedback_messages": analysis.feedback_messages
                            },
                            "display": display_text,
                            "timestamp": datetime.now().isoformat()
                        }, websocket)
                        
                        # Clear buffer
                        session["audio_buffer"] = []
                        
                        # Store utterance
                        session["utterances"].append({
                            "timestamp": datetime.now().isoformat(),
                            "real_text": real_text,
                            "analysis": {
                                "clarity": analysis.clarity_score,
                                "articulation": analysis.articulation_score
                            }
                        })
            
            elif data["type"] == "end_session":
                # End session
                if session_id and session_id in active_sessions:
                    summary = asr_engine.end_session()
                    
                    await manager.send_personal_message({
                        "type": "session_summary",
                        "summary": summary,
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
                    
                    # Remove session
                    del active_sessions[session_id]
            
            elif data["type"] == "get_exercises":
                # Get personalized exercises
                exercises = feedback_engine.generate_personalized_exercises()
                
                await manager.send_personal_message({
                    "type": "exercises_list",
                    "exercises": exercises,
                    "timestamp": datetime.now().isoformat()
                }, websocket)
            
            elif data["type"] == "export_data":
                # Export session data
                if session_id and session_id in active_sessions:
                    session_data = active_sessions[session_id]
                    export_data = format_export_data(session_data)
                    
                    await manager.send_personal_message({
                        "type": "export_ready",
                        "data": export_data,
                        "filename": f"amac_session_{session_id}.json",
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        
        # Clean up session
        if session_id and session_id in active_sessions:
            # End session if active
            asr_engine.end_session()
            del active_sessions[session_id]

def generate_realtime_feedback(features: Dict) -> str:
    """Generate real-time feedback based on audio features"""
    feedback = []
    
    if features.get('energy', 0) < 0.005:
        feedback.append("Speak louder")
    elif features.get('energy', 0) > 0.05:
        feedback.append("Good volume!")
    
    if features.get('zero_crossing_rate', 0) > 0.1:
        feedback.append("Clear articulation")
    
    if features.get('speech_rate', 0) < 5:
        feedback.append("Increase speech rate")
    elif features.get('speech_rate', 0) > 15:
        feedback.append("Slow down slightly")
    
    if not feedback:
        feedback.append("Keep speaking clearly")
    
    return feedback[0]

def get_next_exercise_prompt(session_id: str) -> str:
    """Get next exercise prompt for the session"""
    # In a real app, this would come from a database of exercises
    # For now, use a predefined list
    exercises = [
        "bin blue at b eight now",
        "she sells seashells by the seashore",
        "peter piper picked a peck of pickled peppers",
        "the quick brown fox jumps over the lazy dog",
        "how much wood would a woodchuck chuck",
        "red lorry yellow lorry",
        "unique New York",
        "toy boat toy boat toy boat"
    ]
    
    session = active_sessions.get(session_id, {})
    utterance_count = len(session.get("utterances", []))
    
    return exercises[utterance_count % len(exercises)]

def format_export_data(session_data: Dict) -> Dict:
    """Format session data for export"""
    return {
        "session_id": list(active_sessions.keys())[0] if active_sessions else "unknown",
        "start_time": session_data.get("start_time").isoformat() if session_data.get("start_time") else "",
        "end_time": datetime.now().isoformat(),
        "total_utterances": len(session_data.get("utterances", [])),
        "utterances": session_data.get("utterances", []),
        "analysis_summary": asr_engine.get_realtime_metrics(),
        "system_info": {
            "model": "Whisper + Adaptive Feedback Engine",
            "version": "1.0",
            "export_time": datetime.now().isoformat()
        }
    }

@router.get("/sessions/active")
async def get_active_sessions():
    """Get list of active sessions"""
    return {
        "active_sessions": len(active_sessions),
        "session_ids": list(active_sessions.keys())
    }

@router.get("/session/{session_id}")
async def get_session_data(session_id: str):
    """Get data for specific session"""
    if session_id in active_sessions:
        session = active_sessions[session_id]
        return {
            "session_id": session_id,
            "start_time": session.get("start_time").isoformat(),
            "utterance_count": len(session.get("utterances", [])),
            "audio_buffer_size": len(session.get("audio_buffer", [])),
            "duration": (datetime.now() - session.get("start_time")).seconds
        }
    return {"error": "Session not found"}
