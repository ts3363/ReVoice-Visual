from fastapi import APIRouter, HTTPException
from typing import Dict, List
import json
import time
import random

router = APIRouter()

# Exercise database
EXERCISES = [
    {"id": 1, "target": "AH", "instructions": "Open mouth wide, say 'AH'", "type": "phoneme", "difficulty": 1},
    {"id": 2, "target": "EE", "instructions": "Smile wide, say 'EE'", "type": "phoneme", "difficulty": 1},
    {"id": 3, "target": "Hello", "instructions": "Say 'Hello' clearly", "type": "word", "difficulty": 2},
    {"id": 4, "target": "Good morning", "instructions": "Greet naturally", "type": "sentence", "difficulty": 3},
    {"id": 5, "target": "The sun is bright", "instructions": "Clear 'S' and 'T' sounds", "type": "sentence", "difficulty": 3},
    {"id": 6, "target": "Mama", "instructions": "Strong 'M' sound", "type": "word", "difficulty": 2},
    {"id": 7, "target": "She sells seashells", "instructions": "Tongue twister - go slow", "type": "challenge", "difficulty": 4},
]

# User data storage - per user
users_db = {}

# Default user data template
def get_default_user_data(username: str):
    """Create default user data for a new user"""
    return {
        "user_id": username.lower().replace(" ", "_"),
        "name": username,
    "impairment_level": "moderate",
        "total_sessions": 0,
        "day_streak": 0,
        "overall_score": 0,
        "created_at": time.ctime(time.time())
}

# Mock session data
sessions_db = {}
current_sessions = {}

@router.get("/user/profile")
async def get_user_profile(username: str = None):
    """Get user profile data"""
    if not username:
        # Default to test_user if no username provided (for backward compatibility)
        username = "test_user"
    
    username_lower = username.lower()
    
    # Get or create user data
    if username_lower not in users_db:
        users_db[username_lower] = get_default_user_data(username)
    
    user_data = users_db[username_lower].copy()
    
    # Update from actual sessions
    user_sessions = [s for s in sessions_db.values() if s.get("user_id", "").lower() == username_lower]
    if user_sessions:
        user_data["total_sessions"] = len(user_sessions)
        all_scores = []
        for session in user_sessions:
            all_scores.extend(session.get("scores", []))
        if all_scores:
            user_data["overall_score"] = round(sum(all_scores) / len(all_scores), 1)
    
    return user_data

@router.get("/user/progress")
async def get_user_progress(username: str = None):
    """Get user progress data"""
    if not username:
        username = "test_user"
    
    username_lower = username.lower()
    
    # Get user sessions
    user_sessions = [s for s in sessions_db.values() if s.get("user_id", "").lower() == username_lower]
    
    if not user_sessions:
        # Return default data for new users
    return {
            "best_score": 0,
            "average_clarity": 0,
            "improvement_amount": 0,
            "total_sessions": 0,
            "current_streak": 0,
        "milestones": [
                {"name": "First Session", "achieved": False},
                {"name": "5 Sessions", "achieved": False, "progress": "0/5"},
                {"name": "70+ Score", "achieved": False},
                {"name": "10 Sessions", "achieved": False, "progress": "0/10"}
            ]
        }
    
    # Calculate actual progress from sessions
    all_scores = []
    for session in user_sessions:
        all_scores.extend(session.get("scores", []))
    
    total_sessions = len(user_sessions)
    best_score = max(all_scores) if all_scores else 0
    avg_score = round(sum(all_scores) / len(all_scores), 1) if all_scores else 0
    
    # Calculate milestones
    milestones = [
        {"name": "First Session", "achieved": total_sessions >= 1},
        {"name": "5 Sessions", "achieved": total_sessions >= 5, "progress": f"{total_sessions}/5"},
        {"name": "70+ Score", "achieved": avg_score >= 70},
        {"name": "10 Sessions", "achieved": total_sessions >= 10, "progress": f"{total_sessions}/10"}
        ]
    
    return {
        "best_score": best_score,
        "average_clarity": avg_score,
        "improvement_amount": max(0, round(avg_score - 50, 1)),  # Assuming base of 50
        "total_sessions": total_sessions,
        "current_streak": 1,  # Simplified - could calculate actual streak
        "milestones": milestones
    }

@router.post("/therapy/start")
async def start_therapy_session(request: Dict):
    """Start a new therapy session"""
    user_id = request.get("user_id", "test_user")
    session_id = f"{user_id}_{int(time.time())}"
    
    # Select 3-5 random exercises
    num_exercises = random.randint(3, 5)
    selected_exercises = random.sample(EXERCISES, min(num_exercises, len(EXERCISES)))
    
    # Create session
    session = {
        "session_id": session_id,
        "user_id": user_id,
        "start_time": time.time(),
        "exercises": selected_exercises,
        "current_exercise_index": 0,
        "completed_exercises": [],
        "scores": []
    }
    
    current_sessions[session_id] = session
    
    # Get first exercise
    first_exercise = selected_exercises[0]
    
    return {
        "session_id": session_id,
        "message": "Session started successfully",
        "total_exercises": len(selected_exercises),
        "first_exercise": {
            "exercise_number": 1,
            "total_exercises": len(selected_exercises),
            **first_exercise
        }
    }

@router.get("/therapy/current-exercise/{session_id}")
async def get_current_exercise(session_id: str):
    """Get current exercise for session"""
    if session_id not in current_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = current_sessions[session_id]
    current_idx = session["current_exercise_index"]
    
    if current_idx >= len(session["exercises"]):
        # Session complete
        avg_score = sum(session["scores"]) / len(session["scores"]) if session["scores"] else 0
        return {
            "completed": True,
            "summary": {
                "session_id": session_id,
                "exercises_completed": len(session["completed_exercises"]),
                "average_score": round(avg_score, 1),
                "best_score": max(session["scores"]) if session["scores"] else 0,
                "duration_minutes": round((time.time() - session["start_time"]) / 60, 1)
            }
        }
    
    current_exercise = session["exercises"][current_idx]
    
    return {
        "completed": False,
        "exercise": {
            "exercise_number": current_idx + 1,
            "total_exercises": len(session["exercises"]),
            **current_exercise
        }
    }

@router.post("/therapy/process-attempt")
async def process_attempt(request: Dict):
    """Process a speech attempt"""
    session_id = request.get("session_id")
    
    if not session_id or session_id not in current_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = current_sessions[session_id]
    current_idx = session["current_exercise_index"]
    
    if current_idx >= len(session["exercises"]):
        raise HTTPException(status_code=400, detail="No current exercise")
    
    # Generate random score based on difficulty
    current_exercise = session["exercises"][current_idx]
    base_score = 100 - (current_exercise["difficulty"] * 10)
    random_variation = random.randint(-20, 20)
    score = max(30, min(100, base_score + random_variation))
    
    # Generate feedback
    feedback = generate_feedback(score, current_exercise)
    
    # Store result
    result = {
        "exercise": current_exercise,
        "score": score,
        "feedback": feedback
    }
    
    session["completed_exercises"].append(result)
    session["scores"].append(score)
    session["current_exercise_index"] += 1
    
    # Check if session is complete
    next_exercise = None
    if session["current_exercise_index"] < len(session["exercises"]):
        next_ex = session["exercises"][session["current_exercise_index"]]
        next_exercise = {
            "exercise_number": session["current_exercise_index"] + 1,
            "total_exercises": len(session["exercises"]),
            **next_ex
        }
    
    response = {
        "result": result,
        "exercise_completed": True,
        "next_available": next_exercise is not None
    }
    
    if next_exercise:
        response["next_exercise"] = next_exercise
    else:
        # Session complete
        avg_score = sum(session["scores"]) / len(session["scores"])
        response["session_summary"] = {
            "session_id": session_id,
            "exercises_completed": len(session["completed_exercises"]),
            "average_score": round(avg_score, 1),
            "best_score": max(session["scores"]),
            "duration_minutes": round((time.time() - session["start_time"]) / 60, 1)
        }
        # Move to permanent storage
        sessions_db[session_id] = session
        del current_sessions[session_id]
    
    return response

def generate_feedback(score: int, exercise: Dict):
    """Generate feedback based on score"""
    if score >= 85:
        points = ["Excellent clarity!", "Perfect pronunciation"]
        suggestions = ["Try more complex sentences", "Focus on pacing"]
    elif score >= 70:
        points = ["Good job!", "Clear speech detected"]
        suggestions = ["Work on consonant endings", "Practice with longer phrases"]
    elif score >= 60:
        points = ["Fair attempt", "Some clarity issues"]
        suggestions = ["Speak more slowly", "Take deeper breaths"]
    else:
        points = ["Needs practice", "Focus on basics"]
        suggestions = ["Start with simpler sounds", "Practice daily"]
    
    encouragements = [
        "Keep up the good work!",
        "Every attempt makes you better!",
        "You're making progress!",
        "Consistency is key!"
    ]
    
    return {
        "points": points,
        "suggestions": suggestions[:2],
        "encouragement": random.choice(encouragements)
    }

@router.get("/therapy/sessions")
async def get_user_sessions(user_id: str):
    """Get all sessions for a user"""
    user_id_lower = user_id.lower()
    user_sessions = [s for s in sessions_db.values() if s.get("user_id", "").lower() == user_id_lower]
    
    return {
        "user_id": user_id,
        "total_sessions": len(user_sessions),
        "sessions": [
            {
                "session_id": s["session_id"],
                "date": time.ctime(s["start_time"]),
                "exercises_completed": len(s["completed_exercises"]),
                "average_score": round(sum(s["scores"]) / len(s["scores"]), 1) if s["scores"] else 0
            }
            for s in sorted(user_sessions, key=lambda x: x.get("start_time", 0), reverse=True)[:10]  # Last 10 sessions
        ]
    }
