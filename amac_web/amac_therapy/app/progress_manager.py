import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

class ProgressManager:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.users_file = os.path.join(data_dir, "users.json")
        self.sessions_file = os.path.join(data_dir, "sessions.json")
        self.ensure_data_files()
    
    def ensure_data_files(self):
        """Create data files if they don't exist"""
        os.makedirs(self.data_dir, exist_ok=True)
        
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w') as f:
                json.dump({}, f)
        
        if not os.path.exists(self.sessions_file):
            with open(self.sessions_file, 'w') as f:
                json.dump([], f)
    
    def add_user(self, user_id: str, name: str, impairment_level: str = "moderate"):
        """Add a new user or update existing"""
        with open(self.users_file, 'r') as f:
            users = json.load(f)
        
        if user_id not in users:
            users[user_id] = {
                "name": name,
                "impairment_level": impairment_level,
                "created_at": datetime.now().isoformat(),
                "total_sessions": 0,
                "best_score": 0,
                "average_score": 0,
                "last_session": None
            }
        else:
            users[user_id]["name"] = name
            users[user_id]["impairment_level"] = impairment_level
        
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)
        
        return users[user_id]
    
    def add_session(self, user_id: str, session_data: Dict):
        """Add a new therapy session"""
        # Update users file
        with open(self.users_file, 'r') as f:
            users = json.load(f)
        
        if user_id not in users:
            users[user_id] = {
                "name": "User",
                "impairment_level": "moderate",
                "created_at": datetime.now().isoformat(),
                "total_sessions": 0,
                "best_score": 0,
                "average_score": 0,
                "last_session": None
            }
        
        # Update user stats
        users[user_id]["total_sessions"] += 1
        users[user_id]["last_session"] = datetime.now().isoformat()
        
        if "average_score" in session_data:
            avg_score = session_data["average_score"]
            users[user_id]["best_score"] = max(users[user_id]["best_score"], avg_score)
            
            # Update average score
            if users[user_id]["average_score"] == 0:
                users[user_id]["average_score"] = avg_score
            else:
                total_sessions = users[user_id]["total_sessions"]
                old_avg = users[user_id]["average_score"]
                users[user_id]["average_score"] = ((old_avg * (total_sessions - 1)) + avg_score) / total_sessions
        
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)
        
        # Add to sessions file
        with open(self.sessions_file, 'r') as f:
            sessions = json.load(f)
        
        session_record = {
            "session_id": session_data.get("session_id", ""),
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "exercises_completed": session_data.get("exercises_completed", 0),
            "average_score": session_data.get("average_score", 0),
            "best_score": session_data.get("best_score", 0),
            "duration_minutes": session_data.get("duration_minutes", 0)
        }
        
        sessions.append(session_record)
        
        with open(self.sessions_file, 'w') as f:
            json.dump(sessions, f, indent=2)
        
        return session_record
    
    def get_user_progress(self, user_id: str) -> Dict:
        """Get comprehensive progress for a user"""
        with open(self.users_file, 'r') as f:
            users = json.load(f)
        
        with open(self.sessions_file, 'r') as f:
            sessions = json.load(f)
        
        user_sessions = [s for s in sessions if s["user_id"] == user_id]
        
        if user_id not in users:
            return {
                "error": "User not found",
                "sessions": [],
                "stats": {}
            }
        
        user_data = users[user_id]
        
        # Calculate streak
        streak = self.calculate_streak(user_sessions)
        
        # Get recent scores for chart
        recent_sessions = user_sessions[-10:]  # Last 10 sessions
        scores = [s.get("average_score", 0) for s in recent_sessions]
        dates = [datetime.fromisoformat(s["timestamp"]).strftime("%m/%d") for s in recent_sessions]
        
        # Generate chart
        chart_base64 = self.generate_progress_chart(dates, scores) if scores else None
        
        # Calculate milestones
        milestones = self.calculate_milestones(user_data, user_sessions)
        
        return {
            "user": user_data,
            "total_sessions": len(user_sessions),
            "current_streak": streak,
            "recent_scores": scores[-5:] if scores else [],
            "recent_dates": dates[-5:] if dates else [],
            "progress_chart": chart_base64,
            "milestones": milestones,
            "last_session": user_sessions[-1] if user_sessions else None
        }
    
    def calculate_streak(self, sessions: List[Dict]) -> int:
        """Calculate current daily streak"""
        if not sessions:
            return 0
        
        streak = 0
        today = datetime.now().date()
        
        # Sort by date descending
        sorted_sessions = sorted(sessions, 
                               key=lambda x: datetime.fromisoformat(x["timestamp"]), 
                               reverse=True)
        
        current_date = today
        
        for session in sorted_sessions:
            session_date = datetime.fromisoformat(session["timestamp"]).date()
            
            if session_date == current_date:
                streak += 1
                current_date -= timedelta(days=1)
            elif session_date == current_date + timedelta(days=1):
                # Session was yesterday, continue streak
                streak += 1
                current_date = session_date
            else:
                break
        
        return streak
    
    def generate_progress_chart(self, dates: List[str], scores: List[float]) -> str:
        """Generate progress chart as base64 image"""
        plt.figure(figsize=(10, 4))
        
        # Plot scores
        plt.plot(dates, scores, 'b-o', linewidth=2, markersize=8, label='Clarity Score')
        plt.fill_between(dates, scores, alpha=0.3)
        
        # Add trend line
        if len(scores) > 1:
            x_numeric = list(range(len(scores)))
            z = np.polyfit(x_numeric, scores, 1)
            p = np.poly1d(z)
            plt.plot(dates, p(x_numeric), "r--", alpha=0.5, label=f"Trend")
        
        plt.title("Speech Clarity Progress")
        plt.xlabel("Session Date")
        plt.ylabel("Score (0-100)")
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_str
    
    def calculate_milestones(self, user_data: Dict, sessions: List[Dict]) -> List[Dict]:
        """Calculate achieved milestones"""
        milestones = []
        
        total_sessions = len(sessions)
        best_score = user_data.get("best_score", 0)
        
        # Session milestones
        if total_sessions >= 1:
            milestones.append({"name": "First Session", "achieved": True, "icon": "🥳"})
        if total_sessions >= 5:
            milestones.append({"name": "5 Sessions", "achieved": True, "icon": "🎯"})
        if total_sessions >= 10:
            milestones.append({"name": "10 Sessions", "achieved": True, "icon": "🏆"})
        elif total_sessions >= 8:
            milestones.append({"name": "10 Sessions", "achieved": False, "icon": "🏆", "progress": f"{total_sessions}/10"})
        
        # Score milestones
        if best_score >= 60:
            milestones.append({"name": "60+ Score", "achieved": True, "icon": "👍"})
        if best_score >= 75:
            milestones.append({"name": "75+ Score", "achieved": True, "icon": "👏"})
        if best_score >= 85:
            milestones.append({"name": "85+ Score", "achieved": True, "icon": "🌟"})
        elif best_score >= 80:
            milestones.append({"name": "85+ Score", "achieved": False, "icon": "🌟", "progress": f"{best_score}/85"})
        
        # Streak milestones
        streak = self.calculate_streak(sessions)
        if streak >= 3:
            milestones.append({"name": "3-Day Streak", "achieved": True, "icon": "🔥"})
        if streak >= 7:
            milestones.append({"name": "7-Day Streak", "achieved": True, "icon": "💪"})
        
        return milestones

# Import numpy for trend line calculation
import numpy as np
