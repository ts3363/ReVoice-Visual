import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

class ProgressTracker:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.sessions_file = f"data/users/{user_id}/sessions.json"
        self.load_sessions()
    
    def load_sessions(self):
        """Load user's session history"""
        try:
            with open(self.sessions_file, 'r') as f:
                self.sessions = json.load(f)
        except:
            self.sessions = []
    
    def add_session(self, session_data: Dict):
        """Add a new session to history"""
        session_data["timestamp"] = datetime.now().isoformat()
        self.sessions.append(session_data)
        self.save_sessions()
    
    def save_sessions(self):
        """Save sessions to file"""
        os.makedirs(os.path.dirname(self.sessions_file), exist_ok=True)
        with open(self.sessions_file, 'w') as f:
            json.dump(self.sessions, f, indent=2)
    
    def get_progress_summary(self) -> Dict:
        """Get comprehensive progress summary"""
        if not self.sessions:
            return {"message": "No sessions yet"}
        
        # Calculate trends
        clarity_scores = [s.get("average_clarity", 0) for s in self.sessions]
        dates = [datetime.fromisoformat(s["timestamp"]) for s in self.sessions]
        
        # Generate progress chart
        chart_base64 = self.generate_progress_chart(dates, clarity_scores)
        
        # Calculate statistics
        recent_sessions = self.sessions[-5:]  # Last 5 sessions
        if len(recent_sessions) >= 2:
            improvement = recent_sessions[-1]["average_clarity"] - recent_sessions[0]["average_clarity"]
        else:
            improvement = 0
        
        return {
            "total_sessions": len(self.sessions),
            "current_streak": self.calculate_streak(),
            "average_clarity": np.mean(clarity_scores),
            "best_score": max(clarity_scores),
            "improvement_trend": "positive" if improvement > 0 else "neutral",
            "improvement_amount": round(improvement, 2),
            "progress_chart": chart_base64,
            "milestones": self.check_milestones(),
            "recommended_focus": self.recommend_focus_areas()
        }
    
    def generate_progress_chart(self, dates, scores):
        """Generate progress chart as base64 image"""
        plt.figure(figsize=(10, 4))
        plt.plot(dates, scores, 'b-o', linewidth=2, markersize=8)
        plt.fill_between(dates, scores, alpha=0.3)
        
        # Add trend line
        if len(scores) > 1:
            x_numeric = np.arange(len(scores))
            z = np.polyfit(x_numeric, scores, 1)
            p = np.poly1d(z)
            plt.plot(dates, p(x_numeric), "r--", alpha=0.5, label=f"Trend")
        
        plt.title(f"Speech Clarity Progress - {self.user_id}")
        plt.xlabel("Session Date")
        plt.ylabel("Clarity Score")
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_str
    
    def calculate_streak(self) -> int:
        """Calculate current daily streak"""
        streak = 0
        today = datetime.now().date()
        
        for session in reversed(self.sessions):
            session_date = datetime.fromisoformat(session["timestamp"]).date()
            if session_date == today - timedelta(days=streak):
                streak += 1
            else:
                break
        
        return streak
    
    def check_milestones(self) -> List[Dict]:
        """Check achieved milestones"""
        milestones = []
        
        total_sessions = len(self.sessions)
        if total_sessions >= 10:
            milestones.append({"name": "10 Sessions", "achieved": True})
        if total_sessions >= 25:
            milestones.append({"name": "25 Sessions", "achieved": True})
        
        # Check score milestones
        best_score = max([s.get("average_clarity", 0) for s in self.sessions])
        if best_score >= 70:
            milestones.append({"name": "70+ Clarity Score", "achieved": True})
        if best_score >= 85:
            milestones.append({"name": "85+ Clarity Score", "achieved": True})
        
        return milestones
    
    def recommend_focus_areas(self) -> List[str]:
        """Recommend areas to focus on based on history"""
        # Analyze common issues across sessions
        focus_areas = []
        
        # This would analyze session data to find patterns
        # For now, return generic recommendations
        focus_areas.append("Consonant clarity - especially at word endings")
        focus_areas.append("Speech pacing - maintain consistent speed")
        
        return focus_areas