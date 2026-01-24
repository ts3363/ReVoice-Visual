# app/api/recording_endpoints.py
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import Optional
import os
import json
import uuid
from datetime import datetime
from pathlib import Path
import shutil

router = APIRouter(prefix="/recording", tags=["recording"])

# Configuration
RECORDINGS_DIR = Path("saved_recordings")
SESSIONS_DIR = Path("therapy_sessions")
RECORDINGS_DIR.mkdir(exist_ok=True)
SESSIONS_DIR.mkdir(exist_ok=True)

@router.post("/save")
async def save_recording(
    background_tasks: BackgroundTasks,
    session_id: str,
    video_file: UploadFile = File(...),
    timestamp: Optional[str] = None,
    duration: Optional[int] = None
):
    """Save a video recording from therapy session"""
    try:
        # Validate file type
        if not video_file.filename.endswith(('.webm', '.mp4', '.mkv')):
            raise HTTPException(status_code=400, detail="Invalid video format")
        
        # Generate unique filename
        file_ext = os.path.splitext(video_file.filename)[1] or '.webm'
        timestamp_str = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{session_id}_{timestamp_str}{file_ext}"
        filepath = RECORDINGS_DIR / filename
        
        # Save the file
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        
        # Get file info
        file_size = os.path.getsize(filepath)
        
        # Create session metadata
        session_file = SESSIONS_DIR / f"{session_id}.json"
        session_data = {
            "session_id": session_id,
            "recording_file": str(filepath),
            "filename": filename,
            "file_size": file_size,
            "duration_seconds": duration,
            "timestamp": timestamp or datetime.now().isoformat(),
            "saved_at": datetime.now().isoformat()
        }
        
        # Save/update session metadata
        if session_file.exists():
            with open(session_file, 'r') as f:
                existing_data = json.load(f)
            existing_data["recordings"].append(session_data)
            session_data = existing_data
        else:
            session_data = {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(),
                "recordings": [session_data]
            }
        
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        # Background task: Process video (e.g., extract audio, generate thumbnail)
        background_tasks.add_task(process_video_background, filepath, session_id)
        
        return {
            "status": "success",
            "message": "Recording saved successfully",
            "filename": filename,
            "file_path": str(filepath),
            "file_size": file_size,
            "session_id": session_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving recording: {str(e)}")

@router.get("/list")
async def list_recordings(
    session_id: Optional[str] = None,
    limit: int = 20
):
    """List saved recordings"""
    try:
        recordings = []
        
        # List files in recordings directory
        for file_path in RECORDINGS_DIR.glob("*"):
            if file_path.is_file() and file_path.suffix in ['.webm', '.mp4', '.mkv']:
                file_info = {
                    "filename": file_path.name,
                    "path": str(file_path),
                    "size": os.path.getsize(file_path),
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    "session_id": file_path.stem.split('_')[0] if '_' in file_path.stem else "unknown"
                }
                
                # Filter by session_id if provided
                if session_id and session_id != file_info["session_id"]:
                    continue
                    
                recordings.append(file_info)
        
        # Sort by modification time (newest first)
        recordings.sort(key=lambda x: x["modified"], reverse=True)
        
        return {
            "status": "success",
            "count": len(recordings),
            "recordings": recordings[:limit],
            "directory": str(RECORDINGS_DIR.absolute())
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}")
async def get_session_recordings(session_id: str):
    """Get all recordings for a specific session"""
    try:
        session_file = SESSIONS_DIR / f"{session_id}.json"
        
        if session_file.exists():
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Also find actual files
            recordings = []
            for recording in session_data.get("recordings", []):
                filepath = Path(recording.get("recording_file", ""))
                if filepath.exists():
                    recordings.append({
                        **recording,
                        "exists": True,
                        "url": f"/recordings/{filepath.name}"  # URL for downloading
                    })
                else:
                    recordings.append({**recording, "exists": False})
            
            return {
                "status": "success",
                "session_id": session_id,
                "recordings": recordings,
                "total_recordings": len(recordings)
            }
        else:
            # Try to find by filename pattern
            matching_files = list(RECORDINGS_DIR.glob(f"{session_id}_*"))
            recordings = []
            
            for filepath in matching_files:
                recordings.append({
                    "filename": filepath.name,
                    "path": str(filepath),
                    "size": os.path.getsize(filepath),
                    "modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat()
                })
            
            return {
                "status": "success",
                "session_id": session_id,
                "recordings": recordings,
                "total_recordings": len(recordings),
                "note": "Session file not found, listed matching files"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete/{filename}")
async def delete_recording(filename: str):
    """Delete a recording"""
    try:
        filepath = RECORDINGS_DIR / filename
        
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="Recording not found")
        
        # Also remove from session metadata
        session_id = filename.split('_')[0]
        session_file = SESSIONS_DIR / f"{session_id}.json"
        
        if session_file.exists():
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Remove recording from session data
            session_data["recordings"] = [
                r for r in session_data.get("recordings", [])
                if r.get("filename") != filename
            ]
            
            # Update or delete session file
            if session_data["recordings"]:
                with open(session_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
            else:
                os.remove(session_file)
        
        # Delete the file
        os.remove(filepath)
        
        return {
            "status": "success",
            "message": f"Recording '{filename}' deleted successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{filename}")
async def download_recording(filename: str):
    """Download a recording"""
    from fastapi.responses import FileResponse
    
    filepath = RECORDINGS_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Recording not found")
    
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type='video/webm'
    )

@router.post("/convert/{filename}")
async def convert_recording(filename: str, format: str = "mp4"):
    """Convert recording to different format"""
    # Note: This requires ffmpeg installed
    # For now, just return info
    filepath = RECORDINGS_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Recording not found")
    
    return {
        "status": "info",
        "message": f"Conversion to {format} would require ffmpeg",
        "filename": filename,
        "current_format": filepath.suffix,
        "target_format": f".{format}"
    }

def process_video_background(filepath: Path, session_id: str):
    """Background task to process saved video"""
    try:
        print(f"Background processing: {filepath.name}")
        
        # 1. Generate thumbnail
        generate_thumbnail(filepath, session_id)
        
        # 2. Extract audio for analysis
        extract_audio_for_analysis(filepath, session_id)
        
        # 3. Log processing completion
        log_file = SESSIONS_DIR / f"{session_id}_processing.log"
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()}: Processed {filepath.name}\n")
            
    except Exception as e:
        print(f"Background processing error: {e}")

def generate_thumbnail(video_path: Path, session_id: str):
    """Generate thumbnail from video"""
    try:
        import cv2
        
        # Create thumbnails directory
        thumb_dir = RECORDINGS_DIR / "thumbnails"
        thumb_dir.mkdir(exist_ok=True)
        
        # Capture first frame
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            thumb_path = thumb_dir / f"{video_path.stem}_thumb.jpg"
            cv2.imwrite(str(thumb_path), frame)
            print(f"Generated thumbnail: {thumb_path.name}")
            
    except ImportError:
        print("OpenCV not available for thumbnail generation")
    except Exception as e:
        print(f"Thumbnail generation error: {e}")

def extract_audio_for_analysis(video_path: Path, session_id: str):
    """Extract audio for speech analysis"""
    try:
        # Create audio directory
        audio_dir = RECORDINGS_DIR / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        # In a real implementation, use ffmpeg to extract audio
        # For now, just create placeholder
        audio_path = audio_dir / f"{video_path.stem}.wav"
        
        # Create empty file as placeholder
        with open(audio_path, 'w') as f:
            f.write(f"Audio would be extracted from {video_path.name}")
        
        print(f"Audio extraction placeholder: {audio_path.name}")
        
    except Exception as e:
        print(f"Audio extraction error: {e}")
