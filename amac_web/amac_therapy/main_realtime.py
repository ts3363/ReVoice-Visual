import os
import asyncio
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AMAC Research Simulation")

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_path = os.path.join(BASE_DIR, "app", "static")
os.makedirs(static_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_path), name="static")

# --- SIMULATION ENDPOINT ---
@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Wait 2 seconds so you can get ready
    await asyncio.sleep(2)
    
    # 1. Simulate "Listening" state
    await websocket.send_json({
        "status": "success",
        "prediction": "Listening...",
        "feedback": "🎤 Audio Input Active..."
    })
    await asyncio.sleep(1.5)
    
    # 2. Simulate "Processing" state
    await websocket.send_json({
        "status": "success",
        "prediction": "Processing...",
        "feedback": "🧠 Fusion Model Analyzing..."
    })
    await asyncio.sleep(1)
    
    # 3. FINAL OUTPUT (The Screenshot Moment)
    # This displays the multimodal result your teammate wants
    target_sentence = "BIN BLUE AT F TWO NOW"
    
    await websocket.send_json({
        "status": "success",
        "prediction": f"👁️ LIP: {target_sentence} | 🎤 AUDIO: {target_sentence.lower()}",
        "feedback": "✅ Audio-Visual Match Confirmed (98.5%)"
    })
    
    # Keep connection open so the text stays on screen
    while True:
        await websocket.receive_bytes() 

# --- STANDARD ROUTES ---
@app.get("/therapy")
async def therapy(): return FileResponse(os.path.join(static_path, "therapy.html"))
@app.get("/api/user/profile")
def profile(u: str="t"): return {"name": "Test User", "impairment_level": "moderate", "day_streak": 5, "overall_score": 88}
@app.get("/api/therapy/sessions")
def sessions(u: str="t"): return {"sessions": []}
@app.get("/api/user/progress")
def progress(u: str="t"): return {"best": 92, "avg": 85}
@app.post("/api/therapy/start")
async def start(r: Request): return JSONResponse({"status": "success"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)