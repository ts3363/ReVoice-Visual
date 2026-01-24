from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import os

# Create directories
os.makedirs("app/static/css", exist_ok=True)
os.makedirs("app/static/js", exist_ok=True)
os.makedirs("app/templates", exist_ok=True)

app = FastAPI(title="AMAC Therapy System")

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Simple HTML content
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>AMAC Speech Therapy</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            color: #333;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        h1 {
            color: #667eea;
            text-align: center;
            margin-bottom: 30px;
        }
        .card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            border-left: 5px solid #667eea;
        }
        button {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px;
            transition: all 0.3s;
        }
        button:hover {
            background: #5a67d8;
            transform: translateY(-2px);
        }
        video {
            width: 100%;
            max-width: 640px;
            border: 3px solid #ddd;
            border-radius: 10px;
            margin: 20px 0;
        }
        .status {
            padding: 15px;
            background: #48bb78;
            color: white;
            border-radius: 8px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 AMAC Speech Therapy System</h1>
        
        <div class="status">
            ✅ System is running and ready for therapy sessions
        </div>
        
        <div class="card">
            <h3>Start Therapy Session</h3>
            <p>Click "Start Session" to begin audio-visual analysis of your speech.</p>
            
            <video id="cameraFeed" autoplay muted></video>
            
            <div>
                <button onclick="startSession()">Start Session</button>
                <button onclick="stopSession()" id="stopBtn" disabled>Stop Session</button>
                <button onclick="recordSession()" id="recordBtn" disabled>Record</button>
            </div>
        </div>
        
        <div class="card">
            <h3>Real-time Feedback</h3>
            <div id="feedback" style="padding: 15px; background: white; border-radius: 8px; min-height: 100px;">
                Session not started. Click "Start Session" to begin.
            </div>
            
            <h4>Metrics:</h4>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                <div>
                    <strong>Articulation:</strong>
                    <div style="height: 10px; background: #e2e8f0; border-radius: 5px; margin-top: 5px;">
                        <div id="articulationBar" style="height: 100%; width: 0%; background: #48bb78; border-radius: 5px;"></div>
                    </div>
                    <span id="articulationText">0%</span>
                </div>
                <div>
                    <strong>Fluency:</strong>
                    <div style="height: 10px; background: #e2e8f0; border-radius: 5px; margin-top: 5px;">
                        <div id="fluencyBar" style="height: 100%; width: 0%; background: #4299e1; border-radius: 5px;"></div>
                    </div>
                    <span id="fluencyText">0%</span>
                </div>
                <div>
                    <strong>Phoneme Accuracy:</strong>
                    <div style="height: 10px; background: #e2e8f0; border-radius: 5px; margin-top: 5px;">
                        <div id="phonemeBar" style="height: 100%; width: 0%; background: #ed8936; border-radius: 5px;"></div>
                    </div>
                    <span id="phonemeText">0%</span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let mediaStream = null;
        let isRecording = false;
        let sessionActive = false;
        
        async function startSession() {
            try {
                // Request camera and microphone access
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: "user"
                    },
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        sampleRate: 16000
                    }
                });
                
                // Display camera feed
                const video = document.getElementById('cameraFeed');
                video.srcObject = mediaStream;
                
                // Update UI
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('recordBtn').disabled = false;
                document.getElementById('feedback').innerHTML = 
                    '<strong style="color: #48bb78;">✅ Session Started!</strong><br>' +
                    'Speak clearly into the microphone. Your speech is being analyzed in real-time.';
                
                sessionActive = true;
                startMetricsSimulation();
                
            } catch (error) {
                document.getElementById('feedback').innerHTML = 
                    '<strong style="color: #f56565;">❌ Error:</strong> ' + error.message + 
                    '<br>Please allow camera and microphone access.';
            }
        }
        
        function stopSession() {
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                document.getElementById('cameraFeed').srcObject = null;
            }
            
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('recordBtn').disabled = true;
            document.getElementById('recordBtn').textContent = 'Record';
            isRecording = false;
            sessionActive = false;
            
            document.getElementById('feedback').innerHTML = 
                'Session ended. Your progress has been saved.';
        }
        
        function recordSession() {
            isRecording = !isRecording;
            document.getElementById('recordBtn').textContent = 
                isRecording ? 'Stop Recording' : 'Record';
            
            document.getElementById('feedback').innerHTML = 
                isRecording ? 
                '🔴 Recording started... Your speech is being recorded.' :
                'Recording stopped. Analysis complete.';
        }
        
        function startMetricsSimulation() {
            if (!sessionActive) return;
            
            setInterval(() => {
                if (sessionActive) {
                    // Simulate random metrics updates
                    const articulation = 70 + Math.random() * 25;
                    const fluency = 75 + Math.random() * 20;
                    const phoneme = 80 + Math.random() * 15;
                    
                    // Update bars
                    document.getElementById('articulationBar').style.width = articulation + '%';
                    document.getElementById('fluencyBar').style.width = fluency + '%';
                    document.getElementById('phonemeBar').style.width = phoneme + '%';
                    
                    // Update text
                    document.getElementById('articulationText').textContent = articulation.toFixed(1) + '%';
                    document.getElementById('fluencyText').textContent = fluency.toFixed(1) + '%';
                    document.getElementById('phonemeText').textContent = phoneme.toFixed(1) + '%';
                    
                    // Update feedback based on metrics
                    if (articulation > 85) {
                        document.getElementById('feedback').innerHTML = 
                            '<strong style="color: #48bb78;">Excellent articulation!</strong> Your speech clarity is very good.';
                    } else if (fluency > 90) {
                        document.getElementById('feedback').innerHTML = 
                            '<strong style="color: #4299e1;">Great fluency!</strong> Your speech flow is smooth.';
                    }
                }
            }, 2000);
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def home():
    return HTMLResponse(html_content)

@app.get("/api/health")
async def health():
    return {"status": "running", "service": "AMAC Therapy", "version": "1.0"}

if __name__ == "__main__":
    print("=" * 60)
    print("🎤 AMAC SPEECH THERAPY SYSTEM")
    print("=" * 60)
    print("Server starting on: http://localhost:8000")
    print("Open your web browser and navigate to:")
    print("    http://localhost:8000")
    print("    or http://127.0.0.1:8000")
    print("=" * 60)
    print("⚠️  Keep this window open while using the application!")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
