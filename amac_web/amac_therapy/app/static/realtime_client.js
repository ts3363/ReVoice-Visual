// Connect to the WebSocket
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsUrl = `${protocol}//${window.location.host}/ws/predict`;
console.log(`[*] Connecting to WebSocket at: ${wsUrl}`);

const ws = new WebSocket(wsUrl);
let stream = null;
let isRecording = false;

// 1. Connection Status
ws.onopen = () => {
    console.log("[*] Connected to ReVoice AI Server");
    updateStatus("AI System Ready", "green");
};

ws.onmessage = (event) => {
    try {
        const data = JSON.parse(event.data);
        if (data.status === "success") {
            // Show the raw prediction index for now
            const feedbackElement = document.getElementById("feedback-text");
            if (feedbackElement) {
                feedbackElement.innerText = "Prediction Index: " + data.prediction;
                feedbackElement.style.color = "#2c5aa0";
            }
        }
    } catch (e) {
        console.error("Error parsing message:", e);
    }
};

ws.onerror = (error) => {
    console.error("WebSocket Error:", error);
    updateStatus("Connection Error", "red");
};

// 2. Start Camera & Loop
async function startRealTimeSession() {
    console.log("[*] Starting Real-Time Session...");
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        
        const videoElement = document.getElementById("user-video");
        if (videoElement) videoElement.srcObject = stream;

        isRecording = true;
        updateStatus("Session Active - Analyzing...", "blue");
        
        // Update Button
        toggleButton(true);

        // Start the recording loop
        recordClip();

    } catch (err) {
        console.error("Error accessing webcam:", err);
        alert("Could not access camera. Please allow permissions.");
    }
}

// 3. The "Valid File" Loop
function recordClip() {
    if (!isRecording) return;

    // Create a NEW recorder for every clip to ensure it has a valid header
    const recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
    const chunks = [];

    recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
    };

    recorder.onstop = () => {
        // Bundle chunks into a single valid Blob
        const blob = new Blob(chunks, { type: 'video/webm' });
        
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(blob);
            console.log(`[>] Sent valid clip: ${blob.size} bytes`);
        }
        
        // Immediately start the next clip if still recording
        if (isRecording) {
            recordClip();
        }
    };

    // Record for 2 seconds, then stop to force file generation
    recorder.start();
    setTimeout(() => {
        if (recorder.state === "recording") recorder.stop();
    }, 2000);
}

// 4. Stop Session
function stopRealTimeSession() {
    console.log("[*] Stopping Session...");
    isRecording = false;
    
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    updateStatus("Session Paused", "grey");
    toggleButton(false);
}

function toggleButton(recording) {
    const btn = document.getElementById("start-btn");
    if (!btn) return;

    if (recording) {
        btn.innerText = "Stop Session";
        btn.onclick = stopRealTimeSession;
        btn.classList.remove("btn-success");
        btn.classList.add("btn-danger");
    } else {
        btn.innerText = "Start New Session";
        btn.onclick = startRealTimeSession;
        btn.classList.remove("btn-danger");
        btn.classList.add("btn-success");
    }
}

function updateStatus(text, color) {
    const statusEl = document.getElementById("connection-status");
    if (statusEl) {
        statusEl.innerText = text;
        // Map simple colors to Bootstrap classes
        const map = { 'red': 'danger', 'green': 'success', 'blue': 'primary', 'grey': 'secondary' };
        statusEl.className = `badge bg-${map[color] || 'secondary'} mb-3`;
    }
}

// Expose to HTML
window.startRealTimeSession = startRealTimeSession;
window.stopRealTimeSession = stopRealTimeSession;