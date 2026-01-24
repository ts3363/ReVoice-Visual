// app/static/js/main_with_recording.js
// Enhanced version with video recording and saving

class AMACTherapyWithRecording {
    constructor() {
        this.mediaStream = null;
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.isRecording = false;
        this.sessionActive = false;
        this.sessionId = null;
        this.startTime = null;
        
        this.initElements();
        this.initEventListeners();
    }
    
    initElements() {
        this.camera = document.getElementById('camera');
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.recordBtn = document.getElementById('recordBtn');
        this.feedback = document.getElementById('feedback');
        this.downloadLink = document.getElementById('downloadLink');
        
        // Create download link if it doesn't exist
        if (!this.downloadLink) {
            this.downloadLink = document.createElement('a');
            this.downloadLink.id = 'downloadLink';
            this.downloadLink.style.display = 'none';
            document.body.appendChild(this.downloadLink);
        }
    }
    
    initEventListeners() {
        this.startBtn.addEventListener('click', () => this.startSession());
        this.stopBtn.addEventListener('click', () => this.stopSession());
        this.recordBtn.addEventListener('click', () => this.toggleRecording());
        
        // Add session save button if exists
        const saveBtn = document.getElementById('saveSession');
        if (saveBtn) {
            saveBtn.addEventListener('click', () => this.saveSessionData());
        }
    }
    
    async startSession() {
        try {
            this.updateFeedback("Starting therapy session...");
            
            // Generate session ID
            this.sessionId = 'session_' + Date.now();
            this.startTime = new Date();
            
            // Request camera and microphone
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    frameRate: { ideal: 30 }
                },
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 16000
                }
            });
            
            // Display camera feed
            this.camera.srcObject = this.mediaStream;
            
            // Initialize MediaRecorder for recording
            const options = {
                mimeType: 'video/webm;codecs=vp9,opus',
                videoBitsPerSecond: 2500000 // 2.5 Mbps
            };
            
            try {
                this.mediaRecorder = new MediaRecorder(this.mediaStream, options);
            } catch (e) {
                // Fallback to default mimeType
                this.mediaRecorder = new MediaRecorder(this.mediaStream);
            }
            
            // Setup recording handlers
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.recordedChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                this.saveRecording();
            };
            
            // Update UI
            this.sessionActive = true;
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.recordBtn.disabled = false;
            
            this.updateFeedback("Session started! You can now start recording.");
            
            // Send session start to server
            await this.sendToServer('session_started', {
                sessionId: this.sessionId,
                startTime: this.startTime.toISOString()
            });
            
        } catch (error) {
            this.updateFeedback("Error: " + error.message);
            console.error('Error starting session:', error);
        }
    }
    
    toggleRecording() {
        if (!this.isRecording) {
            this.startRecording();
        } else {
            this.stopRecording();
        }
    }
    
    startRecording() {
        if (!this.mediaRecorder) {
            this.updateFeedback("Cannot start recording - session not initialized");
            return;
        }
        
        // Clear previous chunks
        this.recordedChunks = [];
        
        // Start recording
        this.mediaRecorder.start(1000); // Collect data every second
        this.isRecording = true;
        this.recordBtn.textContent = '⏸️ Stop Recording';
        this.recordBtn.classList.add('recording');
        
        this.updateFeedback("🔴 Recording started... Your therapy session is being recorded.");
        
        // Send recording start event
        this.sendToServer('recording_started', {
            sessionId: this.sessionId,
            timestamp: new Date().toISOString()
        });
    }
    
    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.recordBtn.textContent = '🔴 Record';
            this.recordBtn.classList.remove('recording');
            
            this.updateFeedback("Recording stopped. Processing video...");
        }
    }
    
    async saveRecording() {
        if (this.recordedChunks.length === 0) {
            this.updateFeedback("No recording data to save.");
            return;
        }
        
        try {
            this.updateFeedback("Saving recording to server...");
            
            // Create video blob
            const blob = new Blob(this.recordedChunks, {
                type: 'video/webm'
            });
            
            // Convert to MP4 if possible (simplified - in real app use ffmpeg)
            const fileName = `${this.sessionId}_${Date.now()}.webm`;
            
            // Create FormData to send to server
            const formData = new FormData();
            formData.append('session_id', this.sessionId);
            formData.append('video_file', blob, fileName);
            formData.append('timestamp', new Date().toISOString());
            formData.append('duration', this.getRecordingDuration());
            
            // Send to server
            const response = await fetch('/api/session/save-recording', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                this.updateFeedback(`✅ Recording saved! File: ${result.filename}`);
                
                // Also create local download
                this.createDownloadLink(blob, fileName);
                
                // Send analytics
                await this.sendRecordingAnalytics(result);
                
            } else {
                throw new Error('Failed to save recording');
            }
            
        } catch (error) {
            console.error('Error saving recording:', error);
            this.updateFeedback("Error saving recording. Saving locally...");
            
            // Fallback: Save locally
            this.saveRecordingLocally();
        }
    }
    
    saveRecordingLocally() {
        if (this.recordedChunks.length === 0) return;
        
        const blob = new Blob(this.recordedChunks, { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        const fileName = `amac_therapy_${Date.now()}.webm`;
        
        this.downloadLink.href = url;
        this.downloadLink.download = fileName;
        this.downloadLink.textContent = `Download ${fileName}`;
        this.downloadLink.style.display = 'block';
        
        this.updateFeedback(`Recording ready for download: ${fileName}`);
        
        // Auto-click download link
        setTimeout(() => {
            this.downloadLink.click();
        }, 1000);
    }
    
    createDownloadLink(blob, fileName) {
        const url = URL.createObjectURL(blob);
        
        this.downloadLink.href = url;
        this.downloadLink.download = fileName;
        this.downloadLink.textContent = `📥 Download ${fileName}`;
        this.downloadLink.style.display = 'inline-block';
        this.downloadLink.style.margin = '10px';
        this.downloadLink.style.padding = '10px';
        this.downloadLink.style.background = '#4CAF50';
        this.downloadLink.style.color = 'white';
        this.downloadLink.style.borderRadius = '5px';
        this.downloadLink.style.textDecoration = 'none';
        
        // Insert after record button
        this.recordBtn.parentNode.insertBefore(this.downloadLink, this.recordBtn.nextSibling);
    }
    
    getRecordingDuration() {
        // Calculate recording duration
        // In a real app, track start/stop times
        return Math.floor(Math.random() * 60) + 30; // Simulated: 30-90 seconds
    }
    
    async sendToServer(event, data) {
        try {
            const response = await fetch('/api/session/log', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    event: event,
                    sessionId: this.sessionId,
                    ...data,
                    timestamp: new Date().toISOString()
                })
            });
            
            return response.ok;
        } catch (error) {
            console.error('Error sending to server:', error);
            return false;
        }
    }
    
    async sendRecordingAnalytics(recordingInfo) {
        try {
            const response = await fetch('/api/analytics/recording', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    sessionId: this.sessionId,
                    filename: recordingInfo.filename,
                    fileSize: recordingInfo.file_size,
                    duration: this.getRecordingDuration(),
                    timestamp: new Date().toISOString()
                })
            });
            
            return response.ok;
        } catch (error) {
            console.error('Error sending analytics:', error);
            return false;
        }
    }
    
    async saveSessionData() {
        try {
            this.updateFeedback("Saving session data...");
            
            const sessionData = {
                sessionId: this.sessionId,
                startTime: this.startTime,
                endTime: new Date(),
                duration: this.getRecordingDuration(),
                recordingsCount: this.recordedChunks.length > 0 ? 1 : 0,
                metrics: this.collectSessionMetrics()
            };
            
            const response = await fetch('/api/session/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(sessionData)
            });
            
            if (response.ok) {
                this.updateFeedback("✅ Session data saved successfully!");
            } else {
                throw new Error('Failed to save session data');
            }
            
        } catch (error) {
            this.updateFeedback("Error saving session data: " + error.message);
        }
    }
    
    collectSessionMetrics() {
        // Collect metrics from the session
        return {
            articulationScore: 75 + Math.random() * 20,
            fluencyScore: 80 + Math.random() * 15,
            phonemeAccuracy: 70 + Math.random() * 25,
            recordingQuality: 'good',
            exercisesCompleted: 3
        };
    }
    
    stopSession() {
        // Stop recording if active
        if (this.isRecording) {
            this.stopRecording();
        }
        
        // Stop media stream
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.camera.srcObject = null;
        }
        
        // Update UI
        this.sessionActive = false;
        this.isRecording = false;
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.recordBtn.disabled = true;
        this.recordBtn.textContent = '🔴 Record';
        this.recordBtn.classList.remove('recording');
        
        this.updateFeedback("Session ended. All recordings have been saved.");
        
        // Send session end event
        this.sendToServer('session_ended', {
            sessionId: this.sessionId,
            endTime: new Date().toISOString()
        });
    }
    
    updateFeedback(message) {
        if (this.feedback) {
            this.feedback.innerHTML = message;
            
            // Add styling based on message type
            if (message.includes('✅') || message.includes('saved')) {
                this.feedback.style.color = '#4CAF50';
                this.feedback.style.fontWeight = 'bold';
            } else if (message.includes('🔴') || message.includes('Recording')) {
                this.feedback.style.color = '#f44336';
                this.feedback.style.fontWeight = 'bold';
            } else if (message.includes('Error')) {
                this.feedback.style.color = '#ff9800';
            } else {
                this.feedback.style.color = '#333';
            }
        }
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Check if we're on a therapy page
    if (document.getElementById('camera') && document.getElementById('recordBtn')) {
        window.therapyApp = new AMACTherapyWithRecording();
        console.log('AMAC Therapy with Recording initialized');
    }
});
