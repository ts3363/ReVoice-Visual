class AMACTherapy {
    constructor() {
        this.mediaStream = null;
        this.isRecording = false;
        this.initElements();
        this.initEvents();
    }
    
    initElements() {
        this.camera = document.getElementById('camera');
        this.startBtn = document.getElementById('start');
        this.stopBtn = document.getElementById('stop');
        this.recordBtn = document.getElementById('record');
        this.feedback = document.getElementById('feedback-text');
        this.articulation = document.getElementById('articulation');
        this.fluency = document.getElementById('fluency');
        this.phoneme = document.getElementById('phoneme');
    }
    
    initEvents() {
        this.startBtn.addEventListener('click', () => this.startSession());
        this.stopBtn.addEventListener('click', () => this.stopSession());
        this.recordBtn.addEventListener('click', () => this.toggleRecording());
    }
    
    async startSession() {
        try {
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: true
            });
            this.camera.srcObject = this.mediaStream;
            
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.recordBtn.disabled = false;
            
            this.updateFeedback("Session started! Speak clearly.");
            this.simulateMetrics();
            
        } catch (error) {
            this.updateFeedback("Error accessing camera/microphone: " + error.message);
        }
    }
    
    stopSession() {
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        this.camera.srcObject = null;
        
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        this.recordBtn.disabled = true;
        this.recordBtn.textContent = 'Record';
        this.isRecording = false;
        
        this.updateFeedback("Session ended.");
    }
    
    toggleRecording() {
        this.isRecording = !this.isRecording;
        this.recordBtn.textContent = this.isRecording ? 'Stop Recording' : 'Record';
        this.updateFeedback(this.isRecording ? "Recording started..." : "Recording stopped.");
    }
    
    updateFeedback(message) {
        this.feedback.textContent = message;
    }
    
    simulateMetrics() {
        // Simulate updating metrics
        setInterval(() => {
            if (this.startBtn.disabled) {
                const articulation = 70 + Math.random() * 20;
                const fluency = 75 + Math.random() * 20;
                const phoneme = 80 + Math.random() * 15;
                
                this.articulation.textContent = articulation.toFixed(1) + '%';
                this.fluency.textContent = fluency.toFixed(1) + '%';
                this.phoneme.textContent = phoneme.toFixed(1) + '%';
                
                if (articulation > 85) {
                    this.updateFeedback("Excellent articulation!");
                } else if (fluency > 90) {
                    this.updateFeedback("Great fluency!");
                }
            }
        }, 2000);
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    new AMACTherapy();
});
