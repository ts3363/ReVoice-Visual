@echo off
echo Installing Python dependencies for Dysarthria Analysis System...
echo.

echo Step 1: Installing basic dependencies...
pip install numpy==1.24.3 scipy==1.10.1

echo Step 2: Installing audio processing libraries...
pip install librosa==0.10.0 soundfile==0.12.1 pydub==0.25.1

echo Step 3: Installing PyTorch (CPU version)...
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

echo Step 4: Installing computer vision libraries...
pip install opencv-python==4.8.0.74 mediapipe==0.10.3

echo Step 5: Installing Whisper for speech recognition...
pip install openai-whisper==20231117

echo Step 6: Installing Flask and web dependencies...
pip install Flask==2.3.3 Flask-CORS==4.0.0 python-dotenv==1.0.0

echo Step 7: Installing additional utilities...
pip install moviepy==1.0.3 scikit-learn==1.3.0

echo.
echo Installation complete!
echo.
echo Now create the required directories:
echo mkdir uploads
echo mkdir training_data
echo mkdir templates
echo mkdir static
pause