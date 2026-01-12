@echo off
echo Setting up Dysarthria Analysis System...
echo.

echo Updating pip...
python -m pip install --upgrade pip

echo Installing NumPy...
pip install numpy==1.24.3

echo Installing SciPy...
pip install scipy==1.10.1

echo Installing librosa...
pip install librosa==0.10.0

echo Installing PyTorch...
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

echo Installing OpenCV...
pip install opencv-python==4.8.0.74

echo Installing Whisper...
pip install openai-whisper==20231117

echo Installing MediaPipe...
pip install mediapipe==0.10.3

echo Installing Flask...
pip install Flask==2.3.3

echo Installing other dependencies...
pip install moviepy==1.0.3
pip install scikit-learn==1.3.0
pip install pydub==0.25.1
pip install soundfile==0.12.1
pip install python-dotenv==1.0.0
pip install Flask-CORS==4.0.0

echo.
echo Setup complete!
pause