@echo off
echo Fixing NumPy compatibility issues...
echo.

echo Step 1: Uninstalling incompatible packages...
pip uninstall numpy scipy torch torchvision torchaudio -y

echo Step 2: Installing compatible NumPy version...
pip install numpy==1.24.3

echo Step 3: Installing SciPy...
pip install scipy==1.10.1

echo Step 4: Installing PyTorch with compatible NumPy...
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

echo Step 5: Reinstalling other dependencies...
pip install --force-reinstall librosa==0.10.0
pip install --force-reinstall opencv-python==4.8.0.74
pip install --force-reinstall mediapipe==0.10.3
pip install --force-reinstall openai-whisper==20231117

echo.
echo Fix complete! NumPy should now be compatible.
pause