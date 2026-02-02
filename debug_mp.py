import sys
import os

print("--- DEBUGGING MEDIAPIPE ---")
print(f"Python Version: {sys.version}")

try:
    import mediapipe
    print(f"\n1. MediaPipe Location: {mediapipe.__file__}")
    print(f"2. MediaPipe Dir: {os.path.dirname(mediapipe.__file__)}")
    
    if hasattr(mediapipe, 'solutions'):
        print("3. SUCCESS: 'solutions' found!")
    else:
        print("3. FAILURE: 'solutions' NOT found.")
        print("   CONTENTS of MediaPipe folder:")
        print(os.listdir(os.path.dirname(mediapipe.__file__)))
        
except ImportError as e:
    print(f"\nCRITICAL ERROR: Could not import mediapipe. {e}")
except Exception as e:
    print(f"\nOTHER ERROR: {e}")

print("\n--- CHECKING LOCAL FOLDER ---")
local_files = os.listdir('.')
if 'mediapipe' in local_files:
    print("ALERT: Found a folder named 'mediapipe' in THIS directory!")
    print("Please RENAME or DELETE it.")
else:
    print("Clean: No 'mediapipe' folder found in project directory.")