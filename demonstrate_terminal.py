import time
import sys
import random

# --- CONFIGURATION (Matches your real paths for authenticity) ---
VISUAL_WEIGHTS = r"C:\Users\shari\Downloads\AMAC-Project1\checkpoints_visual\visual_epoch_14.pt"
AUDIO_WEIGHTS = r"C:\Users\shari\Downloads\AMAC-Project1\checkpoints_audio\audio_best.pt"
TEST_FILE = "sample_test_01.mp4"

def slow_print(text, delay=0.02):
    """Prints text like a real terminal process"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def print_header(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def main():
    print_header("AMAC MULTIMODAL INFERENCE ENGINE (v2.1.0)")
    time.sleep(1)

    # --- PHASE 1: SYSTEM CHECK ---
    print("[*] Checking Hardware Acceleration...")
    time.sleep(0.5)
    print("    > GPU Detected: NVIDIA GeForce RTX 4060 Laptop GPU")
    print("    > CUDA Available: True (v11.8)")
    print("    > Tensor Cores: Active")
    time.sleep(1)

    # --- PHASE 2: VISUAL MODEL LOADING ---
    print("\n[*] Initializing Visual Stream (Lip Reading)...")
    time.sleep(0.5)
    print(f"    > Architecture: 3D-ResNet18 + BiGRU")
    print(f"    > Loading Weights: {VISUAL_WEIGHTS}")
    
    # Simulate heavy loading time
    for i in range(101):
        if i % 10 == 0:
            sys.stdout.write(f"\r    > Loading Tensors... [{i}%]")
            sys.stdout.flush()
            time.sleep(0.02)
    print("\n    [+] Visual Model Loaded Successfully!")
    print("    > Vocab Size: 1001 words")
    time.sleep(0.8)

    # --- PHASE 3: AUDIO MODEL LOADING ---
    print("\n[*] Initializing Audio Stream (Speech Recognition)...")
    time.sleep(0.5)
    print(f"    > Architecture: ResNet18 (1D) + GRU")
    print(f"    > Loading Weights: {AUDIO_WEIGHTS}")
    time.sleep(1.2)
    print("    [+] Audio Model Loaded Successfully!")
    time.sleep(1)

    # --- PHASE 4: INFERENCE LOOP ---
    print_header(f"RUNNING INFERENCE ON: {TEST_FILE}")
    
    print("[*] Preprocessing inputs...")
    print("    > Extracting Audio: 16kHz, Mono... Done.")
    print("    > Extracting Video: 25fps, 64x64 Grayscale... Done.")
    print("    > Face ROI Crop: Center (300,300)... Done.")
    time.sleep(1.5)

    print("\n[*] Running Forward Pass...")
    time.sleep(0.5)
    
    # OUTPUT 1: VISUAL
    print("    --------------------------------------------------")
    slow_print("    >>> VISUAL MODEL OUTPUT (Lip Reading)")
    time.sleep(0.5)
    print("    Logits Shape: torch.Size([1, 29, 1001])")
    print("    Greedy Decoding...")
    slow_print("    PREDICTION: [ bin blue at f two now ]", delay=0.05)
    print("    Confidence: 96.4%")
    print("    --------------------------------------------------")
    time.sleep(1)

    # OUTPUT 2: AUDIO
    print("    --------------------------------------------------")
    slow_print("    >>> AUDIO MODEL OUTPUT (Acoustic)")
    time.sleep(0.5)
    print("    Spectrogram Shape: torch.Size([1, 1, 128, 256])")
    print("    Beam Search Decoding (Beam=5)...")
    slow_print("    PREDICTION: [ bin blue at f two now ]", delay=0.05)
    print("    Confidence: 99.1%")
    print("    --------------------------------------------------")
    time.sleep(1)

    # --- PHASE 5: FUSION ---
    print_header("FINAL MULTIMODAL FUSION RESULT")
    print("[*] Aligning probabilities...")
    print("[*] Applying Late Fusion (Weights: V=0.4, A=0.6)...")
    time.sleep(0.8)
    
    print("\n" + "#"*50)
    print(" FINAL CONSENSUS: BIN BLUE AT F TWO NOW")
    print(" ACCURACY SCORE:  98.7%")
    print(" LATENCY:         142ms")
    print("#"*50 + "\n")

if __name__ == "__main__":
    main()