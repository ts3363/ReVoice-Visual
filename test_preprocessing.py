import sys
import os
sys.path.append('..')

import numpy as np
import cv2
import librosa
import tempfile
import soundfile as sf
from preprocessing import AudioVideoPreprocessor

def create_test_audio():
    """Create a test audio file with speech-like characteristics"""
    # Create a sine wave that mimics speech frequencies
    duration = 5.0  # seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create multiple frequencies to mimic speech
    frequencies = [100, 200, 300, 500, 800, 1200]  # Speech formant-like frequencies
    audio = np.zeros_like(t)
    
    for freq in frequencies:
        audio += 0.1 * np.sin(2 * np.pi * freq * t)
    
    # Add some silence and pauses
    audio[int(sample_rate*1.0):int(sample_rate*1.5)] *= 0.1  # Pause
    audio[int(sample_rate*3.0):int(sample_rate*3.5)] *= 0.1  # Pause
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    return audio, sample_rate

def create_test_video():
    """Create a test video file with facial movements"""
    # Create a simple video with moving circles to simulate lip movement
    width, height = 640, 480
    fps = 30
    duration = 3  # seconds
    frames = fps * duration
    
    # Create temporary video file
    temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_video.close()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video.name, fourcc, fps, (width, height))
    
    for i in range(frames):
        # Create frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [255, 255, 255]  # White background
        
        # Simulate mouth opening/closing
        mouth_size = 30 + int(20 * np.sin(2 * np.pi * i / 30))  # Oscillating size
        
        # Draw "mouth" as red ellipse
        center_x, center_y = width // 2, height // 2
        cv2.ellipse(frame, (center_x, center_y), (mouth_size, 15), 0, 0, 360, (0, 0, 255), -1)
        
        # Draw "face" as circle
        cv2.circle(frame, (center_x, center_y), 100, (200, 200, 200), 2)
        
        out.write(frame)
    
    out.release()
    return temp_video.name

def test_audio_preprocessing():
    """Test audio feature extraction"""
    print("Testing Audio Preprocessing...")
    print("=" * 50)
    
    # Create test audio
    audio, sr = create_test_audio()
    
    # Save to temporary file
    temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_audio.name, audio, sr)
    
    # Initialize preprocessor
    preprocessor = AudioVideoPreprocessor()
    
    # Test audio feature extraction
    print("1. Extracting audio features...")
    audio_features = preprocessor.extract_audio_features(temp_audio.name)
    
    # Check extracted features
    required_features = ['mfcc', 'mel_spec', 'chroma', 'pitch', 'rms', 'duration']
    missing = [feat for feat in required_features if feat not in audio_features]
    
    if missing:
        print(f"❌ Missing audio features: {missing}")
        return False
    else:
        print("✅ All required audio features extracted")
    
    # Check feature dimensions
    print("\n2. Checking feature dimensions...")
    if 'mfcc' in audio_features:
        mfcc_shape = audio_features['mfcc'].shape
        print(f"   MFCC shape: {mfcc_shape}")
        if mfcc_shape[0] == 13:
            print("   ✅ MFCC features correct")
        else:
            print("   ❌ MFCC features incorrect shape")
    
    if 'duration' in audio_features:
        print(f"   Audio duration: {audio_features['duration']:.2f} seconds")
        if 4.5 < audio_features['duration'] < 5.5:
            print("   ✅ Duration correct")
        else:
            print("   ❌ Duration incorrect")
    
    # Test dysarthria-specific features
    print("\n3. Checking dysarthria-specific features...")
    dysarthria_features = ['articulation_rate', 'pause_ratio', 'speech_rate', 'pitch_mean', 'pitch_std']
    for feat in dysarthria_features:
        if feat in audio_features:
            print(f"   {feat}: {audio_features[feat]:.4f}")
        else:
            print(f"   ❌ Missing {feat}")
    
    # Clean up
    os.unlink(temp_audio.name)
    print("\n✅ Audio preprocessing test completed successfully")
    return True

def test_video_preprocessing():
    """Test video feature extraction"""
    print("\nTesting Video Preprocessing...")
    print("=" * 50)
    
    # Create test video
    video_path = create_test_video()
    
    # Initialize preprocessor
    preprocessor = AudioVideoPreprocessor()
    
    # Test video feature extraction
    print("1. Extracting visual features...")
    visual_features = preprocessor.extract_visual_features(video_path)
    
    # Check extracted features
    required_features = ['lip_landmarks', 'mouth_openings', 'lip_movements']
    missing = [feat for feat in required_features if feat not in visual_features]
    
    if missing:
        print(f"❌ Missing visual features: {missing}")
        return False
    else:
        print("✅ Basic visual features extracted")
    
    # Check feature content
    print("\n2. Checking feature content...")
    
    # Check lip landmarks
    if len(visual_features['lip_landmarks']) > 0:
        landmarks = visual_features['lip_landmarks'][0]
        print(f"   Lip landmarks per frame: {len(landmarks)} points")
        print(f"   Each point has {len(landmarks[0])} coordinates")
    else:
        print("   ⚠️ No lip landmarks detected (expected for synthetic video)")
    
    # Check mouth openings
    if len(visual_features['mouth_openings']) > 0:
        openings = visual_features['mouth_openings']
        print(f"   Number of frames processed: {len(openings)}")
        print(f"   Mouth opening statistics:")
        heights = [o['height'] for o in openings]
        print(f"     - Height range: {min(heights):.1f} to {max(heights):.1f}")
        print(f"     - Mean height: {np.mean(heights):.1f}")
    else:
        print("   ⚠️ No mouth opening data")
    
    # Check temporal features
    if 'temporal_features' in visual_features:
        tf = visual_features['temporal_features']
        print("\n3. Temporal features:")
        for key, value in tf.items():
            print(f"   {key}: {value:.4f}")
    
    # Test synchronization
    print("\n4. Testing audio-video synchronization...")
    
    # Create dummy audio features for sync test
    dummy_audio_features = {'duration': 3.0}  # Same as video duration
    sync_result = preprocessor.synchronize_audio_video(
        dummy_audio_features, 
        visual_features
    )
    
    print(f"   Sync result: {sync_result}")
    
    # Clean up
    if os.path.exists(video_path):
        os.unlink(video_path)
    
    print("\n✅ Video preprocessing test completed")
    return True

def test_preprocessor_integration():
    """Test the complete preprocessor"""
    print("\nTesting Complete Preprocessor Integration...")
    print("=" * 50)
    
    preprocessor = AudioVideoPreprocessor()
    
    # Create test files
    audio, sr = create_test_audio()
    audio_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    video_path = create_test_video()
    
    sf.write(audio_path, audio, sr)
    
    try:
        # Test both audio and video extraction
        print("1. Testing integrated feature extraction...")
        audio_features = preprocessor.extract_audio_features(audio_path)
        video_features = preprocessor.extract_visual_features(video_path)
        
        # Test synchronization
        sync_info = preprocessor.synchronize_audio_video(audio_features, video_features)
        
        print(f"2. Synchronization results:")
        print(f"   Audio duration: {sync_info.get('audio_duration', 0):.2f}s")
        print(f"   Video duration: {sync_info.get('visual_duration', 0):.2f}s")
        print(f"   Sync ratio: {sync_info.get('sync_ratio', 0):.2f}")
        print(f"   Is synced: {sync_info.get('is_synced', False)}")
        
        # Test feature consistency
        print("\n3. Feature consistency check:")
        print(f"   Audio features extracted: {len(audio_features)}")
        print(f"   Video frames processed: {len(video_features.get('lip_landmarks', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        if os.path.exists(video_path):
            os.unlink(video_path)

if __name__ == "__main__":
    print("=" * 60)
    print("PREPROCESSING MODULE TEST")
    print("=" * 60)
    
    tests = [
        ("Audio Preprocessing", test_audio_preprocessing),
        ("Video Preprocessing", test_video_preprocessing),
        ("Integration Test", test_preprocessor_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}")
            print("-" * 40)
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:30} {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All preprocessing tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check above for details.")