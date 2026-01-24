import sys
import os
sys.path.append('..')

import numpy as np
import torch
import librosa
import tempfile
import soundfile as sf
from audio_model import AudioAnalysisModel

def create_dysarthria_test_audio(severity='moderate'):
    """Create test audio with simulated dysarthria characteristics"""
    duration = 10.0
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration), False)
    
    # Base speech signal
    base_freq = 120  # Base pitch for male voice
    audio = 0.5 * np.sin(2 * np.pi * base_freq * t)
    
    # Add formants (vowel-like resonances)
    formants = {
        'normal': [500, 1500, 2500],
        'mild': [450, 1400, 2300],
        'moderate': [400, 1300, 2100],
        'severe': [350, 1200, 1900]
    }
    
    formant_freqs = formants.get(severity, formants['moderate'])
    for i, freq in enumerate(formant_freqs):
        audio += 0.1 * np.sin(2 * np.pi * freq * t)
    
    # Add dysarthria characteristics based on severity
    if severity == 'mild':
        # Slight pitch instability
        pitch_variation = 0.05 * np.sin(2 * np.pi * 2 * t)  # 2Hz variation
        audio *= (1 + pitch_variation)
        
        # Slight amplitude instability (shimmer)
        amplitude_variation = 0.1 * np.sin(2 * np.pi * 3 * t)
        audio *= (1 + amplitude_variation)
        
    elif severity == 'moderate':
        # Moderate pitch instability
        pitch_variation = 0.1 * np.sin(2 * np.pi * 3 * t)
        audio *= (1 + pitch_variation)
        
        # Moderate amplitude instability
        amplitude_variation = 0.2 * np.sin(2 * np.pi * 4 * t)
        audio *= (1 + amplitude_variation)
        
        # Add some breathiness (noise)
        noise = 0.05 * np.random.randn(len(t))
        audio += noise
        
    elif severity == 'severe':
        # Severe pitch instability
        pitch_variation = 0.2 * np.sin(2 * np.pi * 5 * t)
        audio *= (1 + pitch_variation)
        
        # Severe amplitude instability
        amplitude_variation = 0.3 * np.sin(2 * np.pi * 6 * t)
        audio *= (1 + amplitude_variation)
        
        # More noise (breathiness)
        noise = 0.1 * np.random.randn(len(t))
        audio += noise
        
        # Add irregular pauses
        for pause_start in [2.0, 5.0, 8.0]:
            start_idx = int(pause_start * sr)
            end_idx = start_idx + int(0.3 * sr)  # 300ms pause
            if end_idx < len(audio):
                audio[start_idx:end_idx] *= 0.1
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    return audio, sr

def test_audio_model_initialization():
    """Test model initialization and architecture"""
    print("Testing Audio Model Initialization...")
    print("=" * 50)
    
    try:
        # Initialize model
        model = AudioAnalysisModel()
        
        # Check model structure
        print("1. Checking model architecture...")
        
        # Count layers
        conv_layers = sum(1 for _ in model.modules() if isinstance(_, torch.nn.Conv1d))
        lstm_layers = sum(1 for _ in model.modules() if isinstance(_, torch.nn.LSTM))
        linear_layers = sum(1 for _ in model.modules() if isinstance(_, torch.nn.Linear))
        
        print(f"   Convolutional layers: {conv_layers}")
        print(f"   LSTM layers: {lstm_layers}")
        print(f"   Linear layers: {linear_layers}")
        
        # Check attention mechanism
        if hasattr(model, 'attention'):
            print("   ✅ Attention mechanism present")
        else:
            print("   ❌ Attention mechanism missing")
        
        # Check analysis heads
        required_heads = ['clarity_head', 'articulation_head', 'fluency_head', 
                         'pitch_stability_head', 'severity_classifier']
        missing_heads = [head for head in required_heads if not hasattr(model, head)]
        
        if missing_heads:
            print(f"   ❌ Missing analysis heads: {missing_heads}")
            return False
        else:
            print("   ✅ All analysis heads present")
        
        # Test model forward pass with random input
        print("\n2. Testing forward pass...")
        batch_size = 2
        time_steps = 100
        features = 128
        
        dummy_input = torch.randn(batch_size, features, time_steps)
        with torch.no_grad():
            output = model(dummy_input)
        
        # Check output structure
        required_outputs = ['clarity_score', 'articulation_score', 'fluency_score',
                          'pitch_stability', 'severity', 'features']
        
        missing_outputs = [key for key in required_outputs if key not in output]
        
        if missing_outputs:
            print(f"   ❌ Missing outputs: {missing_outputs}")
            return False
        else:
            print("   ✅ Forward pass successful")
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    print(f"     {key}: shape {value.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Initialization test failed: {e}")
        return False

def test_audio_feature_extraction():
    """Test feature extraction for dysarthria analysis"""
    print("\nTesting Audio Feature Extraction...")
    print("=" * 50)
    
    try:
        # Create test audio
        print("1. Creating test audio with dysarthria characteristics...")
        audio, sr = create_dysarthria_test_audio('moderate')
        
        # Save to temporary file
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(temp_audio.name, audio, sr)
        
        # Extract features using librosa (simulating preprocessing)
        print("2. Extracting audio features...")
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        
        # Create feature dictionary
        features = {
            'mfcc': mfcc,
            'mel_spec': mel_spec,
            'chroma': chroma,
            'spectral_centroid': spectral_centroid,
            'pitch_mean': 120.0,
            'pitch_std': 15.0,
            'energy_mean': 0.5,
            'energy_variance': 0.1,
            'articulation_rate': 4.5,
            'pause_ratio': 0.3,
            'speech_rate': 3.8,
            'duration': 10.0,
            'jitter': 0.8,
            'shimmer': 0.12
        }
        
        print(f"   MFCC shape: {mfcc.shape}")
        print(f"   Mel spectrogram shape: {mel_spec.shape}")
        print(f"   Chroma shape: {chroma.shape}")
        
        # Clean up
        os.unlink(temp_audio.name)
        
        return features
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return None

def test_audio_analysis():
    """Test complete audio analysis pipeline"""
    print("\nTesting Audio Analysis Pipeline...")
    print("=" * 50)
    
    try:
        # Initialize model
        model = AudioAnalysisModel()
        model.eval()  # Set to evaluation mode
        
        # Get test features
        features = test_audio_feature_extraction()
        if features is None:
            return False
        
        print("\n3. Running audio analysis...")
        
        # Test analysis method
        analysis_result = model.analyze(features)
        
        # Check analysis results
        print("4. Checking analysis results...")
        
        required_keys = [
            'clarity_score', 'articulation_score', 'fluency_score',
            'pitch_stability', 'severity_level', 'severity_label',
            'additional_metrics', 'pitch_variability', 'energy_consistency'
        ]
        
        missing_keys = [key for key in required_keys if key not in analysis_result]
        
        if missing_keys:
            print(f"   ❌ Missing analysis keys: {missing_keys}")
            return False
        
        print("   ✅ All analysis keys present")
        
        # Display results
        print("\n5. Analysis Results:")
        print("-" * 30)
        print(f"   Clarity Score: {analysis_result['clarity_score']:.3f}")
        print(f"   Articulation Score: {analysis_result['articulation_score']:.3f}")
        print(f"   Fluency Score: {analysis_result['fluency_score']:.3f}")
        print(f"   Pitch Stability: {analysis_result['pitch_stability']:.3f}")
        print(f"   Severity Level: {analysis_result['severity_level']}")
        print(f"   Severity Label: {analysis_result['severity_label']}")
        print(f"   Pitch Variability: {analysis_result['pitch_variability']:.3f}")
        print(f"   Energy Consistency: {analysis_result['energy_consistency']:.3f}")
        
        # Check additional metrics
        if 'additional_metrics' in analysis_result:
            print(f"\n   Additional Metrics:")
            for key, value in analysis_result['additional_metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"     {key}: {value:.4f}")
        
        # Test with different severity levels
        print("\n6. Testing different severity levels...")
        severity_levels = ['mild', 'moderate', 'severe']
        
        for severity in severity_levels:
            audio, sr = create_dysarthria_test_audio(severity)
            
            # Simple feature extraction for test
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            test_features = {
                'mfcc': mfcc,
                'pitch_mean': 120.0,
                'pitch_std': 10.0 + severity_levels.index(severity) * 5,
                'duration': 10.0
            }
            
            result = model.analyze(test_features)
            print(f"   {severity.capitalize():10} -> Clarity: {result['clarity_score']:.3f}, "
                  f"Severity: {result['severity_label']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_robustness():
    """Test model robustness with edge cases"""
    print("\nTesting Model Robustness...")
    print("=" * 50)
    
    try:
        model = AudioAnalysisModel()
        model.eval()
        
        print("1. Testing with empty features...")
        empty_features = {}
        result1 = model.analyze(empty_features)
        print(f"   Result with empty features: {result1.get('clarity_score', 'N/A'):.3f}")
        
        print("\n2. Testing with minimal features...")
        minimal_features = {
            'mfcc': np.random.randn(13, 100),
            'duration': 5.0
        }
        result2 = model.analyze(minimal_features)
        print(f"   Result with minimal features: {result2.get('clarity_score', 'N/A'):.3f}")
        
        print("\n3. Testing with noisy input...")
        noisy_features = {
            'mfcc': np.random.randn(13, 100) * 10,  # High variance
            'pitch_mean': 1000,  # Unusually high
            'pitch_std': 500,    # Very unstable
            'duration': 1.0      # Very short
        }
        result3 = model.analyze(noisy_features)
        print(f"   Result with noisy features: {result3.get('clarity_score', 'N/A'):.3f}")
        
        print("\n4. Testing model device compatibility...")
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
        
        for device in devices:
            torch_device = torch.device(device)
            model_device = AudioAnalysisModel().to(torch_device)
            model_device.eval()
            
            # Test forward pass on device
            test_input = torch.randn(1, 128, 100).to(torch_device)
            with torch.no_grad():
                output = model_device(test_input)
            
            print(f"   Model works on {device.upper()}: ✓")
        
        return True
        
    except Exception as e:
        print(f"❌ Robustness test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("AUDIO MODEL TEST")
    print("=" * 60)
    
    tests = [
        ("Model Initialization", test_audio_model_initialization),
        ("Audio Analysis", test_audio_analysis),
        ("Model Robustness", test_model_robustness)
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
        print("\n🎉 All audio model tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check above for details.")