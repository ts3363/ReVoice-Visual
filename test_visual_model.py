import sys
import os
sys.path.append('..')

import numpy as np
import torch
import cv2
import tempfile
from visual_model import VisualAnalysisModel

def create_synthetic_lip_landmarks(num_frames=100, severity='moderate'):
    """Create synthetic lip landmarks with dysarthria characteristics"""
    landmarks_sequence = []
    
    # Base lip points (20 points with x,y coordinates)
    # Outer lip: 10 points, Inner lip: 10 points
    base_outer = np.array([
        [300, 240], [320, 235], [340, 230], [360, 235], [380, 240],  # Upper outer
        [380, 260], [360, 265], [340, 270], [320, 265], [300, 260]   # Lower outer
    ])
    
    base_inner = np.array([
        [320, 245], [335, 242], [350, 240], [365, 242], [380, 245],  # Upper inner
        [380, 255], [365, 257], [350, 259], [335, 257], [320, 255]   # Lower inner
    ])
    
    base_points = np.vstack([base_outer, base_inner])  # 20 points total
    
    # Dysarthria characteristics based on severity
    severity_params = {
        'normal': {'movement_range': 5, 'irregularity': 0.1, 'asymmetry': 0.05},
        'mild': {'movement_range': 8, 'irregularity': 0.2, 'asymmetry': 0.1},
        'moderate': {'movement_range': 12, 'irregularity': 0.3, 'asymmetry': 0.2},
        'severe': {'movement_range': 20, 'irregularity': 0.5, 'asymmetry': 0.3}
    }
    
    params = severity_params.get(severity, severity_params['moderate'])
    
    for frame in range(num_frames):
        # Base movement (speech rhythm)
        speech_cycle = np.sin(2 * np.pi * frame / 25)  # ~4Hz speech rate
        
        # Create frame landmarks
        frame_landmarks = base_points.copy()
        
        # Apply vertical movement (mouth opening/closing)
        vertical_movement = params['movement_range'] * speech_cycle
        
        # Upper lip moves up, lower lip moves down for opening
        frame_landmarks[:10, 1] -= vertical_movement * 0.5  # Upper lip
        frame_landmarks[10:, 1] += vertical_movement * 0.5  # Lower lip
        
        # Apply horizontal movement
        horizontal_movement = params['movement_range'] * 0.3 * speech_cycle
        frame_landmarks[:, 0] += horizontal_movement
        
        # Add irregularity (dysarthria characteristic)
        irregularity = params['irregularity'] * np.random.randn(20, 2)
        frame_landmarks += irregularity
        
        # Add asymmetry (common in dysarthria)
        if params['asymmetry'] > 0:
            # Right side (higher indices) moves less
            right_indices = [4,5,6,7,8,9,14,15,16,17,18,19]
            frame_landmarks[right_indices, 0] += params['asymmetry'] * 10
            frame_landmarks[right_indices, 1] += params['asymmetry'] * 5
        
        landmarks_sequence.append(frame_landmarks)
    
    return np.array(landmarks_sequence)

def create_synthetic_video_from_landmarks(landmarks_sequence, output_path):
    """Create a video visualization from landmarks"""
    height, width = 480, 640
    fps = 30
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_idx, landmarks in enumerate(landmarks_sequence):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw face outline
        cv2.circle(frame, (width//2, height//2), 150, (200, 200, 200), 2)
        
        # Draw lip landmarks
        for i, (x, y) in enumerate(landmarks):
            color = (0, 0, 255) if i < 10 else (255, 0, 0)  # Red for outer, Blue for inner
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        
        # Connect outer lip points
        outer_points = landmarks[:10].astype(int)
        cv2.polylines(frame, [outer_points], True, (0, 0, 255), 2)
        
        # Connect inner lip points
        inner_points = landmarks[10:].astype(int)
        cv2.polylines(frame, [inner_points], True, (255, 0, 0), 1)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        out.write(frame)
    
    out.release()
    return output_path

def test_visual_model_initialization():
    """Test visual model initialization"""
    print("Testing Visual Model Initialization...")
    print("=" * 50)
    
    try:
        # Initialize model
        model = VisualAnalysisModel()
        
        # Check model structure
        print("1. Checking model architecture...")
        
        # Count layers
        linear_layers = sum(1 for _ in model.modules() if isinstance(_, torch.nn.Linear))
        lstm_layers = sum(1 for _ in model.modules() if isinstance(_, torch.nn.LSTM))
        
        print(f"   Linear layers: {linear_layers}")
        print(f"   LSTM layers: {lstm_layers}")
        
        # Check attention mechanism
        if hasattr(model, 'temporal_attention'):
            print("   ✅ Temporal attention present")
        else:
            print("   ❌ Temporal attention missing")
        
        # Check analysis heads
        required_heads = [
            'lip_sync_head', 'articulation_head', 'mouth_opening_head',
            'movement_smoothness_head', 'expression_symmetry_head',
            'severity_classifier'
        ]
        
        missing_heads = [head for head in required_heads if not hasattr(model, head)]
        
        if missing_heads:
            print(f"   ❌ Missing analysis heads: {missing_heads}")
            return False
        else:
            print("   ✅ All analysis heads present")
        
        # Test forward pass
        print("\n2. Testing forward pass...")
        batch_size = 2
        seq_len = 50
        num_points = 20
        features = num_points * 2  # x,y coordinates
        
        dummy_input = torch.randn(batch_size, seq_len, features)
        with torch.no_grad():
            output = model(dummy_input)
        
        # Check output structure
        required_outputs = [
            'lip_sync_score', 'articulation_score', 'mouth_opening_score',
            'movement_smoothness', 'expression_symmetry', 'severity',
            'attention_weights', 'global_features'
        ]
        
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

def test_visual_feature_processing():
    """Test visual feature processing"""
    print("\nTesting Visual Feature Processing...")
    print("=" * 50)
    
    try:
        # Create synthetic landmarks
        print("1. Creating synthetic lip landmarks...")
        landmarks_seq = create_synthetic_lip_landmarks(100, 'moderate')
        
        print(f"   Landmarks shape: {landmarks_seq.shape}")
        print(f"   Sequence length: {len(landmarks_seq)}")
        print(f"   Points per frame: {len(landmarks_seq[0])}")
        
        # Create visual features dictionary
        visual_features = {
            'lip_landmarks': landmarks_seq.tolist(),
            'mouth_openings': [],
            'lip_movements': [],
            'face_orientations': [],
            'expression_features': [],
            'temporal_features': {}
        }
        
        # Calculate mouth openings
        print("\n2. Calculating mouth openings...")
        for landmarks in landmarks_seq:
            # Upper lip points (first 10)
            upper_lip = landmarks[:10]
            # Lower lip points (next 10)
            lower_lip = landmarks[10:20]
            
            upper_center = np.mean(upper_lip, axis=0)
            lower_center = np.mean(lower_lip, axis=0)
            
            height = np.linalg.norm(upper_center - lower_center)
            width = np.linalg.norm(landmarks[0] - landmarks[5])  # Left to right corner
            
            visual_features['mouth_openings'].append({
                'height': height,
                'width': width,
                'area': height * width
            })
        
        print(f"   Calculated {len(visual_features['mouth_openings'])} mouth openings")
        
        # Calculate lip movements
        print("\n3. Calculating lip movements...")
        for i in range(1, len(landmarks_seq)):
            prev = landmarks_seq[i-1]
            curr = landmarks_seq[i]
            movement = np.mean(np.linalg.norm(curr - prev, axis=1))
            visual_features['lip_movements'].append(movement)
        
        print(f"   Calculated {len(visual_features['lip_movements'])} movement values")
        
        # Calculate temporal features
        print("\n4. Calculating temporal features...")
        if visual_features['mouth_openings']:
            openings = [o['height'] for o in visual_features['mouth_openings']]
            movements = visual_features['lip_movements']
            
            visual_features['temporal_features'] = {
                'mouth_opening_mean': np.mean(openings),
                'mouth_opening_std': np.std(openings),
                'mouth_opening_range': np.max(openings) - np.min(openings),
                'movement_smoothness': 1.0 - (np.std(movements) / max(np.mean(movements), 1)),
                'sync_consistency': 0.7  # Placeholder
            }
        
        print("   Temporal features calculated")
        
        return visual_features
        
    except Exception as e:
        print(f"❌ Feature processing test failed: {e}")
        return None

def test_visual_analysis():
    """Test complete visual analysis"""
    print("\nTesting Visual Analysis...")
    print("=" * 50)
    
    try:
        # Initialize model
        model = VisualAnalysisModel()
        model.eval()
        
        # Get test features
        visual_features = test_visual_feature_processing()
        if visual_features is None:
            return False
        
        print("\n5. Running visual analysis...")
        
        # Test analysis method
        analysis_result = model.analyze(visual_features)
        
        # Check analysis results
        print("6. Checking analysis results...")
        
        required_keys = [
            'lip_sync_score', 'articulation_score', 'mouth_opening_score',
            'movement_smoothness', 'expression_symmetry', 'severity_level',
            'severity_label', 'additional_metrics'
        ]
        
        missing_keys = [key for key in required_keys if key not in analysis_result]
        
        if missing_keys:
            print(f"   ❌ Missing analysis keys: {missing_keys}")
            return False
        
        print("   ✅ All analysis keys present")
        
        # Display results
        print("\n7. Analysis Results:")
        print("-" * 30)
        print(f"   Lip Sync Score: {analysis_result['lip_sync_score']:.3f}")
        print(f"   Articulation Score: {analysis_result['articulation_score']:.3f}")
        print(f"   Mouth Opening Score: {analysis_result['mouth_opening_score']:.3f}")
        print(f"   Movement Smoothness: {analysis_result['movement_smoothness']:.3f}")
        print(f"   Expression Symmetry: {analysis_result['expression_symmetry']:.3f}")
        print(f"   Severity Level: {analysis_result['severity_level']}")
        print(f"   Severity Label: {analysis_result['severity_label']}")
        
        # Check additional metrics
        if 'additional_metrics' in analysis_result:
            print(f"\n   Additional Metrics:")
            for key, value in analysis_result['additional_metrics'].items():
                if isinstance(value, (int, float)):
                    print(f"     {key}: {value:.4f}")
        
        # Test with different severity levels
        print("\n8. Testing different severity levels...")
        severity_levels = ['mild', 'moderate', 'severe']
        
        for severity in severity_levels:
            landmarks_seq = create_synthetic_lip_landmarks(50, severity)
            
            # Create simple features
            test_features = {
                'lip_landmarks': landmarks_seq.tolist(),
                'mouth_openings': [{'height': 30 + severity_levels.index(severity) * 10} for _ in range(50)],
                'lip_movements': [5 + severity_levels.index(severity) * 3 for _ in range(49)]
            }
            
            result = model.analyze(test_features)
            print(f"   {severity.capitalize():10} -> Lip Sync: {result['lip_sync_score']:.3f}, "
                  f"Severity: {result['severity_label']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_video_visualization():
    """Create visualization of synthetic data"""
    print("\nCreating Video Visualization...")
    print("=" * 50)
    
    try:
        # Create landmarks for different severities
        severities = ['normal', 'mild', 'moderate', 'severe']
        
        for severity in severities:
            print(f"\nCreating {severity} dysarthria simulation...")
            
            # Create landmarks
            landmarks_seq = create_synthetic_lip_landmarks(150, severity)
            
            # Create video
            temp_video = tempfile.NamedTemporaryFile(suffix=f'_{severity}.mp4', delete=False)
            temp_video.close()
            
            video_path = create_synthetic_video_from_landmarks(landmarks_seq, temp_video.name)
            
            print(f"   Video saved: {video_path}")
            print(f"   Landmarks: {len(landmarks_seq)} frames")
            
            # Optional: Display first frame info
            if len(landmarks_seq) > 0:
                print(f"   First landmark position: {landmarks_seq[0][0]}")
        
        print("\n✅ Video visualization created for all severities")
        print("Note: Check the generated MP4 files to see the lip movement simulations")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def test_model_robustness():
    """Test model robustness with edge cases"""
    print("\nTesting Visual Model Robustness...")
    print("=" * 50)
    
    try:
        model = VisualAnalysisModel()
        model.eval()
        
        print("1. Testing with empty features...")
        empty_features = {}
        result1 = model.analyze(empty_features)
        print(f"   Result with empty features - Lip Sync: {result1.get('lip_sync_score', 'N/A'):.3f}")
        
        print("\n2. Testing with minimal features...")
        minimal_features = {
            'lip_landmarks': [np.random.randn(20, 2).tolist() for _ in range(10)]
        }
        result2 = model.analyze(minimal_features)
        print(f"   Result with minimal features - Lip Sync: {result2.get('lip_sync_score', 'N/A'):.3f}")
        
        print("\n3. Testing with very short sequence...")
        short_features = {
            'lip_landmarks': [np.random.randn(20, 2).tolist() for _ in range(3)]
        }
        result3 = model.analyze(short_features)
        print(f"   Result with short sequence - Lip Sync: {result3.get('lip_sync_score', 'N/A'):.3f}")
        
        print("\n4. Testing with noisy landmarks...")
        noisy_features = {
            'lip_landmarks': [(np.random.randn(20, 2) * 100).tolist() for _ in range(20)]
        }
        result4 = model.analyze(noisy_features)
        print(f"   Result with noisy landmarks - Lip Sync: {result4.get('lip_sync_score', 'N/A'):.3f}")
        
        print("\n5. Testing model device compatibility...")
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
        
        for device in devices:
            torch_device = torch.device(device)
            model_device = VisualAnalysisModel().to(torch_device)
            model_device.eval()
            
            # Test forward pass
            test_input = torch.randn(1, 30, 40).to(torch_device)  # 30 frames, 20 points * 2
            with torch.no_grad():
                output = model_device(test_input)
            
            print(f"   Model works on {device.upper()}: ✓")
        
        return True
        
    except Exception as e:
        print(f"❌ Robustness test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("VISUAL MODEL TEST")
    print("=" * 60)
    
    tests = [
        ("Model Initialization", test_visual_model_initialization),
        ("Visual Analysis", test_visual_analysis),
        ("Video Visualization", test_video_visualization),
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
        print("\n🎉 All visual model tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check above for details.")