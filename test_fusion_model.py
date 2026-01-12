import sys
import os
sys.path.append('..')

import numpy as np
import torch
from fusion_model import DysarthriaFusionModel

def create_test_audio_features():
    """Create synthetic audio features for testing"""
    features = {
        'mfcc_mean': np.random.randn(13) * 0.5 + np.array([-5, -3, -1, 0, 1, 3, 5, 4, 2, 0, -2, -4, -6]),
        'pitch_mean': 120.0 + np.random.randn() * 20,
        'pitch_std': 15.0 + np.random.randn() * 5,
        'energy_mean': 0.5 + np.random.randn() * 0.1,
        'energy_variance': 0.1 + np.random.randn() * 0.05,
        'articulation_rate': 4.5 + np.random.randn() * 0.5,
        'pause_ratio': 0.3 + np.random.randn() * 0.1,
        'speech_rate': 3.8 + np.random.randn() * 0.3,
        'duration': 10.0
    }
    
    # Add dysarthria patterns based on severity
    return features

def create_test_visual_features():
    """Create synthetic visual features for testing"""
    features = {
        'temporal_features': {
            'mouth_opening_mean': 30.0 + np.random.randn() * 5,
            'mouth_opening_std': 5.0 + np.random.randn() * 2,
            'movement_smoothness': 0.7 + np.random.randn() * 0.2,
            'sync_consistency': 0.6 + np.random.randn() * 0.15
        },
        'lip_landmarks': [np.random.randn(20, 2).tolist() for _ in range(50)]
    }
    
    return features

def create_test_analysis_results(severity='moderate'):
    """Create synthetic analysis results"""
    severity_map = {
        'normal': {'clarity': 0.9, 'articulation': 0.85, 'fluency': 0.88, 'lip_sync': 0.9},
        'mild': {'clarity': 0.7, 'articulation': 0.65, 'fluency': 0.68, 'lip_sync': 0.7},
        'moderate': {'clarity': 0.5, 'articulation': 0.45, 'fluency': 0.48, 'lip_sync': 0.5},
        'severe': {'clarity': 0.3, 'articulation': 0.25, 'fluency': 0.28, 'lip_sync': 0.3}
    }
    
    params = severity_map.get(severity, severity_map['moderate'])
    
    audio_analysis = {
        'clarity_score': params['clarity'] + np.random.randn() * 0.05,
        'articulation_score': params['articulation'] + np.random.randn() * 0.05,
        'fluency_score': params['fluency'] + np.random.randn() * 0.05,
        'pitch_stability': 0.6 + np.random.randn() * 0.1,
        'pitch_variability': 0.3 + np.random.randn() * 0.1,
        'severity_level': ['normal', 'mild', 'moderate', 'severe'].index(severity)
    }
    
    visual_analysis = {
        'lip_sync_score': params['lip_sync'] + np.random.randn() * 0.05,
        'articulation_score': params['articulation'] + np.random.randn() * 0.05,
        'mouth_opening_score': 0.5 + np.random.randn() * 0.1,
        'movement_smoothness': 0.6 + np.random.randn() * 0.1,
        'expression_symmetry': 0.7 + np.random.randn() * 0.1,
        'severity_level': ['normal', 'mild', 'moderate', 'severe'].index(severity)
    }
    
    return audio_analysis, visual_analysis

def test_fusion_model_initialization():
    """Test fusion model initialization"""
    print("Testing Fusion Model Initialization...")
    print("=" * 50)
    
    try:
        # Initialize model
        model = DysarthriaFusionModel()
        
        # Check model structure
        print("1. Checking model architecture...")
        
        # Check projection layers
        if hasattr(model, 'audio_projection') and hasattr(model, 'visual_projection'):
            print("   ✅ Audio and visual projection layers present")
        else:
            print("   ❌ Missing projection layers")
            return False
        
        # Check attention mechanism
        if hasattr(model, 'cross_attention'):
            print("   ✅ Cross-modal attention present")
        else:
            print("   ❌ Cross-modal attention missing")
            return False
        
        # Check fusion layers
        if hasattr(model, 'fusion_layer'):
            print("   ✅ Fusion layer present")
        else:
            print("   ❌ Fusion layer missing")
            return False
        
        # Check prediction heads
        required_heads = [
            'sync_confidence_head', 'overall_severity_head',
            'intelligibility_head', 'compensation_pattern_head',
            'diagnostic_features'
        ]
        
        missing_heads = [head for head in required_heads if not hasattr(model, head)]
        
        if missing_heads:
            print(f"   ❌ Missing prediction heads: {missing_heads}")
            return False
        else:
            print("   ✅ All prediction heads present")
        
        # Test forward pass
        print("\n2. Testing forward pass...")
        batch_size = 2
        audio_features = torch.randn(batch_size, 256)  # Audio feature vector
        visual_features = torch.randn(batch_size, 512)  # Visual feature vector
        
        with torch.no_grad():
            output = model(audio_features, visual_features)
        
        # Check output structure
        required_outputs = [
            'sync_confidence', 'overall_severity', 'intelligibility_score',
            'compensation_pattern', 'diagnostic_features', 'fused_features'
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
        
        # Check output ranges
        print("\n3. Checking output ranges...")
        sync_conf = output['sync_confidence'].item()
        intelligibility = output['intelligibility_score'].item()
        
        if 0 <= sync_conf <= 1:
            print(f"   ✅ Sync confidence in range [0,1]: {sync_conf:.3f}")
        else:
            print(f"   ❌ Sync confidence out of range: {sync_conf}")
            return False
        
        if 0 <= intelligibility <= 1:
            print(f"   ✅ Intelligibility in range [0,1]: {intelligibility:.3f}")
        else:
            print(f"   ❌ Intelligibility out of range: {intelligibility}")
            return False
        
        # Check severity probabilities sum to ~1
        severity_probs = output['overall_severity'].sum(dim=1).item()
        if 0.99 <= severity_probs <= 1.01:
            print(f"   ✅ Severity probabilities sum to ~1: {severity_probs:.3f}")
        else:
            print(f"   ❌ Severity probabilities don't sum to 1: {severity_probs}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Initialization test failed: {e}")
        return False

def test_feature_extraction():
    """Test feature extraction methods"""
    print("\nTesting Feature Extraction...")
    print("=" * 50)
    
    try:
        model = DysarthriaFusionModel()
        
        # Create test data
        audio_features = create_test_audio_features()
        visual_features = create_test_visual_features()
        audio_analysis, visual_analysis = create_test_analysis_results('moderate')
        
        print("1. Testing audio feature extraction...")
        audio_vector = model._extract_audio_feature_vector(audio_features, audio_analysis)
        
        print(f"   Audio feature vector shape: {audio_vector.shape}")
        print(f"   Expected size: 256, Actual size: {len(audio_vector)}")
        
        if len(audio_vector) == 256:
            print("   ✅ Audio feature extraction successful")
        else:
            print("   ❌ Audio feature vector wrong size")
            return False
        
        print("\n2. Testing visual feature extraction...")
        visual_vector = model._extract_visual_feature_vector(visual_features, visual_analysis)
        
        print(f"   Visual feature vector shape: {visual_vector.shape}")
        print(f"   Expected size: 512, Actual size: {len(visual_vector)}")
        
        if len(visual_vector) == 512:
            print("   ✅ Visual feature extraction successful")
        else:
            print("   ❌ Visual feature vector wrong size")
            return False
        
        # Check feature statistics
        print("\n3. Checking feature statistics...")
        print(f"   Audio features - Mean: {np.mean(audio_vector):.4f}, Std: {np.std(audio_vector):.4f}")
        print(f"   Visual features - Mean: {np.mean(visual_vector):.4f}, Std: {np.std(visual_vector):.4f}")
        
        # Test with missing data
        print("\n4. Testing with missing features...")
        minimal_audio = {'duration': 5.0}
        minimal_visual = {}
        
        audio_minimal = model._extract_audio_feature_vector(minimal_audio, None)
        visual_minimal = model._extract_visual_feature_vector(minimal_visual, None)
        
        print(f"   Minimal audio vector size: {len(audio_minimal)}")
        print(f"   Minimal visual vector size: {len(visual_minimal)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False

def test_fusion_analysis():
    """Test complete fusion analysis"""
    print("\nTesting Fusion Analysis...")
    print("=" * 50)
    
    try:
        model = DysarthriaFusionModel()
        model.eval()
        
        # Create test data for different severities
        severities = ['normal', 'mild', 'moderate', 'severe']
        
        print("1. Testing fusion analysis across severities...")
        results = {}
        
        for severity in severities:
            audio_features = create_test_audio_features()
            visual_features = create_test_visual_features()
            audio_analysis, visual_analysis = create_test_analysis_results(severity)
            
            # Run fusion analysis
            analysis_result = model.analyze(
                audio_features, 
                visual_features,
                audio_analysis,
                visual_analysis
            )
            
            results[severity] = analysis_result
            
            print(f"\n   {severity.capitalize()} Dysarthria:")
            print(f"     Sync Confidence: {analysis_result.get('sync_confidence', 0):.3f}")
            print(f"     Intelligibility: {analysis_result.get('intelligibility_score', 0):.3f}")
            print(f"     Severity: {analysis_result.get('severity_label', 'Unknown')}")
            print(f"     Compensation: {analysis_result.get('compensation_label', 'Unknown')}")
        
        # Check consistency
        print("\n2. Checking analysis consistency...")
        
        # Severity should increase with dysarthria severity
        severity_scores = {}
        for severity in severities:
            result = results[severity]
            severity_scores[severity] = result.get('overall_severity', 0)
        
        # Normal should have lowest severity score
        if severity_scores['normal'] < severity_scores['severe']:
            print("   ✅ Severity scores consistent (normal < severe)")
        else:
            print("   ❌ Severity scores inconsistent")
        
        # Check derived metrics
        print("\n3. Checking derived metrics...")
        moderate_result = results['moderate']
        
        if 'derived_metrics' in moderate_result:
            derived = moderate_result['derived_metrics']
            print(f"   Audio-visual correlation: {derived.get('audio_visual_correlation', 0):.3f}")
            print(f"   Timing alignment: {derived.get('timing_alignment', 0):.3f}")
            print(f"   Articulation consistency: {derived.get('articulation_consistency', 0):.3f}")
        
        # Check interpretation functions
        print("\n4. Testing interpretation functions...")
        
        test_confidences = [0.9, 0.7, 0.5, 0.3, 0.1]
        print("   Sync confidence interpretations:")
        for conf in test_confidences:
            interpretation = model._interpret_sync_confidence(conf)
            print(f"     {conf:.1f} -> {interpretation}")
        
        print("\n   Intelligibility interpretations:")
        for score in test_confidences:
            interpretation = model._interpret_intelligibility(score)
            print(f"     {score:.1f} -> {interpretation}")
        
        # Test compensation patterns
        print("\n5. Testing compensation patterns...")
        for i in range(3):
            label = model._get_compensation_label(i)
            print(f"   Pattern {i} -> {label}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fusion analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_derived_metrics():
    """Test derived metrics calculation"""
    print("\nTesting Derived Metrics Calculation...")
    print("=" * 50)
    
    try:
        model = DysarthriaFusionModel()
        
        # Create test data
        audio_features = {'duration': 8.5}
        visual_features = {
            'lip_landmarks': [np.random.randn(20, 2).tolist() for _ in range(100)],
            'temporal_features': {}
        }
        
        audio_analysis = {
            'clarity_score': 0.6,
            'articulation_score': 0.55,
            'fluency_score': 0.65
        }
        
        visual_analysis = {
            'lip_sync_score': 0.58,
            'articulation_score': 0.52,
            'mouth_opening_score': 0.62
        }
        
        print("1. Calculating derived metrics...")
        metrics = model._calculate_derived_metrics(
            audio_features,
            visual_features,
            audio_analysis,
            visual_analysis
        )
        
        print(f"   Calculated {len(metrics)} metrics")
        
        # Display metrics
        print("\n2. Metric values:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:.4f}")
            elif isinstance(value, dict):
                print(f"   {key}:")
                for sub_key, sub_value in value.items():
                    print(f"     {sub_key}: {sub_value:.4f}")
        
        # Test edge cases
        print("\n3. Testing edge cases...")
        
        # Test with missing analysis
        metrics_missing = model._calculate_derived_metrics(
            {}, {}, None, None
        )
        print(f"   Metrics with missing analysis: {len(metrics_missing)} calculated")
        
        # Test with inconsistent data
        inconsistent_audio = {'duration': 0}  # Zero duration
        inconsistent_visual = {'lip_landmarks': []}  # Empty landmarks
        
        metrics_inconsistent = model._calculate_derived_metrics(
            inconsistent_audio,
            inconsistent_visual,
            audio_analysis,
            visual_analysis
        )
        print(f"   Metrics with inconsistent data: {len(metrics_inconsistent)} calculated")
        
        # Check priority calculation
        print("\n4. Testing priority calculation...")
        priorities = model._calculate_priority(audio_analysis, visual_analysis)
        print(f"   Calculated priorities: {priorities}")
        
        return True
        
    except Exception as e:
        print(f"❌ Derived metrics test failed: {e}")
        return False

def test_model_robustness():
    """Test model robustness"""
    print("\nTesting Fusion Model Robustness...")
    print("=" * 50)
    
    try:
        model = DysarthriaFusionModel()
        model.eval()
        
        print("1. Testing with empty inputs...")
        try:
            result1 = model.analyze({}, {}, None, None)
            print(f"   Analysis with empty inputs completed")
            print(f"   Result - Sync: {result1.get('sync_confidence', 'N/A'):.3f}")
        except Exception as e:
            print(f"   ⚠️ Empty input test raised exception: {e}")
        
        print("\n2. Testing with extreme values...")
        extreme_audio = {
            'mfcc_mean': np.array([1000] * 13),  # Very high values
            'pitch_mean': 10000,  # Extremely high pitch
            'pitch_std': 5000,    # Extremely unstable
            'duration': 0.001     # Extremely short
        }
        
        extreme_visual = {
            'temporal_features': {
                'mouth_opening_mean': 1000,
                'mouth_opening_std': 500,
                'movement_smoothness': -1,  # Invalid
                'sync_consistency': 2.0     # Invalid
            }
        }
        
        extreme_audio_analysis = {
            'clarity_score': 2.0,  # > 1
            'articulation_score': -0.5,  # < 0
            'fluency_score': 1.5  # > 1
        }
        
        try:
            result2 = model.analyze(
                extreme_audio,
                extreme_visual,
                extreme_audio_analysis,
                {}
            )
            print(f"   Analysis with extreme values completed")
            print(f"   Result - Sync: {result2.get('sync_confidence', 'N/A'):.3f}")
        except Exception as e:
            print(f"   ⚠️ Extreme value test raised exception: {e}")
        
        print("\n3. Testing with NaN and Inf values...")
        nan_audio = {
            'mfcc_mean': np.array([np.nan] * 13),
            'pitch_mean': np.inf,
            'pitch_std': -np.inf,
            'duration': np.nan
        }
        
        try:
            result3 = model.analyze(nan_audio, {}, {}, {})
            print(f"   Analysis with NaN/Inf completed")
        except Exception as e:
            print(f"   ⚠️ NaN/Inf test raised exception (expected): {e}")
        
        print("\n4. Testing model device compatibility...")
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')
        
        for device in devices:
            torch_device = torch.device(device)
            model_device = DysarthriaFusionModel().to(torch_device)
            model_device.eval()
            
            # Test forward pass
            audio_features = torch.randn(1, 256).to(torch_device)
            visual_features = torch.randn(1, 512).to(torch_device)
            
            with torch.no_grad():
                output = model_device(audio_features, visual_features)
            
            print(f"   Model works on {device.upper()}: ✓")
        
        return True
        
    except Exception as e:
        print(f"❌ Robustness test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("FUSION MODEL TEST")
    print("=" * 60)
    
    tests = [
        ("Model Initialization", test_fusion_model_initialization),
        ("Feature Extraction", test_feature_extraction),
        ("Fusion Analysis", test_fusion_analysis),
        ("Derived Metrics", test_derived_metrics),
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
        print("\n🎉 All fusion model tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check above for details.")