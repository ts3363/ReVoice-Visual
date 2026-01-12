import sys
import os
sys.path.append('..')

import numpy as np
import tempfile
import json
from MAJOR import DysarthriaAnalysisSystem, app
from preprocessing import AudioVideoPreprocessor
from audio_model import AudioAnalysisModel
from visual_model import VisualAnalysisModel
from fusion_model import DysarthriaFusionModel

def test_system_initialization():
    """Test complete system initialization"""
    print("Testing System Initialization...")
    print("=" * 50)
    
    try:
        # Initialize the complete system
        print("1. Initializing Dysarthria Analysis System...")
        system = DysarthriaAnalysisSystem()
        
        # Check all components are initialized
        print("\n2. Checking system components...")
        
        components = [
            ('Preprocessor', system.preprocessor),
            ('Audio Model', system.audio_model),
            ('Visual Model', system.visual_model),
            ('Fusion Model', system.fusion_model),
            ('Whisper Model', system.whisper_model)
        ]
        
        all_initialized = True
        for name, component in components:
            if component is not None:
                print(f"   ✅ {name} initialized")
            else:
                print(f"   ❌ {name} not initialized")
                all_initialized = False
        
        if not all_initialized:
            return False
        
        # Check device
        print(f"\n3. System running on: {system.device}")
        
        # Check model modes
        print("\n4. Checking model modes...")
        if system.audio_model.training == False:
            print("   ✅ Audio model in evaluation mode")
        else:
            print("   ❌ Audio model not in evaluation mode")
        
        if system.visual_model.training == False:
            print("   ✅ Visual model in evaluation mode")
        else:
            print("   ❌ Visual model not in evaluation mode")
        
        if system.fusion_model.training == False:
            print("   ✅ Fusion model in evaluation mode")
        else:
            print("   ❌ Fusion model not in evaluation mode")
        
        return True
        
    except Exception as e:
        print(f"❌ System initialization test failed: {e}")
        return False

def create_test_session():
    """Create a complete test session"""
    import librosa
    import soundfile as sf
    import cv2
    
    # Create test audio
    duration = 3.0
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration), False)
    
    # Create speech-like signal
    audio = 0.5 * np.sin(2 * np.pi * 120 * t)  # Base pitch
    audio += 0.1 * np.sin(2 * np.pi * 500 * t)  # Formant 1
    audio += 0.05 * np.sin(2 * np.pi * 1500 * t)  # Formant 2
    
    # Add some dysarthria characteristics
    audio *= (1 + 0.1 * np.sin(2 * np.pi * 3 * t))  # Amplitude modulation
    audio += 0.02 * np.random.randn(len(t))  # Some noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    # Save audio
    audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(audio_file.name, audio, sr)
    
    # Create simple test video
    width, height = 640, 480
    fps = 30
    frames = int(duration * fps)
    
    video_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    video_file.close()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_file.name, fourcc, fps, (width, height))
    
    for i in range(frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Simulate mouth movement
        mouth_size = 20 + int(15 * np.sin(2 * np.pi * i / 15))
        center_x, center_y = width // 2, height // 2
        
        # Draw mouth
        cv2.ellipse(frame, (center_x, center_y), (mouth_size, 10), 0, 0, 360, (0, 0, 255), -1)
        
        # Draw face
        cv2.circle(frame, (center_x, center_y), 100, (200, 200, 200), 2)
        
        out.write(frame)
    
    out.release()
    
    return audio_file.name, video_file.name

def test_analysis_pipeline():
    """Test complete analysis pipeline"""
    print("\nTesting Complete Analysis Pipeline...")
    print("=" * 50)
    
    try:
        # Initialize system
        system = DysarthriaAnalysisSystem()
        
        # Create test files
        print("1. Creating test audio and video...")
        audio_path, video_path = create_test_session()
        
        print(f"   Audio file: {audio_path}")
        print(f"   Video file: {video_path}")
        
        # Test complete analysis
        print("\n2. Running complete analysis...")
        results = system.analyze_audio_video(audio_path, video_path)
        
        # Check results
        print("\n3. Checking analysis results...")
        
        if not results.get('success', False):
            print(f"   ❌ Analysis failed: {results.get('error', 'Unknown error')}")
            return False
        
        print("   ✅ Analysis completed successfully")
        
        # Check all required fields
        required_fields = [
            'transcription', 'audio_analysis', 'visual_analysis',
            'fusion_analysis', 'recommendations', 'confidence_score'
        ]
        
        missing_fields = [field for field in required_fields if field not in results]
        
        if missing_fields:
            print(f"   ❌ Missing fields: {missing_fields}")
            return False
        
        print("   ✅ All required fields present")
        
        # Display results
        print("\n4. Analysis Results Summary:")
        print("-" * 30)
        print(f"   Transcription: {results['transcription'][:100]}...")
        print(f"   Confidence Score: {results['confidence_score']:.3f}")
        
        # Audio analysis
        audio = results['audio_analysis']
        print(f"\n   Audio Analysis:")
        print(f"     Clarity: {audio.get('clarity_score', 0):.3f}")
        print(f"     Articulation: {audio.get('articulation_score', 0):.3f}")
        print(f"     Fluency: {audio.get('fluency_score', 0):.3f}")
        print(f"     Severity: {audio.get('severity_label', 'Unknown')}")
        
        # Visual analysis
        visual = results['visual_analysis']
        print(f"\n   Visual Analysis:")
        print(f"     Lip Sync: {visual.get('lip_sync_score', 0):.3f}")
        print(f"     Mouth Opening: {visual.get('mouth_opening_score', 0):.3f}")
        print(f"     Movement Smoothness: {visual.get('movement_smoothness', 0):.3f}")
        
        # Fusion analysis
        fusion = results['fusion_analysis']
        print(f"\n   Fusion Analysis:")
        print(f"     Sync Confidence: {fusion.get('sync_confidence', 0):.3f}")
        print(f"     Intelligibility: {fusion.get('intelligibility_score', 0):.3f}")
        print(f"     Overall Severity: {fusion.get('severity_label', 'Unknown')}")
        
        # Recommendations
        recommendations = results['recommendations']
        print(f"\n   Recommendations ({len(recommendations)}):")
        for i, rec in enumerate(recommendations[:3], 1):  # Show first 3
            print(f"     {i}. {rec.get('exercise', 'Unknown')}")
        
        # Clean up
        if os.path.exists(audio_path):
            os.unlink(audio_path)
        if os.path.exists(video_path):
            os.unlink(video_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_session():
    """Test training session creation"""
    print("\nTesting Training Session Creation...")
    print("=" * 50)
    
    try:
        system = DysarthriaAnalysisSystem()
        
        # Create mock analysis results
        mock_results = {
            'success': True,
            'transcription': "Test speech sample",
            'audio_analysis': {
                'clarity_score': 0.6,
                'articulation_score': 0.55,
                'fluency_score': 0.65,
                'severity_label': 'Moderate'
            },
            'visual_analysis': {
                'lip_sync_score': 0.58,
                'articulation_score': 0.52,
                'mouth_opening_score': 0.62,
                'severity_label': 'Moderate'
            },
            'fusion_analysis': {
                'sync_confidence': 0.59,
                'intelligibility_score': 0.61,
                'severity_label': 'Moderate'
            },
            'recommendations': [
                {
                    'type': 'audio',
                    'exercise': 'Slow pronunciation practice',
                    'description': 'Practice speaking slowly',
                    'duration': '10 minutes',
                    'difficulty': 'Beginner'
                },
                {
                    'type': 'visual',
                    'exercise': 'Lip movement synchronization',
                    'description': 'Practice lip movements',
                    'duration': '15 minutes',
                    'difficulty': 'Beginner'
                }
            ],
            'confidence_score': 0.6
        }
        
        print("1. Creating training session...")
        user_id = "test_user_001"
        training_session = system.create_training_session(user_id, mock_results)
        
        # Check session structure
        print("\n2. Checking session structure...")
        
        required_fields = [
            'user_id', 'session_id', 'created_at', 'baseline',
            'exercises', 'progress', 'current_exercise'
        ]
        
        missing_fields = [field for field in required_fields if field not in training_session]
        
        if missing_fields:
            print(f"   ❌ Missing fields: {missing_fields}")
            return False
        
        print("   ✅ All session fields present")
        
        # Check values
        print("\n3. Checking session values...")
        print(f"   User ID: {training_session['user_id']}")
        print(f"   Session ID: {training_session['session_id']}")
        print(f"   Created: {training_session['created_at']}")
        print(f"   Exercises: {len(training_session['exercises'])}")
        print(f"   Progress entries: {len(training_session['progress'])}")
        print(f"   Current exercise: {training_session['current_exercise']}")
        
        # Check session ID format
        if training_session['session_id'].startswith('train_'):
            print("   ✅ Session ID format correct")
        else:
            print("   ❌ Session ID format incorrect")
        
        # Check exercises
        exercises = training_session['exercises']
        if len(exercises) == len(mock_results['recommendations']):
            print(f"   ✅ All recommendations converted to exercises")
        else:
            print(f"   ❌ Exercise count mismatch")
        
        # Test saving to file
        print("\n4. Testing session persistence...")
        import tempfile
        
        temp_dir = tempfile.mkdtemp()
        app.config['TRAINING_DATA'] = temp_dir
        
        # Create session with temporary directory
        session = system.create_training_session(user_id, mock_results)
        
        # Check if file was created
        session_file = os.path.join(temp_dir, f"{session['session_id']}.json")
        if os.path.exists(session_file):
            print(f"   ✅ Session file created: {session_file}")
            
            # Verify file content
            with open(session_file, 'r') as f:
                loaded_session = json.load(f)
            
            if loaded_session['session_id'] == session['session_id']:
                print("   ✅ Session file content verified")
            else:
                print("   ❌ Session file content mismatch")
        else:
            print(f"   ❌ Session file not created")
            return False
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Training session test failed: {e}")
        return False

def test_recommendation_generation():
    """Test recommendation generation"""
    print("\nTesting Recommendation Generation...")
    print("=" * 50)
    
    try:
        system = DysarthriaAnalysisSystem()
        
        # Test cases with different analysis results
        test_cases = [
            {
                'name': 'Poor Clarity',
                'audio': {'clarity_score': 0.4, 'pitch_variability': 0.7},
                'visual': {'lip_sync_score': 0.7, 'mouth_opening_score': 0.6},
                'fusion': {'sync_confidence': 0.8}
            },
            {
                'name': 'Poor Lip Sync',
                'audio': {'clarity_score': 0.8, 'pitch_variability': 0.6},
                'visual': {'lip_sync_score': 0.4, 'mouth_opening_score': 0.7},
                'fusion': {'sync_confidence': 0.5}
            },
            {
                'name': 'Poor Mouth Opening',
                'audio': {'clarity_score': 0.7, 'pitch_variability': 0.5},
                'visual': {'lip_sync_score': 0.6, 'mouth_opening_score': 0.3},
                'fusion': {'sync_confidence': 0.7}
            },
            {
                'name': 'Multiple Issues',
                'audio': {'clarity_score': 0.4, 'pitch_variability': 0.3},
                'visual': {'lip_sync_score': 0.3, 'mouth_opening_score': 0.4},
                'fusion': {'sync_confidence': 0.4}
            }
        ]
        
        print("1. Testing recommendation generation...")
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n   Test Case {i}: {test_case['name']}")
            print(f"   {'-' * (len(test_case['name']) + 12)}")
            
            recommendations = system.generate_recommendations(
                test_case['audio'],
                test_case['visual'],
                test_case['fusion']
            )
            
            print(f"   Generated {len(recommendations)} recommendations")
            
            for j, rec in enumerate(recommendations, 1):
                print(f"     {j}. [{rec.get('type', 'unknown')}] {rec.get('exercise', 'Unknown')}")
                print(f"        Difficulty: {rec.get('difficulty', 'Unknown')}")
                print(f"        Duration: {rec.get('duration', 'Unknown')}")
        
        # Test with minimal input
        print("\n2. Testing with minimal input...")
        minimal_recs = system.generate_recommendations({}, {}, {})
        print(f"   Recommendations with minimal input: {len(minimal_recs)}")
        
        # Test exercise instruction generation
        print("\n3. Testing exercise instruction generation...")
        
        test_exercises = [
            {'type': 'audio', 'exercise': 'Slow pronunciation practice'},
            {'type': 'visual', 'exercise': 'Lip movement synchronization'},
            {'type': 'combined', 'exercise': 'Audio-visual synchronization'},
            {'type': 'unknown', 'exercise': 'Unknown exercise'}
        ]
        
        for exercise in test_exercises:
            instructions = system._generate_exercise_instructions(exercise)
            print(f"\n   Exercise: {exercise['exercise']}")
            print(f"   Instructions: {len(instructions)} steps")
            if instructions:
                print(f"   First step: {instructions[0][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Recommendation generation test failed: {e}")
        return False

def test_transcription():
    """Test audio transcription"""
    print("\nTesting Audio Transcription...")
    print("=" * 50)
    
    try:
        system = DysarthriaAnalysisSystem()
        
        # Create test audio with known content
        import librosa
        import soundfile as sf
        import tempfile
        
        print("1. Creating test audio...")
        
        # Create a simple tone that won't be transcribed (to test error handling)
        duration = 2.0
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration), False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # Pure tone
        
        # Save to file
        audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(audio_file.name, audio, sr)
        
        print("2. Testing transcription...")
        transcription = system.transcribe_audio(audio_file.name)
        
        print(f"   Transcription result: {transcription}")
        
        if transcription:
            print("   ✅ Transcription completed")
        else:
            print("   ⚠️ Empty transcription (expected for pure tone)")
        
        # Test with non-existent file
        print("\n3. Testing error handling...")
        try:
            result = system.transcribe_audio("non_existent_file.wav")
            print(f"   Result for non-existent file: {result}")
        except Exception as e:
            print(f"   ⚠️ Exception for non-existent file (expected): {e}")
        
        # Clean up
        if os.path.exists(audio_file.name):
            os.unlink(audio_file.name)
        
        return True
        
    except Exception as e:
        print(f"❌ Transcription test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("COMPLETE SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        ("System Initialization", test_system_initialization),
        ("Analysis Pipeline", test_analysis_pipeline),
        ("Training Session", test_training_session),
        ("Recommendation Generation", test_recommendation_generation),
        ("Transcription", test_transcription)
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
        print("\n🎉 All system tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check above for details.")