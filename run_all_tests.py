#!/usr/bin/env python3
"""
Run all backend tests in sequence
"""

import os
import sys
import subprocess
import time

def run_test(test_file, test_name):
    """Run a single test file"""
    print(f"\n{'='*70}")
    print(f"RUNNING: {test_name}")
    print(f"{'='*70}")
    
    try:
        # Run the test
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        # Check result
        if result.returncode == 0:
            # Check if tests passed by looking for failure indicators
            if "FAIL" in result.stdout or "❌" in result.stdout:
                print(f"\n❌ {test_name} - TESTS FAILED")
                return False
            else:
                print(f"\n✅ {test_name} - ALL TESTS PASSED")
                return True
        else:
            print(f"\n❌ {test_name} - SCRIPT EXITED WITH CODE {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n⏰ {test_name} - TIMED OUT AFTER 5 MINUTES")
        return False
    except Exception as e:
        print(f"\n❌ {test_name} - ERROR: {e}")
        return False

def main():
    """Run all tests"""
    print("="*70)
    print("DYSARTHRIA ANALYSIS SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    # Define test files and their names
    tests = [
        ("test_preprocessing.py", "Preprocessing Module Test"),
        ("test_audio_model.py", "Audio Model Test"),
        ("test_visual_model.py", "Visual Model Test"),
        ("test_fusion_model.py", "Fusion Model Test"),
        ("test_major.py", "Complete System Test")
    ]
    
    # Change to test directory
    original_dir = os.getcwd()
    test_dir = os.path.join(original_dir, "test_backend")
    
    if not os.path.exists(test_dir):
        print(f"Creating test directory: {test_dir}")
        os.makedirs(test_dir, exist_ok=True)
    
    os.chdir(test_dir)
    
    # Run tests
    results = []
    start_time = time.time()
    
    for test_file, test_name in tests:
        # Check if test file exists
        if not os.path.exists(test_file):
            print(f"\n⚠️ Test file not found: {test_file}")
            print("Please make sure you've placed the test files in test_backend/ directory")
            results.append((test_name, False))
            continue
            
        success = run_test(test_file, test_name)
        results.append((test_name, success))
        
        # Small delay between tests
        time.sleep(1)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Return to original directory
    os.chdir(original_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL TEST SUMMARY")
    print("="*70)
    
    passed = 0
    failed = 0
    
    for test_name, success in results:
        if success:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1
        print(f"{test_name:35} {status}")
    
    print("\n" + "-"*70)
    print(f"TOTAL: {len(results)} tests")
    print(f"PASSED: {passed}")
    print(f"FAILED: {failed}")
    print(f"TIME: {total_time:.1f} seconds")
    print("-"*70)
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! System is ready for integration.")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please fix issues before integration.")
    
    # Create requirements.txt if missing
    requirements_file = os.path.join(original_dir, "requirements.txt")
    if not os.path.exists(requirements_file):
        print("\n📝 Creating requirements.txt file...")
        requirements_content = """Flask==2.3.3
Flask-CORS==4.0.0
numpy==1.24.3
opencv-python==4.8.0.74
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
librosa==0.10.0
scipy==1.10.1
mediapipe==0.10.3
moviepy==1.0.3
whisper==1.1.10
scikit-learn==1.3.0
pydub==0.25.1
wave==0.0.2
soundfile==0.12.1
python-dotenv==1.0.0
"""
        
        with open(requirements_file, 'w') as f:
            f.write(requirements_content)
        print(f"✅ Created {requirements_file}")
    
    # Create setup instructions
    print("\n" + "="*70)
    print("NEXT STEPS FOR INTEGRATION")
    print("="*70)
    print("""
1. Install dependencies:
   pip install -r requirements.txt

2. Set up the directory structure:
   mkdir -p uploads training_data templates static

3. Run the main application:
   python MAJOR.py

4. Access the web interface:
   http://localhost:5000

5. Integration with your web UI:
   - Copy the backend code to your project
   - Update API endpoints in your JavaScript
   - Ensure file uploads are sent to /record endpoint
   - Display results from /analyze endpoint
   - Implement training interface using /training/* endpoints
    """)

if __name__ == "__main__":
    main()