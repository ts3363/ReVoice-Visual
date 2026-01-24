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
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300  # 5 minute timeout
        )

        out = result.stdout or ""
        err = result.stderr or ""

        if out.strip():
            print(out)
        if err.strip():
            print("STDERR:")
            print(err)

        # Return code handling
        if result.returncode == 0:
            # Avoid emoji check, just look for FAIL
            if "FAIL" in out:
                print(f"\n{test_name} - TESTS FAILED")
                return False
            else:
                print(f"\n{test_name} - ALL TESTS PASSED")
                return True
        else:
            print(f"\n{test_name} - SCRIPT EXITED WITH CODE {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print(f"\n{test_name} - TIMED OUT AFTER 5 MINUTES")
        return False
    except Exception as e:
        print(f"\n{test_name} - ERROR: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("DYSARTHRIA ANALYSIS SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    tests = [
        ("test_preprocessing.py", "Preprocessing Module Test"),
        ("test_audio_model.py", "Audio Model Test"),
        ("test_visual_model.py", "Visual Model Test"),
        ("test_fusion_model.py", "Fusion Model Test"),
        ("test_major.py", "Complete System Test")
    ]

    original_dir = os.getcwd()
    test_dir = os.path.join(original_dir, "test_backend")

    if not os.path.exists(test_dir):
        print(f"Creating test directory: {test_dir}")
        os.makedirs(test_dir, exist_ok=True)

    os.chdir(test_dir)

    results = []
    start_time = time.time()

    for test_file, test_name in tests:
        if not os.path.exists(test_file):
            print(f"\n⚠️ Test file not found: {test_file}")
            print("Please make sure you've placed the test files in test_backend/ directory")
            results.append((test_name, False))
            continue

        success = run_test(test_file, test_name)
        results.append((test_name, success))
        time.sleep(1)

    total_time = time.time() - start_time

    os.chdir(original_dir)

    print("\n" + "=" * 70)
    print("FINAL TEST SUMMARY")
    print("=" * 70)

    passed = 0
    failed = 0

    for test_name, success in results:
        if success:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1
        print(f"{test_name:35} {status}")

    print("\n" + "-" * 70)
    print(f"TOTAL: {len(results)} tests")
    print(f"PASSED: {passed}")
    print(f"FAILED: {failed}")
    print(f"TIME: {total_time:.1f} seconds")
    print("-" * 70)

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! System is ready for integration.")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please fix issues before integration.")


if __name__ == "__main__":
    main()
