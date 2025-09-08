#!/usr/bin/env python
"""
Test runner script for PainCare Model
"""

import subprocess
import sys
import os

def run_tests():
    """Run the test suite"""
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Python executable path
    python_exe = os.path.join(project_root, ".venv", "Scripts", "python.exe")
    
    # Test commands to run
    commands = [
        # Run all tests with verbose output
        [python_exe, "-m", "pytest", "tests/", "-v"],
        
        # Run tests with coverage (if pytest-cov is installed)
        # [python_exe, "-m", "pytest", "tests/", "--cov=src", "--cov-report=term-missing"],
        
        # Run only fast tests (excluding slow marked tests)
        # [python_exe, "-m", "pytest", "tests/", "-m", "not slow"],
    ]
    
    print("Running PainCare Model Test Suite")
    print("=" * 50)
    
    for i, cmd in enumerate(commands, 1):
        print(f"\nRunning command {i}: {' '.join(cmd[2:])}")
        print("-" * 40)
        
        try:
            result = subprocess.run(cmd, cwd=project_root, check=False)
            if result.returncode != 0:
                print(f"Command {i} failed with exit code {result.returncode}")
                return result.returncode
            else:
                print(f"Command {i} completed successfully")
        except Exception as e:
            print(f"Error running command {i}: {e}")
            return 1
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(run_tests())
