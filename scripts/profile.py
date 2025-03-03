#!/usr/bin/env python3
"""
Main profiling script for tensorgrad.

This script provides a unified interface to profile different aspects of tensorgrad.

Usage:
    python scripts/profile.py [options]

Options:
    --ops          Profile tensor operations
    --test FILE    Profile a specific test file
    --all          Run all profiling jobs
    --clean        Clean up profiling output
"""
import os
import sys
import shutil
import subprocess

def create_profiling_dir():
    """Create the profiling directory if it doesn't exist."""
    os.makedirs("profiling", exist_ok=True)

def clean_profiling_outputs():
    """Remove all profiling outputs."""
    if os.path.exists("profiling"):
        shutil.rmtree("profiling")
        print("Cleaned profiling outputs.")

def profile_ops():
    """Run the tensor operations profiling."""
    create_profiling_dir()
    print("\n=== Profiling Tensor Operations ===")
    subprocess.run([sys.executable, "scripts/profile_ops.py"])

def profile_test(test_path):
    """Profile a specific test file."""
    create_profiling_dir()
    if not test_path.startswith("tests/"):
        test_path = f"tests/{test_path}"
    print(f"\n=== Profiling Test: {test_path} ===")
    subprocess.run([sys.executable, "scripts/profile_test.py", test_path])

def print_usage():
    """Print usage instructions."""
    print(__doc__)

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        return
    
    arg = sys.argv[1]
    
    if arg == "--clean":
        clean_profiling_outputs()
        return
    
    if arg == "--ops":
        profile_ops()
        return
    
    if arg == "--test" and len(sys.argv) > 2:
        profile_test(sys.argv[2])
        return
    
    if arg == "--all":
        # Run all profiling jobs
        profile_ops()
        profile_test("test_isomorphism.py")
        profile_test("test_tensor.py")
        profile_test("test_ml.py")
        return
    
    # If we got here, the arguments were invalid
    print("Invalid arguments.")
    print_usage()

if __name__ == "__main__":
    main()