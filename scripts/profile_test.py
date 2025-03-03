"""
Python script to run profiling on specific pytest tests.

Usage:
    python scripts/profile_test.py tests/test_file.py [test_function_name]
    
Examples:
    # Profile a specific test file
    python scripts/profile_test.py tests/test_tensor.py
    
    # Profile a specific test function
    python scripts/profile_test.py tests/test_tensor.py::test_rename
    
    # Use analyze_profile.py to examine the results
    python scripts/analyze_profile.py test_output.prof
"""
import cProfile
import pytest
import sys
import os

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/profile_test.py <test_file_path> [test_function]")
        sys.exit(1)
    
    test_path = sys.argv[1]
    
    # Create a output filename based on the test path
    profile_output = os.path.join(
        "profiling", 
        os.path.basename(test_path).replace('.py', '') + '.prof'
    )
    
    # Make sure the output directory exists
    os.makedirs("profiling", exist_ok=True)
    
    # Run the test with cProfile
    cProfile.run(f'pytest.main(["{test_path}", "-v"])', profile_output)
    
    print(f"\nProfile data saved to {profile_output}")
    print(f"To analyze the profile, run:")
    print(f"python scripts/analyze_profile.py {profile_output}")