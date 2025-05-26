#!/usr/bin/env python3
"""Debug script to test Docker build and run issues."""

import subprocess
import sys
import time

def run_command(cmd, cwd=None):
    """Run a command and stream output in real-time."""
    print(f"\n>>> Running: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd
    )
    
    # Stream output in real-time
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    
    process.wait()
    return process.returncode

def main():
    print("Testing Docker build process...")
    
    # Check if Docker is running
    print("\n1. Checking Docker status...")
    ret = run_command(["docker", "version"])
    if ret != 0:
        print("ERROR: Docker is not running or not installed!")
        return 1
    
    # Check for existing containers
    print("\n2. Checking for existing containers...")
    run_command(["docker", "ps", "-a", "--filter", "name=lambda_local_test"])
    
    # Try to remove any existing container
    print("\n3. Cleaning up any existing container...")
    run_command(["docker", "rm", "-f", "lambda_local_test"])
    
    # Build with progress output
    print("\n4. Building Docker image (this may take several minutes)...")
    start_time = time.time()
    ret = run_command([
        "docker", "build",
        "--progress=plain",  # Show detailed progress
        "-t", "tensorgrad",
        "-f", "docker/Dockerfile",
        "."
    ], cwd="..")
    
    if ret != 0:
        print(f"\nERROR: Docker build failed after {time.time() - start_time:.1f} seconds!")
        return 1
    
    print(f"\nDocker build completed in {time.time() - start_time:.1f} seconds")
    
    # Try running the container
    print("\n5. Running container...")
    ret = run_command([
        "docker", "run",
        "-d",
        "--name", "lambda_local_test",
        "-p", "9000:8080",
        "tensorgrad"
    ])
    
    if ret != 0:
        print("\nERROR: Failed to run container!")
        return 1
    
    # Check if container is running
    print("\n6. Checking container status...")
    time.sleep(2)
    run_command(["docker", "ps", "--filter", "name=lambda_local_test"])
    
    # Check logs
    print("\n7. Container logs:")
    run_command(["docker", "logs", "lambda_local_test"])
    
    # Cleanup
    print("\n8. Cleaning up...")
    run_command(["docker", "stop", "lambda_local_test"])
    run_command(["docker", "rm", "lambda_local_test"])
    
    print("\nDone!")
    return 0

if __name__ == "__main__":
    sys.exit(main())