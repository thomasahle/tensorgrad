"""Quick test to check if Docker tests can run without building."""

import subprocess
import sys

def check_docker():
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            print("ERROR: Docker is not running!")
            print(result.stderr)
            return False
        print("✓ Docker is running")
        return True
    except Exception as e:
        print(f"ERROR: Failed to check Docker: {e}")
        return False

def check_existing_image():
    """Check if the tensorgrad image already exists."""
    result = subprocess.run(
        ["docker", "images", "tensorgrad", "-q"],
        capture_output=True,
        text=True
    )
    if result.stdout.strip():
        print("✓ tensorgrad image already exists")
        return True
    else:
        print("✗ tensorgrad image not found - would need to build")
        return False

def check_port():
    """Check if port 9000 is available."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 9000))
    sock.close()
    if result == 0:
        print("✗ Port 9000 is already in use!")
        # Check what's using it
        subprocess.run(["lsof", "-i", ":9000"], capture_output=False)
        return False
    else:
        print("✓ Port 9000 is available")
        return True

def check_buildx():
    """Check if docker buildx is available."""
    result = subprocess.run(
        ["docker", "buildx", "version"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✓ docker buildx is available")
        return True
    else:
        print("✗ docker buildx not available - will fall back to regular build")
        return False

def main():
    print("Checking Docker test prerequisites...\n")
    
    checks = [
        check_docker(),
        check_existing_image(),
        check_port(),
        check_buildx()
    ]
    
    if not checks[0]:  # Docker not running
        print("\nDocker is required to run these tests!")
        return 1
    
    if not checks[1]:  # Image doesn't exist
        print("\nThe Docker image needs to be built first.")
        print("This can take 10-20 minutes due to TeX installation.")
        print("Run: docker build -t tensorgrad -f docker/Dockerfile .")
        
    if not checks[2]:  # Port in use
        print("\nPort 9000 is in use. Kill the process or use a different port.")
        
    return 0

if __name__ == "__main__":
    sys.exit(main())