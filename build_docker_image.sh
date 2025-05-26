#!/bin/bash
# Build the tensorgrad Docker image with progress

echo "Building tensorgrad Docker image..."
echo "This will take 10-20 minutes due to TeX package installation"
echo ""

# Check if image already exists
if docker images tensorgrad -q | grep -q .; then
    echo "Warning: tensorgrad image already exists"
    read -p "Rebuild anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping build"
        exit 0
    fi
fi

# Build with progress
docker build \
    --progress=plain \
    -t tensorgrad \
    -f docker/Dockerfile \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Build successful!"
    docker images tensorgrad
else
    echo ""
    echo "❌ Build failed!"
    exit 1
fi