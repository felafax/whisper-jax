#!/bin/bash

# Set error handling
set -e

# Common variables
IMAGE_NAME="gcr.io/felafax-training/whisper-jax-server"
LOCAL_CONTAINER_NAME="whisper-jax-server"

# Function to build Docker image
build() {
    local platform=$1
    echo "Building docker image..."
    if [ "$platform" == "arm64" ]; then
        echo "Building for Apple Silicon (ARM64)..."
        docker buildx create --use --name multi-platform-builder 2>/dev/null || true
        docker buildx build --platform linux/arm64 \
            --pull \
            --no-cache \
            -t $IMAGE_NAME:latest \
            --load .
    else
        echo "Building for default platform..."
        docker build --pull --no-cache -t $IMAGE_NAME:latest .
    fi
}

# Function to build and push
build_and_push() {
    echo "Building docker image for AMD64..."
    docker buildx create --use --name multi-platform-builder 2>/dev/null || true
    docker buildx build --platform linux/amd64 \
        --pull \
        --no-cache \
        -t $IMAGE_NAME:latest \
        --load .

    echo "Pushing docker image... $IMAGE_NAME"
    docker push $IMAGE_NAME:latest
}

# Main script logic
case "$1" in
    "build")
        if [ "$2" == "arm64" ]; then
            build "arm64"
        else
            build
        fi
        ;;
    "push")
        build_and_push
        ;;
    *)
        echo "Usage: $0 {build|push}"
        echo "  build [arm64] - Build the Docker image (optionally for ARM64)"
        echo "  push         - Build and push to container registry"
        exit 1
        ;;
esac 