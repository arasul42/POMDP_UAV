#!/bin/bash

# Script to build and run a Julia 1.9 Docker container for POMDPs

# Name of the Docker image
IMAGE_NAME="julia-pomdp"

# Build the image (if not already built)
echo "🔧 Building Docker image '$IMAGE_NAME'..."
docker build -t $IMAGE_NAME .

# Run the container
echo "🚀 Starting container with current directory mounted to /workspace..."
docker run -it --rm -v "$(pwd)":/workspace $IMAGE_NAME bash
