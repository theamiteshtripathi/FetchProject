#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t sentence-multitask:latest -f docker/Dockerfile .

# Run the Docker container
echo "Running Docker container..."
docker run -p 8000:8000 --name sentence-multitask-container sentence-multitask:latest

# Note: To stop the container, run:
# docker stop sentence-multitask-container
 