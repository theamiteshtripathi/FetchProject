#!/bin/bash

# Enhanced Docker run script with options for API or frontend

# Default values
MODE="api"
API_PORT=8000
FRONTEND_PORT=8501
IMAGE_NAME="sentence-multitask"
CONTAINER_NAME="sentence-multitask-container"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --api-port)
      API_PORT="$2"
      shift 2
      ;;
    --frontend-port)
      FRONTEND_PORT="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --mode <api|frontend>  Run mode (default: api)"
      echo "  --api-port <port>      API port (default: 8000)"
      echo "  --frontend-port <port> Frontend port (default: 8501)"
      echo "  --help                 Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Build the Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:latest -f docker/Dockerfile .

# Run the Docker container based on the mode
if [ "$MODE" = "api" ]; then
  echo "Running Docker container in API mode..."
  docker run -p ${API_PORT}:8000 --name ${CONTAINER_NAME} ${IMAGE_NAME}:latest
elif [ "$MODE" = "frontend" ]; then
  echo "Running Docker container in Frontend mode..."
  docker run -p ${FRONTEND_PORT}:8501 --name ${CONTAINER_NAME} ${IMAGE_NAME}:latest streamlit run /app/src/frontend/app.py
else
  echo "Invalid mode: $MODE"
  exit 1
fi

echo "Container is running. To stop it, run: docker stop ${CONTAINER_NAME}"
 