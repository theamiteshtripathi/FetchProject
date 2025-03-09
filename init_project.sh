#!/bin/bash

# Initialize the project
echo "Initializing the Sentence Transformer Multi-Task Learning project..."

# Create necessary directories
mkdir -p outputs

# Create Python package files
touch src/__init__.py
touch src/models/__init__.py
touch src/data/__init__.py
touch src/utils/__init__.py
touch src/api/__init__.py
touch tests/__init__.py

# Make shell scripts executable
chmod +x docker/run_docker.sh docker/deploy_aws.sh

# Instructions for conda environment
echo "Project initialized successfully!"
echo ""
echo "To set up the conda environment, run the following commands:"
echo "conda create -n fetch python=3.9"
echo "conda activate fetch"
echo "conda install pytorch -c pytorch"
echo "pip install -r requirements.txt"
echo ""
echo "To get started, see the documentation in the docs/ directory." 