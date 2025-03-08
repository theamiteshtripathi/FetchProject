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

# Create virtual environment (optional)
if command -v python3 &>/dev/null; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment
    if [ -f venv/bin/activate ]; then
        source venv/bin/activate
        
        # Install dependencies
        echo "Installing dependencies..."
        pip install -r requirements.txt
        
        echo "Virtual environment created and dependencies installed."
    else
        echo "Failed to create virtual environment."
    fi
else
    echo "Python 3 not found. Please install Python 3 and try again."
fi

echo "Project initialized successfully!"
echo "To get started, see the documentation in the docs/ directory." 