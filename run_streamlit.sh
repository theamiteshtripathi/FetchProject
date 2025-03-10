#!/bin/bash

# Consolidated script to run the Streamlit app

# Activate the conda environment
echo "Activating conda environment 'fetch'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate fetch

# Set environment variables to prevent issues with PyTorch and Streamlit
export STREAMLIT_WATCH_EXCLUDE_MODULES="torch,tensorflow"
export STREAMLIT_SERVER_HEADLESS=true
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Force PyTorch to use CPU to avoid CUDA issues with Streamlit
export CUDA_VISIBLE_DEVICES=""

# Optional: Update dependencies if needed (uncomment if required)
# echo "Updating dependencies..."
# pip install transformers==4.36.2 tokenizers==0.21.0

# Run the Streamlit app
echo "Starting the Streamlit app..."
streamlit run src/frontend/app.py --server.port 8501

echo "Done!" 