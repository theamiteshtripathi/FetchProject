#!/bin/bash

# Activate the conda environment
echo "Activating conda environment 'fetch'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate fetch

# Update the environment with the fixed dependencies
echo "Updating dependencies..."
pip install transformers==4.36.2 tokenizers==0.21.0

# Run the test Streamlit app
echo "Starting the test Streamlit app..."
streamlit run test_streamlit.py

echo "Done!" 