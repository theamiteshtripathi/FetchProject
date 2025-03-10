#!/bin/bash

# Cleanup script to remove unnecessary files

echo "Cleaning up unnecessary files..."

# Remove redundant Streamlit run scripts
echo "Removing redundant Streamlit run scripts..."
rm -f run_streamlit_debug.sh
rm -f run_streamlit_fixed.sh
rm -f run_streamlit_main.sh
rm -f run_test_streamlit.sh
rm -f fix_and_run.sh

# Remove test files
echo "Removing test files..."
rm -f test_streamlit.py
rm -f test_imports.py

# Remove __pycache__ directories
echo "Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} +

# Remove .pyc files
echo "Removing .pyc files..."
find . -name "*.pyc" -delete

echo "Cleanup complete!" 