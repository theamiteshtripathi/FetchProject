# Makefile for Sentence Transformer Multi-Task Learning project

.PHONY: setup run-model run-real-data run-frontend run-api test clean docker-api docker-frontend

# Setup the project
setup:
	./init_project.sh

# Run the model with dummy data
run-model:
	python src/run_model.py

# Run the model with real data
run-real-data:
	python src/run_model_real_data.py

# Run the frontend
run-frontend:
	./run_streamlit.sh

# Run the API
run-api:
	python src/run_api.py

# Run tests
test:
	python -m unittest discover tests

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name "*.pyd" -delete
	find . -name ".DS_Store" -delete
	find . -name "*.so" -delete
	find . -name "*.ipynb_checkpoints" -exec rm -rf {} +

# Docker commands
docker-api:
	./docker/run_docker.sh --mode api

docker-frontend:
	./docker/run_docker.sh --mode frontend

# Install the package in development mode
dev-install:
	pip install -e .

# Create a distribution package
dist:
	python setup.py sdist bdist_wheel 