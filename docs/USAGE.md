# Usage Guide

This document provides instructions on how to use the Sentence Transformer Multi-Task Learning project.

## Table of Contents

- [Installation](#installation)
- [Running the Model](#running-the-model)
- [Training the Model](#training-the-model)
- [Using the API](#using-the-api)
- [Deployment](#deployment)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FetchProject.git
   cd FetchProject
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Model

You can run the model directly using the `run_model.py` script:

```bash
python src/run_model.py --model_name all-MiniLM-L6-v2 --pooling mean
```

This will:
1. Load the sentence encoder
2. Encode example sentences and show their similarities
3. Run the multi-task model on the sentences and show predictions

### Options

- `--model_name`: Pre-trained model name for the encoder (default: all-MiniLM-L6-v2)
- `--pooling`: Pooling strategy for sentence embeddings (choices: mean, cls, max; default: mean)
- `--model_path`: Path to a saved model checkpoint (optional)

## Training the Model

To train the model, use the `train.py` script:

```bash
python src/train.py --model_name all-MiniLM-L6-v2 --pooling mean --epochs 5
```

This will:
1. Create a dummy dataset for demonstration
2. Train the model for the specified number of epochs
3. Save the best model to the output directory

### Training Options

- `--model_name`: Pre-trained model name for the encoder (default: all-MiniLM-L6-v2)
- `--pooling`: Pooling strategy for sentence embeddings (choices: mean, cls, max; default: mean)
- `--task_a_classes`: Number of classes for Task A (default: 3)
- `--task_b_labels`: Number of labels for Task B (default: 5)
- `--max_length`: Maximum sequence length (default: 128)
- `--batch_size`: Batch size for training (default: 16)
- `--epochs`: Number of training epochs (default: 5)
- `--lr`: Learning rate (default: 2e-5)
- `--weight_decay`: Weight decay (default: 0.01)
- `--task_a_weight`: Weight for Task A loss (default: 1.0)
- `--task_b_weight`: Weight for Task B loss (default: 1.0)
- `--freeze_encoder`: Freeze the encoder parameters (flag)
- `--freeze_task_a`: Freeze Task A head parameters (flag)
- `--freeze_task_b`: Freeze Task B head parameters (flag)
- `--seed`: Random seed (default: 42)
- `--device`: Device to use (cuda or cpu; default: cuda if available, otherwise cpu)
- `--output_dir`: Directory to save model and results (default: ./outputs)

## Using the API

### Starting the API Server

To start the API server, use the `run_api.py` script:

```bash
python src/run_api.py --host 0.0.0.0 --port 8000
```

This will start a FastAPI server that provides endpoints for sentence encoding and multi-task predictions.

### API Options

- `--host`: Host to run the server on (default: 0.0.0.0)
- `--port`: Port to run the server on (default: 8000)
- `--model_path`: Path to a saved model checkpoint (optional)
- `--reload`: Enable auto-reload for development (flag)

### Testing the API

To test the API, use the `test_api.py` script:

```bash
python src/test_api.py --url http://localhost:8000
```

This will:
1. Test the health endpoint
2. Test the encode endpoint with example sentences
3. Test the predict endpoint with example sentences

### API Endpoints

- `GET /`: Root endpoint
- `GET /health`: Health check endpoint
- `POST /encode`: Encode sentences to embeddings
- `POST /predict`: Make predictions for both tasks

## Deployment

### Docker

To build and run the Docker container locally:

```bash
./docker/run_docker.sh
```

This will:
1. Build the Docker image
2. Run the Docker container, exposing port 8000

### AWS Deployment

To deploy the Docker container to AWS:

```bash
./docker/deploy_aws.sh
```

This will:
1. Build the Docker image
2. Push the image to Amazon ECR
3. Create or update an ECS task definition
4. Create or update an ECS service

#### Prerequisites for AWS Deployment

- AWS CLI installed and configured
- Docker installed
- Appropriate AWS permissions

#### AWS Configuration

You may need to modify the following variables in `docker/deploy_aws.sh`:

- `AWS_REGION`: AWS region to deploy to
- `ECR_REPOSITORY_NAME`: Name of the ECR repository
- `ECS_CLUSTER_NAME`: Name of the ECS cluster
- `ECS_SERVICE_NAME`: Name of the ECS service
- `ECS_TASK_FAMILY`: Name of the ECS task family 