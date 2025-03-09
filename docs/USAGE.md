# Usage Guide

This document provides instructions on how to use the Sentence Transformer Multi-Task Learning project.

## Table of Contents

- [Installation](#installation)
- [Running the Model](#running-the-model)
  - [Running with Dummy Data](#running-with-dummy-data)
  - [Running with Real Data](#running-with-real-data)
- [Training the Model](#training-the-model)
  - [Training with Dummy Data](#training-with-dummy-data)
  - [Training with Real Datasets](#training-with-real-datasets)
- [Using the API](#using-the-api)
- [Using the Frontend](#using-the-frontend)
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

### Running with Dummy Data

You can run the model with dummy data using the `run_model.py` script:

```bash
python src/run_model.py --model_name sentence-transformers/all-MiniLM-L6-v2 --pooling mean
```

This will:
1. Load the sentence encoder
2. Encode example sentences and show their similarities
3. Run the multi-task model on the sentences and show predictions

#### Options

- `--model_name`: Pre-trained model name for the encoder (default: sentence-transformers/all-MiniLM-L6-v2)
- `--pooling`: Pooling strategy for sentence embeddings (choices: mean, cls, max; default: mean)
- `--model_path`: Path to a saved model checkpoint (optional)

### Running with Real Data

You can run the model with real data using the `run_model_real_data.py` script:

```bash
python src/run_model_real_data.py --model_name sentence-transformers/all-MiniLM-L6-v2 --pooling mean
```

This will:
1. Load sample sentences from SST-2 and CoNLL-2003 datasets
2. Load the sentence encoder and encode the sentences
3. Run the multi-task model on the sentences and show predictions for both sentiment analysis and named entity recognition

#### Options

- `--model_name`: Pre-trained model name for the encoder (default: sentence-transformers/all-MiniLM-L6-v2)
- `--pooling`: Pooling strategy for sentence embeddings (choices: mean, cls, max; default: mean)
- `--model_path`: Path to a saved model checkpoint (optional)

## Training the Model

### Training with Dummy Data

To train the model with dummy data, use the `train.py` script:

```bash
python src/train.py --model_name sentence-transformers/all-MiniLM-L6-v2 --pooling mean --epochs 5
```

This will:
1. Create a dummy dataset for demonstration
2. Train the model for the specified number of epochs
3. Save the best model to the output directory

#### Training Options

- `--model_name`: Pre-trained model name for the encoder (default: sentence-transformers/all-MiniLM-L6-v2)
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

### Training with Real Datasets

To train the model with real datasets (SST-2 for sentiment analysis and CoNLL-2003 for NER), use the `train_real_data.py` script:

```bash
python src/train_real_data.py --model_name sentence-transformers/all-MiniLM-L6-v2 --pooling mean --epochs 3 --sample_size 1000
```

This will:
1. Load the SST-2 dataset for sentiment analysis (Task A)
2. Load the CoNLL-2003 dataset for named entity recognition (Task B)
3. Train the model for the specified number of epochs
4. Save the best model to the output directory
5. Evaluate the model on the test set

#### Real Data Training Options

- `--model_name`: Pre-trained model name for the encoder (default: sentence-transformers/all-MiniLM-L6-v2)
- `--pooling`: Pooling strategy for sentence embeddings (choices: mean, cls, max; default: mean)
- `--max_length`: Maximum sequence length (default: 128)
- `--batch_size`: Batch size for training (default: 16)
- `--epochs`: Number of training epochs (default: 3)
- `--lr`: Learning rate (default: 2e-5)
- `--weight_decay`: Weight decay (default: 0.01)
- `--task_a_weight`: Weight for Task A loss (default: 1.0)
- `--task_b_weight`: Weight for Task B loss (default: 1.0)
- `--sample_size`: Number of examples to use from each dataset (default: 1000, use None for full datasets)
- `--freeze_encoder`: Freeze the encoder parameters (flag)
- `--freeze_task_a`: Freeze Task A head parameters (flag)
- `--freeze_task_b`: Freeze Task B head parameters (flag)
- `--seed`: Random seed (default: 42)
- `--device`: Device to use (cuda or cpu; default: cuda if available, otherwise cpu)
- `--output_dir`: Directory to save model and results (default: ./outputs)

#### Datasets

The real data training uses the following datasets:

1. **SST-2 (Stanford Sentiment Treebank)**: A dataset for sentiment analysis with binary labels (positive/negative).
2. **CoNLL-2003**: A dataset for named entity recognition with 9 entity types (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC).

These datasets are automatically downloaded using the Hugging Face Datasets library.

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

## Using the Frontend

The project includes a web-based frontend built with Streamlit that provides a user-friendly interface for interacting with the model.

### Starting the Frontend

To start the frontend, use the `run_frontend.py` script:

```bash
python src/run_frontend.py --port 8501
```

This will start a Streamlit server that serves the frontend application. You can access it by opening your browser and navigating to `http://localhost:8501`.

### Frontend Options

- `--port`: Port to run the Streamlit app on (default: 8501)

### Frontend Features

The frontend provides the following features:

1. **Model Configuration**: Select the pre-trained model and pooling strategy.
2. **Input Sentences**: Enter your own sentences or select from example sentences.
3. **Sentence Embeddings**: Visualize the sentence embeddings and their dimensions.
4. **Task Predictions**: View the predictions for both tasks (sentiment analysis and named entity recognition).
5. **Similarity Matrix**: Explore the similarity between different sentences.

The frontend is designed to be intuitive and user-friendly, making it easy to interact with the model and understand its outputs.

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