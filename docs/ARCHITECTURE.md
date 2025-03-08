# Architecture

This document describes the architecture of the Sentence Transformer Multi-Task Learning project.

## Overview

The project implements a multi-task learning system using sentence transformers. The system is designed to encode sentences into fixed-size vectors and perform multiple tasks simultaneously using a shared transformer backbone.

## System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Sentences                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Text Preprocessing                        │
│                  (Tokenization, Input Formatting)               │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Shared Transformer Encoder                    │
│                 (Pre-trained BERT/RoBERTa/etc.)                 │
└───────────────┬─────────────────────────────────┬───────────────┘
                │                                 │
                ▼                                 ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│        Task A Head            │   │        Task B Head            │
│  (Sentence Classification)    │   │  (NER/Sentiment/etc.)         │
└───────────────┬───────────────┘   └───────────────┬───────────────┘
                │                                   │
                ▼                                   ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│      Task A Predictions       │   │      Task B Predictions       │
└───────────────────────────────┘   └───────────────────────────────┘
```

## Components

### Sentence Encoder

The `SentenceEncoder` class is responsible for converting sentences into fixed-size embeddings. It wraps a pre-trained transformer model and adds pooling to generate sentence-level embeddings from token-level representations.

Key features:
- Supports different pooling strategies (mean, CLS, max)
- Uses pre-trained models from Hugging Face
- Returns both sentence embeddings and token embeddings

### Multi-Task Model

The `MultiTaskModel` class combines the sentence encoder with task-specific heads for multi-task learning. It has the following components:

1. **Shared Encoder**: A pre-trained transformer that encodes input sentences into embeddings.
2. **Task A Head**: A classification head for sentence-level tasks.
3. **Task B Head**: A token classification head for sequence labeling tasks.

The model can be configured to freeze different parts during training, allowing for various transfer learning scenarios.

### Task-Specific Heads

1. **ClassificationHead**: A simple feed-forward network for sentence classification tasks.
2. **TokenClassificationHead**: A token-level classifier for sequence labeling tasks like NER.

## Training Process

The training process involves:

1. **Data Preparation**: Loading and preprocessing data for both tasks.
2. **Forward Pass**: Passing input through the shared encoder and task-specific heads.
3. **Loss Computation**: Computing losses for each task and combining them.
4. **Backpropagation**: Updating model parameters based on the combined loss.
5. **Evaluation**: Tracking metrics for each task to monitor progress.

The training loop supports different freezing scenarios:
- Freezing the entire network
- Freezing only the transformer backbone
- Freezing only one task-specific head

## API

The API is built using FastAPI and provides endpoints for:

1. **Sentence Encoding**: Converting sentences into embeddings.
2. **Multi-Task Predictions**: Making predictions for both tasks.

The API is designed to be deployed as a containerized service using Docker and AWS.

## Deployment Architecture

The deployment architecture consists of:

1. **Docker Container**: Packaging the application and its dependencies.
2. **AWS ECR**: Storing the Docker image.
3. **AWS ECS/Fargate**: Running the containerized application.

This architecture allows for scalable and reliable deployment of the model as a service. 