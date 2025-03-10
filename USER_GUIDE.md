# Sentence Transformer Multi-Task Learning Project User Guide

This guide will help you understand how to use the Sentence Transformer Multi-Task Learning project, including running the model, fine-tuning it for better accuracy, and explaining the project to an interviewer.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Getting Started](#getting-started)
3. [Running the Model](#running-the-model)
4. [Using the Frontend](#using-the-frontend)
5. [Fine-Tuning the Model](#fine-tuning-the-model)
6. [Explaining to an Interviewer](#explaining-to-an-interviewer)
7. [Troubleshooting](#troubleshooting)

## Project Overview

This project implements a multi-task learning system using sentence transformers. The system encodes sentences into fixed-size vectors and performs multiple tasks simultaneously:

- **Task A**: Sentence Classification - Classifies sentences into predefined categories (Technology, Weather, Other)
- **Task B**: Token Classification - Assigns tags to each token in a sentence (O, B-TECH, I-TECH, B-WEATHER, I-WEATHER)

The system uses a shared transformer backbone (a pre-trained model like BERT or RoBERTa) and task-specific heads for each task.

## Getting Started

### Prerequisites

- Python 3.9+
- PyTorch
- Transformers library
- Streamlit (for the frontend)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/FetchProject.git
   cd FetchProject
   ```

2. Set up the conda environment:

   ```bash
   # Create and activate the conda environment
   conda env create -f environment.yml
   conda activate fetch
   ```

3. Initialize the project:
   ```bash
   ./init_project.sh
   # Or use make
   make setup
   ```

## Running the Model

You can run the model with either dummy data or real data:

### Using Dummy Data

```bash
# Run with dummy data
python src/run_model.py
# Or use make
make run-model
```

This will:

1. Load the sentence encoder
2. Encode example sentences
3. Calculate similarity between sentences
4. Run the multi-task model to get predictions for both tasks

### Using Real Data

```bash
# Run with real data
python src/run_model_real_data.py
# Or use make
make run-real-data
```

This uses real datasets (SST-2 for sentiment analysis and CoNLL-2003 for named entity recognition) instead of dummy data.

## Using the Frontend

The project includes a Streamlit frontend that allows you to interact with the model:

```bash
# Start the frontend
./run_streamlit.sh
# Or use make
make run-frontend
```

The frontend provides:

1. **Model Configuration**:

   - Choose different pre-trained models
   - Select pooling strategies
   - Adjust confidence thresholds

2. **Input**:

   - Enter your own sentences
   - See sentence embeddings

3. **Results**:
   - View sentence classification results
   - See token classification with color-coded tags
   - Explore similarity between sentences
   - View confidence scores for predictions

## Fine-Tuning the Model

If you're not satisfied with the model's performance, you can fine-tune it on your own data:

### Creating Example Data

```bash
# Create example data
python src/fine_tune.py --create_example_data
```

This creates a JSON file with example data in `outputs/data/example_data.json`.

### Fine-Tuning

```bash
# Fine-tune the model
python src/fine_tune.py --num_epochs 5 --batch_size 8
```

You can customize the fine-tuning process with these parameters:

- `--model_name`: Pre-trained model to use (default: "sentence-transformers/all-MiniLM-L6-v2")
- `--pooling_strategy`: Pooling strategy (default: "mean")
- `--num_epochs`: Number of training epochs (default: 5)
- `--batch_size`: Batch size for training (default: 8)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--task_a_weight`: Weight for Task A loss (default: 1.0)
- `--task_b_weight`: Weight for Task B loss (default: 1.0)
- `--output_dir`: Directory to save the model (default: "outputs/models/fine_tuned")

### Using Your Fine-Tuned Model

After fine-tuning, you can use your model by specifying the model path:

```bash
python src/run_model.py --model_path outputs/models/fine_tuned/model_final.pt
```

Or in the frontend, you can select your fine-tuned model from the dropdown.

## Explaining to an Interviewer

When explaining this project to an interviewer, focus on these key aspects:

### 1. Architecture Overview

"This project implements a multi-task learning system using sentence transformers. The architecture consists of a shared transformer backbone that encodes sentences into fixed-size vectors, followed by task-specific heads for different NLP tasks."

### 2. Technical Implementation

"I used PyTorch and the Hugging Face Transformers library to implement the model. The system leverages pre-trained models like BERT or RoBERTa as the backbone, which provides a strong foundation for the downstream tasks."

### 3. Multi-Task Learning Benefits

"The multi-task learning approach allows the model to share knowledge across tasks, which can lead to better generalization and more efficient use of data. By training on multiple tasks simultaneously, the model learns more robust representations."

### 4. Design Decisions

"I made several key design decisions:

- Using a modular architecture that separates the encoder from task-specific heads
- Implementing different pooling strategies (mean, max, CLS) to get sentence embeddings
- Creating a flexible training loop that balances multiple task losses
- Building an interactive frontend for easy exploration and demonstration"

### 5. Transfer Learning Considerations

"I carefully considered transfer learning aspects:

- For the encoder, I used a pre-trained model to leverage knowledge from large datasets
- I implemented options to freeze different parts of the model during fine-tuning
- The system allows for task-specific fine-tuning while preserving general language understanding"

### 6. Challenges and Solutions

"Some challenges I faced included:

- Aligning token labels with wordpiece tokens, which I solved by implementing a mapping algorithm
- Balancing the losses of different tasks, which I addressed by adding task weights
- Optimizing the model for both accuracy and inference speed, which required careful hyperparameter tuning"

## Troubleshooting

### Common Issues

#### PyTorch and Streamlit Conflict

If you encounter errors like:

```
RuntimeError: no running event loop
RuntimeError: Tried to instantiate class '__path__._path', but it does not exist!
```

Solution:

```bash
export STREAMLIT_WATCH_EXCLUDE_MODULES="torch,tensorflow"
export STREAMLIT_SERVER_HEADLESS=true
```

These are already set in the `run_streamlit.sh` script.

#### Model Misclassification

If the model is misclassifying everything:

1. Try fine-tuning the model on your specific data:

   ```bash
   python src/fine_tune.py
   ```

2. Adjust the confidence thresholds in the frontend

3. Try a different pre-trained model:
   ```bash
   python src/run_model.py --model_name sentence-transformers/all-mpnet-base-v2
   ```

#### Out of Memory Errors

If you encounter out of memory errors:

1. Reduce the batch size:

   ```bash
   python src/fine_tune.py --batch_size 4
   ```

2. Use a smaller model:

   ```bash
   python src/run_model.py --model_name sentence-transformers/all-MiniLM-L6-v2
   ```

3. Reduce the maximum sequence length:
   ```bash
   python src/run_model.py --max_length 64
   ```

### Getting Help

If you encounter any issues not covered here, please check the [TROUBLESHOOTING.md](TROUBLESHOOTING.md) file or open an issue on GitHub.
