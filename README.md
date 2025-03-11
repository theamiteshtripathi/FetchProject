# Sentence Transformer Multi-Task Learning Project

## Project Overview

This project implements a multi-task learning system using sentence transformers. The system is designed to encode sentences into fixed-size vectors and perform multiple tasks simultaneously using a shared transformer backbone. It demonstrates how transfer learning and multi-task learning can be combined to create a powerful NLP system capable of handling multiple tasks with a single model.

## System Design Workflow

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
│  (Sentence Classification)    │   │  (Token Classification)       │
└───────────────┬───────────────┘   └───────────────┬───────────────┘
                │                                   │
                ▼                                   ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│      Task A Predictions       │   │      Task B Predictions       │
└───────────────────────────────┘   └───────────────────────────────┘
```

## Project Structure

```
FetchProject/
├── src/                    # Source code
│   ├── models/             # Model definitions
│   │   ├── sentence_encoder.py    # Sentence embedding model
│   │   └── multi_task_model.py    # Multi-task learning model
│   ├── data/               # Data loading and processing
│   ├── utils/              # Utility functions
│   ├── api/                # API for model serving
│   ├── frontend/           # Streamlit frontend
│   ├── run_model.py        # Script to run model with example data
│   ├── run_model_real_data.py # Script to run model with real data
│   └── fine_tune.py        # Script for fine-tuning the model
├── notebooks/              # Jupyter notebooks with examples and tutorials
├── tests/                  # Unit and integration tests
├── docker/                 # Docker-related files for containerization
├── docs/                   # Documentation files
├── outputs/                # Model outputs, checkpoints, and data
├── FREEZING_GUIDE.md       # Guide for using model freezing parameters
├── USER_GUIDE.md           # Comprehensive user guide
├── FINE_TUNING_REPORT.md   # Report on fine-tuning results
├── requirements.txt        # Python dependencies
├── environment.yml         # Conda environment specification
├── Makefile                # Make commands for common operations
├── run_streamlit.sh        # Script to run the Streamlit frontend
└── README.md               # Main project documentation
```

## Documentation Structure

This project includes several README files and documentation resources:

1. **Main README (this file)**: Overview of the project, setup instructions, and general information
2. **[USER_GUIDE.md](USER_GUIDE.md)**: Comprehensive guide on using the project, including how to explain it to an interviewer
3. **[FREEZING_GUIDE.md](FREEZING_GUIDE.md)**: Detailed guide on using the model freezing parameters for transfer learning
4. **[FINE_TUNING_REPORT.md](FINE_TUNING_REPORT.md)**: Report on fine-tuning experiments and results
5. **[docs/README.md](docs/README.md)**: Index of additional documentation resources
6. **[tests/README.md](tests/README.md)**: Information about the test suite and how to run tests
7. **[notebooks/README.md](notebooks/README.md)**: Guide to the Jupyter notebooks included in the project
8. **[outputs/README.md](outputs/README.md)**: Information about model outputs and checkpoints

## Implementation Plan

### Understanding Key Concepts

#### Sentence Transformers

A sentence transformer converts an entire sentence into a fixed-size numerical vector (embedding) that captures its meaning. This is achieved by passing a sentence through a transformer model like BERT and then applying pooling operations to obtain a single vector representation. These embeddings can be used for various downstream tasks such as classification, clustering, and similarity comparison.

#### Multi-Task Learning

Multi-task learning (MTL) is a training paradigm where a single model is trained to perform multiple tasks simultaneously. By sharing knowledge across tasks, MTL often leads to improved learning efficiency and better generalization. In this project, we implement:

- **Task A**: Sentence Classification - Categorizing sentences into predefined classes (Technology, Weather, Other)
- **Task B**: Token Classification - Assigning tags to individual tokens in a sentence (NER tags)

### Tech Stack & Tools

- **PyTorch**: Primary deep learning framework for model implementation
- **Hugging Face Transformers**: For pre-trained transformer models and tokenizers
- **Hugging Face Datasets**: For loading and processing datasets
- **Streamlit**: For building the interactive frontend with visualization capabilities
- **FastAPI**: For API development and model serving
- **Docker**: For containerization and deployment
- **AWS**: For cloud deployment and hosting

### Implementation Steps

1. **Setting Up the Environment** (Day 1)

   - Configure development environment with conda and pip
   - Install necessary libraries and dependencies
   - Set up version control and project structure

2. **Implementing the Sentence Transformer** (Days 2-3)

   - Leverage pre-trained models from Hugging Face
   - Implement sentence encoding functionality with different pooling strategies
   - Test with sample sentences and visualize embeddings

3. **Expanding to Multi-Task Learning** (Days 3-4)

   - Implement task-specific heads for classification and token tagging
   - Integrate with the sentence encoder using a shared backbone
   - Test with dummy data to verify functionality

4. **Training Considerations** (Day 5)

   - Explore different freezing scenarios for transfer learning
   - Implement transfer learning strategies with parameter freezing
   - Document training approaches and best practices

5. **Writing the Training Loop** (Day 6)

   - Implement multi-task training logic with weighted losses
   - Handle task-specific losses and optimization
   - Monitor training progress with validation metrics

6. **Testing and Evaluating Results** (Day 7)

   - Evaluate model performance on test datasets
   - Analyze multi-task learning benefits compared to single-task models
   - Prepare for deployment with model serialization

7. **Building the Frontend** (Day 8)
   - Create an interactive Streamlit app with intuitive UI
   - Visualize sentence embeddings and similarities with heatmaps
   - Display task predictions with confidence scores and color coding

### Datasets

The project supports training with both dummy data and real datasets:

1. **Dummy Data**: Generated synthetic data for quick testing and demonstration, created with the `create_example_data` function.

2. **Real Datasets**:
   - **SST-2 (Stanford Sentiment Treebank)**: A dataset for sentiment analysis with binary labels (positive/negative).
   - **CoNLL-2003**: A dataset for named entity recognition with 9 entity types (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC).
   - **Custom Data**: The project also supports fine-tuning on custom data with the format specified in `fine_tune.py`.

These datasets are automatically downloaded using the Hugging Face Datasets library or can be created using the provided scripts.

### Deployment Strategy

- Containerize the application with Docker using the provided Dockerfile
- Deploy on AWS using ECS or Fargate for scalable hosting
- Set up API endpoints for inference with FastAPI
- Configure CI/CD pipelines for automated testing and deployment

### Best Practices & Optimization

- Incremental development and testing with unit tests
- GPU optimization for training with mixed precision
- Efficient data loading with batching and caching
- Monitoring and early stopping to prevent overfitting
- Balancing multi-task training with task weights
- Code modularity and comprehensive documentation

## Getting Started

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

   # Alternatively, you can set up manually:
   # conda create -n fetch python=3.9
   # conda activate fetch
   # conda install pytorch -c pytorch
   # pip install -r requirements.txt
   ```

3. Initialize the project:

   ```bash
   ./init_project.sh
   # Or use make
   make setup
   ```

4. Run the model:

   ```bash
   # Run with dummy data
   python src/run_model.py
   # Or use make
   make run-model

   # Run with real data
   python src/run_model_real_data.py
   # Or use make
   make run-real-data
   ```

5. Start the frontend:

   ```bash
   # Use the consolidated script
   ./run_streamlit.sh
   # Or use make
   make run-frontend
   ```

6. Run with Docker:

   ```bash
   # Run the API
   ./docker/run_docker.sh --mode api
   # Or use make
   make docker-api

   # Run the frontend
   ./docker/run_docker.sh --mode frontend
   # Or use make
   make docker-frontend

   # For more options
   ./docker/run_docker.sh --help
   ```

7. Run tests:

   ```bash
   python -m unittest discover tests
   # Or use make
   make test
   ```

8. Fine-tune the model:

   ```bash
   # Create example data
   make create-example-data

   # Fine-tune the model
   make fine-tune

   # Run the model with the fine-tuned model
   make run-fine-tuned
   ```

9. Clean up:

   ```bash
   # Clean up cache files
   make clean
   ```

10. Follow the notebooks in the `notebooks/` directory for examples and tutorials

11. See the documentation in the `docs/` directory for detailed usage instructions

## Key Features

- **Multi-Task Learning**: Train a single model to perform multiple NLP tasks simultaneously
- **Transfer Learning**: Leverage pre-trained transformer models and fine-tune for specific tasks
- **Interactive Frontend**: Visualize embeddings, similarities, and predictions with Streamlit
- **Model Freezing**: Control which parts of the model are updated during fine-tuning
- **Customizable Training**: Adjust learning rates, batch sizes, and task weights
- **Comprehensive Documentation**: Detailed guides and explanations for all aspects of the project

## User Guide

For a comprehensive guide on how to use this project, including how to explain it to an interviewer, see the [USER_GUIDE.md](USER_GUIDE.md) file.

## License

[MIT License](LICENSE)

## Acknowledgments

- Hugging Face for their excellent Transformers library
- Sentence-Transformers project for inspiration
- PyTorch team for the deep learning framework
- Streamlit for the interactive frontend capabilities
