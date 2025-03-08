"""
Script to run the model with real data for quick testing.
"""
import os
import argparse
import torch
import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset

from models.sentence_encoder import SentenceEncoder
from models.multi_task_model import MultiTaskModel


def encode_sentences(encoder: SentenceEncoder, sentences: List[str]) -> None:
    """
    Encode sentences and print their embeddings and similarities.
    
    Args:
        encoder: The sentence encoder
        sentences: List of sentences to encode
    """
    print("\n=== Encoding Sentences ===\n")
    
    # Encode sentences
    with torch.no_grad():
        embeddings = encoder.encode(sentences)
    
    # Print embedding shapes
    print(f"Embedding shape: {embeddings.shape}")
    
    # Print a sample of the first embedding
    print(f"\nSample of first embedding (first 10 dimensions):")
    print(embeddings[0, :10].cpu().numpy())
    
    # Compute similarity matrix
    embeddings_np = embeddings.cpu().numpy()
    similarity_matrix = cosine_similarity(embeddings_np)
    
    # Print similarities
    print("\nSimilarity Matrix:")
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            print(f"Similarity between \"{sentences[i]}\" and \"{sentences[j]}\": {similarity_matrix[i][j]:.4f}")


def run_multi_task_model(model: MultiTaskModel, sentences: List[str]) -> None:
    """
    Run the multi-task model on sentences and print predictions.
    
    Args:
        model: The multi-task model
        sentences: List of sentences to process
    """
    print("\n=== Multi-Task Model Predictions ===\n")
    
    # Define class names and NER tags for interpretation
    task_a_labels = ["Negative", "Positive"]  # SST-2 labels
    task_b_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]  # CoNLL-2003 labels
    
    # Make predictions
    predictions = model.predict(sentences)
    
    # Get predictions for each task
    task_a_preds = predictions['task_a']
    task_b_preds = predictions['task_b']
    
    # Print predictions
    for i, sentence in enumerate(sentences):
        print(f"Sentence: {sentence}")
        
        # Task A prediction (sentiment classification)
        task_a_pred = task_a_preds[i]
        print(f"Task A (Sentiment): {task_a_labels[task_a_pred]}")
        
        # Task B prediction (token classification)
        tokens = sentence.split()
        task_b_pred = task_b_preds[i][:len(tokens)]  # Get predictions for actual tokens
        
        # Map numeric predictions to labels
        token_labels = [task_b_labels[pred] for pred in task_b_pred]
        
        # Print token-level predictions
        print("Task B (NER):")
        for token, label in zip(tokens, token_labels):
            print(f"  {token}: {label}")
        
        print("\n" + "-"*50 + "\n")


def get_real_data_samples() -> List[str]:
    """
    Get sample sentences from real datasets.
    
    Returns:
        List of sample sentences
    """
    # Load SST-2 dataset
    sst2 = load_dataset("glue", "sst2", split="validation")
    
    # Load CoNLL-2003 dataset
    conll = load_dataset("conll2003", split="validation")
    
    # Get sample sentences from SST-2
    sst2_samples = [
        sst2[0]["sentence"],  # Negative example
        sst2[1]["sentence"],  # Positive example
    ]
    
    # Get sample sentences from CoNLL-2003
    conll_samples = [
        " ".join(conll[0]["tokens"]),  # Example with entities
        " ".join(conll[1]["tokens"]),  # Example with entities
    ]
    
    # Add some custom sentences
    custom_samples = [
        "Apple Inc. is headquartered in Cupertino, California.",
        "Barack Obama was the 44th president of the United States.",
        "I really enjoyed the movie, it was fantastic!",
        "The weather in New York is terrible today."
    ]
    
    # Combine all samples
    return sst2_samples + conll_samples + custom_samples


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run the sentence transformer and multi-task model with real data')
    
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='Pre-trained model name for the encoder')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'cls', 'max'],
                        help='Pooling strategy for sentence embeddings')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a saved model checkpoint')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get sample sentences from real datasets
    sentences = get_real_data_samples()
    
    # Load sentence encoder
    print(f"\nLoading sentence encoder with {args.model_name} and {args.pooling} pooling...")
    encoder = SentenceEncoder(model_name=args.model_name, pooling_strategy=args.pooling)
    encoder = encoder.to(device)
    encoder.eval()
    
    # Encode sentences
    encode_sentences(encoder, sentences)
    
    # Load multi-task model
    print(f"\nLoading multi-task model...")
    model = MultiTaskModel(
        encoder_model_name=args.model_name,
        pooling_strategy=args.pooling,
        task_a_num_classes=2,  # SST-2: 2 classes (negative, positive)
        task_b_num_labels=9    # CoNLL-2003: 9 labels (O, B-PER, I-PER, etc.)
    )
    
    # Load saved model if specified
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    # Run multi-task model
    run_multi_task_model(model, sentences)


if __name__ == "__main__":
    main() 