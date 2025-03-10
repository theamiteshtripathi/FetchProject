"""
Script to run the model for quick testing.
"""
import os
import argparse
import torch
import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

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
    Run the multi-task model on the given sentences.
    
    Args:
        model: The multi-task model
        sentences: List of sentences to process
    """
    print("\n=== Multi-Task Model Predictions ===\n")
    
    # Make predictions
    task_a_preds, task_a_probs, task_b_preds, task_b_probs = model.predict(sentences)
    
    # Print predictions
    for i, sentence in enumerate(sentences):
        print(f"Sentence: {sentence}")
        
        # Task A: Sentence Classification
        print(f"Task A (Classification): {task_a_preds[i]}")
        
        # Task B: Token Classification
        print("Task B (Token Classification):")
        tokens = sentence.split()
        if len(tokens) == len(task_b_preds[i]):
            for token, tag in zip(tokens, task_b_preds[i]):
                print(f"  {token}: {tag}")
        else:
            print(f"  Token mismatch: {len(tokens)} tokens vs {len(task_b_preds[i])} predictions")
        
        print("\n--------------------------------------------------\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run the sentence transformer and multi-task model')
    
    parser.add_argument('--model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Pre-trained model name for the encoder')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'cls', 'max'],
                        help='Pooling strategy for sentence embeddings')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to a saved model checkpoint')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Path to a file containing sentences (one per line)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Example sentences or load from file
    if args.input_file and os.path.exists(args.input_file):
        print(f"Loading sentences from {args.input_file}")
        with open(args.input_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(sentences)} sentences")
    else:
        sentences = [
            "I love machine learning and natural language processing.",
            "Deep learning models are revolutionizing NLP applications.",
            "The weather is beautiful today.",
            "It's a sunny day with clear skies.",
            "Python is my favorite programming language."
        ]
    
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
        encoder=encoder,
        encoder_model_name=args.model_name,
        pooling_strategy=args.pooling
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