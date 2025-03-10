"""
Script to test the API.
"""
import argparse
import requests
import json
from typing import List, Dict, Any


def test_health(base_url: str) -> None:
    """
    Test the health endpoint.
    
    Args:
        base_url: Base URL of the API
    """
    url = f"{base_url}/health"
    print(f"Testing health endpoint: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        print(f"Health check successful: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")


def test_encode(base_url: str, sentences: List[str]) -> None:
    """
    Test the encode endpoint.
    
    Args:
        base_url: Base URL of the API
        sentences: List of sentences to encode
    """
    url = f"{base_url}/encode"
    print(f"Testing encode endpoint: {url}")
    
    data = {
        "sentences": sentences,
        "return_probabilities": False
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        
        print(f"Encode successful!")
        print(f"Embedding shape: {len(result['embeddings'])} sentences, {len(result['embeddings'][0])} dimensions")
        print(f"Sample of first embedding (first 5 dimensions): {result['embeddings'][0][:5]}")
    except requests.exceptions.RequestException as e:
        print(f"Encode failed: {e}")


def test_predict(base_url: str, sentences: List[str]) -> None:
    """
    Test the predict endpoint.
    
    Args:
        base_url: Base URL of the API
        sentences: List of sentences to predict
    """
    url = f"{base_url}/predict"
    print(f"Testing predict endpoint: {url}")
    
    # Test with class labels
    data = {
        "sentences": sentences,
        "return_probabilities": False
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        
        print(f"Predict (labels) successful!")
        print("Task A predictions:")
        for i, sentence in enumerate(sentences):
            print(f"  {sentence}: {result['task_a_predictions'][i]}")
        
        print("\nTask B predictions (first sentence):")
        tokens = sentences[0].split()
        for token, label in zip(tokens, result['task_b_predictions'][0][:len(tokens)]):
            print(f"  {token}: {label}")
    except requests.exceptions.RequestException as e:
        print(f"Predict (labels) failed: {e}")
    
    # Test with probabilities
    data = {
        "sentences": sentences,
        "return_probabilities": True
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        
        print(f"\nPredict (probabilities) successful!")
        print("Task A probabilities (first sentence):")
        print(f"  {result['task_a_predictions'][0]}")
    except requests.exceptions.RequestException as e:
        print(f"Predict (probabilities) failed: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test the API')
    
    parser.add_argument('--url', type=str, default='http://localhost:8000',
                        help='Base URL of the API')
    
    args = parser.parse_args()
    
    # Example sentences
    sentences = [
        "I love machine learning and natural language processing.",
        "Deep learning models are revolutionizing NLP applications.",
        "The weather is beautiful today."
    ]
    
    # Test endpoints
    test_health(args.url)
    print("\n" + "-"*50 + "\n")
    
    test_encode(args.url, sentences)
    print("\n" + "-"*50 + "\n")
    
    test_predict(args.url, sentences)


if __name__ == "__main__":
    main() 