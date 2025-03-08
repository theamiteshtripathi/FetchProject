"""
Tests for the sentence encoder and multi-task model.
"""
import sys
import os
import unittest
import torch
import numpy as np
from typing import List, Dict

# Add the parent directory to the path so we can import the models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.sentence_encoder import SentenceEncoder
from src.models.multi_task_model import MultiTaskModel, ClassificationHead, TokenClassificationHead


class TestSentenceEncoder(unittest.TestCase):
    """Tests for the SentenceEncoder class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.encoder = SentenceEncoder(model_name="all-MiniLM-L6-v2", pooling_strategy="mean")
        self.test_sentences = [
            "This is a test sentence.",
            "Another test sentence with different words."
        ]
    
    def test_initialization(self):
        """Test that the encoder initializes correctly."""
        self.assertEqual(self.encoder.model_name, "all-MiniLM-L6-v2")
        self.assertEqual(self.encoder.pooling_strategy, "mean")
        self.assertEqual(self.encoder.max_length, 128)
        
    def test_encode_shape(self):
        """Test that the encoder produces embeddings of the expected shape."""
        embeddings = self.encoder.encode(self.test_sentences)
        
        # Check that we get one embedding per sentence
        self.assertEqual(embeddings.shape[0], len(self.test_sentences))
        
        # Check that the embedding dimension is as expected
        # For all-MiniLM-L6-v2, it should be 384
        self.assertEqual(embeddings.shape[1], 384)
        
    def test_encode_similarity(self):
        """Test that similar sentences have higher similarity than dissimilar ones."""
        similar_sentences = [
            "I love machine learning.",
            "Machine learning is my favorite subject."
        ]
        
        dissimilar_sentences = [
            "I love machine learning.",
            "The weather is nice today."
        ]
        
        # Encode similar sentences
        similar_embeddings = self.encoder.encode(similar_sentences)
        similar_embeddings_np = similar_embeddings.cpu().numpy()
        
        # Encode dissimilar sentences
        dissimilar_embeddings = self.encoder.encode(dissimilar_sentences)
        dissimilar_embeddings_np = dissimilar_embeddings.cpu().numpy()
        
        # Compute cosine similarities
        similar_similarity = np.dot(similar_embeddings_np[0], similar_embeddings_np[1]) / (
            np.linalg.norm(similar_embeddings_np[0]) * np.linalg.norm(similar_embeddings_np[1])
        )
        
        dissimilar_similarity = np.dot(dissimilar_embeddings_np[0], dissimilar_embeddings_np[1]) / (
            np.linalg.norm(dissimilar_embeddings_np[0]) * np.linalg.norm(dissimilar_embeddings_np[1])
        )
        
        # Similar sentences should have higher similarity
        self.assertGreater(similar_similarity, dissimilar_similarity)


class TestMultiTaskModel(unittest.TestCase):
    """Tests for the MultiTaskModel class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.model = MultiTaskModel(
            encoder_model_name="all-MiniLM-L6-v2",
            pooling_strategy="mean",
            task_a_num_classes=3,
            task_b_num_labels=5
        )
        self.test_sentences = [
            "This is a test sentence.",
            "Another test sentence with different words."
        ]
    
    def test_initialization(self):
        """Test that the model initializes correctly."""
        # Check that the encoder is initialized
        self.assertIsNotNone(self.model.encoder)
        
        # Check that the task heads are initialized
        self.assertIsNotNone(self.model.task_a_head)
        self.assertIsNotNone(self.model.task_b_head)
        
        # Check that the task heads have the correct output dimensions
        self.assertEqual(self.model.task_a_head.classifier[-1].out_features, 3)
        self.assertEqual(self.model.task_b_head.classifier.out_features, 5)
    
    def test_forward_shape(self):
        """Test that the model produces outputs of the expected shape."""
        # Tokenize sentences
        tokenized = self.model.encoder.tokenizer(
            self.test_sentences,
            padding=True,
            truncation=True,
            max_length=self.model.encoder.max_length,
            return_tensors='pt'
        )
        
        # Forward pass
        outputs = self.model(**tokenized)
        
        # Check Task A output shape
        self.assertEqual(outputs['task_a'].shape[0], len(self.test_sentences))
        self.assertEqual(outputs['task_a'].shape[1], 3)  # 3 classes
        
        # Check Task B output shape
        self.assertEqual(outputs['task_b'].shape[0], len(self.test_sentences))
        self.assertEqual(outputs['task_b'].shape[2], 5)  # 5 labels
    
    def test_predict(self):
        """Test the predict method."""
        predictions = self.model.predict(self.test_sentences)
        
        # Check that we get predictions for both tasks
        self.assertIn('task_a', predictions)
        self.assertIn('task_b', predictions)
        
        # Check that we get one prediction per sentence for Task A
        self.assertEqual(len(predictions['task_a']), len(self.test_sentences))
        
        # Check that we get one sequence of predictions per sentence for Task B
        self.assertEqual(len(predictions['task_b']), len(self.test_sentences))
    
    def test_freezing(self):
        """Test the freezing functionality."""
        # Initially, all parameters should be trainable
        for param in self.model.parameters():
            self.assertTrue(param.requires_grad)
        
        # Freeze encoder
        self.model.freeze_encoder(freeze=True)
        
        # Check that encoder parameters are frozen
        for param in self.model.encoder.parameters():
            self.assertFalse(param.requires_grad)
        
        # Check that task head parameters are still trainable
        for param in self.model.task_a_head.parameters():
            self.assertTrue(param.requires_grad)
        
        for param in self.model.task_b_head.parameters():
            self.assertTrue(param.requires_grad)
        
        # Freeze Task A head
        self.model.freeze_task_head(task='a', freeze=True)
        
        # Check that Task A head parameters are frozen
        for param in self.model.task_a_head.parameters():
            self.assertFalse(param.requires_grad)
        
        # Check that Task B head parameters are still trainable
        for param in self.model.task_b_head.parameters():
            self.assertTrue(param.requires_grad)
        
        # Unfreeze everything
        self.model.freeze_encoder(freeze=False)
        self.model.freeze_task_head(task='a', freeze=False)
        
        # Check that all parameters are trainable again
        for param in self.model.parameters():
            self.assertTrue(param.requires_grad)


if __name__ == "__main__":
    unittest.main() 