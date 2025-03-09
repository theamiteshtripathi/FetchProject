"""
Multi-Task Learning model with a shared sentence encoder and task-specific heads.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union

from .sentence_encoder import SentenceEncoder


class ClassificationHead(nn.Module):
    """
    Classification head for sentence-level tasks.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        num_classes: int, 
        hidden_dim: Optional[int] = None,
        dropout_prob: float = 0.1
    ):
        """
        Initialize the classification head.
        
        Args:
            input_dim: Dimension of input embeddings
            num_classes: Number of output classes
            hidden_dim: Optional hidden layer dimension (if None, no hidden layer is used)
            dropout_prob: Dropout probability
        """
        super(ClassificationHead, self).__init__()
        
        layers = []
        
        if hidden_dim is not None:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            ])
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classification head.
        
        Args:
            embeddings: Sentence embeddings from the encoder
            
        Returns:
            Logits for each class
        """
        return self.classifier(embeddings)


class TokenClassificationHead(nn.Module):
    """
    Token classification head for sequence labeling tasks like NER.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        num_labels: int, 
        dropout_prob: float = 0.1
    ):
        """
        Initialize the token classification head.
        
        Args:
            input_dim: Dimension of token embeddings
            num_labels: Number of token labels
            dropout_prob: Dropout probability
        """
        super(TokenClassificationHead, self).__init__()
        
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(input_dim, num_labels)
        
    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the token classification head.
        
        Args:
            token_embeddings: Token-level embeddings from the encoder
            
        Returns:
            Logits for each token and label
        """
        token_embeddings = self.dropout(token_embeddings)
        return self.classifier(token_embeddings)


class MultiTaskModel(nn.Module):
    """
    Multi-task learning model with a shared sentence encoder and task-specific heads.
    """
    
    def __init__(
        self,
        encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        pooling_strategy: str = "mean",
        task_a_num_classes: int = 3,
        task_b_num_labels: int = 5,
        task_a_hidden_dim: Optional[int] = None,
        max_length: int = 128,
        dropout_prob: float = 0.1
    ):
        """
        Initialize the multi-task model.
        
        Args:
            encoder_model_name: Name of the pre-trained model to use for the encoder
            pooling_strategy: Strategy for pooling token embeddings
            task_a_num_classes: Number of classes for Task A (sentence classification)
            task_b_num_labels: Number of labels for Task B (token classification)
            task_a_hidden_dim: Optional hidden dimension for Task A head
            max_length: Maximum sequence length
            dropout_prob: Dropout probability
        """
        super(MultiTaskModel, self).__init__()
        
        # Initialize the shared sentence encoder
        self.encoder = SentenceEncoder(
            model_name=encoder_model_name,
            pooling_strategy=pooling_strategy,
            max_length=max_length
        )
        
        # Get embedding dimension from the encoder
        # This is a bit of a hack, but it works for Hugging Face models
        if hasattr(self.encoder.transformer.config, 'hidden_size'):
            embed_dim = self.encoder.transformer.config.hidden_size
        else:
            # Default for MiniLM-L6
            embed_dim = 384
            
        # Initialize task-specific heads
        self.task_a_head = ClassificationHead(
            input_dim=embed_dim,
            num_classes=task_a_num_classes,
            hidden_dim=task_a_hidden_dim,
            dropout_prob=dropout_prob
        )
        
        self.task_b_head = TokenClassificationHead(
            input_dim=embed_dim,
            num_labels=task_b_num_labels,
            dropout_prob=dropout_prob
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multi-task model.
        
        Args:
            input_ids: Tensor of token ids
            attention_mask: Tensor indicating which tokens to attend to
            token_type_ids: Optional tensor for token type ids
            
        Returns:
            Dictionary containing task outputs:
                - 'task_a': Logits for Task A (sentence classification)
                - 'task_b': Logits for Task B (token classification)
        """
        # Get embeddings from the encoder
        sentence_embedding, token_embeddings = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Task A: Sentence Classification
        task_a_output = self.task_a_head(sentence_embedding)
        
        # Task B: Token Classification
        task_b_output = self.task_b_head(token_embeddings)
        
        return {
            'task_a': task_a_output,
            'task_b': task_b_output
        }
    
    def freeze_encoder(self, freeze: bool = True):
        """
        Freeze or unfreeze the encoder parameters.
        
        Args:
            freeze: If True, freeze the encoder; if False, unfreeze
        """
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
            
    def freeze_task_head(self, task: str, freeze: bool = True):
        """
        Freeze or unfreeze a specific task head.
        
        Args:
            task: Task identifier ('a' or 'b')
            freeze: If True, freeze the head; if False, unfreeze
        """
        if task.lower() == 'a':
            for param in self.task_a_head.parameters():
                param.requires_grad = not freeze
        elif task.lower() == 'b':
            for param in self.task_b_head.parameters():
                param.requires_grad = not freeze
        else:
            raise ValueError(f"Unknown task: {task}. Expected 'a' or 'b'.")
            
    def predict(
        self,
        sentences: Union[str, List[str]],
        return_probabilities: bool = False
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """
        Make predictions for both tasks on the given sentences.
        
        Args:
            sentences: A single sentence or list of sentences
            return_probabilities: If True, return probabilities instead of class indices
            
        Returns:
            Dictionary containing predictions for both tasks
        """
        # Handle single sentence
        if isinstance(sentences, str):
            sentences = [sentences]
            
        # Tokenize sentences
        encoded_input = self.encoder.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.encoder.max_length,
            return_tensors='pt'
        )
        
        # Move to the same device as the model
        device = next(self.parameters()).device
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # Get predictions
        self.eval()
        with torch.no_grad():
            outputs = self.forward(**encoded_input)
            
            # Task A: Sentence Classification
            if return_probabilities:
                task_a_preds = torch.softmax(outputs['task_a'], dim=1)
            else:
                task_a_preds = torch.argmax(outputs['task_a'], dim=1)
                
            # Task B: Token Classification
            if return_probabilities:
                task_b_preds = torch.softmax(outputs['task_b'], dim=2)
            else:
                task_b_preds = torch.argmax(outputs['task_b'], dim=2)
                
        return {
            'task_a': task_a_preds.cpu().numpy(),
            'task_b': task_b_preds.cpu().numpy()
        } 