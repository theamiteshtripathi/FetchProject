"""
Multi-Task Learning model with a shared sentence encoder and task-specific heads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

from .sentence_encoder import SentenceEncoder

# Define task-specific classes and tags
TASK_A_CLASSES = ["Technology", "Weather", "Other"]
TASK_B_TAGS = ["O", "B-TECH", "I-TECH", "B-WEATHER", "I-WEATHER"]

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
            embeddings: Input embeddings of shape (batch_size, embedding_dim)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.classifier(embeddings)


class TokenClassificationHead(nn.Module):
    """
    Token classification head for token-level tasks like NER.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        num_labels: int, 
        hidden_dim: Optional[int] = None,
        dropout_prob: float = 0.1
    ):
        """
        Initialize the token classification head.
        
        Args:
            input_dim: Dimension of input token embeddings
            num_labels: Number of output labels
            hidden_dim: Optional hidden layer dimension
            dropout_prob: Dropout probability
        """
        super(TokenClassificationHead, self).__init__()
        
        layers = []
        
        if hidden_dim is not None:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            ])
            input_dim = hidden_dim
            
        layers.append(nn.Linear(input_dim, num_labels))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the token classification head.
        
        Args:
            token_embeddings: Input token embeddings of shape (batch_size, seq_len, embedding_dim)
            
        Returns:
            Logits of shape (batch_size, seq_len, num_labels)
        """
        return self.classifier(token_embeddings)


class MultiTaskModel(nn.Module):
    """
    Multi-task learning model with a shared transformer encoder and task-specific heads.
    """
    
    def __init__(
        self,
        encoder: Optional[SentenceEncoder] = None,
        encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        pooling_strategy: str = "mean",
        task_a_classes: List[str] = TASK_A_CLASSES,
        task_b_tags: List[str] = TASK_B_TAGS,
        task_a_hidden_dim: Optional[int] = 128,
        task_b_hidden_dim: Optional[int] = 128,
        max_length: int = 128,
        dropout_prob: float = 0.1,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the multi-task model.
        
        Args:
            encoder: Optional pre-initialized SentenceEncoder
            encoder_model_name: Name of the pre-trained model (used if encoder is None)
            pooling_strategy: Pooling strategy for sentence embeddings
            task_a_classes: List of class names for Task A
            task_b_tags: List of tag names for Task B
            task_a_hidden_dim: Hidden dimension for Task A classification head
            task_b_hidden_dim: Hidden dimension for Task B classification head
            max_length: Maximum sequence length
            dropout_prob: Dropout probability
            device: Device to use (cpu or cuda)
        """
        super(MultiTaskModel, self).__init__()
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize or use provided encoder
        if encoder is None:
            self.encoder = SentenceEncoder(
                model_name=encoder_model_name,
                pooling_strategy=pooling_strategy,
                max_length=max_length
            )
        else:
            self.encoder = encoder
            
        # Move encoder to device
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()  # Set to evaluation mode
        
        # Get embedding dimension from encoder
        self.embedding_dim = self.encoder.get_embedding_dimension()
        
        # Store class and tag names
        self.task_a_classes = task_a_classes
        self.task_b_tags = task_b_tags
        
        # Initialize task-specific heads
        self.task_a_head = ClassificationHead(
            input_dim=self.embedding_dim,
            num_classes=len(task_a_classes),
            hidden_dim=task_a_hidden_dim,
            dropout_prob=dropout_prob
        ).to(self.device)
        
        self.task_b_head = TokenClassificationHead(
            input_dim=self.embedding_dim,
            num_labels=len(task_b_tags),
            hidden_dim=task_b_hidden_dim,
            dropout_prob=dropout_prob
        ).to(self.device)
        
        # Set to evaluation mode
        self.eval()
        
        # Set default thresholds
        self.task_a_threshold = 0.5
        self.task_b_threshold = 0.5
        
        # Maximum sequence length
        self.max_length = max_length
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the multi-task model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            
        Returns:
            Dictionary with task-specific outputs
        """
        # Get embeddings from encoder
        outputs = self.encoder.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Get token embeddings
        token_embeddings = outputs.last_hidden_state
        
        # Get sentence embedding using the specified pooling strategy
        if self.encoder.pooling_strategy == "cls":
            sentence_embedding = token_embeddings[:, 0]  # CLS token
        elif self.encoder.pooling_strategy == "mean":
            # Mean pooling: mask out padding tokens
            sentence_embedding = torch.sum(token_embeddings * attention_mask.unsqueeze(-1), dim=1)
            sentence_embedding = sentence_embedding / torch.sum(attention_mask, dim=1, keepdim=True)
        elif self.encoder.pooling_strategy == "max":
            # Max pooling: mask out padding tokens
            token_embeddings[attention_mask == 0] = -1e9  # Set padding tokens to large negative value
            sentence_embedding = torch.max(token_embeddings, dim=1)[0]
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.encoder.pooling_strategy}")
        
        # Task A: Sentence Classification
        task_a_logits = self.task_a_head(sentence_embedding)
        
        # Task B: Token Classification
        task_b_logits = self.task_b_head(token_embeddings)
        
        return {
            "task_a_logits": task_a_logits,
            "task_b_logits": task_b_logits,
            "token_embeddings": token_embeddings,
            "sentence_embedding": sentence_embedding
        }
    
    def freeze_encoder(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze the encoder parameters.
        
        Args:
            freeze: Whether to freeze (True) or unfreeze (False) the encoder
        """
        for param in self.encoder.parameters():
            param.requires_grad = not freeze
    
    def freeze_task_head(self, task: str, freeze: bool = True) -> None:
        """
        Freeze or unfreeze a specific task head.
        
        Args:
            task: Task name ('a' or 'b')
            freeze: Whether to freeze (True) or unfreeze (False) the task head
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
        sentences: Union[str, List[str]]
    ) -> Tuple[List[str], List[np.ndarray], List[List[str]], List[List[np.ndarray]]]:
        """
        Make predictions for the given sentences.
        
        Args:
            sentences: Input sentence or list of sentences
            
        Returns:
            Tuple containing:
            - task_a_preds: List of Task A class predictions
            - task_a_probs: List of Task A class probabilities
            - task_b_preds: List of Task B tag predictions for each token
            - task_b_probs: List of Task B tag probabilities for each token
        """
        # Ensure we have a list of sentences
        if isinstance(sentences, str):
            sentences = [sentences]
        
        # Set model to evaluation mode
        self.eval()
        
        # Process sentences in batches to avoid OOM
        batch_size = 8
        task_a_preds = []
        task_a_probs = []
        task_b_preds = []
        task_b_probs = []
        
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            
            # Tokenize sentences
            encoded_input = self.encoder.tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=encoded_input["input_ids"],
                    attention_mask=encoded_input["attention_mask"],
                    token_type_ids=encoded_input.get("token_type_ids", None)
                )
            
            # Task A: Sentence Classification
            task_a_logits = outputs["task_a_logits"]
            task_a_probs_batch = F.softmax(task_a_logits, dim=1).cpu().numpy()
            task_a_preds_batch = [self.task_a_classes[idx] for idx in task_a_probs_batch.argmax(axis=1)]
            
            task_a_preds.extend(task_a_preds_batch)
            task_a_probs.extend(task_a_probs_batch)
            
            # Task B: Token Classification
            task_b_logits = outputs["task_b_logits"]
            task_b_probs_batch = F.softmax(task_b_logits, dim=2).cpu().numpy()
            
            # Process each sentence in the batch
            for j, sentence in enumerate(batch_sentences):
                tokens = sentence.split()
                attention_mask = encoded_input["attention_mask"][j].cpu().numpy()
                
                # Get token predictions, skipping special tokens
                token_probs = []
                token_preds = []
                
                # Get wordpiece tokens
                wordpiece_tokens = self.encoder.tokenizer.convert_ids_to_tokens(encoded_input["input_ids"][j])
                
                # Map wordpiece tokens back to original tokens
                token_idx = 0
                for k, wp_token in enumerate(wordpiece_tokens):
                    # Skip special tokens and padding
                    if attention_mask[k] == 0 or wp_token in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"]:
                        continue
                    
                    # If this is a continuation of a wordpiece, skip
                    if wp_token.startswith("##") or wp_token.startswith("Ġ") or wp_token.startswith("▁"):
                        continue
                    
                    # If we've processed all tokens, break
                    if token_idx >= len(tokens):
                        break
                    
                    # Get probabilities for this token
                    probs = task_b_probs_batch[j, k]
                    token_probs.append(probs)
                    
                    # Get prediction for this token
                    pred_idx = np.argmax(probs)
                    token_preds.append(self.task_b_tags[pred_idx])
                    
                    token_idx += 1
                
                # If we have more tokens than predictions, use "O" for the rest
                while len(token_preds) < len(tokens):
                    token_preds.append("O")
                    token_probs.append(np.zeros(len(self.task_b_tags)))
                    token_probs[-1][0] = 1.0  # Set probability of "O" to 1.0
                
                # If we have more predictions than tokens, truncate
                token_preds = token_preds[:len(tokens)]
                token_probs = token_probs[:len(tokens)]
                
                task_b_preds.append(token_preds)
                task_b_probs.append(token_probs)
        
        return task_a_preds, task_a_probs, task_b_preds, task_b_probs
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        task_a_weight: float = 1.0,
        task_b_weight: float = 1.0
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data
            optimizer: Optimizer
            task_a_weight: Weight for Task A loss
            task_b_weight: Weight for Task B loss
            
        Returns:
            Dictionary with loss values
        """
        self.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids", None)
        )
        
        # Task A loss
        task_a_loss = F.cross_entropy(
            outputs["task_a_logits"],
            batch["task_a_labels"]
        )
        
        # Task B loss (only for non-padding tokens)
        active_loss = batch["attention_mask"].view(-1) == 1
        active_logits = outputs["task_b_logits"].view(-1, len(self.task_b_tags))
        active_labels = batch["task_b_labels"].view(-1)
        active_labels = torch.where(
            active_loss,
            active_labels,
            torch.tensor(-100).to(self.device)
        )
        task_b_loss = F.cross_entropy(
            active_logits,
            active_labels,
            ignore_index=-100
        )
        
        # Combined loss
        loss = task_a_weight * task_a_loss + task_b_weight * task_b_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return {
            "loss": loss.item(),
            "task_a_loss": task_a_loss.item(),
            "task_b_loss": task_b_loss.item()
        }
    
    def save(self, path: str) -> None:
        """
        Save the model to the given path.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            "model_state_dict": self.state_dict(),
            "task_a_classes": self.task_a_classes,
            "task_b_tags": self.task_b_tags,
            "task_a_threshold": self.task_a_threshold,
            "task_b_threshold": self.task_b_threshold,
            "encoder_model_name": self.encoder.model_name,
            "pooling_strategy": self.encoder.pooling_strategy,
            "max_length": self.max_length
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "MultiTaskModel":
        """
        Load a model from the given path.
        
        Args:
            path: Path to load the model from
            device: Device to load the model to
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location="cpu")
        
        # Create model
        model = cls(
            encoder_model_name=checkpoint.get("encoder_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            pooling_strategy=checkpoint.get("pooling_strategy", "mean"),
            task_a_classes=checkpoint.get("task_a_classes", TASK_A_CLASSES),
            task_b_tags=checkpoint.get("task_b_tags", TASK_B_TAGS),
            max_length=checkpoint.get("max_length", 128),
            device=device
        )
        
        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Set thresholds
        model.task_a_threshold = checkpoint.get("task_a_threshold", 0.5)
        model.task_b_threshold = checkpoint.get("task_b_threshold", 0.5)
        
        return model 