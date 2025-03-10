"""
Sentence Encoder module for generating sentence embeddings.
"""
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Union, Tuple, Optional, Any


class SentenceEncoder(nn.Module):
    """
    A sentence encoder that converts sentences into fixed-size embeddings.
    
    This class wraps a pre-trained transformer model and adds pooling to generate
    sentence-level embeddings from token-level representations.
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
        pooling_strategy: str = "mean",
        max_length: int = 128
    ):
        """
        Initialize the sentence encoder.
        
        Args:
            model_name: Name of the pre-trained model to use
            pooling_strategy: Strategy for pooling token embeddings ('mean', 'cls', or 'max')
            max_length: Maximum sequence length for tokenization
        """
        super(SentenceEncoder, self).__init__()
        
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.max_length = max_length
        
        # Load pre-trained model and tokenizer
        print(f"Loading sentence encoder with {model_name} and {pooling_strategy} pooling...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Set to evaluation mode by default
        self.model.eval()
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            input_ids: Token IDs of shape (batch_size, seq_length)
            attention_mask: Attention mask of shape (batch_size, seq_length)
            token_type_ids: Optional token type IDs of shape (batch_size, seq_length)
            
        Returns:
            Tuple containing:
            - sentence_embedding: Sentence embeddings of shape (batch_size, embedding_dim)
            - token_embeddings: Token embeddings of shape (batch_size, seq_length, embedding_dim)
        """
        # Get the model outputs
        model_kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None:
            model_kwargs['token_type_ids'] = token_type_ids
            
        outputs = self.model(**model_kwargs, return_dict=True)
        
        # Get the token embeddings
        token_embeddings = outputs.last_hidden_state
        
        # Apply pooling strategy
        if self.pooling_strategy == 'cls':
            # Use the CLS token embedding as the sentence embedding
            sentence_embedding = token_embeddings[:, 0]
        elif self.pooling_strategy == 'mean':
            # Mean pooling: average all token embeddings
            # Mask out padding tokens
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.sum(input_mask_expanded, 1)
            sentence_embedding = sum_embeddings / sum_mask
        elif self.pooling_strategy == 'max':
            # Max pooling: take the max value over the sequence length dimension
            # Mask out padding tokens
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            sentence_embedding = torch.max(token_embeddings, 1)[0]
        else:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")
        
        return sentence_embedding, token_embeddings
    
    def encode(
        self, 
        sentences: Union[str, List[str]], 
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = False,
        device: Optional[torch.device] = None
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Encode sentences to embeddings.
        
        Args:
            sentences: A single sentence or a list of sentences
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show a progress bar
            convert_to_numpy: Whether to convert the output to numpy array
            device: Device to use for encoding
            
        Returns:
            Tensor of sentence embeddings with shape (n_sentences, embedding_dim)
        """
        # Handle single sentence
        if isinstance(sentences, str):
            sentences = [sentences]
            
        # Set device
        if device is None:
            device = next(self.parameters()).device
            
        # Set model to evaluation mode
        self.eval()
        
        # Initialize embeddings
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(device)
            
            # Encode
            with torch.no_grad():
                sentence_embedding, _ = self.forward(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    token_type_ids=inputs.get('token_type_ids', None)
                )
                
            all_embeddings.append(sentence_embedding.detach())
            
        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Convert to numpy if requested
        if convert_to_numpy:
            all_embeddings = all_embeddings.cpu().numpy()
            
        return all_embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            Dimension of the embeddings
        """
        if hasattr(self.model.config, 'hidden_size'):
            return self.model.config.hidden_size
        else:
            # Default for MiniLM-L6
            return 384
    
    def to(self, device: torch.device) -> "SentenceEncoder":
        """
        Move the model to the specified device.
        
        Args:
            device: Device to move the model to
            
        Returns:
            Self for chaining
        """
        self.model = self.model.to(device)
        return super().to(device)
    
    def parameters(self):
        """
        Get the parameters of the model.
        
        Returns:
            Iterator over the parameters
        """
        return self.model.parameters()
    
    def save(self, path: str) -> None:
        """
        Save the model to the given path.
        
        Args:
            path: Path to save the model
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save additional information
        torch.save({
            "model_name": self.model_name,
            "pooling_strategy": self.pooling_strategy,
            "max_length": self.max_length
        }, f"{path}/encoder_config.pt")
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "SentenceEncoder":
        """
        Load a model from the given path.
        
        Args:
            path: Path to load the model from
            device: Device to load the model to
            
        Returns:
            Loaded model
        """
        # Load config
        config = torch.load(f"{path}/encoder_config.pt", map_location="cpu")
        
        # Create model
        model = cls(
            model_name=path,  # Use the path as the model name
            pooling_strategy=config["pooling_strategy"],
            max_length=config["max_length"]
        )
        
        # Move to device
        if device is not None:
            model = model.to(device)
            
        return model 