"""
Sentence Encoder module for generating sentence embeddings.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Union, Tuple


class SentenceEncoder(nn.Module):
    """
    A sentence encoder that converts sentences into fixed-size embeddings.
    
    This class wraps a pre-trained transformer model and adds pooling to generate
    sentence-level embeddings from token-level representations.
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2", 
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        token_type_ids: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            input_ids: Tensor of token ids
            attention_mask: Tensor indicating which tokens to attend to
            token_type_ids: Optional tensor for token type ids (for BERT-like models)
            
        Returns:
            sentence_embedding: Pooled sentence embedding
            token_embeddings: Token-level embeddings (useful for token classification tasks)
        """
        # Prepare inputs
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        if token_type_ids is not None:
            model_inputs['token_type_ids'] = token_type_ids
            
        # Get transformer outputs
        outputs = self.transformer(**model_inputs)
        token_embeddings = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]
        
        # Apply pooling to get sentence embeddings
        if self.pooling_strategy == 'cls':
            # Use [CLS] token embedding as sentence representation
            sentence_embedding = token_embeddings[:, 0]
        elif self.pooling_strategy == 'max':
            # Max pooling over tokens
            sentence_embedding = torch.max(
                token_embeddings * attention_mask.unsqueeze(-1), 
                dim=1
            )[0]
        else:  # Default: mean pooling
            # Mean pooling: average all token embeddings
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * attention_mask_expanded, 1)
            sum_mask = torch.sum(attention_mask_expanded, 1)
            sentence_embedding = sum_embeddings / sum_mask
            
        return sentence_embedding, token_embeddings
    
    def encode(self, sentences: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode sentences to embeddings.
        
        Args:
            sentences: A single sentence or list of sentences to encode
            
        Returns:
            Tensor of sentence embeddings
        """
        # Handle single sentence
        if isinstance(sentences, str):
            sentences = [sentences]
            
        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to the same device as the model
        device = next(self.parameters()).device
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # Get embeddings
        with torch.no_grad():
            sentence_embedding, _ = self.forward(**encoded_input)
            
        return sentence_embedding 