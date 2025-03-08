"""
Dataset loading and preprocessing for the multi-task model.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from datasets import load_dataset
import numpy as np


class SentimentNERDataset(Dataset):
    """
    Dataset for multi-task learning with sentiment analysis and named entity recognition.
    
    This dataset combines SST-2 (sentiment analysis) and CoNLL-2003 (NER) datasets.
    """
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 128,
        split: str = "train",
        sample_size: Optional[int] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            tokenizer: Tokenizer to use for encoding sentences
            max_length: Maximum sequence length
            split: Dataset split (train, validation, test)
            sample_size: Number of examples to use (for quick testing)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load SST-2 dataset for sentiment analysis (Task A)
        self.sst2 = load_dataset("glue", "sst2", split=split)
        
        # Load CoNLL-2003 dataset for NER (Task B)
        self.conll = load_dataset("conll2003", split=split)
        
        # Take a sample if specified
        if sample_size is not None:
            self.sst2 = self.sst2.select(range(min(sample_size, len(self.sst2))))
            self.conll = self.conll.select(range(min(sample_size, len(self.conll))))
        
        # Map NER tags to indices
        self.ner_tags = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.ner_tags)}
        
        # For simplicity, we'll use the smaller dataset's size
        self.size = min(len(self.sst2), len(self.conll))
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Dict:
        # Get SST-2 example (Task A)
        sst2_example = self.sst2[idx % len(self.sst2)]
        task_a_sentence = sst2_example["sentence"]
        task_a_label = sst2_example["label"]  # 0: negative, 1: positive
        
        # Get CoNLL example (Task B)
        conll_example = self.conll[idx % len(self.conll)]
        task_b_tokens = conll_example["tokens"]
        task_b_sentence = " ".join(task_b_tokens)
        
        # Convert NER tags to indices
        task_b_tags = [self.tag2idx[self.ner_tags[tag]] for tag in conll_example["ner_tags"]]
        
        return {
            "task_a_sentence": task_a_sentence,
            "task_a_label": task_a_label,
            "task_b_sentence": task_b_sentence,
            "task_b_tokens": task_b_tokens,
            "task_b_tags": torch.tensor(task_b_tags)
        }


class MultiTaskDataModule:
    """
    Data module for multi-task learning.
    
    This class handles loading and preprocessing data for both tasks.
    """
    
    def __init__(
        self,
        tokenizer,
        batch_size: int = 16,
        max_length: int = 128,
        sample_size: Optional[int] = None
    ):
        """
        Initialize the data module.
        
        Args:
            tokenizer: Tokenizer to use for encoding sentences
            batch_size: Batch size for dataloaders
            max_length: Maximum sequence length
            sample_size: Number of examples to use (for quick testing)
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.sample_size = sample_size
    
    def setup(self):
        """Set up the datasets."""
        self.train_dataset = SentimentNERDataset(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            split="train",
            sample_size=self.sample_size
        )
        
        self.val_dataset = SentimentNERDataset(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            split="validation",
            sample_size=self.sample_size
        )
        
        self.test_dataset = SentimentNERDataset(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            split="test",
            sample_size=self.sample_size
        )
    
    def collate_fn(self, batch: List[Dict]) -> Dict:
        """
        Collate function for the DataLoader.
        
        Args:
            batch: List of examples from the dataset
            
        Returns:
            Batch dictionary with tokenized inputs and labels
        """
        # Task A (Sentiment Analysis)
        task_a_sentences = [item["task_a_sentence"] for item in batch]
        task_a_labels = torch.tensor([item["task_a_label"] for item in batch])
        
        # Task B (NER)
        task_b_sentences = [item["task_b_sentence"] for item in batch]
        task_b_tags = [item["task_b_tags"] for item in batch]
        
        # Tokenize Task A sentences
        task_a_encodings = self.tokenizer(
            task_a_sentences,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize Task B sentences
        task_b_encodings = self.tokenizer(
            task_b_sentences,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            is_split_into_words=False  # We're passing the full sentence, not tokens
        )
        
        # Prepare Task B labels (NER tags)
        # This is more complex because we need to align tokens with tags
        task_b_labels = torch.zeros(
            (len(batch), self.max_length), dtype=torch.long
        )
        
        for i, (encoding, tags) in enumerate(zip(task_b_encodings.encodings, task_b_tags)):
            # Convert token ids to tokens
            tokens = self.tokenizer.convert_ids_to_tokens(encoding.ids)
            
            # Map tags to tokens (simplified approach)
            tag_idx = 0
            for j, token in enumerate(tokens):
                if token in ["[CLS]", "[SEP]", "[PAD]"] or token.startswith("##"):
                    # Special tokens get tag 0 (O)
                    task_b_labels[i, j] = 0
                else:
                    if tag_idx < len(tags):
                        task_b_labels[i, j] = tags[tag_idx]
                        tag_idx += 1
                    else:
                        task_b_labels[i, j] = 0
        
        return {
            "task_a_input_ids": task_a_encodings.input_ids,
            "task_a_attention_mask": task_a_encodings.attention_mask,
            "task_a_labels": task_a_labels,
            "task_b_input_ids": task_b_encodings.input_ids,
            "task_b_attention_mask": task_b_encodings.attention_mask,
            "task_b_labels": task_b_labels
        }
    
    def train_dataloader(self) -> DataLoader:
        """Get train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )


# For quick testing with dummy data
class DummyMultiTaskDataset(Dataset):
    """
    A dummy dataset for multi-task learning with sentence classification and token classification.
    
    This is just for demonstration purposes when real datasets are not available.
    """
    
    def __init__(self, size: int = 100, max_length: int = 20):
        """
        Initialize the dummy dataset.
        
        Args:
            size: Number of examples to generate
            max_length: Maximum sequence length
        """
        self.size = size
        self.max_length = max_length
        
        # Generate dummy sentences
        self.sentences = [
            f"This is a dummy sentence number {i} for multi-task learning."
            for i in range(size)
        ]
        
        # Generate dummy labels for Task A (sentence classification)
        # Assume 3 classes: 0, 1, 2
        self.task_a_labels = torch.randint(0, 3, (size,))
        
        # Generate dummy labels for Task B (token classification)
        # Assume 5 classes: 0, 1, 2, 3, 4 (e.g., O, B-PER, I-PER, B-LOC, I-LOC)
        self.task_b_labels = [
            torch.randint(0, 5, (min(len(s.split()), max_length),))
            for s in self.sentences
        ]
        
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Dict:
        return {
            'sentence': self.sentences[idx],
            'task_a_label': self.task_a_labels[idx],
            'task_b_label': self.task_b_labels[idx]
        } 