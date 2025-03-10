"""
Script to fine-tune the multi-task model on custom data.
"""
import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from models.sentence_encoder import SentenceEncoder
from models.multi_task_model import MultiTaskModel, TASK_A_CLASSES, TASK_B_TAGS


class CustomDataset(Dataset):
    """
    Custom dataset for multi-task learning.
    """
    
    def __init__(
        self,
        sentences: List[str],
        task_a_labels: List[int],
        task_b_labels: List[List[int]],
        tokenizer,
        max_length: int = 128
    ):
        """
        Initialize the dataset.
        
        Args:
            sentences: List of sentences
            task_a_labels: List of Task A labels (class indices)
            task_b_labels: List of Task B labels (token label indices)
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.sentences = sentences
        self.task_a_labels = task_a_labels
        self.task_b_labels = task_b_labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        """Get the number of samples."""
        return len(self.sentences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with input_ids, attention_mask, task_a_label, and task_b_labels
        """
        sentence = self.sentences[idx]
        task_a_label = self.task_a_labels[idx]
        task_b_label = self.task_b_labels[idx]
        
        # Tokenize the sentence
        encoding = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Add labels
        encoding["task_a_label"] = torch.tensor(task_a_label)
        
        # Align token labels with wordpieces
        tokens = sentence.split()
        aligned_labels = []
        
        # Get wordpiece tokens
        wordpiece_tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        
        # Initialize with special token label (ignored in loss)
        task_b_labels = [-100] * len(wordpiece_tokens)
        
        # Map original tokens to wordpieces
        token_idx = 0
        for i, wp_token in enumerate(wordpiece_tokens):
            # Skip special tokens
            if wp_token in ["[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"]:
                continue
                
            # If this is a continuation of a wordpiece, skip
            if wp_token.startswith("##") or wp_token.startswith("Ġ") or wp_token.startswith("▁"):
                continue
                
            # If we've processed all tokens, break
            if token_idx >= len(tokens):
                break
                
            # Assign the label
            if token_idx < len(task_b_label):
                task_b_labels[i] = task_b_label[token_idx]
                
            token_idx += 1
        
        encoding["task_b_labels"] = torch.tensor(task_b_labels)
        
        return encoding


def load_custom_data(
    data_path: str,
    task_a_classes: List[str] = TASK_A_CLASSES,
    task_b_tags: List[str] = TASK_B_TAGS
) -> Tuple[List[str], List[int], List[List[int]]]:
    """
    Load custom data from a JSON file.
    
    Args:
        data_path: Path to the JSON file
        task_a_classes: List of Task A classes
        task_b_tags: List of Task B tags
        
    Returns:
        Tuple containing:
        - sentences: List of sentences
        - task_a_labels: List of Task A labels (class indices)
        - task_b_labels: List of Task B labels (token label indices)
    """
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    sentences = []
    task_a_labels = []
    task_b_labels = []
    
    for item in data:
        sentences.append(item["sentence"])
        
        # Convert class name to index
        task_a_label = task_a_classes.index(item["task_a_label"])
        task_a_labels.append(task_a_label)
        
        # Convert token tags to indices
        tokens = item["sentence"].split()
        token_labels = item["task_b_labels"]
        
        # Ensure we have labels for all tokens
        if len(token_labels) < len(tokens):
            token_labels.extend(["O"] * (len(tokens) - len(token_labels)))
        elif len(token_labels) > len(tokens):
            token_labels = token_labels[:len(tokens)]
            
        # Convert to indices
        token_label_indices = [task_b_tags.index(label) for label in token_labels]
        task_b_labels.append(token_label_indices)
    
    return sentences, task_a_labels, task_b_labels


def create_example_data(output_path: str) -> None:
    """
    Create example data for fine-tuning.
    
    Args:
        output_path: Path to save the example data
    """
    print("Creating example data...")
    
    # Example sentences with their task A and task B labels
    data = [
        # Technology examples
        {
            "sentence": "I love machine learning and natural language processing.",
            "task_a_label": "Technology",
            "task_b_labels": ["O", "O", "B-TECH", "I-TECH", "O", "B-TECH", "I-TECH", "I-TECH", "O"]
        },
        {
            "sentence": "Python is my favorite programming language.",
            "task_a_label": "Technology",
            "task_b_labels": ["B-TECH", "O", "O", "O", "B-TECH", "I-TECH", "O"]
        },
        {
            "sentence": "Deep learning models are revolutionizing NLP applications.",
            "task_a_label": "Technology",
            "task_b_labels": ["B-TECH", "I-TECH", "O", "O", "O", "B-TECH", "I-TECH", "O"]
        },
        {
            "sentence": "Artificial intelligence is transforming healthcare.",
            "task_a_label": "Technology",
            "task_b_labels": ["B-TECH", "I-TECH", "O", "O", "O", "O"]
        },
        {
            "sentence": "JavaScript frameworks make web development easier.",
            "task_a_label": "Technology",
            "task_b_labels": ["B-TECH", "B-TECH", "O", "B-TECH", "I-TECH", "O", "O"]
        },
        {
            "sentence": "Cloud computing has revolutionized business operations.",
            "task_a_label": "Technology",
            "task_b_labels": ["B-TECH", "I-TECH", "O", "O", "O", "O", "O"]
        },
        {
            "sentence": "Quantum computing will solve complex problems faster.",
            "task_a_label": "Technology",
            "task_b_labels": ["B-TECH", "I-TECH", "O", "O", "O", "O", "O", "O"]
        },
        {
            "sentence": "Data science combines statistics and programming.",
            "task_a_label": "Technology",
            "task_b_labels": ["B-TECH", "I-TECH", "O", "O", "O", "B-TECH", "O"]
        },
        
        # Weather examples
        {
            "sentence": "The weather is beautiful today.",
            "task_a_label": "Weather",
            "task_b_labels": ["O", "B-WEATHER", "O", "O", "O", "O"]
        },
        {
            "sentence": "It's a sunny day with clear skies.",
            "task_a_label": "Weather",
            "task_b_labels": ["O", "O", "B-WEATHER", "I-WEATHER", "O", "B-WEATHER", "I-WEATHER", "O"]
        },
        {
            "sentence": "The forecast predicts rain for the weekend.",
            "task_a_label": "Weather",
            "task_b_labels": ["O", "B-WEATHER", "O", "B-WEATHER", "O", "O", "O", "O"]
        },
        {
            "sentence": "The humidity is very high today.",
            "task_a_label": "Weather",
            "task_b_labels": ["O", "B-WEATHER", "O", "O", "O", "O", "O"]
        },
        {
            "sentence": "A cold front is moving in from the north.",
            "task_a_label": "Weather",
            "task_b_labels": ["O", "B-WEATHER", "I-WEATHER", "O", "O", "O", "O", "O", "O", "O"]
        },
        {
            "sentence": "The temperature dropped significantly overnight.",
            "task_a_label": "Weather",
            "task_b_labels": ["O", "B-WEATHER", "O", "O", "O", "O"]
        },
        {
            "sentence": "The wind speed has increased to 30 mph.",
            "task_a_label": "Weather",
            "task_b_labels": ["O", "B-WEATHER", "I-WEATHER", "O", "O", "O", "O", "O", "O"]
        },
        {
            "sentence": "Today's temperature is 25 degrees Celsius.",
            "task_a_label": "Weather",
            "task_b_labels": ["O", "B-WEATHER", "O", "O", "O", "O", "O"]
        },
        {
            "sentence": "The barometric pressure is 1013 hPa.",
            "task_a_label": "Weather",
            "task_b_labels": ["O", "B-WEATHER", "I-WEATHER", "O", "O", "O", "O"]
        },
        {
            "sentence": "Wind gusts of up to 45 mph are expected.",
            "task_a_label": "Weather",
            "task_b_labels": ["B-WEATHER", "I-WEATHER", "O", "O", "O", "O", "O", "O", "O"]
        },
        
        # Other examples
        {
            "sentence": "I went to the store to buy some groceries.",
            "task_a_label": "Other",
            "task_b_labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O"]
        },
        {
            "sentence": "The movie we watched last night was amazing.",
            "task_a_label": "Other",
            "task_b_labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O"]
        }
    ]
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the data to a JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Example data saved to {output_path}")
    print(f"Created {len(data)} examples")
    print("Example data created. You can now fine-tune the model on this data.")


def train(
    model: MultiTaskModel,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    num_epochs: int = 5,
    learning_rate: float = 2e-5,
    task_a_weight: float = 1.0,
    task_b_weight: float = 1.0,
    device: Optional[torch.device] = None,
    output_dir: str = "outputs/models/fine_tuned"
) -> Dict[str, List[float]]:
    """
    Train the model.
    
    Args:
        model: The model to train
        train_dataloader: DataLoader for training data
        val_dataloader: Optional DataLoader for validation data
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate for the optimizer
        task_a_weight: Weight for Task A loss
        task_b_weight: Weight for Task B loss
        device: Device to train on
        output_dir: Directory to save the model
        
    Returns:
        Dictionary with training history
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Initialize history
    history = {
        "train_loss": [],
        "train_task_a_loss": [],
        "train_task_b_loss": [],
        "val_loss": [],
        "val_task_a_loss": [],
        "val_task_b_loss": []
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_task_a_loss = 0.0
        train_task_b_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Rename labels for compatibility with train_step
            batch["task_a_labels"] = batch.pop("task_a_label")
            
            # Train step
            loss_dict = model.train_step(
                batch=batch,
                optimizer=optimizer,
                task_a_weight=task_a_weight,
                task_b_weight=task_b_weight
            )
            
            # Update metrics
            train_loss += loss_dict["loss"]
            train_task_a_loss += loss_dict["task_a_loss"]
            train_task_b_loss += loss_dict["task_b_loss"]
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss_dict["loss"],
                "task_a_loss": loss_dict["task_a_loss"],
                "task_b_loss": loss_dict["task_b_loss"]
            })
        
        # Calculate average losses
        train_loss /= len(train_dataloader)
        train_task_a_loss /= len(train_dataloader)
        train_task_b_loss /= len(train_dataloader)
        
        # Update history
        history["train_loss"].append(train_loss)
        history["train_task_a_loss"].append(train_task_a_loss)
        history["train_task_b_loss"].append(train_task_b_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Task A Loss: {train_task_a_loss:.4f}, Task B Loss: {train_task_b_loss:.4f}")
        
        # Validation
        if val_dataloader is not None:
            model.eval()
            val_loss = 0.0
            val_task_a_loss = 0.0
            val_task_b_loss = 0.0
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    # Move batch to device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    # Rename labels for compatibility with forward
                    batch["task_a_labels"] = batch.pop("task_a_label")
                    
                    # Forward pass
                    outputs = model.forward(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch.get("token_type_ids", None)
                    )
                    
                    # Task A loss
                    task_a_loss = torch.nn.functional.cross_entropy(
                        outputs["task_a_logits"],
                        batch["task_a_labels"]
                    )
                    
                    # Task B loss
                    active_loss = batch["attention_mask"].view(-1) == 1
                    active_logits = outputs["task_b_logits"].view(-1, len(model.task_b_tags))
                    active_labels = batch["task_b_labels"].view(-1)
                    active_labels = torch.where(
                        active_loss,
                        active_labels,
                        torch.tensor(-100).to(device)
                    )
                    task_b_loss = torch.nn.functional.cross_entropy(
                        active_logits,
                        active_labels,
                        ignore_index=-100
                    )
                    
                    # Combined loss
                    loss = task_a_weight * task_a_loss + task_b_weight * task_b_loss
                    
                    # Update metrics
                    val_loss += loss.item()
                    val_task_a_loss += task_a_loss.item()
                    val_task_b_loss += task_b_loss.item()
            
            # Calculate average losses
            val_loss /= len(val_dataloader)
            val_task_a_loss /= len(val_dataloader)
            val_task_b_loss /= len(val_dataloader)
            
            # Update history
            history["val_loss"].append(val_loss)
            history["val_task_a_loss"].append(val_task_a_loss)
            history["val_task_b_loss"].append(val_task_b_loss)
            
            print(f"Val Loss: {val_loss:.4f}, Task A Loss: {val_task_a_loss:.4f}, Task B Loss: {val_task_b_loss:.4f}")
        
        # Save model
        model_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pt")
        model.save(model_path)
        print(f"Model saved to {model_path}")
    
    # Save final model
    model_path = os.path.join(output_dir, "model_final.pt")
    model.save(model_path)
    print(f"Final model saved to {model_path}")
    
    # Plot training history
    plot_history(history, output_dir)
    
    return history


def plot_history(history: Dict[str, List[float]], output_dir: str) -> None:
    """
    Plot training history.
    
    Args:
        history: Dictionary with training history
        output_dir: Directory to save the plots
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot total loss
    ax1.plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history and history["val_loss"]:
        ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Total Loss")
    ax1.legend()
    
    # Plot task-specific losses
    ax2.plot(history["train_task_a_loss"], label="Train Task A Loss")
    ax2.plot(history["train_task_b_loss"], label="Train Task B Loss")
    if "val_task_a_loss" in history and history["val_task_a_loss"]:
        ax2.plot(history["val_task_a_loss"], label="Val Task A Loss")
        ax2.plot(history["val_task_b_loss"], label="Val Task B Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Task-Specific Losses")
    ax2.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_history.png"))
    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fine-tune the multi-task model on custom data.")
    
    parser.add_argument("--data_path", type=str, default="outputs/data/example_data.json",
                        help="Path to the data file")
    parser.add_argument("--create_example_data", action="store_true",
                        help="Create example data for fine-tuning")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Name of the pre-trained model")
    parser.add_argument("--pooling_strategy", type=str, default="mean",
                        choices=["mean", "cls", "max"],
                        help="Pooling strategy for sentence embeddings")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate for the optimizer")
    parser.add_argument("--task_a_weight", type=float, default=1.0,
                        help="Weight for Task A loss")
    parser.add_argument("--task_b_weight", type=float, default=1.0,
                        help="Weight for Task B loss")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default="outputs/models/fine_tuned",
                        help="Directory to save the model")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of data to use for validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze the encoder (transformer backbone)")
    parser.add_argument("--freeze_task_a", action="store_true",
                        help="Freeze the Task A head (sentence classification)")
    parser.add_argument("--freeze_task_b", action="store_true",
                        help="Freeze the Task B head (token classification)")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to a pre-trained model to continue fine-tuning")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create example data if requested
    if args.create_example_data:
        create_example_data(args.data_path)
        return
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"Data file {args.data_path} does not exist.")
        print("You can create example data with --create_example_data")
        return
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    sentences, task_a_labels, task_b_labels = load_custom_data(args.data_path)
    print(f"Loaded {len(sentences)} samples")
    
    # Initialize encoder and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading pre-trained model from {args.load_model}...")
        model = MultiTaskModel.load(args.load_model, device=device)
        encoder = model.encoder
    else:
        print(f"Initializing model with {args.model_name} and {args.pooling_strategy} pooling...")
        encoder = SentenceEncoder(
            model_name=args.model_name,
            pooling_strategy=args.pooling_strategy,
            max_length=args.max_length
        )
        
        model = MultiTaskModel(
            encoder=encoder,
            task_a_hidden_dim=128,
            task_b_hidden_dim=128,
            max_length=args.max_length,
            device=device
        )
    
    # Apply freezing settings
    if args.freeze_encoder:
        print("Freezing encoder (transformer backbone)...")
        model.freeze_encoder(True)
    
    if args.freeze_task_a:
        print("Freezing Task A head (sentence classification)...")
        model.freeze_task_head('a', True)
    
    if args.freeze_task_b:
        print("Freezing Task B head (token classification)...")
        model.freeze_task_head('b', True)
    
    # Print freezing status
    print("\nFreezing Status:")
    print(f"  Encoder: {'Frozen' if args.freeze_encoder else 'Trainable'}")
    print(f"  Task A Head: {'Frozen' if args.freeze_task_a else 'Trainable'}")
    print(f"  Task B Head: {'Frozen' if args.freeze_task_b else 'Trainable'}")
    
    if args.freeze_encoder and args.freeze_task_a and args.freeze_task_b:
        print("\nWARNING: All components are frozen. No weights will be updated during training.")
        if not args.load_model:
            print("Consider unfreezing at least one component or loading a pre-trained model.")
    
    # Split data into train and validation sets
    if args.val_split > 0:
        # Shuffle indices
        indices = np.arange(len(sentences))
        np.random.shuffle(indices)
        
        # Split indices
        val_size = int(len(sentences) * args.val_split)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        # Create datasets
        train_sentences = [sentences[i] for i in train_indices]
        train_task_a_labels = [task_a_labels[i] for i in train_indices]
        train_task_b_labels = [task_b_labels[i] for i in train_indices]
        
        val_sentences = [sentences[i] for i in val_indices]
        val_task_a_labels = [task_a_labels[i] for i in val_indices]
        val_task_b_labels = [task_b_labels[i] for i in val_indices]
        
        print(f"Train set: {len(train_sentences)} samples")
        print(f"Validation set: {len(val_sentences)} samples")
        
        # Create datasets
        train_dataset = CustomDataset(
            sentences=train_sentences,
            task_a_labels=train_task_a_labels,
            task_b_labels=train_task_b_labels,
            tokenizer=encoder.tokenizer,
            max_length=args.max_length
        )
        
        val_dataset = CustomDataset(
            sentences=val_sentences,
            task_a_labels=val_task_a_labels,
            task_b_labels=val_task_b_labels,
            tokenizer=encoder.tokenizer,
            max_length=args.max_length
        )
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
    else:
        # Use all data for training
        train_dataset = CustomDataset(
            sentences=sentences,
            task_a_labels=task_a_labels,
            task_b_labels=task_b_labels,
            tokenizer=encoder.tokenizer,
            max_length=args.max_length
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )
        
        val_dataloader = None
    
    # Train the model
    print("Starting training...")
    history = train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        task_a_weight=args.task_a_weight,
        task_b_weight=args.task_b_weight,
        output_dir=args.output_dir,
        device=device
    )
    
    print("Training complete!")


if __name__ == "__main__":
    main() 