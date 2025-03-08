"""
Training script for the multi-task learning model.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from tqdm import tqdm

from models.multi_task_model import MultiTaskModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DummyMultiTaskDataset(Dataset):
    """
    A dummy dataset for multi-task learning with sentence classification and token classification.
    
    This is just for demonstration purposes. In a real scenario, you would load actual data.
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
        
        # Generate dummy sentences (just for demonstration)
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


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for the DataLoader.
    
    Args:
        batch: List of examples from the dataset
        
    Returns:
        Batch dictionary with sentences and labels
    """
    sentences = [item['sentence'] for item in batch]
    task_a_labels = torch.stack([item['task_a_label'] for item in batch])
    
    # For task B, we need to pad the sequences
    task_b_labels = [item['task_b_label'] for item in batch]
    
    return {
        'sentences': sentences,
        'task_a_labels': task_a_labels,
        'task_b_labels': task_b_labels
    }


def train_epoch(
    model: MultiTaskModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_a_criterion: nn.Module,
    task_b_criterion: nn.Module,
    task_a_weight: float = 1.0,
    task_b_weight: float = 1.0
) -> Dict[str, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: The multi-task model
        dataloader: DataLoader for training data
        optimizer: Optimizer for parameter updates
        device: Device to train on (CPU or GPU)
        task_a_criterion: Loss function for Task A
        task_b_criterion: Loss function for Task B
        task_a_weight: Weight for Task A loss
        task_b_weight: Weight for Task B loss
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    task_a_total_loss = 0.0
    task_b_total_loss = 0.0
    task_a_correct = 0
    task_b_correct = 0
    task_b_total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Get batch data
        sentences = batch['sentences']
        task_a_labels = batch['task_a_labels'].to(device)
        task_b_labels_list = batch['task_b_labels']
        
        # Tokenize sentences
        tokenized = model.encoder.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=model.encoder.max_length,
            return_tensors='pt'
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in tokenized.items()}
        
        # Forward pass
        outputs = model(**inputs)
        
        # Task A loss (sentence classification)
        task_a_loss = task_a_criterion(outputs['task_a'], task_a_labels)
        
        # Task B loss (token classification)
        # This is more complex because we need to handle variable sequence lengths
        task_b_loss = 0.0
        batch_size = len(task_b_labels_list)
        
        for i in range(batch_size):
            # Get the actual length of this sequence
            seq_len = min(len(task_b_labels_list[i]), inputs['input_ids'].size(1))
            
            # Get predictions for this sequence
            pred = outputs['task_b'][i, :seq_len, :]
            
            # Get labels for this sequence (padded to match pred)
            labels = task_b_labels_list[i][:seq_len].to(device)
            
            # Compute loss for this sequence
            task_b_loss += task_b_criterion(pred, labels)
            
            # Track accuracy
            task_b_correct += (pred.argmax(dim=1) == labels).sum().item()
            task_b_total += seq_len
        
        # Average the task B loss over the batch
        task_b_loss /= batch_size
        
        # Combine losses with weights
        loss = task_a_weight * task_a_loss + task_b_weight * task_b_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        task_a_total_loss += task_a_loss.item()
        task_b_total_loss += task_b_loss.item()
        
        # Track Task A accuracy
        task_a_correct += (outputs['task_a'].argmax(dim=1) == task_a_labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'task_a_loss': task_a_loss.item(),
            'task_b_loss': task_b_loss.item()
        })
    
    # Calculate average metrics
    avg_loss = total_loss / len(dataloader)
    avg_task_a_loss = task_a_total_loss / len(dataloader)
    avg_task_b_loss = task_b_total_loss / len(dataloader)
    task_a_accuracy = task_a_correct / len(dataloader.dataset)
    task_b_accuracy = task_b_correct / task_b_total if task_b_total > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'task_a_loss': avg_task_a_loss,
        'task_b_loss': avg_task_b_loss,
        'task_a_accuracy': task_a_accuracy,
        'task_b_accuracy': task_b_accuracy
    }


def evaluate(
    model: MultiTaskModel,
    dataloader: DataLoader,
    device: torch.device,
    task_a_criterion: nn.Module,
    task_b_criterion: nn.Module
) -> Dict[str, float]:
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The multi-task model
        dataloader: DataLoader for evaluation data
        device: Device to evaluate on (CPU or GPU)
        task_a_criterion: Loss function for Task A
        task_b_criterion: Loss function for Task B
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    task_a_total_loss = 0.0
    task_b_total_loss = 0.0
    task_a_correct = 0
    task_b_correct = 0
    task_b_total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Get batch data
            sentences = batch['sentences']
            task_a_labels = batch['task_a_labels'].to(device)
            task_b_labels_list = batch['task_b_labels']
            
            # Tokenize sentences
            tokenized = model.encoder.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=model.encoder.max_length,
                return_tensors='pt'
            )
            
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in tokenized.items()}
            
            # Forward pass
            outputs = model(**inputs)
            
            # Task A loss (sentence classification)
            task_a_loss = task_a_criterion(outputs['task_a'], task_a_labels)
            
            # Task B loss (token classification)
            task_b_loss = 0.0
            batch_size = len(task_b_labels_list)
            
            for i in range(batch_size):
                # Get the actual length of this sequence
                seq_len = min(len(task_b_labels_list[i]), inputs['input_ids'].size(1))
                
                # Get predictions for this sequence
                pred = outputs['task_b'][i, :seq_len, :]
                
                # Get labels for this sequence
                labels = task_b_labels_list[i][:seq_len].to(device)
                
                # Compute loss for this sequence
                task_b_loss += task_b_criterion(pred, labels)
                
                # Track accuracy
                task_b_correct += (pred.argmax(dim=1) == labels).sum().item()
                task_b_total += seq_len
            
            # Average the task B loss over the batch
            task_b_loss /= batch_size
            
            # Combine losses
            loss = task_a_loss + task_b_loss
            
            # Track metrics
            total_loss += loss.item()
            task_a_total_loss += task_a_loss.item()
            task_b_total_loss += task_b_loss.item()
            
            # Track Task A accuracy
            task_a_correct += (outputs['task_a'].argmax(dim=1) == task_a_labels).sum().item()
    
    # Calculate average metrics
    avg_loss = total_loss / len(dataloader)
    avg_task_a_loss = task_a_total_loss / len(dataloader)
    avg_task_b_loss = task_b_total_loss / len(dataloader)
    task_a_accuracy = task_a_correct / len(dataloader.dataset)
    task_b_accuracy = task_b_correct / task_b_total if task_b_total > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'task_a_loss': avg_task_a_loss,
        'task_b_loss': avg_task_b_loss,
        'task_a_accuracy': task_a_accuracy,
        'task_b_accuracy': task_b_accuracy
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train a multi-task model')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2',
                        help='Pre-trained model name for the encoder')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'cls', 'max'],
                        help='Pooling strategy for sentence embeddings')
    parser.add_argument('--task_a_classes', type=int, default=3,
                        help='Number of classes for Task A')
    parser.add_argument('--task_b_labels', type=int, default=5,
                        help='Number of labels for Task B')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--task_a_weight', type=float, default=1.0,
                        help='Weight for Task A loss')
    parser.add_argument('--task_b_weight', type=float, default=1.0,
                        help='Weight for Task B loss')
    
    # Freezing options
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze the encoder parameters')
    parser.add_argument('--freeze_task_a', action='store_true',
                        help='Freeze Task A head parameters')
    parser.add_argument('--freeze_task_b', action='store_true',
                        help='Freeze Task B head parameters')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Directory to save model and results')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info(f"Creating model with {args.model_name} encoder")
    model = MultiTaskModel(
        encoder_model_name=args.model_name,
        pooling_strategy=args.pooling,
        task_a_num_classes=args.task_a_classes,
        task_b_num_labels=args.task_b_labels,
        max_length=args.max_length
    )
    
    # Apply freezing if specified
    if args.freeze_encoder:
        logger.info("Freezing encoder parameters")
        model.freeze_encoder(freeze=True)
    
    if args.freeze_task_a:
        logger.info("Freezing Task A head parameters")
        model.freeze_task_head(task='a', freeze=True)
    
    if args.freeze_task_b:
        logger.info("Freezing Task B head parameters")
        model.freeze_task_head(task='b', freeze=True)
    
    # Move model to device
    model = model.to(device)
    
    # Create dummy dataset (in a real scenario, you would load actual data)
    logger.info("Creating dummy dataset")
    dataset = DummyMultiTaskDataset(size=500, max_length=args.max_length)
    
    # Split dataset into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Define loss functions
    task_a_criterion = nn.CrossEntropyLoss()
    task_b_criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Train the model
    logger.info(f"Starting training for {args.epochs} epochs")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            task_a_criterion=task_a_criterion,
            task_b_criterion=task_b_criterion,
            task_a_weight=args.task_a_weight,
            task_b_weight=args.task_b_weight
        )
        
        # Evaluate
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            task_a_criterion=task_a_criterion,
            task_b_criterion=task_b_criterion
        )
        
        # Log metrics
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                   f"Task A Loss: {train_metrics['task_a_loss']:.4f}, "
                   f"Task B Loss: {train_metrics['task_b_loss']:.4f}, "
                   f"Task A Acc: {train_metrics['task_a_accuracy']:.4f}, "
                   f"Task B Acc: {train_metrics['task_b_accuracy']:.4f}")
        
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                   f"Task A Loss: {val_metrics['task_a_loss']:.4f}, "
                   f"Task B Loss: {val_metrics['task_b_loss']:.4f}, "
                   f"Task A Acc: {val_metrics['task_a_accuracy']:.4f}, "
                   f"Task B Acc: {val_metrics['task_b_accuracy']:.4f}")
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            logger.info(f"New best model with validation loss: {best_val_loss:.4f}")
            
            # Save model
            model_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'args': vars(args)
            }, model_path)
            logger.info(f"Model saved to {model_path}")
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main() 