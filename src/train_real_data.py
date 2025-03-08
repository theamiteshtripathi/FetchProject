"""
Training script for the multi-task learning model using real datasets.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from tqdm import tqdm

from models.multi_task_model import MultiTaskModel
from data.datasets import MultiTaskDataModule

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_epoch(
    model: MultiTaskModel,
    dataloader: torch.utils.data.DataLoader,
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
        # Move inputs to device
        task_a_input_ids = batch["task_a_input_ids"].to(device)
        task_a_attention_mask = batch["task_a_attention_mask"].to(device)
        task_a_labels = batch["task_a_labels"].to(device)
        
        task_b_input_ids = batch["task_b_input_ids"].to(device)
        task_b_attention_mask = batch["task_b_attention_mask"].to(device)
        task_b_labels = batch["task_b_labels"].to(device)
        
        # Forward pass for Task A
        outputs_a = model(
            input_ids=task_a_input_ids,
            attention_mask=task_a_attention_mask
        )
        
        # Forward pass for Task B
        outputs_b = model(
            input_ids=task_b_input_ids,
            attention_mask=task_b_attention_mask
        )
        
        # Task A loss (sentiment classification)
        task_a_loss = task_a_criterion(outputs_a["task_a"], task_a_labels)
        
        # Task B loss (NER)
        # Reshape predictions to [batch_size * seq_length, num_labels]
        task_b_logits = outputs_b["task_b"].view(-1, outputs_b["task_b"].size(-1))
        task_b_labels_flat = task_b_labels.view(-1)
        
        # Only consider non-padding tokens for loss
        active_mask = task_b_labels_flat != 0  # Assuming 0 is padding
        active_logits = task_b_logits[active_mask]
        active_labels = task_b_labels_flat[active_mask]
        
        task_b_loss = task_b_criterion(active_logits, active_labels)
        
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
        task_a_preds = outputs_a["task_a"].argmax(dim=1)
        task_a_correct += (task_a_preds == task_a_labels).sum().item()
        
        # Track Task B accuracy (token-level)
        task_b_preds = outputs_b["task_b"].argmax(dim=-1)
        active_mask_b = task_b_labels != 0
        task_b_correct += ((task_b_preds == task_b_labels) & active_mask_b).sum().item()
        task_b_total += active_mask_b.sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'task_a_loss': task_a_loss.item(),
            'task_b_loss': task_b_loss.item()
        })
    
    # Calculate average metrics
    num_batches = len(dataloader)
    num_examples = len(dataloader.dataset)
    
    avg_loss = total_loss / num_batches
    avg_task_a_loss = task_a_total_loss / num_batches
    avg_task_b_loss = task_b_total_loss / num_batches
    task_a_accuracy = task_a_correct / num_examples
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
    dataloader: torch.utils.data.DataLoader,
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
            # Move inputs to device
            task_a_input_ids = batch["task_a_input_ids"].to(device)
            task_a_attention_mask = batch["task_a_attention_mask"].to(device)
            task_a_labels = batch["task_a_labels"].to(device)
            
            task_b_input_ids = batch["task_b_input_ids"].to(device)
            task_b_attention_mask = batch["task_b_attention_mask"].to(device)
            task_b_labels = batch["task_b_labels"].to(device)
            
            # Forward pass for Task A
            outputs_a = model(
                input_ids=task_a_input_ids,
                attention_mask=task_a_attention_mask
            )
            
            # Forward pass for Task B
            outputs_b = model(
                input_ids=task_b_input_ids,
                attention_mask=task_b_attention_mask
            )
            
            # Task A loss (sentiment classification)
            task_a_loss = task_a_criterion(outputs_a["task_a"], task_a_labels)
            
            # Task B loss (NER)
            # Reshape predictions to [batch_size * seq_length, num_labels]
            task_b_logits = outputs_b["task_b"].view(-1, outputs_b["task_b"].size(-1))
            task_b_labels_flat = task_b_labels.view(-1)
            
            # Only consider non-padding tokens for loss
            active_mask = task_b_labels_flat != 0  # Assuming 0 is padding
            active_logits = task_b_logits[active_mask]
            active_labels = task_b_labels_flat[active_mask]
            
            task_b_loss = task_b_criterion(active_logits, active_labels)
            
            # Combine losses
            loss = task_a_loss + task_b_loss
            
            # Track metrics
            total_loss += loss.item()
            task_a_total_loss += task_a_loss.item()
            task_b_total_loss += task_b_loss.item()
            
            # Track Task A accuracy
            task_a_preds = outputs_a["task_a"].argmax(dim=1)
            task_a_correct += (task_a_preds == task_a_labels).sum().item()
            
            # Track Task B accuracy (token-level)
            task_b_preds = outputs_b["task_b"].argmax(dim=-1)
            active_mask_b = task_b_labels != 0
            task_b_correct += ((task_b_preds == task_b_labels) & active_mask_b).sum().item()
            task_b_total += active_mask_b.sum().item()
    
    # Calculate average metrics
    num_batches = len(dataloader)
    num_examples = len(dataloader.dataset)
    
    avg_loss = total_loss / num_batches
    avg_task_a_loss = task_a_total_loss / num_batches
    avg_task_b_loss = task_b_total_loss / num_batches
    task_a_accuracy = task_a_correct / num_examples
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
    parser = argparse.ArgumentParser(description='Train a multi-task model on real datasets')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                        help='Pre-trained model name for the encoder')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'cls', 'max'],
                        help='Pooling strategy for sentence embeddings')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--task_a_weight', type=float, default=1.0,
                        help='Weight for Task A loss')
    parser.add_argument('--task_b_weight', type=float, default=1.0,
                        help='Weight for Task B loss')
    parser.add_argument('--sample_size', type=int, default=1000,
                        help='Number of examples to use (for quick testing)')
    
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
    
    # For SST-2 (sentiment analysis), we have 2 classes
    task_a_num_classes = 2
    
    # For CoNLL-2003 (NER), we have 9 classes (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC)
    task_b_num_labels = 9
    
    model = MultiTaskModel(
        encoder_model_name=args.model_name,
        pooling_strategy=args.pooling,
        task_a_num_classes=task_a_num_classes,
        task_b_num_labels=task_b_num_labels,
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
    
    # Create data module
    logger.info("Loading datasets")
    data_module = MultiTaskDataModule(
        tokenizer=model.encoder.tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        sample_size=args.sample_size
    )
    data_module.setup()
    
    # Create data loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Define loss functions
    task_a_criterion = nn.CrossEntropyLoss()
    task_b_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
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
            model_path = os.path.join(args.output_dir, 'best_model_real_data.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'args': vars(args)
            }, model_path)
            logger.info(f"Model saved to {model_path}")
    
    logger.info("Training completed!")
    
    # Test the model
    logger.info("Evaluating on test set")
    test_loader = data_module.test_dataloader()
    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        task_a_criterion=task_a_criterion,
        task_b_criterion=task_b_criterion
    )
    
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}, "
               f"Task A Loss: {test_metrics['task_a_loss']:.4f}, "
               f"Task B Loss: {test_metrics['task_b_loss']:.4f}, "
               f"Task A Acc: {test_metrics['task_a_accuracy']:.4f}, "
               f"Task B Acc: {test_metrics['task_b_accuracy']:.4f}")


if __name__ == "__main__":
    main() 