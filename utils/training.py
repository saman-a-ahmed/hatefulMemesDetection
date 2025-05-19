# utils/training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import time
from .metrics import compute_metrics, MetricTracker

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, 
                metric_tracker=None, grad_clip=None):
    """
    Train model for one epoch
    
    Args:
        model (nn.Module): Model to train
        dataloader (DataLoader): Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        epoch (int): Current epoch
        metric_tracker (MetricTracker, optional): Metric tracker for logging
        grad_clip (float, optional): Gradient clipping value
        
    Returns:
        dict: Training metrics
    """
    model.train()
    
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    all_scores = []
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Train]")
    
    for batch in pbar:
        # Get batch data
        images = batch['image'].to(device)
        labels = batch['label'].to(device).float()
        
        # Text input depends on the model (BERT or LSTM)
        if isinstance(batch['text'], dict):  # BERT
            text = {
                'input_ids': batch['text']['input_ids'].to(device),
                'attention_mask': batch['text']['attention_mask'].to(device)
            }
        else:  # LSTM
            text = batch['text'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(text, images)
        outputs = outputs.squeeze()
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping if specified
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Update parameters
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item() * labels.size(0)
        
        # Convert outputs to predictions
        scores = torch.sigmoid(outputs).detach().cpu().numpy()
        predictions = (scores >= 0.5).astype(int)
        
        # Store for metrics calculation
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions)
        all_scores.extend(scores)
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    train_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_predictions, all_scores)
    metrics['loss'] = train_loss
    
    # Log metrics
    if metric_tracker:
        metric_tracker.update_metrics(metrics, 'train', epoch)
        metric_tracker.log_pr_curve(all_labels, all_scores, 'train', epoch)
    
    # Print metrics
    print(f"Train Loss: {train_loss:.4f}, "
          f"Precision: {metrics['precision']:.4f}, "
          f"Recall: {metrics['recall']:.4f}, "
          f"F1: {metrics['f1']:.4f}, "
          f"AUROC: {metrics.get('auroc', 0.0):.4f}")
    
    return metrics

def validate(model, dataloader, criterion, device, epoch, metric_tracker=None):
    """
    Validate model
    
    Args:
        model (nn.Module): Model to validate
        dataloader (DataLoader): Validation data loader
        criterion: Loss function
        device: Device to use
        epoch (int): Current epoch
        metric_tracker (MetricTracker, optional): Metric tracker for logging
        
    Returns:
        dict: Validation metrics
    """
    model.eval()
    
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    all_scores = []
    
    with torch.no_grad():
        # Progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Val]")
        
        for batch in pbar:
            # Get batch data
            images = batch['image'].to(device)
            labels = batch['label'].to(device).float()
            
            # Text input depends on the model (BERT or LSTM)
            if isinstance(batch['text'], dict):  # BERT
                text = {
                    'input_ids': batch['text']['input_ids'].to(device),
                    'attention_mask': batch['text']['attention_mask'].to(device)
                }
            else:  # LSTM
                text = batch['text'].to(device)
            
            # Forward pass
            outputs = model(text, images)
            outputs = outputs.squeeze()
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Update statistics
            running_loss += loss.item() * labels.size(0)
            
            # Convert outputs to predictions
            scores = torch.sigmoid(outputs).cpu().numpy()
            predictions = (scores >= 0.5).astype(int)
            
            # Store for metrics calculation
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions)
            all_scores.extend(scores)
    
    # Calculate metrics
    val_loss = running_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_predictions, all_scores)
    metrics['loss'] = val_loss
    
    # Log metrics
    if metric_tracker:
        metric_tracker.update_metrics(metrics, 'val', epoch)
        metric_tracker.log_pr_curve(all_labels, all_scores, 'val', epoch)
    
    # Print metrics
    print(f"Val Loss: {val_loss:.4f}, "
          f"Precision: {metrics['precision']:.4f}, "
          f"Recall: {metrics['recall']:.4f}, "
          f"F1: {metrics['f1']:.4f}, "
          f"AUROC: {metrics.get('auroc', 0.0):.4f}")
    
    return metrics

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler=None, device='cuda', num_epochs=10, save_dir='checkpoints',
                model_name='model', grad_clip=None, patience=5, log_dir='runs'):
    """
    Train and validate model
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to use
        num_epochs (int): Number of epochs
        save_dir (str): Directory to save model checkpoints
        model_name (str): Name for model checkpoints
        grad_clip (float, optional): Gradient clipping value
        patience (int): Early stopping patience
        log_dir (str): Directory for TensorBoard logs
        
    Returns:
        model: Trained model
        dict: Training history
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize metric tracker
    metric_tracker = MetricTracker(log_dir, model_name)
    
    # Move model to device
    model = model.to(device)
    
    # Initialize variables
    best_auroc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, 
                                    device, epoch, metric_tracker, grad_clip)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch, 
                             metric_tracker)
        
        # Update learning rate if scheduler is provided
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics.get('auroc', val_metrics['f1']))
            else:
                scheduler.step()
        
        # Check if this is the best model
        current_auroc = val_metrics.get('auroc', val_metrics['f1'])
        if current_auroc > best_auroc:
            best_auroc = current_auroc
            best_epoch = epoch
            patience_counter = 0
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'train_metrics': train_metrics
            }, os.path.join(save_dir, f"{model_name}_best.pt"))
        else:
            patience_counter += 1
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'train_metrics': train_metrics
            }, os.path.join(save_dir, f"{model_name}_epoch{epoch+1}.pt"))
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Print best result
    print(f"Best performance at epoch {best_epoch+1} with AUROC: {best_auroc:.4f}")
    
    # Close metric tracker
    metric_tracker.close()
    
    # Load best model
    checkpoint = torch.load(os.path.join(save_dir, f"{model_name}_best.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model