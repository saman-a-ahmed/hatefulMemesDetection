# main.py
import os
import argparse
import multiprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from models.text_models import BERTTextProcessor, LSTMTextProcessor
from models.image_models import CNNImageProcessor, ResNetImageProcessor
from models.fusion_models import LateFusionModel, EarlyFusionModel
from utils.dataset import get_dataloaders
from utils.training import train_model
from utils.metrics import plot_confusion_matrix, plot_roc_curve, compute_metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import json

def train_and_evaluate_model(args, model_type, text_model_type, image_model_type, use_attention=False):
    """
    Train and evaluate a model
    
    Args:
        args: Command line arguments
        model_type (str): 'late' or 'early' fusion
        text_model_type (str): 'bert' or 'lstm'
        image_model_type (str): 'resnet' or 'cnn'
        use_attention (bool): Whether to use attention (only for early fusion)
        
    Returns:
        dict: Validation metrics
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create model name
    model_name = f"{model_type}_fusion_{text_model_type}_{image_model_type}"
    if model_type == 'early' and use_attention:
        model_name += "_attention"
    
    print(f"\n{'='*20} Training {model_name} {'='*20}\n")
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(
        jsonl_train=args.train_jsonl,
        jsonl_val=args.val_jsonl,
        img_dir=args.img_dir,
        batch_size=args.batch_size,
        text_processor=text_model_type,
        augment=not args.no_augment,
        num_workers=args.num_workers
    )
    
    # Create text model
    print(f"Creating {text_model_type} text model...")
    if text_model_type == 'bert':
        text_model = BERTTextProcessor(output_dim=None, dropout=args.dropout, 
                                      freeze_bert=args.freeze_text)
    else:  # LSTM
        # Simplified LSTM initialization (in a real project, you'd need to handle vocab etc.)
        text_model = LSTMTextProcessor(vocab_size=10000, embedding_dim=300, 
                                       hidden_dim=256, output_dim=None, 
                                       dropout=args.dropout)
    
    # Create image model
    print(f"Creating {image_model_type} image model...")
    if image_model_type == 'resnet':
        image_model = ResNetImageProcessor(output_dim=None, dropout=args.dropout, 
                                          pretrained=True, freeze_backbone=args.freeze_image)
    else:  # CNN
        image_model = CNNImageProcessor(output_dim=None, dropout=args.dropout)
    
    # Create fusion model
    print(f"Creating {model_type} fusion model...")
    if model_type == 'late':
        model = LateFusionModel(
            text_model=text_model,
            image_model=image_model,
            output_dim=1,
            dropout=args.dropout
        )
    else:  # Early fusion
        model = EarlyFusionModel(
            text_model=text_model,
            image_model=image_model,
            output_dim=1,
            hidden_dim=512,
            dropout=args.dropout,
            use_attention=use_attention
        )
    
    # Define loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3, verbose=True
    )
    
    # Create checkpoint directory for this model
    model_save_dir = os.path.join(args.save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Create figure directory for this model
    model_figure_dir = os.path.join(args.figure_dir, model_name)
    os.makedirs(model_figure_dir, exist_ok=True)
    
    # Start timer
    start_time = time.time()
    
    # Train model
    print("Training model...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        save_dir=model_save_dir,
        model_name=model_name,
        grad_clip=args.grad_clip,
        patience=args.patience,
        log_dir=args.log_dir
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    # Evaluate best model on validation set
    print("Evaluating best model...")
    model.eval()
    all_labels = []
    all_predictions = []
    all_scores = []
    
    with torch.no_grad():
        for batch in val_loader:
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
            
            # Convert outputs to predictions
            scores = torch.sigmoid(outputs).cpu().numpy()
            predictions = (scores >= 0.5).astype(int)
            
            # Store for metrics calculation
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions)
            all_scores.extend(scores)
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_predictions, all_scores)
    
    # Add model info and training time
    metrics['model'] = model_name
    metrics['text_model'] = text_model_type
    metrics['image_model'] = image_model_type
    metrics['fusion_type'] = model_type
    metrics['use_attention'] = use_attention if model_type == 'early' else False
    metrics['training_time'] = training_time
    
    # Print metrics
    print("\nValidation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        all_labels, all_predictions,
        save_path=os.path.join(model_figure_dir, f"{model_name}_cm.png")
    )
    
    # Plot ROC curve
    plot_roc_curve(
        all_labels, all_scores,
        save_path=os.path.join(model_figure_dir, f"{model_name}_roc.png")
    )
    
    # Save metrics to JSON
    with open(os.path.join(model_figure_dir, f"{model_name}_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model {model_name} training and evaluation complete!")
    
    return metrics

def main(args):
    # Create directories if they don't exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Train and evaluate all model combinations if specified
    all_metrics = []
    
    # Configure models to train
    models_to_train = []
    
    if args.all_models:
        # Train all model combinations
        text_models = ['bert', 'lstm']
        image_models = ['resnet', 'cnn']
        
        # Late fusion models
        for text_model in text_models:
            for image_model in image_models:
                models_to_train.append(('late', text_model, image_model, False))
        
        # Early fusion models
        for text_model in text_models:
            for image_model in image_models:
                models_to_train.append(('early', text_model, image_model, False))
                
                # With attention
                models_to_train.append(('early', text_model, image_model, True))
    else:
        # Train only the specified model
        models_to_train.append((args.fusion_type, args.text_model, args.image_model, args.use_attention))
    
    # Train each model
    for fusion_type, text_model, image_model, use_attention in models_to_train:
        metrics = train_and_evaluate_model(
            args, fusion_type, text_model, image_model, use_attention
        )
        all_metrics.append(metrics)
    
    # If multiple models were trained, create comparison
    if len(all_metrics) > 1:
        # Create DataFrame
        df_metrics = pd.DataFrame(all_metrics)
        
        # Print comparison table
        print("\nModel Comparison:")
        print(df_metrics[['model', 'auroc', 'precision', 'recall', 'f1', 'training_time']].to_string(index=False))
        
        # Save metrics to CSV
        df_metrics.to_csv(os.path.join(args.figure_dir, "model_comparison.csv"), index=False)
        
        # Create comparison figures
        metrics_to_plot = ['auroc', 'precision', 'recall', 'f1']
        
        # Bar chart of all metrics
        plt.figure(figsize=(12, 8))
        df_plot = df_metrics.melt(id_vars=['model'], value_vars=metrics_to_plot, 
                                  var_name='Metric', value_name='Value')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(args.figure_dir, "model_comparison_all.png"))
        
        # Save to JSON for easy loading
        with open(os.path.join(args.figure_dir, "model_comparison.json"), 'w') as f:
            json.dump(df_metrics.to_dict('records'), f, indent=2)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train models for hateful memes detection")
    
    # Data arguments
    parser.add_argument('--train_jsonl', type=str, required=True, help='Path to training JSONL file')
    parser.add_argument('--val_jsonl', type=str, required=True, help='Path to validation JSONL file')
    parser.add_argument('--img_dir', type=str, required=True, help='Path to image directory')
    
    # Model arguments
    parser.add_argument('--fusion_type', type=str, choices=['late', 'early'], default='late', 
                        help='Fusion type to use')
    parser.add_argument('--text_model', type=str, choices=['bert', 'lstm'], default='bert', 
                        help='Text model to use')
    parser.add_argument('--image_model', type=str, choices=['resnet', 'cnn'], default='resnet', 
                        help='Image model to use')
    parser.add_argument('--freeze_text', action='store_true', help='Freeze text model parameters')
    parser.add_argument('--freeze_image', action='store_true', help='Freeze image model parameters')
    parser.add_argument('--use_attention', action='store_true', 
                        help='Use cross-modal attention (only for early fusion)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--no_augment', action='store_true', help='Disable data augmentation')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker threads for data loading')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory for TensorBoard logs')
    parser.add_argument('--figure_dir', type=str, default='figures', help='Directory to save figures')
    
    # Other arguments
    parser.add_argument('--all_models', action='store_true', 
                        help='Train all model combinations (ignores specific model arguments)')
    
    args = parser.parse_args()
    
    # Handle Windows multiprocessing
    multiprocessing.freeze_support()
    
    # Run main function
    main(args)