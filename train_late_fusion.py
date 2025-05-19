# train_late_fusion.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import multiprocessing
from models.text_models import BERTTextProcessor, LSTMTextProcessor
from models.image_models import CNNImageProcessor, ResNetImageProcessor
from models.fusion_models import LateFusionModel
from utils.dataset import get_dataloaders
from utils.training import train_model
from utils.metrics import plot_confusion_matrix, plot_roc_curve
import matplotlib.pyplot as plt

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create model save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create figure save directory if it doesn't exist
    os.makedirs(args.figure_dir, exist_ok=True)
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(
        jsonl_train=args.train_jsonl,
        jsonl_val=args.val_jsonl,
        img_dir=args.img_dir,
        batch_size=args.batch_size,
        text_processor=args.text_model,
        augment=not args.no_augment,
        num_workers=args.num_workers
    )
    
    # Create text model
    print(f"Creating {args.text_model} text model...")
    if args.text_model == 'bert':
        text_model = BERTTextProcessor(output_dim=None, dropout=args.dropout, 
                                      freeze_bert=args.freeze_text)
    else:  # LSTM
        # Simplified LSTM initialization (in a real project, you'd need to handle vocab etc.)
        text_model = LSTMTextProcessor(vocab_size=10000, embedding_dim=300, 
                                       hidden_dim=256, output_dim=None, 
                                       dropout=args.dropout)
    
    # Create image model
    print(f"Creating {args.image_model} image model...")
    if args.image_model == 'resnet':
        image_model = ResNetImageProcessor(output_dim=None, dropout=args.dropout, 
                                          pretrained=True, freeze_backbone=args.freeze_image)
    else:  # CNN
        image_model = CNNImageProcessor(output_dim=None, dropout=args.dropout)
    
    # Create fusion model
    print("Creating late fusion model...")
    model = LateFusionModel(
        text_model=text_model,
        image_model=image_model,
        output_dim=1,
        dropout=args.dropout
    )
    
    # Define loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=3, verbose=True
    )
    
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
        save_dir=args.save_dir,
        model_name=f"late_fusion_{args.text_model}_{args.image_model}",
        grad_clip=args.grad_clip,
        patience=args.patience,
        log_dir=args.log_dir
    )
    
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
    
    # Plot confusion matrix
    plot_confusion_matrix(
        all_labels, all_predictions,
        save_path=os.path.join(args.figure_dir, f"late_fusion_cm.png")
    )
    
    # Plot ROC curve
    plot_roc_curve(
        all_labels, all_scores,
        save_path=os.path.join(args.figure_dir, f"late_fusion_roc.png")
    )
    
    print("Training and evaluation complete!")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train late fusion model for hateful memes detection")
    
    # Data arguments
    parser.add_argument('--train_jsonl', type=str, required=True, help='Path to training JSONL file')
    parser.add_argument('--val_jsonl', type=str, required=True, help='Path to validation JSONL file')
    parser.add_argument('--img_dir', type=str, required=True, help='Path to image directory')
    
    # Model arguments
    parser.add_argument('--text_model', type=str, choices=['bert', 'lstm'], default='bert', 
                        help='Text model to use')
    parser.add_argument('--image_model', type=str, choices=['resnet', 'cnn'], default='resnet', 
                        help='Image model to use')
    parser.add_argument('--freeze_text', action='store_true', help='Freeze text model parameters')
    parser.add_argument('--freeze_image', action='store_true', help='Freeze image model parameters')
    
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
    
    args = parser.parse_args()
    
    # Handle Windows multiprocessing
    multiprocessing.freeze_support()
    
    # Run main function
    main(args)