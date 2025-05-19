# create_visualizations.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer

def load_model(model_class, checkpoint_path, device='cpu'):
    """
    Load a model from checkpoint
    
    Args:
        model_class: Model class to instantiate
        checkpoint_path (str): Path to checkpoint file
        device (str): Device to load model on
        
    Returns:
        nn.Module: Loaded model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model (would need to match the exact initialization parameters)
    # For simplicity, assuming the model is passed as an already instantiated object
    model = model_class
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def visualize_attention(model, image_path, text, tokenizer, output_dir, device='cpu'):
    """
    Visualize attention weights (if the model supports it)
    
    Args:
        model: Model to use for prediction
        image_path (str): Path to image file
        text (str): Text to process
        tokenizer: Tokenizer for text
        output_dir (str): Directory to save visualizations
        device (str): Device to use
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Process text
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get model prediction
    with torch.no_grad():
        # This is a placeholder, would need to be adapted to the actual model
        # and if it supports attention visualization
        if hasattr(model, 'get_attention_weights'):
            attention_weights = model.get_attention_weights(
                {'input_ids': input_ids, 'attention_mask': attention_mask}, 
                image_tensor
            )
            
            # Visualize attention
            plt.figure(figsize=(12, 8))
            sns.heatmap(attention_weights.squeeze().cpu().numpy(), cmap='viridis')
            plt.title('Cross-Modal Attention Weights')
            plt.xlabel('Image Features')
            plt.ylabel('Text Features')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "attention_visualization.png"))
        else:
            print("Model does not support attention visualization.")

def visualize_model_predictions(model, image_paths, texts, tokenizer, output_dir, device='cpu'):
    """
    Visualize model predictions on sample images
    
    Args:
        model: Model to use for prediction
        image_paths (list): List of paths to image files
        texts (list): List of texts corresponding to images
        tokenizer: Tokenizer for text
        output_dir (str): Directory to save visualizations
        device (str): Device to use
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Process each sample
    for i, (image_path, text) in enumerate(zip(image_paths, texts)):
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Process text
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get model prediction
        with torch.no_grad():
            if hasattr(model, 'forward'):
                output = model({'input_ids': input_ids, 'attention_mask': attention_mask}, 
                               image_tensor)
                
                # Convert to probability
                prob = torch.sigmoid(output).item()
                
                # Create visualization
                plt.figure(figsize=(10, 6))
                plt.imshow(image)
                plt.title(f"Text: {text}\nPrediction: {'Hateful' if prob >= 0.5 else 'Non-hateful'} ({prob:.2f})")
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"prediction_{i}.png"))
            else:
                print("Model does not support the expected forward method.")

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Note: In a real project, you would load your model here
    # For this example, we'll just show the function signatures
    
    print("To use this script, you need to:")
    print("1. Load your trained model")
    print("2. Prepare sample images and texts")
    print("3. Call the visualization functions with your model")
    
    # Example usage (commented out since we don't have a real model):
    """
    # Load your model
    from models.fusion_models import EarlyFusionModel
    from models.text_models import BERTTextProcessor
    from models.image_models import ResNetImageProcessor
    
    # Create model instances
    text_model = BERTTextProcessor()
    image_model = ResNetImageProcessor()
    model = EarlyFusionModel(text_model, image_model, use_attention=True)
    
    # Load weights
    model = load_model(model, args.model_path, device)
    
    # Sample data
    image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg']
    texts = ['Sample text 1', 'Sample text 2']
    
    # Visualize predictions
    visualize_model_predictions(model, image_paths, texts, tokenizer, args.output_dir, device)
    
    # Visualize attention (if supported)
    visualize_attention(model, image_paths[0], texts[0], tokenizer, args.output_dir, device)
    """

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Create model visualizations")
    
    # Arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Run main function
    main(args)