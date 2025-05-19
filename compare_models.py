# compare_models.py
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def load_metrics(checkpoint_path):
    """
    Load metrics from a checkpoint file
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        
    Returns:
        dict: Metrics from checkpoint
    """
    if not os.path.exists(checkpoint_path):
        return None
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint.get('val_metrics', None)

def compare_models(model_dirs, model_names, output_dir):
    """
    Compare model performance
    
    Args:
        model_dirs (list): List of model checkpoint directories
        model_names (list): List of model names
        output_dir (str): Directory to save comparison figures
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load best metrics for each model
    metrics = []
    for model_dir, model_name in zip(model_dirs, model_names):
        best_checkpoint = os.path.join(model_dir, f"{model_name}_best.pt")
        model_metrics = load_metrics(best_checkpoint)
        
        if model_metrics is not None:
            metrics.append({
                'model': model_name,
                **model_metrics
            })
    
    # Convert to DataFrame
    df_metrics = pd.DataFrame(metrics)
    
    # Skip if no metrics are found
    if len(df_metrics) == 0:
        print("No metrics found for the specified models!")
        return
    
    # Print comparison table
    print("Model Comparison:")
    print(df_metrics.to_string(index=False))
    
    # Save metrics to CSV
    df_metrics.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    
    # Create comparison figures
    metrics_to_plot = ['auroc', 'precision', 'recall', 'f1']
    
    # Bar chart of all metrics
    plt.figure(figsize=(12, 8))
    df_plot = df_metrics.melt(id_vars=['model'], value_vars=metrics_to_plot, 
                              var_name='Metric', value_name='Value')
    sns.barplot(x='model', y='Value', hue='Metric', data=df_plot)
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_all.png"))
    
    # Individual bar charts for each metric
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model', y=metric, data=df_metrics)
        plt.title(f'{metric.upper()} Comparison')
        plt.ylabel(metric.upper())
        plt.xlabel('Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"model_comparison_{metric}.png"))
    
    # Save to JSON for easy loading
    with open(os.path.join(output_dir, "model_comparison.json"), 'w') as f:
        json.dump(df_metrics.to_dict('records'), f, indent=2)
    
    print(f"Comparison results saved to {output_dir}")

def main(args):
    # Check if number of model_dirs and model_names match
    if len(args.model_dirs) != len(args.model_names):
        print("Error: Number of model directories and model names must match!")
        return
    
    # Compare models
    compare_models(args.model_dirs, args.model_names, args.output_dir)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Compare model performance")
    
    # Arguments
    parser.add_argument('--model_dirs', type=str, nargs='+', required=True,
                        help='List of model checkpoint directories')
    parser.add_argument('--model_names', type=str, nargs='+', required=True,
                        help='List of model names')
    parser.add_argument('--output_dir', type=str, default='comparison',
                        help='Directory to save comparison figures')
    
    args = parser.parse_args()
    
    # Run main function
    main(args)