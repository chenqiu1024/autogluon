"""
Evaluation script for jointly trained Conv-LoRA + Layer Selection model.

This script evaluates the model and analyzes the learned layer selection patterns.

Usage:
    python evaluate_joint_training.py \
        --task isic2017 \
        --ckpt_path joint_training_output \
        --output_dir evaluation_results
"""

import argparse
import os
import sys
from pathlib import Path
import json

import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "multimodal" / "src"))

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.models.sam_joint_training import SAMWithJointLayerSelection


def expand_path(df, dataset_dir):
    """Expand relative paths."""
    for col in ["image", "label"]:
        df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


def load_data(task_name, data_dir="datasets"):
    """Load test data."""
    dataset_dir = os.path.join(data_dir, task_name, task_name)
    test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, "test.csv")), dataset_dir)
    return test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='isic2017')
    parser.add_argument('--data_dir', type=str, default='datasets')
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='evaluation_joint')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Evaluating Jointly Trained Model")
    print(f"{'='*70}\n")
    
    # Load data
    print("Loading test data...")
    test_df = load_data(args.task, args.data_dir)
    print(f"Test set: {len(test_df)} images")
    
    # Load model
    print(f"\nLoading model from {args.ckpt_path}...")
    predictor = MultiModalPredictor.load(args.ckpt_path)
    model = predictor._learner._model
    
    # Check if it's a joint training model
    is_joint = isinstance(model, SAMWithJointLayerSelection)
    print(f"Model type: {'Joint Training' if is_joint else 'Standard'}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = predictor.evaluate(test_df, metrics=['iou', 'dice'])
    
    print(f"\nTest Results:")
    print(f"  IoU:  {metrics['iou']:.4f}")
    print(f"  DICE: {metrics['dice']:.4f}")
    
    # Analyze layer selection if joint model
    if is_joint:
        print("\nAnalyzing layer selection patterns...")
        
        # Extract layer masks for test set
        layer_masks_list = []
        model.eval()
        
        for i in range(len(test_df)):
            single_data = test_df.iloc[[i]]
            with torch.no_grad():
                _, layer_mask, _ = model.forward({'image': single_data})
                layer_masks_list.append(layer_mask[0].cpu())
        
        layer_masks = torch.stack(layer_masks_list)  # [N, 32]
        
        # Statistics
        layer_freq = layer_masks.float().mean(dim=0).numpy()
        num_active = layer_masks.sum(dim=1).float()
        
        stats = {
            'mean_active_layers': num_active.mean().item(),
            'std_active_layers': num_active.std().item(),
            'min_active_layers': num_active.min().item(),
            'max_active_layers': num_active.max().item(),
            'layer_activation_freq': layer_freq.tolist(),
        }
        
        print(f"\nLayer Selection Statistics:")
        print(f"  Mean active: {stats['mean_active_layers']:.1f}/32")
        print(f"  Std:  {stats['std_active_layers']:.1f}")
        print(f"  Range: [{stats['min_active_layers']:.0f}, {stats['max_active_layers']:.0f}]")
        
        print(f"\nLayer Activation Frequencies:")
        for i in range(0, 32, 4):
            freqs = [f"{layer_freq[j]:.3f}" for j in range(i, min(i+4, 32))]
            print(f"  Layers {i:2d}-{min(i+3,31):2d}: {', '.join(freqs)}")
        
        # Visualization
        plt.figure(figsize=(14, 6))
        
        # Subplot 1: Layer frequencies
        plt.subplot(1, 2, 1)
        plt.bar(range(32), layer_freq, color='steelblue', alpha=0.7)
        plt.axhline(y=0.5, color='r', linestyle='--', label='50% threshold')
        plt.xlabel('Layer Index')
        plt.ylabel('Activation Frequency')
        plt.title('Layer Activation Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # Subplot 2: Distribution of active layers
        plt.subplot(1, 2, 2)
        plt.hist(num_active.numpy(), bins=20, color='green', alpha=0.7, edgecolor='black')
        plt.axvline(x=stats['mean_active_layers'], color='r', linestyle='--', 
                   label=f"Mean: {stats['mean_active_layers']:.1f}")
        plt.xlabel('Number of Active Layers')
        plt.ylabel('Frequency')
        plt.title('Distribution of Active Layers per Image')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'layer_selection_analysis.png'), dpi=300)
        print(f"\nVisualization saved to {args.output_dir}/layer_selection_analysis.png")
        
        # Save statistics
        results = {
            'test_metrics': metrics,
            'layer_selection_stats': stats,
        }
    else:
        results = {'test_metrics': metrics}
    
    # Save results
    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}/evaluation_results.json")
    print(f"\n{'='*70}")
    print("Evaluation completed!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

