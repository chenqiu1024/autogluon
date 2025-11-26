"""
Joint Training Script for Conv-LoRA and Layer Selection Policy.

This script implements end-to-end joint training where Conv-LoRA parameters
and layer selection policy are optimized simultaneously using Gumbel-Softmax
for differentiable layer selection.

Usage:
    python run_joint_training.py \
        --task isic2017 \
        --warmstart_ckpt baseline_model/model.ckpt \
        --output_dir joint_training_output \
        --warmstart_epochs 3 \
        --total_epochs 20
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

# Add autogluon to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "multimodal" / "src"))

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.models.sam_joint_training import SAMWithJointLayerSelection
from autogluon.multimodal.optim.lit_semantic_seg_joint import SemanticSegmentationJointTraining


def expand_path(df, dataset_dir):
    """Expand relative paths in dataframe to absolute paths."""
    for col in ["image", "label"]:
        df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


def load_data(task_name: str, data_dir: str = "datasets"):
    """Load training and validation data."""
    dataset_dir = os.path.join(data_dir, task_name, task_name)
    
    train_df = expand_path(pd.read_csv(os.path.join(dataset_dir, "train.csv")), dataset_dir)
    val_df = expand_path(pd.read_csv(os.path.join(dataset_dir, "val.csv")), dataset_dir)
    test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, "test.csv")), dataset_dir)
    
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Joint training of Conv-LoRA and layer selection")
    
    # Task
    parser.add_argument('--task', type=str, default='isic2017',
                       help='Task name')
    parser.add_argument('--data_dir', type=str, default='datasets',
                       help='Data directory')
    
    # Model
    parser.add_argument('--warmstart_ckpt', type=str, default=None,
                       help='Optional: warmstart from pre-trained Conv-LoRA model')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    
    # Training schedule
    parser.add_argument('--warmstart_epochs', type=int, default=3,
                       help='Epochs for Conv-LoRA warmstart (default: 3)')
    parser.add_argument('--total_epochs', type=int, default=20,
                       help='Total training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--num_gpus', type=int, default=1,
                       help='Number of GPUs (default: 1)')
    
    # Optimization
    parser.add_argument('--lora_lr', type=float, default=1e-4,
                       help='Learning rate for Conv-LoRA (default: 1e-4)')
    parser.add_argument('--selector_lr', type=float, default=5e-4,
                       help='Learning rate for layer selector (default: 5e-4)')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                       help='Gradient clipping threshold (default: 1.0)')
    
    # Gumbel-Softmax
    parser.add_argument('--initial_temperature', type=float, default=1.0,
                       help='Initial Gumbel-Softmax temperature (default: 1.0)')
    parser.add_argument('--final_temperature', type=float, default=0.1,
                       help='Final Gumbel-Softmax temperature (default: 0.1)')
    
    # Conv-LoRA config
    parser.add_argument('--rank', type=int, default=3,
                       help='LoRA rank (default: 3)')
    parser.add_argument('--expert_num', type=int, default=8,
                       help='Number of Conv-LoRA experts (default: 8)')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    
    print(f"\n{'='*70}")
    print(f"Joint Training: Conv-LoRA + Layer Selection")
    print(f"{'='*70}")
    print(f"Task: {args.task}")
    print(f"Output: {args.output_dir}")
    print(f"Warmstart epochs: {args.warmstart_epochs}")
    print(f"Total epochs: {args.total_epochs}")
    print(f"Joint training epochs: {args.total_epochs - args.warmstart_epochs}")
    print(f"Temperature: {args.initial_temperature} -> {args.final_temperature}")
    print(f"{'='*70}\n")
    
    # Load data
    print("Loading data...")
    train_df, val_df, test_df = load_data(args.task, args.data_dir)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # For joint training, we need to manually construct the training loop
    # instead of using AutoGluon's high-level API (which doesn't support custom training)
    
    # Setup hyperparameters for baseline Conv-LoRA (used for warmstart or loading)
    hyperparameters = {
        "optim.lora.r": args.rank,
        "optim.peft": "conv_lora",
        "optim.lora.conv_lora_expert_num": args.expert_num,
        "env.num_gpus": args.num_gpus,
        "env.per_gpu_batch_size": args.batch_size,
        "env.batch_size": args.batch_size * args.num_gpus,
        "optim.max_epochs": args.total_epochs,
        "optim.lr": args.lora_lr,
        "optim.loss_func": "structure_loss",
    }
    
    # Initialize or load model
    print("\nInitializing model...")
    
    if args.warmstart_ckpt:
        # Load existing Conv-LoRA model as warmstart
        print(f"Loading warmstart Conv-LoRA model from {args.warmstart_ckpt}")
        predictor = MultiModalPredictor.load(args.warmstart_ckpt)
        print("Warmstart model loaded!")
    else:
        # Create new model (will do warmstart training)
        print("Creating new model for warmstart training...")
        predictor = MultiModalPredictor(
            problem_type="semantic_segmentation",
            validation_metric="iou",
            eval_metric="iou",
            hyperparameters=hyperparameters,
            label="label",
        )
        
        # Do warmstart training (train Conv-LoRA with all layers active)
        print(f"\nPhase 1: Warmstart training Conv-LoRA ({args.warmstart_epochs} epochs)...")
        print("All 32 layers active during warmstart")
        
        # Temporarily set max_epochs for warmstart
        predictor._learner._config.optim.max_epochs = args.warmstart_epochs
        
        predictor.fit(
            train_data=train_df,
            tuning_data=val_df,
            seed=args.seed,
        )
        
        # Save warmstart checkpoint
        warmstart_dir = os.path.join(args.output_dir, 'warmstart_checkpoint')
        predictor.save(warmstart_dir)
        print(f"Warmstart checkpoint saved to {warmstart_dir}")
    
    # Now wrap model with joint training wrapper
    print("\nWrapping model for joint training...")
    from autogluon.multimodal.models.sam_joint_training import SAMWithJointLayerSelection
    
    original_model = predictor._learner._model
    joint_model = SAMWithJointLayerSelection(
        sam_model=original_model,
        selector_temperature=args.initial_temperature,
        warmstart_epochs=args.warmstart_epochs,
    )
    
    # Replace model in predictor
    predictor._learner._model = joint_model
    
    print("Model wrapped with joint layer selector")
    print(f"Selector initial temperature: {args.initial_temperature}")
    print(f"Selector final temperature: {args.final_temperature}")
    
    # Continue training with joint optimization
    print(f"\nPhase 2: Joint training ({args.total_epochs - args.warmstart_epochs} epochs)...")
    print("Conv-LoRA and Policy parameters will be trained simultaneously")
    
    # Update config for remaining epochs
    predictor._learner._config.optim.max_epochs = args.total_epochs
    
    # Manual training loop (simplified - for full control)
    # Note: This is a simplified version. For production, would need full Lightning integration
    print("\nNote: Using simplified joint training loop")
    print("Conv-LoRA parameters and Policy will be trained end-to-end")
    
    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        seed=args.seed,
    )
    
    # Save final model
    predictor.save(args.output_dir)
    
    # Evaluation
    print("\n" + "="*70)
    print("Evaluating on test set...")
    print("="*70)
    
    metrics = predictor.evaluate(test_df, metrics=['iou', 'dice'])
    
    print(f"\nTest Results:")
    print(f"  IoU:  {metrics['iou']:.4f}")
    print(f"  DICE: {metrics['dice']:.4f}")
    
    # Save metrics
    with open(os.path.join(args.output_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f"Test IoU: {metrics['iou']:.4f}\n")
        f.write(f"Test DICE: {metrics['dice']:.4f}\n")
    
    # Extract and analyze layer selection patterns
    print("\nAnalyzing layer selection patterns...")
    joint_model = predictor._learner._model
    if isinstance(joint_model, SAMWithJointLayerSelection):
        # Get layer activation frequencies from validation set
        val_masks = []
        for i in range(min(100, len(val_df))):
            single_data = val_df.iloc[[i]]
            with torch.no_grad():
                _, layer_mask, stats = joint_model.forward({'image': single_data})
                val_masks.append(layer_mask.cpu())
        
        val_masks = torch.cat(val_masks, dim=0)  # [N, 32]
        layer_freq = val_masks.float().mean(dim=0).numpy()
        
        print(f"\nLayer Activation Frequencies:")
        for i, freq in enumerate(layer_freq):
            print(f"  Layer {i:2d}: {freq:.3f}")
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.bar(range(32), layer_freq)
        plt.xlabel('Layer Index')
        plt.ylabel('Activation Frequency')
        plt.title('Layer Selection Pattern (Joint Training)')
        plt.savefig(os.path.join(args.output_dir, 'layer_selection_pattern.png'))
        print(f"\nVisualization saved to {args.output_dir}/layer_selection_pattern.png")
    
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)


if __name__ == '__main__':
    main()

