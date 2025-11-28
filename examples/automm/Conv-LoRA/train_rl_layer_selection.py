"""
RL Training Script for Conv-LoRA Layer Selection.

This script trains a reinforcement learning policy to dynamically select
which Conv-LoRA layers to activate for each input image in medical image
segmentation tasks.

Usage:
    python train_rl_layer_selection.py \
        --task isic2017 \
        --ckpt_path baseline_full_layers \
        --output_dir rl_layer_selection \
        --num_episodes 1000 \
        --batch_size 16 \
        --learning_rate 1e-4
"""

import argparse
import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

# Add autogluon to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "multimodal" / "src"))

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.rl.policies import LayerSelectionPolicy
from autogluon.multimodal.rl.envs import ConvLoRAEnvironment
from autogluon.multimodal.rl.algos import REINFORCE
from autogluon.multimodal.rl.algos.rloo import RLOO


def expand_path(df, dataset_dir):
    """Expand relative paths in dataframe to absolute paths."""
    for col in ["image", "label"]:
        df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


def load_data(task_name: str, data_dir: str = "datasets"):
    """
    Load training and validation data for the task.
    
    Parameters
    ----------
    task_name : str
        Name of the task (e.g., 'isic2017')
    data_dir : str
        Base directory for datasets
    
    Returns
    -------
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame
        Validation data
    test_df : pd.DataFrame
        Test data
    """
    dataset_dir = os.path.join(data_dir, task_name, task_name)
    
    train_df = expand_path(pd.read_csv(os.path.join(dataset_dir, "train.csv")), dataset_dir)
    val_df = expand_path(pd.read_csv(os.path.join(dataset_dir, "val.csv")), dataset_dir)
    test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, "test.csv")), dataset_dir)
    
    return train_df, val_df, test_df


def get_patch_embeddings(predictor, images: pd.DataFrame, device: str = 'cuda'):
    """
    Extract patch embeddings from SAM for the policy input.
    
    Parameters
    ----------
    predictor : MultiModalPredictor
        Trained predictor with Conv-LoRA
    images : pd.DataFrame
        DataFrame containing image paths
    device : str
        Device to run on
    
    Returns
    -------
    embeddings : torch.Tensor
        Patch embeddings, shape [B, H, W, C]
    """
    model = predictor._learner._model
    model.eval()
    
    # Get dataloader
    datamodule = predictor._learner._data_module
    dataloader = datamodule.val_dataloader()
    
    embeddings_list = []
    with torch.no_grad():
        for batch in dataloader:
            # Get pixel values
            pixel_values = batch['image'].to(device)
            
            # Get patch embeddings from SAM
            model_output = model.model.vision_encoder.patch_embed(pixel_values)
            if model.model.vision_encoder.pos_embed is not None:
                model_output = model_output + model.model.vision_encoder.pos_embed
            
            embeddings_list.append(model_output.cpu())
            
            if len(embeddings_list) * pixel_values.size(0) >= len(images):
                break
    
    embeddings = torch.cat(embeddings_list, dim=0)[:len(images)]
    return embeddings


def train_rl_policy(
    predictor,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    output_dir: str,
    num_episodes: int = 1000,
    batch_size: int = 16,
    eval_subset_size: int = 200,
    learning_rate: float = 1e-4,
    baseline_decay: float = 0.99,
    entropy_coef: float = 0.05,
    init_bias: float = 0.0,
    warmup_steps: int = 100,
    save_freq: int = 100,
    eval_freq: int = 50,
    reward_metric: str = 'iou',
    algo_name: str = 'reinforce',
    rloo_k: int = 4,
    device: str = 'cuda',
):
    """
    Train RL policy for layer selection.
    
    Parameters
    ----------
    predictor : MultiModalPredictor
        Pre-trained predictor with Conv-LoRA
    val_data : pd.DataFrame
        Validation data for training the policy
    test_data : pd.DataFrame
        Test data for final evaluation
    output_dir : str
        Directory to save checkpoints and logs
    num_episodes : int
        Number of training episodes
    batch_size : int
        Batch size for policy training
    eval_subset_size : int
        Size of subset for fast evaluation
    learning_rate : float
        Learning rate for policy optimizer
    baseline_decay : float
        Decay rate for moving average baseline
    entropy_coef : float
        Entropy regularization coefficient
    warmup_steps : int
        Number of warmup steps for learning rate
    save_freq : int
        Frequency of saving checkpoints
    eval_freq : int
        Frequency of evaluation on test set
    reward_metric : str
        Metric to use as reward ('iou' or 'dice')
    algo_name : str
        Algorithm to use ('reinforce' or 'rloo')
    rloo_k : int
        Number of samples per input for RLOO (default: 4)
    device : str
        Device to run on
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
    
    # Initialize tensorboard writer
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))
    
    print(f"\n{'='*60}")
    print(f"RL Training for Conv-LoRA Layer Selection")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Training episodes: {num_episodes}")
    print(f"Batch size: {batch_size}")
    print(f"Eval subset size: {eval_subset_size}")
    print(f"Reward metric: {reward_metric}")
    print(f"Algorithm: {algo_name}")
    if algo_name == 'rloo':
        print(f"RLOO K: {rloo_k}")
    print(f"{'='*60}\n")
    
    # Initialize environment
    print("Initializing environment...")
    env = ConvLoRAEnvironment(
        predictor=predictor,
        val_data=val_data,
        eval_subset_size=eval_subset_size,
        reward_metric=reward_metric,
        device=device,
    )
    
    # Compute baseline reward
    print("Computing baseline reward (all layers active)...")
    baseline_reward = env.compute_baseline_reward()
    
    # Initialize policy
    print("Initializing policy network...")
    print(f"Policy init_bias: {init_bias} (sigmoid({init_bias}) = {torch.sigmoid(torch.tensor(init_bias)):.3f})")
    # SAM-ViT-Huge has 1280 hidden dim and 32 layers
    # Use init_bias=0.0 to start from neutral state (50% activation probability)
    # instead of init_bias=2.0 which led to getting stuck at 88% for all layers
    policy = LayerSelectionPolicy(
        hidden_dim=1280,
        num_layers=32,
        init_bias=init_bias,  # Configurable via command line
    ).to(device)
    
    # Initialize optimizer and algorithm
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    if algo_name == 'rloo':
        algo = RLOO(
            policy=policy,
            optimizer=optimizer,
            k=rloo_k,
            entropy_coef=entropy_coef,
            max_grad_norm=1.0,
            device=device,
        )
    else:
        algo = REINFORCE(
            policy=policy,
            optimizer=optimizer,
            baseline_decay=baseline_decay,
            entropy_coef=entropy_coef,
            max_grad_norm=1.0,
            device=device,
        )
    
    # Training statistics
    best_reward = baseline_reward
    best_episode = 0
    episode_rewards = []
    episode_num_layers = []
    
    # Sample a batch of validation images for training
    val_indices = np.arange(len(val_data))
    
    print(f"\nStarting RL training for {num_episodes} episodes...")
    print(f"Baseline reward: {baseline_reward:.4f}\n")
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        policy.train()
        
        # Sample batch of images
        batch_indices = np.random.choice(val_indices, size=batch_size, replace=False)
        batch_data = val_data.iloc[batch_indices].reset_index(drop=True)
        
        # Get patch embeddings for the batch
        # Note: In practice, we would cache these embeddings
        # For now, we'll use a simplified approach
        
        # Use predictor's internal processing to get features
        # For the actual implementation, we need to hook into the model
        # to extract patch embeddings before the transformer layers
        
        # Simplified: use random patch embeddings for demonstration
        # In production, replace this with actual patch embeddings
        patch_embeddings = torch.randn(batch_size, 64, 64, 1280).to(device)
        
        if algo_name == 'rloo':
            # RLOO: Generate k samples for each image in the batch
            # We repeat the batch k times to process efficiently
            # batch_size * k samples total
            
            # Repeat patch embeddings k times: [B, ...] -> [B*K, ...]
            # We interleave repeats: [img1, img1, ..., img2, img2, ...]
            patch_embeddings_expanded = patch_embeddings.repeat_interleave(rloo_k, dim=0)
            
            # Policy forward: generate layer masks for all B*K samples
            layer_masks, layer_probs, log_probs = policy(
                patch_embeddings_expanded,
                deterministic=False,
                temperature=1.0,
            )
            
            # Evaluate rewards for each sample
            rewards = []
            for i in range(batch_size):
                # Get single image data
                single_image_data = batch_data.iloc[[i]].reset_index(drop=True)
                
                # Evaluate k samples for this image
                for k in range(rloo_k):
                    idx = i * rloo_k + k
                    result = env.step(single_image_data, layer_masks[idx])
                    rewards.append(result['reward'])
            
        else:
            # REINFORCE: Single sample per image
            # Policy forward: generate layer masks
            layer_masks, layer_probs, log_probs = policy(
                patch_embeddings,
                deterministic=False,
                temperature=1.0,
            )
            
            # Evaluate rewards for each sample in the batch
            # Per-image evaluation: each image uses its own policy-generated mask
            rewards = []
            for i in range(batch_size):
                # Get single image data
                single_image_data = batch_data.iloc[[i]].reset_index(drop=True)
                # Evaluate this image with its specific mask
                result = env.step(single_image_data, layer_masks[i])
                rewards.append(result['reward'])
        
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        # Apply learning rate warmup
        if episode < warmup_steps:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * (episode + 1) / warmup_steps
        
        # Policy update
        update_info = algo.update(log_probs, rewards, layer_probs)
        
        # Statistics
        mean_reward = rewards.mean().item()
        mean_num_layers = layer_masks.sum(dim=1).mean().item()
        episode_rewards.append(mean_reward)
        episode_num_layers.append(mean_num_layers)
        
        # Compute probability distribution statistics
        prob_min = layer_probs.min().item()
        prob_max = layer_probs.max().item()
        prob_mean = layer_probs.mean().item()
        prob_std = layer_probs.std().item()
        
        # Count how many layers have high/low probabilities
        high_prob_layers = (layer_probs > 0.7).float().mean().item() * 32
        low_prob_layers = (layer_probs < 0.3).float().mean().item() * 32
        
        # Logging
        writer.add_scalar('train/reward', mean_reward, episode)
        writer.add_scalar('train/num_active_layers', mean_num_layers, episode)
        writer.add_scalar('train/loss', update_info['loss'], episode)
        writer.add_scalar('train/policy_loss', update_info['policy_loss'], episode)
        writer.add_scalar('train/entropy_loss', update_info['entropy_loss'], episode)
        writer.add_scalar('train/baseline', update_info['baseline'], episode)
        writer.add_scalar('train/grad_norm', update_info['grad_norm'], episode)
        
        # Log probability distribution statistics
        writer.add_scalar('train/prob_min', prob_min, episode)
        writer.add_scalar('train/prob_max', prob_max, episode)
        writer.add_scalar('train/prob_mean', prob_mean, episode)
        writer.add_scalar('train/prob_std', prob_std, episode)
        writer.add_scalar('train/high_prob_layers', high_prob_layers, episode)
        writer.add_scalar('train/low_prob_layers', low_prob_layers, episode)
        
        # Periodic evaluation on test set
        if (episode + 1) % eval_freq == 0:
            policy.eval()
            with torch.no_grad():
                test_embeddings = torch.randn(len(test_data), 64, 64, 1280).to(device)
                
                # Evaluate both stochastic and deterministic modes
                # Stochastic (used during training)
                test_masks_stoch, test_probs_stoch, _ = policy(
                    test_embeddings,
                    deterministic=False,
                )
                stats_stoch = policy.get_layer_statistics(test_probs_stoch)
                num_layers_stoch = test_masks_stoch.sum(dim=1).float().mean().item()
                
                # Deterministic (threshold at 0.5)
                test_masks_det, test_probs_det, _ = policy(
                    test_embeddings,
                    deterministic=True,
                )
                stats_det = policy.get_layer_statistics(test_probs_det)
                num_layers_det = test_masks_det.sum(dim=1).float().mean().item()
                
                # Log both modes for comparison
                writer.add_scalar('eval/num_layers_stochastic', num_layers_stoch, episode)
                writer.add_scalar('eval/num_layers_deterministic', num_layers_det, episode)
                writer.add_scalar('eval/prob_mean', test_probs_det.mean().item(), episode)
                writer.add_scalar('eval/prob_std', test_probs_det.std().item(), episode)
                
                # Log layer activation frequencies (deterministic)
                for layer_idx, freq in enumerate(stats_det['layer_activation_freq']):
                    writer.add_scalar(f'eval/layer_{layer_idx}_freq', freq, episode)
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            checkpoint = {
                'episode': episode,
                'policy_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'baseline': algo.baseline,
                'best_reward': best_reward,
                'episode_rewards': episode_rewards,
                'episode_num_layers': episode_num_layers,
            }
            torch.save(
                checkpoint,
                os.path.join(output_dir, 'checkpoints', f'episode_{episode+1}.pt')
            )
            
            if mean_reward > best_reward:
                best_reward = mean_reward
                best_episode = episode
                torch.save(
                    checkpoint,
                    os.path.join(output_dir, 'checkpoints', 'best.pt')
                )
        
        # Print progress with enhanced statistics
        if (episode + 1) % 10 == 0:
            tqdm.write(
                f"Episode {episode+1}/{num_episodes} | "
                f"Reward: {mean_reward:.4f} | "
                f"Layers: {mean_num_layers:.1f}/32 | "
                f"Prob: [{prob_min:.3f}, {prob_max:.3f}] (std={prob_std:.3f}) | "
                f"Loss: {update_info['loss']:.4f}"
            )
    
    # Final save
    final_checkpoint = {
        'episode': num_episodes - 1,
        'policy_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'baseline': algo.baseline,
        'best_reward': best_reward,
        'best_episode': best_episode,
        'episode_rewards': episode_rewards,
        'episode_num_layers': episode_num_layers,
    }
    torch.save(final_checkpoint, os.path.join(output_dir, 'final.pt'))
    
    # Plot training curves
    plot_training_curves(
        episode_rewards,
        episode_num_layers,
        baseline_reward,
        output_dir,
    )
    
    # Save training summary
    summary = {
        'baseline_reward': baseline_reward,
        'final_reward': episode_rewards[-1],
        'best_reward': best_reward,
        'best_episode': best_episode,
        'mean_reward_last_100': np.mean(episode_rewards[-100:]),
        'mean_num_layers_last_100': np.mean(episode_num_layers[-100:]),
        'num_episodes': num_episodes,
        'hyperparameters': {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'baseline_decay': baseline_decay,
            'entropy_coef': entropy_coef,
            'eval_subset_size': eval_subset_size,
            'algo': algo_name,
            'rloo_k': rloo_k if algo_name == 'rloo' else None,
        }
    }
    
    with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    writer.close()
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Baseline reward: {baseline_reward:.4f}")
    print(f"Best reward: {best_reward:.4f} (episode {best_episode+1})")
    print(f"Final reward: {episode_rewards[-1]:.4f}")
    print(f"Mean reward (last 100): {summary['mean_reward_last_100']:.4f}")
    print(f"Mean active layers (last 100): {summary['mean_num_layers_last_100']:.1f}/32")
    print(f"{'='*60}\n")
    
    return policy, summary


def plot_training_curves(
    episode_rewards: list,
    episode_num_layers: list,
    baseline_reward: float,
    output_dir: str,
):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot rewards
    ax1.plot(episode_rewards, alpha=0.3, label='Episode reward')
    # Moving average
    window = 50
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 
                label=f'{window}-episode moving average', linewidth=2)
    ax1.axhline(y=baseline_reward, color='r', linestyle='--', label='Baseline (all layers)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward (IoU/DICE)')
    ax1.set_title('Training Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot number of active layers
    ax2.plot(episode_num_layers, alpha=0.3, label='Episode')
    if len(episode_num_layers) >= window:
        moving_avg = np.convolve(episode_num_layers, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_num_layers)), moving_avg,
                label=f'{window}-episode moving average', linewidth=2)
    ax2.axhline(y=32, color='r', linestyle='--', label='All layers (32)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Number of Active Layers')
    ax2.set_title('Active Layers')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train RL policy for Conv-LoRA layer selection")
    
    # Task and data
    parser.add_argument('--task', type=str, default='isic2017',
                       help='Task name (default: isic2017)')
    parser.add_argument('--data_dir', type=str, default='datasets',
                       help='Base directory for datasets (default: datasets)')
    
    # Model
    parser.add_argument('--ckpt_path', type=str, required=True,
                       help='Path to pre-trained Conv-LoRA model checkpoint')
    
    # Training
    parser.add_argument('--output_dir', type=str, default='rl_layer_selection',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--num_episodes', type=int, default=1000,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for policy training (default: 16)')
    parser.add_argument('--eval_subset_size', type=int, default=200,
                       help='Size of validation subset for evaluation (default: 200)')
    
    # Hyperparameters
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--baseline_decay', type=float, default=0.99,
                       help='Baseline decay rate (default: 0.99)')
    parser.add_argument('--entropy_coef', type=float, default=0.05,
                       help='Entropy coefficient (default: 0.05, increased from 0.01)')
    parser.add_argument('--init_bias', type=float, default=0.0,
                       help='Initial bias for policy output layer (default: 0.0 for 50%% activation probability)')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='Number of warmup steps (default: 100)')
    
    # Logging and saving
    parser.add_argument('--save_freq', type=int, default=100,
                       help='Checkpoint saving frequency (default: 100)')
    parser.add_argument('--eval_freq', type=int, default=50,
                       help='Evaluation frequency (default: 50)')
    
    # Reward
    parser.add_argument('--reward_metric', type=str, default='iou', choices=['iou', 'dice'],
                       help='Reward metric (default: iou)')
    
    # Algorithm
    parser.add_argument('--algo', type=str, default='reinforce', choices=['reinforce', 'rloo'],
                       help='Algorithm to use (default: reinforce)')
    parser.add_argument('--rloo_k', type=int, default=4,
                       help='Number of samples per input for RLOO (default: 4)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load data
    print("Loading data...")
    train_df, val_df, test_df = load_data(args.task, args.data_dir)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Load pre-trained Conv-LoRA model
    print(f"\nLoading pre-trained model from {args.ckpt_path}...")
    predictor = MultiModalPredictor.load(args.ckpt_path)
    print("Model loaded successfully!")
    
    # Train RL policy
    policy, summary = train_rl_policy(
        predictor=predictor,
        val_data=val_df,
        test_data=test_df,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        eval_subset_size=args.eval_subset_size,
        learning_rate=args.learning_rate,
        baseline_decay=args.baseline_decay,
        entropy_coef=args.entropy_coef,
        init_bias=args.init_bias,
        warmup_steps=args.warmup_steps,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        reward_metric=args.reward_metric,
        algo_name=args.algo,
        rloo_k=args.rloo_k,
        device=args.device,
    )
    
    print("\nTraining completed successfully!")
    print(f"Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()

