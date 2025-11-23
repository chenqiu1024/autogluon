"""
RL Environment for Conv-LoRA Layer Selection.

This module implements an environment that wraps a trained Conv-LoRA model
and provides reward signals based on segmentation performance.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import numpy as np
from torch.utils.data import DataLoader, Subset
import pandas as pd


class ConvLoRAEnvironment:
    """
    Environment for training RL policy to select Conv-LoRA layers.
    
    The environment wraps a pre-trained SAM model with Conv-LoRA and evaluates
    the segmentation performance (IoU/DICE) when different layers are activated.
    
    Parameters
    ----------
    predictor : MultiModalPredictor
        Pre-trained AutoGluon predictor with Conv-LoRA
    val_data : pd.DataFrame
        Validation dataset for evaluation
    eval_subset_size : int
        Number of samples to use for fast evaluation (default: 200)
    reward_metric : str
        Metric to use for reward: 'iou' or 'dice' (default: 'iou')
    device : str
        Device to run evaluation on (default: 'cuda')
    """
    
    def __init__(
        self,
        predictor,
        val_data: pd.DataFrame,
        eval_subset_size: int = 200,
        reward_metric: str = 'iou',
        device: str = 'cuda',
    ):
        self.predictor = predictor
        self.val_data = val_data
        self.eval_subset_size = min(eval_subset_size, len(val_data))
        self.reward_metric = reward_metric
        self.device = device
        
        # Create evaluation subset
        if self.eval_subset_size < len(val_data):
            indices = np.random.choice(len(val_data), self.eval_subset_size, replace=False)
            self.eval_data = val_data.iloc[indices].reset_index(drop=True)
        else:
            self.eval_data = val_data
        
        # Freeze all Conv-LoRA parameters
        self._freeze_conv_lora_params()
        
        # Statistics
        self.num_evaluations = 0
        self.baseline_reward = None
    
    def _freeze_conv_lora_params(self):
        """Freeze all Conv-LoRA parameters for RL training."""
        model = self.predictor._learner._model
        for name, param in model.named_parameters():
            if 'lora' in name.lower() or 'moe' in name.lower():
                param.requires_grad = False
    
    def compute_baseline_reward(self) -> float:
        """
        Compute baseline reward with all layers activated.
        
        Returns
        -------
        baseline_reward : float
            Reward with all 32 layers activated
        """
        if self.baseline_reward is None:
            # Evaluate with all layers active (layer_masks=None)
            metrics = self.predictor.evaluate(
                self.eval_data,
                metrics=[self.reward_metric],
            )
            self.baseline_reward = metrics[self.reward_metric]
            print(f"Baseline reward ({self.reward_metric}): {self.baseline_reward:.4f}")
        
        return self.baseline_reward
    
    def evaluate_policy(
        self,
        layer_masks: torch.Tensor,
        return_std: bool = False,
    ) -> Tuple[float, Optional[float]]:
        """
        Evaluate segmentation performance with given layer masks on the eval subset.
        
        This method is used for computing baseline reward (all layers active).
        For per-image evaluation during RL training, use evaluate_single_image().
        
        Parameters
        ----------
        layer_masks : torch.Tensor
            Layer activation masks, shape [B, 32] or [32]
        return_std : bool
            If True, also return standard deviation of rewards
        
        Returns
        -------
        reward : float
            Average reward (IoU or DICE) across evaluation set
        reward_std : float, optional
            Standard deviation of rewards (if return_std=True)
        """
        self.num_evaluations += 1
        
        # Get the model from predictor
        model = self.predictor._learner._model
        model.eval()
        
        # Store original forward method
        original_forward = model.forward
        
        # Ensure layer_masks has batch dimension
        if layer_masks.dim() == 1:
            # Single mask for all images: [32] -> [1, 32]
            layer_masks = layer_masks.unsqueeze(0)
        
        # Create a wrapper that injects layer_masks
        def forward_with_masks(*args, **kwargs):
            # Use the same mask for all images in the batch
            kwargs['layer_masks'] = layer_masks
            return original_forward(*args, **kwargs)
        
        # Temporarily replace forward method
        model.forward = forward_with_masks
        
        try:
            # Evaluate with layer masks
            with torch.no_grad():
                metrics = self.predictor.evaluate(
                    self.eval_data,
                    metrics=[self.reward_metric, 'dice'] if self.reward_metric == 'iou' else ['iou', 'dice'],
                )
            
            reward = metrics[self.reward_metric]
            
            if return_std:
                # Note: AutoGluon evaluate doesn't return per-sample metrics
                # For now, we estimate std from the metric value
                # In a more detailed implementation, we could compute per-sample rewards
                reward_std = 0.0
                return reward, reward_std
            
            return reward
        
        finally:
            # Restore original forward method
            model.forward = original_forward
    
    def evaluate_single_image(
        self,
        image_data: pd.DataFrame,
        layer_mask: torch.Tensor,
    ) -> float:
        """
        Evaluate a single image with its specific layer mask.
        
        This is the correct method for per-image RL training, ensuring that
        each image is evaluated with its own policy-generated mask.
        
        Parameters
        ----------
        image_data : pd.DataFrame
            Single image data (should be 1 row)
        layer_mask : torch.Tensor
            Layer activation mask for this specific image, shape [32] or [1, 32]
        
        Returns
        -------
        reward : float
            Reward (IoU or DICE) for this single image
        """
        self.num_evaluations += 1
        
        # Ensure single image
        assert len(image_data) == 1, f"Expected 1 image, got {len(image_data)}"
        
        # Get the model from predictor
        model = self.predictor._learner._model
        model.eval()
        
        # Store original forward method
        original_forward = model.forward
        
        # Ensure layer_mask has batch dimension [1, 32]
        if layer_mask.dim() == 1:
            layer_mask = layer_mask.unsqueeze(0)
        
        # Create a wrapper that injects this specific layer_mask
        def forward_with_mask(*args, **kwargs):
            kwargs['layer_masks'] = layer_mask
            return original_forward(*args, **kwargs)
        
        # Temporarily replace forward method
        model.forward = forward_with_mask
        
        try:
            # Evaluate this single image with its specific mask
            with torch.no_grad():
                metrics = self.predictor.evaluate(
                    image_data,
                    metrics=[self.reward_metric],
                )
            
            reward = metrics[self.reward_metric]
            return reward
        
        finally:
            # Restore original forward method
            model.forward = original_forward
    
    def step(
        self,
        image_data: pd.DataFrame,
        layer_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Execute one environment step: evaluate a single image with its layer mask.
        
        Parameters
        ----------
        image_data : pd.DataFrame
            Single image data (1 row)
        layer_mask : torch.Tensor
            Layer activation mask for this image, shape [32] or [1, 32]
        
        Returns
        -------
        result : dict
            Dictionary containing:
            - reward: reward value (IoU or DICE) for this image
            - num_active_layers: number of activated layers
            - done: always True (episodic task)
        """
        reward = self.evaluate_single_image(image_data, layer_mask)
        
        # Compute number of active layers
        if layer_mask.dim() == 1:
            num_active = layer_mask.sum().item()
        else:
            num_active = layer_mask.sum().item()  # [1, 32] -> scalar
        
        return {
            'reward': reward,
            'num_active_layers': num_active,
            'done': True,
        }
    
    def reset(self):
        """Reset environment (no-op for this environment)."""
        pass
    
    def get_statistics(self) -> Dict:
        """
        Get environment statistics.
        
        Returns
        -------
        stats : dict
            Dictionary containing environment statistics
        """
        return {
            'num_evaluations': self.num_evaluations,
            'baseline_reward': self.baseline_reward,
            'eval_subset_size': self.eval_subset_size,
        }


class BatchedConvLoRAEnvironment:
    """
    Batched version of ConvLoRAEnvironment for efficient parallel evaluation.
    
    This environment supports evaluating multiple layer mask configurations
    in parallel, which is useful for batch RL training.
    
    Parameters
    ----------
    predictor : MultiModalPredictor
        Pre-trained AutoGluon predictor with Conv-LoRA
    val_data : pd.DataFrame
        Validation dataset for evaluation
    batch_size : int
        Number of images per batch (default: 16)
    eval_subset_size : int
        Number of samples to use for evaluation (default: 200)
    reward_metric : str
        Metric to use for reward: 'iou' or 'dice' (default: 'iou')
    device : str
        Device to run evaluation on (default: 'cuda')
    """
    
    def __init__(
        self,
        predictor,
        val_data: pd.DataFrame,
        batch_size: int = 16,
        eval_subset_size: int = 200,
        reward_metric: str = 'iou',
        device: str = 'cuda',
    ):
        self.base_env = ConvLoRAEnvironment(
            predictor=predictor,
            val_data=val_data,
            eval_subset_size=eval_subset_size,
            reward_metric=reward_metric,
            device=device,
        )
        self.batch_size = batch_size
        self.device = device
    
    def evaluate_batch(
        self,
        batch_data: pd.DataFrame,
        layer_masks_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate a batch of images with their corresponding layer masks.
        
        Each image is evaluated with its own specific layer mask, following
        the per-image dynamic selection paradigm.
        
        Parameters
        ----------
        batch_data : pd.DataFrame
            Batch of image data, shape [B, ...]
        layer_masks_batch : torch.Tensor
            Batch of layer masks, shape [B, 32]
            layer_masks_batch[i] is the mask for batch_data.iloc[i]
        
        Returns
        -------
        rewards : torch.Tensor
            Rewards for each image, shape [B]
        """
        B = layer_masks_batch.size(0)
        assert len(batch_data) == B, f"Batch size mismatch: {len(batch_data)} vs {B}"
        
        rewards = []
        
        for i in range(B):
            # Get single image data
            single_image_data = batch_data.iloc[[i]].reset_index(drop=True)
            layer_mask = layer_masks_batch[i]  # [32]
            
            # Evaluate this image with its specific mask
            result = self.base_env.step(single_image_data, layer_mask)
            rewards.append(result['reward'])
        
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)
    
    def compute_baseline_reward(self) -> float:
        """Compute baseline reward with all layers activated."""
        return self.base_env.compute_baseline_reward()
    
    def get_statistics(self) -> Dict:
        """Get environment statistics."""
        return self.base_env.get_statistics()

