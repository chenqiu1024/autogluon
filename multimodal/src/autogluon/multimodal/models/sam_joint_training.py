"""
SAM with Joint Training of Conv-LoRA and Layer Selection Policy.

This module wraps a SAM model with Conv-LoRA and adds a differentiable
layer selection mechanism, enabling joint end-to-end training.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import logging

from .differentiable_layer_selection import GumbelLayerSelector, TemperatureScheduler

logger = logging.getLogger(__name__)


class SAMWithJointLayerSelection(nn.Module):
    """
    SAM + Conv-LoRA with jointly trained layer selection policy.
    
    This model integrates:
    1. Pre-trained SAM backbone (frozen)
    2. Conv-LoRA adaptation layers (trainable)
    3. Gumbel-Softmax layer selector (trainable)
    
    During training, the layer selector generates differentiable masks that
    control which Conv-LoRA layers are activated. Both the masks and Conv-LoRA
    parameters are optimized jointly to maximize segmentation performance.
    
    Parameters
    ----------
    sam_model : nn.Module
        SAM model with Conv-LoRA layers (SAMForSemanticSegmentation)
    selector_temperature : float
        Initial temperature for Gumbel-Softmax (default: 1.0)
    warmstart_epochs : int
        Number of epochs to train only Conv-LoRA before joint training (default: 3)
    """
    
    def __init__(
        self,
        sam_model: nn.Module,
        selector_temperature: float = 1.0,
        warmstart_epochs: int = 3,
    ):
        super().__init__()
        self.sam_model = sam_model
        self.warmstart_epochs = warmstart_epochs
        self.current_epoch = 0
        
        # Layer selector (trainable)
        self.layer_selector = GumbelLayerSelector(
            input_dim=1280,  # SAM-ViT-Huge hidden dim
            num_layers=32,   # SAM-ViT-Huge num layers
            temperature=selector_temperature,
            init_bias=0.0,   # Start from 50% activation probability
        )
        
        # Temperature scheduler
        self.temp_scheduler = TemperatureScheduler(
            initial_temp=1.0,
            final_temp=0.1,
            anneal_start_epoch=warmstart_epochs + 1,
            anneal_end_epoch=16,
            anneal_mode='exponential',
        )
        
        # Track whether we're in warmstart phase
        self._in_warmstart = True
        
        logger.info(f"Initialized SAMWithJointLayerSelection")
        logger.info(f"  Selector temperature: {selector_temperature}")
        logger.info(f"  Warmstart epochs: {warmstart_epochs}")
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_selection_stats: bool = False,
    ) -> Tuple[Dict, Optional[torch.Tensor], Optional[Dict]]:
        """
        Joint forward pass: generate layer masks and apply to SAM.
        
        Parameters
        ----------
        batch : dict
            Batch dictionary containing 'image' and 'label' keys
        return_selection_stats : bool
            If True, return layer selection statistics
        
        Returns
        -------
        output : dict
            SAM model output dictionary
        layer_masks : torch.Tensor, optional
            Generated layer masks, shape [B, 32]
        selection_stats : dict, optional
            Layer selection statistics (if return_selection_stats=True)
        """
        # 1. Extract patch embeddings (input to transformer layers)
        patch_embeddings = self.extract_patch_embeddings(batch)
        
        # 2. Generate layer masks
        if self._in_warmstart:
            # Warmstart: force all layers active
            B = patch_embeddings.size(0)
            layer_masks = torch.ones(B, self.layer_selector.num_layers, 
                                    device=patch_embeddings.device)
            # Still compute logits for monitoring
            _, logits = self.layer_selector(patch_embeddings, force_masks=layer_masks)
        else:
            # Joint training: generate masks with Gumbel-Softmax
            layer_masks, logits = self.layer_selector(patch_embeddings, hard=True)
        
        # 3. Forward through SAM with layer masks
        output = self.sam_model(batch, layer_masks=layer_masks)
        
        # 4. Compute selection statistics if requested
        selection_stats = None
        if return_selection_stats:
            selection_stats = self.layer_selector.get_selection_statistics(
                layer_masks, logits
            )
        
        return output, layer_masks, selection_stats
    
    def extract_patch_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract patch embeddings from SAM (input to transformer layers).
        
        These embeddings serve as the state representation for the layer selector.
        We extract them with gradients disabled to avoid interfering with the
        main task gradients.
        
        Parameters
        ----------
        batch : dict
            Batch containing 'image' key with images [B, 3, H, W]
        
        Returns
        -------
        patch_embeddings : torch.Tensor
            Patch embeddings with position encoding, shape [B, 64, 64, 1280]
        """
        # Get image key from sam_model
        if hasattr(self.sam_model, 'image_key'):
            image_key = self.sam_model.image_key
        else:
            image_key = 'image'
        
        images = batch[image_key]
        
        # Extract patch embeddings (without gradients to save memory)
        with torch.no_grad():
            # Access the vision encoder's patch embedding layer
            vision_encoder = self.sam_model.model.vision_encoder
            embeddings = vision_encoder.patch_embed(images)
            
            # Add positional embeddings if present
            if vision_encoder.pos_embed is not None:
                embeddings = embeddings + vision_encoder.pos_embed
        
        return embeddings  # [B, 64, 64, 1280]
    
    def set_epoch(self, epoch: int):
        """
        Update epoch counter and training phase.
        
        This controls:
        1. Whether we're in warmstart phase (all layers active)
        2. Temperature annealing schedule
        
        Parameters
        ----------
        epoch : int
            Current epoch number (0-indexed)
        """
        self.current_epoch = epoch
        
        # Update warmstart status
        if epoch < self.warmstart_epochs:
            if not self._in_warmstart:
                logger.info(f"Epoch {epoch}: Warmstart phase")
            self._in_warmstart = True
        else:
            if self._in_warmstart:
                logger.info(f"Epoch {epoch}: Starting joint training")
            self._in_warmstart = False
        
        # Update temperature if in joint training phase
        if not self._in_warmstart:
            new_temp = self.temp_scheduler.get_temperature(epoch)
            if new_temp != self.layer_selector.temperature:
                logger.info(f"Epoch {epoch}: Temperature {self.layer_selector.temperature:.3f} -> {new_temp:.3f}")
                self.layer_selector.set_temperature(new_temp)
    
    def freeze_layer_selector(self):
        """Freeze layer selector parameters (for warmstart)."""
        for param in self.layer_selector.parameters():
            param.requires_grad = False
    
    def unfreeze_layer_selector(self):
        """Unfreeze layer selector parameters (for joint training)."""
        for param in self.layer_selector.parameters():
            param.requires_grad = True
    
    def get_lora_parameters(self):
        """Get Conv-LoRA parameters for optimizer."""
        lora_params = []
        for name, param in self.sam_model.named_parameters():
            if 'lora' in name.lower() or 'moe' in name.lower():
                lora_params.append(param)
        return lora_params
    
    def get_selector_parameters(self):
        """Get layer selector parameters for optimizer."""
        return list(self.layer_selector.parameters())

