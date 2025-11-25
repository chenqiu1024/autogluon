"""
Lightning Module for Joint Training of Conv-LoRA and Layer Selection.

This module implements the training logic for simultaneous optimization of
Conv-LoRA parameters and layer selection policy using Gumbel-Softmax.
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from lightning import LightningModule
from omegaconf import DictConfig

from ..models.sam_joint_training import SAMWithJointLayerSelection
from ..constants import LOGITS, LABEL
from .losses.structure_loss import StructureLoss

logger = logging.getLogger(__name__)


class SemanticSegmentationJointTraining(LightningModule):
    """
    Lightning module for joint training of Conv-LoRA and layer selection policy.
    
    Key features:
    - Warmstart phase: train only Conv-LoRA with all layers active
    - Joint phase: train both Conv-LoRA and policy with Gumbel-Softmax masks
    - Temperature annealing: gradually transition from soft to hard masks
    - Differential learning rates: faster for policy, slower for Conv-LoRA
    
    Parameters
    ----------
    model : SAMWithJointLayerSelection
        Joint training model wrapper
    config : DictConfig
        Training configuration
    """
    
    def __init__(
        self,
        model: SAMWithJointLayerSelection,
        config: DictConfig,
    ):
        super().__init__()
        self.model = model
        self.config = config
        
        # Loss function
        self.criterion = StructureLoss()  # or other segmentation loss
        
        # Track training phase
        self.current_epoch_num = 0
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
        
        logger.info("Initialized SemanticSegmentationJointTraining")
    
    def forward(self, batch):
        """Forward pass through the joint model."""
        return self.model(batch, return_selection_stats=True)
    
    def training_step(self, batch, batch_idx):
        """
        Training step with joint optimization.
        
        During warmstart: only Conv-LoRA gradients, all layers forced active
        During joint training: both Conv-LoRA and policy gradients, dynamic masks
        """
        # Forward pass
        output, layer_masks, selection_stats = self(batch)
        
        # Get predictions and labels
        logits = output[self.model.sam_model.prefix][LOGITS]
        labels = batch[self.model.sam_model.label_key]
        
        # Compute segmentation loss
        seg_loss = self.criterion(logits, labels)
        
        # MoE loss (if Conv-LoRA is active)
        moe_loss = output.get(self.model.sam_model.prefix, {}).get('moe_loss', 0.0)
        
        # Total loss
        total_loss = seg_loss + 0.01 * moe_loss
        
        # Logging
        self.log('train/seg_loss', seg_loss, prog_bar=True)
        self.log('train/total_loss', total_loss, prog_bar=True)
        self.log('train/moe_loss', moe_loss)
        
        if selection_stats is not None:
            self.log('train/mean_active_layers', selection_stats['mean_active_layers'], prog_bar=True)
            self.log('train/prob_std', selection_stats['prob_std'], prog_bar=True)
            self.log('train/entropy', selection_stats['entropy'])
            self.log('train/prob_min', selection_stats['prob_min'])
            self.log('train/prob_max', selection_stats['prob_max'])
        
        # Log temperature
        self.log('train/temperature', self.model.layer_selector.temperature)
        
        # Log whether in warmstart
        self.log('train/in_warmstart', float(self.model._in_warmstart))
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Forward pass
        output, layer_masks, selection_stats = self(batch)
        
        # Get predictions and labels
        logits = output[self.model.sam_model.prefix][LOGITS]
        labels = batch[self.model.sam_model.label_key]
        
        # Compute loss
        seg_loss = self.criterion(logits, labels)
        
        # Log
        self.log('val/seg_loss', seg_loss, prog_bar=True)
        
        if selection_stats is not None:
            self.log('val/mean_active_layers', selection_stats['mean_active_layers'])
            self.log('val/prob_std', selection_stats['prob_std'])
        
        return {'val_loss': seg_loss, 'layer_masks': layer_masks}
    
    def configure_optimizers(self):
        """
        Configure optimizers with differential learning rates.
        
        Conv-LoRA parameters: lower LR (1e-4) for stable fine-tuning
        Policy parameters: higher LR (5e-4) for faster adaptation
        """
        # Get parameter groups
        lora_params = self.model.get_lora_parameters()
        selector_params = self.model.get_selector_parameters()
        
        # Create optimizer with parameter groups
        optimizer = torch.optim.AdamW([
            {
                'params': lora_params,
                'lr': self.config.get('lora_lr', 1e-4),
                'weight_decay': self.config.get('lora_weight_decay', 0.01),
            },
            {
                'params': selector_params,
                'lr': self.config.get('selector_lr', 5e-4),
                'weight_decay': self.config.get('selector_weight_decay', 0.0),
            },
        ])
        
        # Learning rate scheduler (optional)
        if self.config.get('use_scheduler', False):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.get('max_epochs', 20),
                eta_min=1e-6,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                }
            }
        
        return optimizer
    
    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        # Update epoch in model (for warmstart/temperature control)
        self.model.set_epoch(self.current_epoch)
        
        # Freeze/unfreeze selector based on phase
        if self.model._in_warmstart:
            self.model.freeze_layer_selector()
            logger.info(f"Epoch {self.current_epoch}: Warmstart - selector frozen")
        else:
            self.model.unfreeze_layer_selector()
            if self.current_epoch == self.model.warmstart_epochs:
                logger.info(f"Epoch {self.current_epoch}: Joint training started - selector unfrozen")
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        self.current_epoch_num += 1
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """
        Optimizer step with gradient clipping.
        """
        # Gradient clipping
        if self.config.get('clip_grad_norm', 0) > 0:
            # Clip Conv-LoRA gradients
            lora_params = self.model.get_lora_parameters()
            torch.nn.utils.clip_grad_norm_(
                lora_params,
                max_norm=self.config.get('clip_grad_norm', 1.0)
            )
            
            # Clip policy gradients (can use different threshold)
            if not self.model._in_warmstart:
                selector_params = self.model.get_selector_parameters()
                torch.nn.utils.clip_grad_norm_(
                    selector_params,
                    max_norm=self.config.get('clip_selector_grad_norm', 0.5)
                )
        
        # Standard optimizer step
        optimizer.step(closure=optimizer_closure)

