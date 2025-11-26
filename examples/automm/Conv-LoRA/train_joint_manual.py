"""
Manual Joint Training Script (Direct PyTorch/Lightning Control).

This script bypasses AutoGluon's high-level API and directly implements
the joint training loop for full control over the optimization process.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "multimodal" / "src"))

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.models.sam_joint_training import SAMWithJointLayerSelection
from autogluon.multimodal.models.differentiable_layer_selection import TemperatureScheduler


def expand_path(df, dataset_dir):
    for col in ["image", "label"]:
        df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


class JointTrainingModule(L.LightningModule):
    """
    Lightning module for joint training with manual control.
    """
    
    def __init__(self, joint_model, config):
        super().__init__()
        self.joint_model = joint_model
        self.config = config
        
        # Temperature scheduler
        self.temp_scheduler = TemperatureScheduler(
            initial_temp=config['initial_temperature'],
            final_temp=config['final_temperature'],
            anneal_start_epoch=config['warmstart_epochs'],
            anneal_end_epoch=config['total_epochs'] - 4,
        )
        
        self.current_epoch_num = 0
        self.warmstart_epochs = config['warmstart_epochs']
    
    def forward(self, batch):
        return self.joint_model(batch, return_selection_stats=True)
    
    def training_step(self, batch, batch_idx):
        # Forward
        output, layer_masks, stats = self(batch)
        
        # Extract logits and labels
        prefix = list(output.keys())[0]
        logits = output[prefix]['logits']
        labels = batch['label']
        
        # Segmentation loss (structure loss or BCE)
        from autogluon.multimodal.optim.losses.structure_loss import StructureLoss
        criterion = StructureLoss()
        seg_loss = criterion(logits, labels)
        
        # MoE loss (if any)
        moe_loss = output[prefix].get('moe_loss', 0.0)
        if isinstance(moe_loss, torch.Tensor):
            moe_loss = moe_loss.item() if moe_loss.numel() == 1 else moe_loss.mean().item()
        
        # Total loss
        total_loss = seg_loss + 0.01 * moe_loss
        
        # Logging
        self.log('train/seg_loss', seg_loss, prog_bar=True)
        self.log('train/total_loss', total_loss, prog_bar=True)
        
        if stats:
            self.log('train/mean_active_layers', stats['mean_active_layers'], prog_bar=True)
            self.log('train/prob_std', stats['prob_std'])
            self.log('train/entropy', stats['entropy'])
        
        self.log('train/temperature', self.joint_model.layer_selector.temperature)
        self.log('train/in_warmstart', float(self.joint_model._in_warmstart))
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        output, layer_masks, stats = self(batch)
        
        prefix = list(output.keys())[0]
        logits = output[prefix]['logits']
        labels = batch['label']
        
        from autogluon.multimodal.optim.losses.structure_loss import StructureLoss
        criterion = StructureLoss()
        seg_loss = criterion(logits, labels)
        
        self.log('val/seg_loss', seg_loss, prog_bar=True)
        if stats:
            self.log('val/mean_active_layers', stats['mean_active_layers'])
        
        return seg_loss
    
    def configure_optimizers(self):
        lora_params = self.joint_model.get_lora_parameters()
        selector_params = self.joint_model.get_selector_parameters()
        
        optimizer = torch.optim.AdamW([
            {'params': lora_params, 'lr': self.config['lora_lr'], 'weight_decay': 0.01},
            {'params': selector_params, 'lr': self.config['selector_lr'], 'weight_decay': 0.0},
        ])
        
        return optimizer
    
    def on_train_epoch_start(self):
        # Update epoch and temperature
        self.joint_model.set_epoch(self.current_epoch)
        
        # Freeze/unfreeze selector
        if self.joint_model._in_warmstart:
            self.joint_model.freeze_layer_selector()
        else:
            self.joint_model.unfreeze_layer_selector()
    
    def on_train_epoch_end(self):
        self.current_epoch_num += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='isic2017')
    parser.add_argument('--data_dir', type=str, default='datasets')
    parser.add_argument('--warmstart_ckpt', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--warmstart_epochs', type=int, default=3)
    parser.add_argument('--total_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--lora_lr', type=float, default=1e-4)
    parser.add_argument('--selector_lr', type=float, default=5e-4)
    parser.add_argument('--initial_temperature', type=float, default=1.0)
    parser.add_argument('--final_temperature', type=float, default=0.1)
    parser.add_argument('--rank', type=int, default=3)
    parser.add_argument('--expert_num', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    L.seed_everything(args.seed)
    
    print(f"\n{'='*70}")
    print("Joint Training: Conv-LoRA + Layer Selection (Manual Mode)")
    print(f"{'='*70}\n")
    
    # Load data
    dataset_dir = os.path.join(args.data_dir, args.task, args.task)
    train_df = expand_path(pd.read_csv(os.path.join(dataset_dir, "train.csv")), dataset_dir)
    val_df = expand_path(pd.read_csv(os.path.join(dataset_dir, "val.csv")), dataset_dir)
    test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, "test.csv")), dataset_dir)
    
    print(f"Data loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Get or train warmstart model
    if args.warmstart_ckpt:
        print(f"\nLoading warmstart model from {args.warmstart_ckpt}")
        predictor = MultiModalPredictor.load(args.warmstart_ckpt)
    else:
        print(f"\nTraining warmstart Conv-LoRA ({args.warmstart_epochs} epochs)...")
        hyperparameters_warmstart = {
            "optim.lora.r": args.rank,
            "optim.peft": "conv_lora",
            "optim.lora.conv_lora_expert_num": args.expert_num,
            "env.num_gpus": args.num_gpus,
            "env.per_gpu_batch_size": args.batch_size,
            "env.batch_size": args.batch_size * args.num_gpus,
            "optim.max_epochs": args.warmstart_epochs,
            "optim.lr": args.lora_lr,
            "optim.loss_func": "structure_loss",
        }
        
        predictor = MultiModalPredictor(
            problem_type="semantic_segmentation",
            validation_metric="iou",
            eval_metric="iou",
            hyperparameters=hyperparameters_warmstart,
            label="label",
        )
        
        predictor.fit(train_data=train_df, tuning_data=val_df, seed=args.seed)
        
        # Save warmstart
        warmstart_path = os.path.join(args.output_dir, 'warmstart_model')
        predictor.save(warmstart_path)
        print(f"Warmstart model saved to {warmstart_path}")
    
    # Step 2: Wrap with joint training model
    print("\nWrapping with joint training layer selector...")
    from autogluon.multimodal.models.sam_joint_training import SAMWithJointLayerSelection
    
    original_model = predictor._learner._model
    joint_model = SAMWithJointLayerSelection(
        sam_model=original_model,
        selector_temperature=args.initial_temperature,
        warmstart_epochs=0,  # We already did warmstart
    )
    joint_model._in_warmstart = False  # Start joint training immediately
    
    print("Joint model created")
    
    # Step 3: Manual joint training
    print(f"\nStarting joint training ({args.total_epochs - args.warmstart_epochs} epochs)...")
    print("This will train Conv-LoRA and Policy simultaneously")
    print("Using Gumbel-Softmax for differentiable layer selection")
    print()
    
    config = {
        'lora_lr': args.lora_lr,
        'selector_lr': args.selector_lr,
        'warmstart_epochs': args.warmstart_epochs,
        'total_epochs': args.total_epochs,
        'initial_temperature': args.initial_temperature,
        'final_temperature': args.final_temperature,
    }
    
    # Create Lightning module
    lightning_module = JointTrainingModule(joint_model, config)
    
    # Get dataloaders from predictor
    datamodule = predictor._learner._data_module
    
    # Setup trainer
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, 'checkpoints'),
            filename='epoch_{epoch:02d}',
            save_top_k=-1,  # Save all
            every_n_epochs=2,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    trainer = Trainer(
        max_epochs=args.total_epochs - args.warmstart_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.num_gpus,
        callbacks=callbacks,
        default_root_dir=args.output_dir,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )
    
    # Train
    trainer.fit(lightning_module, datamodule=datamodule)
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model')
    predictor._learner._model = joint_model  # Update model
    predictor.save(final_path)
    print(f"\nFinal model saved to {final_path}")
    
    # Evaluation
    print("\n" + "="*70)
    print("Evaluating on test set...")
    print("="*70)
    
    predictor._learner._model.eval()
    metrics = predictor.evaluate(test_df, metrics=['iou', 'dice'])
    
    print(f"\nTest Results:")
    print(f"  IoU:  {metrics['iou']:.4f}")
    print(f"  DICE: {metrics['dice']:.4f}")
    
    with open(os.path.join(args.output_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f"IoU: {metrics['iou']:.4f}\n")
        f.write(f"DICE: {metrics['dice']:.4f}\n")
    
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)


if __name__ == '__main__':
    main()

