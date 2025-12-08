#!/bin/bash
# Example: Training with Custom Learning Rate
# This script demonstrates how to use the --lr parameter to customize learning rate

# Example 1: Default learning rate (auto-selected based on task)
# For isic2017: 1e-4
echo "Example 1: Using default learning rate..."
python3 run_semantic_segmentation.py \
  --task isic2017 \
  --decoder_attn_lora_enable \
  --decoder_attn_lora_r 8 \
  --num_gpus 1 \
  --per_gpu_batch_size 1 \
  --batch_size 4 \
  --quick_test 10 \
  --output_dir outputs/example_default_lr

# Example 2: Higher learning rate (recommended for LoRA)
# 2e-4 or 3e-4 often works better for LoRA fine-tuning
echo "Example 2: Using higher learning rate (3e-4)..."
python3 run_semantic_segmentation.py \
  --task isic2017 \
  --decoder_attn_lora_enable \
  --decoder_attn_lora_r 8 \
  --lr 3e-4 \
  --num_gpus 1 \
  --per_gpu_batch_size 1 \
  --batch_size 4 \
  --quick_test 10 \
  --output_dir outputs/example_lr_3e4

# Example 3: Very aggressive learning rate
echo "Example 3: Using aggressive learning rate (5e-4)..."
python3 run_semantic_segmentation.py \
  --task isic2017 \
  --decoder_attn_lora_enable \
  --decoder_attn_lora_r 8 \
  --lr 5e-4 \
  --num_gpus 1 \
  --per_gpu_batch_size 1 \
  --batch_size 4 \
  --quick_test 10 \
  --output_dir outputs/example_lr_5e4

# Example 4: Conservative learning rate (for stability)
echo "Example 4: Using conservative learning rate (5e-5)..."
python3 run_semantic_segmentation.py \
  --task isic2017 \
  --decoder_attn_lora_enable \
  --decoder_attn_lora_r 8 \
  --lr 5e-5 \
  --num_gpus 1 \
  --per_gpu_batch_size 1 \
  --batch_size 4 \
  --quick_test 10 \
  --output_dir outputs/example_lr_5e5

echo "All examples completed!"
echo "Compare results to find the best learning rate for your task."

