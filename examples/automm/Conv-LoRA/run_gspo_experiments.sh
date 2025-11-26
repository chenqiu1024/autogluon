#!/bin/bash
#
# GSPO-ConvLoRA Experiment Script
#
# This script runs comparative experiments between baseline Conv-LoRA
# and GSPO-enhanced Conv-LoRA on ISIC2017 dataset.
#

set -e  # Exit on error

DATASET="isic2017"
SEED=42686693
RANK=3
EXPERT_NUM=8
NUM_GPUS=1
BATCH_SIZE=4

echo "=========================================="
echo "GSPO-ConvLoRA Experiments on ${DATASET}"
echo "=========================================="

# Create output directories
mkdir -p outputs/baseline_convlora
mkdir -p outputs/gspo_convlora_g4
mkdir -p outputs/gspo_convlora_g3
mkdir -p outputs/gspo_convlora_g6

# Experiment 1: Baseline Conv-LoRA
echo ""
echo "1. Running Baseline Conv-LoRA..."
echo "Output: outputs/baseline_convlora"
python run_semantic_segmentation.py \
    --task ${DATASET} \
    --seed ${SEED} \
    --rank ${RANK} \
    --expert_num ${EXPERT_NUM} \
    --num_gpus ${NUM_GPUS} \
    --batch_size ${BATCH_SIZE} \
    --output_dir outputs/baseline_convlora

echo "Baseline completed! Metrics saved to outputs/baseline_convlora/metrics.txt"

# Experiment 2: GSPO-ConvLoRA with group_size=4
echo ""
echo "2. Running GSPO-ConvLoRA (group_size=4)..."
echo "Output: outputs/gspo_convlora_g4"
python run_semantic_segmentation.py \
    --task ${DATASET} \
    --seed ${SEED} \
    --rank ${RANK} \
    --expert_num ${EXPERT_NUM} \
    --num_gpus ${NUM_GPUS} \
    --batch_size ${BATCH_SIZE} \
    --gspo_enable \
    --gspo_group_size 4 \
    --gspo_warmup_epochs 5 \
    --gspo_contrastive_weight 0.1 \
    --output_dir outputs/gspo_convlora_g4

echo "GSPO (g=4) completed! Metrics saved to outputs/gspo_convlora_g4/metrics.txt"

# Experiment 3: Ablation - GSPO with group_size=3
echo ""
echo "3. Running GSPO-ConvLoRA (group_size=3) [Ablation]..."
echo "Output: outputs/gspo_convlora_g3"
python run_semantic_segmentation.py \
    --task ${DATASET} \
    --seed ${SEED} \
    --rank ${RANK} \
    --expert_num ${EXPERT_NUM} \
    --num_gpus ${NUM_GPUS} \
    --batch_size ${BATCH_SIZE} \
    --gspo_enable \
    --gspo_group_size 3 \
    --gspo_warmup_epochs 5 \
    --gspo_contrastive_weight 0.1 \
    --output_dir outputs/gspo_convlora_g3

echo "GSPO (g=3) completed! Metrics saved to outputs/gspo_convlora_g3/metrics.txt"

# Experiment 4: Ablation - GSPO with group_size=6
echo ""
echo "4. Running GSPO-ConvLoRA (group_size=6) [Ablation]..."
echo "Output: outputs/gspo_convlora_g6"
python run_semantic_segmentation.py \
    --task ${DATASET} \
    --seed ${SEED} \
    --rank ${RANK} \
    --expert_num ${EXPERT_NUM} \
    --num_gpus ${NUM_GPUS} \
    --batch_size ${BATCH_SIZE} \
    --gspo_enable \
    --gspo_group_size 6 \
    --gspo_warmup_epochs 5 \
    --gspo_contrastive_weight 0.1 \
    --output_dir outputs/gspo_convlora_g6

echo "GSPO (g=6) completed! Metrics saved to outputs/gspo_convlora_g6/metrics.txt"

# Summary
echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Results summary:"
echo "  Baseline:       outputs/baseline_convlora/metrics.txt"
echo "  GSPO (g=4):     outputs/gspo_convlora_g4/metrics.txt"
echo "  GSPO (g=3):     outputs/gspo_convlora_g3/metrics.txt"
echo "  GSPO (g=6):     outputs/gspo_convlora_g6/metrics.txt"
echo ""
echo "To analyze results, run:"
echo "  python analyze_results.py"

