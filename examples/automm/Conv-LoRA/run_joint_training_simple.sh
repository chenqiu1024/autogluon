#!/bin/bash
# Simplified Joint Training Script
# 
# This script runs the complete joint training workflow with recommended settings

set -e

echo "============================================================"
echo "Joint Training: Conv-LoRA + Layer Selection Policy"
echo "============================================================"
echo ""

# Configuration
TASK="isic2017"
OUTPUT_DIR="joint_training_$(date +%Y%m%d_%H%M)"
WARMSTART_EPOCHS=3
TOTAL_EPOCHS=20
BATCH_SIZE=4
RANK=3
EXPERT_NUM=8

# Optional: use existing warmstart checkpoint
WARMSTART_CKPT=""  # Leave empty to train from scratch
# WARMSTART_CKPT="AutogluonModels/ag-20251123_003958"  # Uncomment to use

echo "Configuration:"
echo "  Task: $TASK"
echo "  Output: $OUTPUT_DIR"
echo "  Warmstart epochs: $WARMSTART_EPOCHS"
echo "  Joint training epochs: $((TOTAL_EPOCHS - WARMSTART_EPOCHS))"
echo "  Total epochs: $TOTAL_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate conv-lora

# Set offline mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Run training
if [ -z "$WARMSTART_CKPT" ]; then
    echo "Training from scratch..."
    python run_joint_training.py \
        --task $TASK \
        --output_dir $OUTPUT_DIR \
        --warmstart_epochs $WARMSTART_EPOCHS \
        --total_epochs $TOTAL_EPOCHS \
        --batch_size $BATCH_SIZE \
        --rank $RANK \
        --expert_num $EXPERT_NUM \
        --lora_lr 1e-4 \
        --selector_lr 5e-4 \
        --initial_temperature 1.0 \
        --final_temperature 0.1 \
        --seed 42
else
    echo "Using warmstart checkpoint: $WARMSTART_CKPT"
    python run_joint_training.py \
        --task $TASK \
        --warmstart_ckpt $WARMSTART_CKPT \
        --output_dir $OUTPUT_DIR \
        --warmstart_epochs $WARMSTART_EPOCHS \
        --total_epochs $TOTAL_EPOCHS \
        --batch_size $BATCH_SIZE \
        --lora_lr 1e-4 \
        --selector_lr 5e-4 \
        --initial_temperature 1.0 \
        --final_temperature 0.1 \
        --seed 42
fi

echo ""
echo "============================================================"
echo "Training completed!"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "To evaluate:"
echo "  python run_semantic_segmentation.py \\"
echo "    --task $TASK \\"
echo "    --ckpt_path $OUTPUT_DIR \\"
echo "    --eval"
echo ""

