#!/bin/bash
# Simplified Joint Training - Ready to Run!
#
# This script trains a fixed layer configuration jointly with Conv-LoRA
# Simpler than per-image selection but solves the core distribution shift problem

set -e

echo "============================================================"
echo "Joint Training: Fixed Layer Configuration (Simplified)"
echo "============================================================"
echo ""

# Activate environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate conv-lora

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Configuration
TASK="isic2017"
WARMSTART_CKPT="AutogluonModels/ag-20251123_003958"
OUTPUT_DIR="joint_fixed_$(date +%Y%m%d_%H%M)"
JOINT_EPOCHS=15
BATCH_SIZE=4
LORA_LR=1e-4
SELECTOR_LR=5e-4
INITIAL_TEMP=1.0
FINAL_TEMP=0.1

echo "Configuration:"
echo "  Task: $TASK"
echo "  Warmstart model: $WARMSTART_CKPT"
echo "  Output: $OUTPUT_DIR"
echo "  Joint epochs: $JOINT_EPOCHS"
echo "  Temperature: $INITIAL_TEMP -> $FINAL_TEMP"
echo "  LoRA LR: $LORA_LR"
echo "  Selector LR: $SELECTOR_LR"
echo ""
echo "Key Features:"
echo "  ✓ Gumbel-Softmax for differentiable layer selection"
echo "  ✓ Joint optimization of Conv-LoRA and layer mask"
echo "  ✓ Temperature annealing for stable convergence"
echo "  ✓ Solves distribution shift problem of two-stage training"
echo ""

# Run training
python train_joint_fixed_mask.py \
    --task $TASK \
    --warmstart_ckpt $WARMSTART_CKPT \
    --output_dir $OUTPUT_DIR \
    --joint_epochs $JOINT_EPOCHS \
    --batch_size $BATCH_SIZE \
    --lora_lr $LORA_LR \
    --selector_lr $SELECTOR_LR \
    --initial_temp $INITIAL_TEMP \
    --final_temp $FINAL_TEMP \
    --device cuda

echo ""
echo "============================================================"
echo "Training Completed!"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR/"
echo ""
echo "Key outputs:"
echo "  - results.json: Performance metrics and learned mask"
echo "  - checkpoints/: Model checkpoints"
echo "  - training_history.json: Training curves"
echo ""
echo "The learned fixed layer configuration is saved in results.json"
echo ""

