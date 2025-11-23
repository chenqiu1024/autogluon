#!/bin/bash
# Evaluation Script with Fixed Layer Mask Usage
# This script properly evaluates the RL policy using per-image layer masks

set -e

echo "============================================================"
echo "RL Policy Evaluation (Fixed Version)"
echo "============================================================"
echo ""

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate conv-lora

# Set offline mode to avoid HuggingFace timeout
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Configuration
TASK="isic2017"
CKPT_PATH="AutogluonModels/ag-20251123_003958"
POLICY_PATH="train_rl_layer-700ep-251123/checkpoints/best.pt"
OUTPUT_DIR="rl_evaluation_fixed_$(date +%Y%m%d_%H%M)"
DEVICE="cuda"

echo "Configuration:"
echo "  Task: $TASK"
echo "  Model: $CKPT_PATH"
echo "  Policy: $POLICY_PATH"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "Starting evaluation..."
echo "This will take approximately 10-15 minutes for 600 test images."
echo ""

# Run evaluation
python evaluate_rl_policy.py \
    --task $TASK \
    --ckpt_path $CKPT_PATH \
    --policy_path $POLICY_PATH \
    --output_dir $OUTPUT_DIR \
    --device $DEVICE

echo ""
echo "============================================================"
echo "Evaluation Complete!"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR/"
echo "  - evaluation_results.json"
echo "  - layer_activation_analysis.png"
echo ""
echo "View results:"
echo "  cat $OUTPUT_DIR/evaluation_results.json | python -m json.tool"
echo ""

