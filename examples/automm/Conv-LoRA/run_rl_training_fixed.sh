#!/bin/bash
# Fixed RL Training Script with Root Cause Fixes
#
# Key Fixes:
# 1. init_bias: 2.0 → 0.0 (start from 50% instead of 88%)
# 2. entropy_coef: 0.01 → 0.05 (more exploration)
# 3. Enhanced monitoring (probability distribution, deterministic vs stochastic)
# 4. Better logging

set -e

echo "============================================================"
echo "RL Training for Conv-LoRA Layer Selection (FIXED VERSION)"
echo "============================================================"
echo ""

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate conv-lora

# Set offline mode
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Configuration
TASK="isic2017"
CKPT_PATH="AutogluonModels/ag-20251123_003958"
OUTPUT_DIR="rl_training_fixed_$(date +%Y%m%d_%H%M)"
NUM_EPISODES=500
BATCH_SIZE=8
EVAL_SUBSET_SIZE=100
LEARNING_RATE=1e-4
ENTROPY_COEF=0.05
INIT_BIAS=0.0
DEVICE="cuda"

echo "Configuration:"
echo "  Task: $TASK"
echo "  Model: $CKPT_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Episodes: $NUM_EPISODES"
echo "  Batch size: $BATCH_SIZE"
echo "  Init bias: $INIT_BIAS (sigmoid = 0.500)"
echo "  Entropy coef: $ENTROPY_COEF"
echo ""
echo "Key Fixes:"
echo "  ✓ Lowered init_bias from 2.0 to 0.0"
echo "  ✓ Increased entropy_coef from 0.01 to 0.05"
echo "  ✓ Enhanced monitoring with probability distribution"
echo "  ✓ Track both stochastic and deterministic modes"
echo ""
echo "Starting training..."
echo "This will take approximately $(echo "$NUM_EPISODES * 27 / 3600" | bc) hours."
echo ""

# Run training
python train_rl_layer_selection.py \
    --task $TASK \
    --ckpt_path $CKPT_PATH \
    --output_dir $OUTPUT_DIR \
    --num_episodes $NUM_EPISODES \
    --batch_size $BATCH_SIZE \
    --eval_subset_size $EVAL_SUBSET_SIZE \
    --learning_rate $LEARNING_RATE \
    --entropy_coef $ENTROPY_COEF \
    --init_bias $INIT_BIAS \
    --warmup_steps 50 \
    --save_freq 50 \
    --eval_freq 25 \
    --reward_metric iou \
    --device $DEVICE

echo ""
echo "============================================================"
echo "Training Complete!"
echo "============================================================"
echo "Results saved to: $OUTPUT_DIR/"
echo "  - checkpoints/best.pt"
echo "  - checkpoints/episode_*.pt"
echo "  - logs/ (TensorBoard)"
echo ""
echo "Monitor training:"
echo "  tensorboard --logdir $OUTPUT_DIR/logs --port 6006"
echo ""
echo "Evaluate policy:"
echo "  python evaluate_rl_policy.py \\"
echo "    --task isic2017 \\"
echo "    --ckpt_path $CKPT_PATH \\"
echo "    --policy_path $OUTPUT_DIR/checkpoints/best.pt \\"
echo "    --output_dir evaluation_results"
echo ""

