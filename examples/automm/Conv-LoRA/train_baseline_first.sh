#!/bin/bash
# Train baseline Conv-LoRA model (required before RL experiments)

TASK=${1:-isic2017}
OUTPUT_DIR=${2:-outputs_baseline}

echo "Training baseline Conv-LoRA model for task: $TASK"
echo "Output directory: $OUTPUT_DIR"
echo ""

python run_semantic_segmentation.py \
  --task $TASK \
  --rank 3 \
  --expert_num 8 \
  --num_gpus 1 \
  --per_gpu_batch_size 1 \
  --batch_size 4 \
  --output_dir $OUTPUT_DIR

echo ""
echo "Baseline training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo ""
echo "Next step: Run BC warmstart"
echo "  python rl_bc_routing_policy.py --task $TASK --model_path $OUTPUT_DIR --output_dir bc_warmstart/"

