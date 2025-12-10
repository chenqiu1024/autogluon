#!/usr/bin/env bash
set -euo pipefail

# Basic grid for TTA / multi-scale consistency-style experiments.
# Configure dataset/checkpoint via environment variables for unattended runs.
#   DATASET: task name (default: isic2017)
#   CKPT:    checkpoint path
#   OUT_ROOT: output directory root (default: outputs/consistency_grid)
#   SEED:    seed for TTA sampling (default: 42686693)

DATASET="${DATASET:-isic2017}"
CKPT="${CKPT:?Set CKPT to your checkpoint path}"
OUT_ROOT="${OUT_ROOT:-outputs/consistency_grid}"
SEED="${SEED:-42686693}"

mkdir -p "${OUT_ROOT}"

run() {
  local name="$1"; shift
  echo "==> Running ${name}"
  python3 run_semantic_segmentation.py \
    --task "${DATASET}" \
    --eval \
    --ckpt_path "${CKPT}" \
    --output_dir "${OUT_ROOT}/${name}" \
    --seed "${SEED}" \
    "$@"
}

# 1) 单尺度 + 轻量 TTA consistency（快速验证）
run tta_single_scale \
  --tta_enable \
  --tta_augment_types horizontal_flip rotate \
  --tta_rotations -10 0 10 \
  --tta_num_aug 4 \
  --tta_prob 0.8 \
  --tta_resize_method bilinear

# 2) 轻量多尺度 consistency（关注额外收益）
run tta_multiscale_light \
  --tta_enable \
  --tta_augment_types horizontal_flip rotate \
  --tta_rotations -10 0 10 \
  --tta_scale_ratios 0.8 1.0 1.2 \
  --tta_resize_method bilinear \
  --tta_multi_scale_weight 0.2 \
  --tta_prob 0.8

# 3) 稍强多尺度（bicubic + 更大比例）
run tta_multiscale_bicubic \
  --tta_enable \
  --tta_augment_types horizontal_flip rotate \
  --tta_rotations -10 0 10 \
  --tta_scale_ratios 0.75 1.0 1.25 \
  --tta_resize_method bicubic \
  --tta_multi_scale_weight 0.15 \
  --tta_prob 1.0

echo "All runs submitted. Outputs saved to ${OUT_ROOT}/*"
