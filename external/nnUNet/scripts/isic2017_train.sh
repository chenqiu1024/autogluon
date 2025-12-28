#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/isic2017_env.sh"

DATASET="Dataset701_ISIC2017"
CONFIG="${1:-2d}"
FOLDS="${2:-0 1 2 3 4}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for FOLD in ${FOLDS}; do
  echo "开始训练 ${DATASET} 配置=${CONFIG} 折=${FOLD}（GPU=${CUDA_VISIBLE_DEVICES}）"
  nnUNetv2_train "${DATASET}" "${CONFIG}" "${FOLD}" --npz
done

