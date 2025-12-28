#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/isic2017_env.sh"

python "${SCRIPT_DIR}/isic2017_prepare.py" \
  --src "/root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA/datasets/isic2017/isic2017" \
  --raw-root "${nnUNet_raw}" \
  --dataset-id 701 \
  --dataset-name "ISIC2017" \
  --include-val

nnUNetv2_plan_and_preprocess -d 701 --verify_dataset_integrity -c 2d

