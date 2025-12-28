#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/isic2017_env.sh"

INPUT="${1:-${nnUNet_raw}/Dataset701_ISIC2017/imagesTs}"
OUTPUT="${2:-${nnUNet_results}/Dataset701_ISIC2017/predictions_2d}"
CONFIG="${3:-2d}"
FOLDS="${4:-0 1 2 3 4}"

echo "输入: ${INPUT}"
echo "输出: ${OUTPUT}"
echo "配置: ${CONFIG}, 折: ${FOLDS}"

nnUNetv2_predict \
  -i "${INPUT}" \
  -o "${OUTPUT}" \
  -d Dataset701_ISIC2017 \
  -c "${CONFIG}" \
  -f ${FOLDS}

POSTPROC_DIR="${nnUNet_results}/Dataset701_ISIC2017/nnUNetTrainer__nnUNetPlans__${CONFIG}"
PP_FILE="${POSTPROC_DIR}/postprocessing.pkl"
PLANS_JSON="${POSTPROC_DIR}/plans.json"
DATASET_JSON="${POSTPROC_DIR}/dataset.json"

if [ -f "${PP_FILE}" ]; then
  echo "检测到 postprocessing.pkl，正在应用后处理..."
  nnUNetv2_apply_postprocessing \
    -i "${OUTPUT}" \
    -o "${OUTPUT}-postproc" \
    --pp_pkl_file "${PP_FILE}" \
    -plans_json "${PLANS_JSON}" \
    -dataset_json "${DATASET_JSON}"
  echo "后处理完成，结果输出至 ${OUTPUT}-postproc"
else
  echo "未找到 postprocessing.pkl，跳过后处理。"
fi

