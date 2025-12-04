#!/bin/bash
# 快速 TTA 配置（无 scale，只用 flip + rotate）
# 速度：~2-3s/图，比完整 TTA 快 3 倍

set -e

TASK="isic2017"
CKPT_PATH=${1:-"AutogluonModels/ag-20251203_075302"}
OUTPUT_DIR=${2:-"outputs/eval_tta_fast"}

echo "========================================"
echo "快速 TTA 评估（无 scale，只用 flip + rotate）"
echo "========================================"
echo "  Checkpoint: $CKPT_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  预计速度: ~2-3s/图"
echo "  600 图总时间: ~20-30 分钟"
echo "========================================"
echo ""

python3 run_semantic_segmentation.py \
    --task $TASK \
    --eval \
    --ckpt_path $CKPT_PATH \
    --tta_enable \
    --tta_scales 1.0 \
    --tta_flips none horizontal \
    --tta_rotations -10 0 10 \
    --tta_fusion weighted_mean \
    --output_dir $OUTPUT_DIR

echo ""
echo "========================================"
echo "快速 TTA 评估完成！"
echo "========================================"
echo "结果："
cat $OUTPUT_DIR/metrics.txt
echo ""
echo "说明："
echo "  - 本配置移除了多尺度变换（scale）"
echo "  - 只使用 flip + rotate，速度快 3 倍"
echo "  - 精度略低于完整 TTA，但仍有提升"
echo "========================================"

