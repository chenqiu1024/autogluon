#!/bin/bash
# 最佳精度 TTA 配置
# 目标：最大化 Dice/IoU 分数，不计算推理时间
# 适用于：论文提交、比赛、最终评测

set -e

TASK="isic2017"
CKPT_PATH=${1:-"AutogluonModels/ag-20251203_075302"}
OUTPUT_DIR=${2:-"outputs/eval_tta_best_accuracy"}

echo "========================================"
echo "最佳精度 TTA 配置"
echo "========================================"
echo "  Checkpoint: $CKPT_PATH"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "配置详情："
echo "  - Scales: [0.75, 1.0, 1.25]（多尺度）"
echo "  - Flips: [none, horizontal]"
echo "  - Rotations: [-10, 0, 10]（小角度）"
echo "  - Fusion: weighted_mean（scale=1.0 权重更高）"
echo "  - Post: 形态学平滑开启"
echo ""
echo "  总变换次数: 3 × 2 × 3 = 18 次/图"
echo "  预计时间: ~15-20s/图，600图约 2.5-3.5 小时"
echo "  预期提升: Dice +0.5~0.8%"
echo "========================================"
echo ""

python3 run_semantic_segmentation.py \
    --task $TASK \
    --eval \
    --ckpt_path $CKPT_PATH \
    --tta_enable \
    --tta_scales 0.75 1.0 1.25 \
    --tta_flips none horizontal \
    --tta_rotations -10 0 10 \
    --tta_fusion weighted_mean \
    --tta_morphology \
    --output_dir $OUTPUT_DIR

echo ""
echo "========================================"
echo "最佳精度 TTA 评估完成！"
echo "========================================"
echo ""
echo "结果："
cat $OUTPUT_DIR/metrics.txt
echo ""
echo "说明："
echo "  ✅ 已使用最全面的 TTA 配置"
echo "  ✅ 包含多尺度、翻转、旋转"
echo "  ✅ 使用加权融合"
echo "  ✅ 应用形态学后处理"
echo "  ✅ 这是我们能达到的最佳精度"
echo ""
echo "与基线对比建议："
echo "  1. 运行基线评估（无 TTA）"
echo "  2. 对比 Dice/IoU 提升"
echo "  3. 预期提升: +0.5~0.8%"
echo "========================================"

