#!/bin/bash
# TTA 功能测试脚本
# 用于验证 TTA 实现的正确性

set -e  # Exit on error

echo "========================================"
echo "TTA (Test-Time Augmentation) 功能测试"
echo "========================================"

# 检查必要文件是否存在
if [ ! -f "run_semantic_segmentation.py" ]; then
    echo "错误：找不到 run_semantic_segmentation.py"
    exit 1
fi

# 设置变量
TASK="isic2017"
DATASET_DIR="datasets/${TASK}/${TASK}"
OUTPUT_BASE="outputs/tta_tests"

# 检查是否有可用的模型checkpoint
echo ""
echo "请输入模型checkpoint路径（例如：AutogluonModels/ag-20251129_083619）："
read CKPT_PATH

if [ ! -d "$CKPT_PATH" ]; then
    echo "错误：checkpoint 目录不存在: $CKPT_PATH"
    exit 1
fi

echo ""
echo "找到checkpoint: $CKPT_PATH"
echo ""

# 测试 1：基线评估（无 TTA）
echo "========================================" 
echo "测试 1: 基线评估（无 TTA）"
echo "========================================"
python3 run_semantic_segmentation.py \
    --task $TASK \
    --eval \
    --ckpt_path $CKPT_PATH \
    --output_dir $OUTPUT_BASE/baseline

echo ""
echo "基线结果："
cat $OUTPUT_BASE/baseline/metrics.txt
echo ""

# 测试 2：基础 TTA（6 次推理）
echo "========================================" 
echo "测试 2: 基础 TTA（6 次推理）"
echo "  - Scales: [0.75, 1.0, 1.25]"
echo "  - Flips: [none, horizontal]"
echo "  - Rotations: [0]"
echo "========================================"
python3 run_semantic_segmentation.py \
    --task $TASK \
    --eval \
    --ckpt_path $CKPT_PATH \
    --tta_enable \
    --output_dir $OUTPUT_BASE/tta_basic

echo ""
echo "基础 TTA 结果："
cat $OUTPUT_BASE/tta_basic/metrics.txt
echo ""

# 测试 3：进阶 TTA（18 次推理）
echo "========================================" 
echo "测试 3: 进阶 TTA（18 次推理）"
echo "  - Scales: [0.75, 1.0, 1.25]"
echo "  - Flips: [none, horizontal]"
echo "  - Rotations: [-10, 0, 10]"
echo "  - Fusion: weighted_mean"
echo "========================================"
python3 run_semantic_segmentation.py \
    --task $TASK \
    --eval \
    --ckpt_path $CKPT_PATH \
    --tta_enable \
    --tta_scales 0.75 1.0 1.25 \
    --tta_flips none horizontal \
    --tta_rotations -10 0 10 \
    --tta_fusion weighted_mean \
    --output_dir $OUTPUT_BASE/tta_advanced

echo ""
echo "进阶 TTA 结果："
cat $OUTPUT_BASE/tta_advanced/metrics.txt
echo ""

# 测试 4：TTA + 形态学后处理
echo "========================================" 
echo "测试 4: TTA + 形态学后处理"
echo "  - Scales: [0.75, 1.0, 1.25]"
echo "  - Flips: [none, horizontal]"
echo "  - Morphology: enabled"
echo "========================================"
python3 run_semantic_segmentation.py \
    --task $TASK \
    --eval \
    --ckpt_path $CKPT_PATH \
    --tta_enable \
    --tta_scales 0.75 1.0 1.25 \
    --tta_flips none horizontal \
    --tta_morphology \
    --output_dir $OUTPUT_BASE/tta_morphology

echo ""
echo "TTA + 形态学后处理结果："
cat $OUTPUT_BASE/tta_morphology/metrics.txt
echo ""

# 结果对比
echo "========================================"
echo "结果对比总结"
echo "========================================"
echo ""
echo "1. 基线（无 TTA）："
cat $OUTPUT_BASE/baseline/metrics.txt
echo ""
echo "2. 基础 TTA（6 次）："
cat $OUTPUT_BASE/tta_basic/metrics.txt
echo ""
echo "3. 进阶 TTA（18 次）："
cat $OUTPUT_BASE/tta_advanced/metrics.txt
echo ""
echo "4. TTA + 形态学："
cat $OUTPUT_BASE/tta_morphology/metrics.txt
echo ""

echo "========================================"
echo "所有测试完成！"
echo "结果已保存到: $OUTPUT_BASE/"
echo "========================================"

# 可选：生成对比表格
echo ""
echo "生成对比表格..."

cat > $OUTPUT_BASE/comparison.md << EOF
# TTA 测试结果对比

## 测试配置

- **任务**: $TASK
- **模型**: $CKPT_PATH
- **测试时间**: $(date)

## 结果对比

| 配置 | 推理次数 | 结果 |
|------|---------|------|
| 基线（无 TTA） | 1 | $(cat $OUTPUT_BASE/baseline/metrics.txt) |
| 基础 TTA | 6 | $(cat $OUTPUT_BASE/tta_basic/metrics.txt) |
| 进阶 TTA | 18 | $(cat $OUTPUT_BASE/tta_advanced/metrics.txt) |
| TTA + 形态学 | 6 | $(cat $OUTPUT_BASE/tta_morphology/metrics.txt) |

## 结论

观察各配置的性能提升情况，选择最适合的 TTA 配置。

EOF

echo "对比表格已保存到: $OUTPUT_BASE/comparison.md"
cat $OUTPUT_BASE/comparison.md

