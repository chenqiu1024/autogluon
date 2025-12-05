#!/bin/bash

# ========================================
# 渐进式 TTA 测试脚本
# 从小到大逐步验证，快速发现问题
# ========================================

TASK="isic2017"
CKPT_PATH="AutogluonModels/ag-20251203_075302"
OUTPUT_DIR="outputs/tta_progressive_test"

echo "========================================"
echo "阶段 1: 单样本完整 TTA (1 image, 18 augmentations)"
echo "目的：快速验证 TTA pipeline 是否正常工作"
echo "========================================"
python run_semantic_segmentation.py \
    --task $TASK \
    --eval \
    --ckpt_path $CKPT_PATH \
    --tta_enable \
    --tta_scales 0.75 1.0 1.25 \
    --tta_flips none horizontal \
    --tta_rotations -10 0 10 \
    --tta_fusion weighted_mean \
    --tta_morphology \
    --output_dir ${OUTPUT_DIR}/stage1 \
    --quick_test 1

if [ $? -ne 0 ]; then
    echo "❌ 阶段 1 失败！请先修复问题"
    exit 1
fi
echo "✅ 阶段 1 成功"

echo ""
echo "========================================"
echo "阶段 2: 小批量测试 (10 images)"
echo "目的：验证多样本处理和内存管理"
echo "========================================"
python run_semantic_segmentation.py \
    --task $TASK \
    --eval \
    --ckpt_path $CKPT_PATH \
    --tta_enable \
    --tta_scales 0.75 1.0 1.25 \
    --tta_flips none horizontal \
    --tta_rotations -10 0 10 \
    --tta_fusion weighted_mean \
    --tta_morphology \
    --output_dir ${OUTPUT_DIR}/stage2 \
    --quick_test 10

if [ $? -ne 0 ]; then
    echo "❌ 阶段 2 失败！请先修复问题"
    exit 1
fi
echo "✅ 阶段 2 成功"

echo ""
echo "========================================"
echo "阶段 3: 中等批量测试带缓存 (50 images)"
echo "目的：验证断点续传功能"
echo "========================================"
python run_semantic_segmentation.py \
    --task $TASK \
    --eval \
    --ckpt_path $CKPT_PATH \
    --tta_enable \
    --tta_scales 0.75 1.0 1.25 \
    --tta_flips none horizontal \
    --tta_rotations -10 0 10 \
    --tta_fusion weighted_mean \
    --tta_morphology \
    --tta_cache_dir ${OUTPUT_DIR}/cache_stage3 \
    --output_dir ${OUTPUT_DIR}/stage3 \
    --quick_test 50

if [ $? -ne 0 ]; then
    echo "❌ 阶段 3 失败！请先修复问题"
    exit 1
fi
echo "✅ 阶段 3 成功"

echo ""
echo "========================================"
echo "阶段 4: 测试断点续传 (重新运行 stage 3)"
echo "目的：验证从缓存恢复功能"
echo "========================================"
echo "重新运行相同的命令，应该跳过所有已处理的图像..."
python run_semantic_segmentation.py \
    --task $TASK \
    --eval \
    --ckpt_path $CKPT_PATH \
    --tta_enable \
    --tta_scales 0.75 1.0 1.25 \
    --tta_flips none horizontal \
    --tta_rotations -10 0 10 \
    --tta_fusion weighted_mean \
    --tta_morphology \
    --tta_cache_dir ${OUTPUT_DIR}/cache_stage3 \
    --output_dir ${OUTPUT_DIR}/stage4 \
    --quick_test 50

if [ $? -ne 0 ]; then
    echo "❌ 阶段 4 失败！"
    exit 1
fi
echo "✅ 阶段 4 成功 - 断点续传验证通过！"

echo ""
echo "========================================"
echo "🎉 所有阶段测试成功！"
echo "========================================"
echo ""
echo "现在可以安全地运行完整评估："
echo "python run_semantic_segmentation.py \\"
echo "    --task $TASK \\"
echo "    --eval \\"
echo "    --ckpt_path $CKPT_PATH \\"
echo "    --tta_enable \\"
echo "    --tta_scales 0.75 1.0 1.25 \\"
echo "    --tta_flips none horizontal \\"
echo "    --tta_rotations -10 0 10 \\"
echo "    --tta_fusion weighted_mean \\"
echo "    --tta_morphology \\"
echo "    --tta_cache_dir outputs/tta_full_cache \\"
echo "    --output_dir outputs/tta_full"

