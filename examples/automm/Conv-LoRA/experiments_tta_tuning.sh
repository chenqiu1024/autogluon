#!/bin/bash
# TTA 超参数调节批量实验脚本
# 用法: bash experiments_tta_tuning.sh [stage]
# stage: all, 1, 2, 3, 4, quick (默认: quick)

set -e  # 遇到错误立即退出

# ==================== 配置区 ====================
CKPT="AutogluonModels/ag-20251208_054753"
TASK="isic2017"
OUTPUT_BASE="outputs"
QUICK_TEST=""  # 设置为 "--quick_test 50" 可快速验证

# 解析命令行参数
STAGE="${1:-quick}"

echo "=========================================="
echo "TTA 超参数调节实验"
echo "任务: $TASK"
echo "检查点: $CKPT"
echo "执行阶段: $STAGE"
echo "=========================================="
echo ""

# ==================== Baseline ====================
run_baseline() {
  echo "[Baseline] 不使用 TTA..."
  python run_semantic_segmentation.py \
    --task $TASK \
    --eval \
    --ckpt_path $CKPT \
    --output_dir $OUTPUT_BASE/baseline \
    $QUICK_TEST
  echo "[Baseline] 完成！"
  echo ""
}

# ==================== 阶段 1: Multi-scale 配置优化 ====================
run_stage1() {
  echo "=========================================="
  echo "阶段 1: Multi-scale 配置优化"
  echo "=========================================="
  
  # 实验 1.1: 不同 scales + bilinear
  for scales in "0.75 1.0 1.25" "0.8 1.0 1.2" "0.9 1.0 1.1"; do
    scales_name=$(echo $scales | tr ' ' '_')
    exp_name="exp_1_scales_${scales_name}_bilinear"
    echo "[1.1] $exp_name"
    
    python run_semantic_segmentation.py \
      --task $TASK \
      --eval \
      --ckpt_path $CKPT \
      --tta_enable \
      --tta_scales $scales \
      --tta_flips none horizontal \
      --tta_rotations 0 \
      --tta_resize_method bilinear \
      --tta_fusion mean \
      --output_dir $OUTPUT_BASE/$exp_name \
      $QUICK_TEST
    
    echo "[1.1] $exp_name 完成！"
    echo ""
  done
  
  # 实验 1.2: 最佳 scales + bicubic
  for scales in "0.75 1.0 1.25" "0.8 1.0 1.2"; do
    scales_name=$(echo $scales | tr ' ' '_')
    exp_name="exp_1_scales_${scales_name}_bicubic"
    echo "[1.2] $exp_name"
    
    python run_semantic_segmentation.py \
      --task $TASK \
      --eval \
      --ckpt_path $CKPT \
      --tta_enable \
      --tta_scales $scales \
      --tta_flips none horizontal \
      --tta_rotations 0 \
      --tta_resize_method bicubic \
      --tta_fusion mean \
      --output_dir $OUTPUT_BASE/$exp_name \
      $QUICK_TEST
    
    echo "[1.2] $exp_name 完成！"
    echo ""
  done
  
  echo "阶段 1 完成！"
  echo ""
}

# ==================== 阶段 2: TTA Augmentation 配置优化 ====================
run_stage2() {
  echo "=========================================="
  echo "阶段 2: TTA Augmentation 配置优化"
  echo "=========================================="
  
  # 假设阶段 1 最佳配置: scales=[0.75, 1.0, 1.25], resize_method=bilinear
  BEST_SCALES="0.75 1.0 1.25"
  BEST_RESIZE="bilinear"
  
  # 实验 2.1: 不同翻转策略
  echo "[2.1] 测试翻转策略..."
  
  # 仅 horizontal flip
  python run_semantic_segmentation.py \
    --task $TASK \
    --eval \
    --ckpt_path $CKPT \
    --tta_enable \
    --tta_scales $BEST_SCALES \
    --tta_flips none horizontal \
    --tta_rotations 0 \
    --tta_resize_method $BEST_RESIZE \
    --tta_fusion mean \
    --output_dir $OUTPUT_BASE/exp_2_flip_h \
    $QUICK_TEST
  
  # horizontal + vertical flips
  python run_semantic_segmentation.py \
    --task $TASK \
    --eval \
    --ckpt_path $CKPT \
    --tta_enable \
    --tta_scales $BEST_SCALES \
    --tta_flips none horizontal vertical \
    --tta_rotations 0 \
    --tta_resize_method $BEST_RESIZE \
    --tta_fusion mean \
    --output_dir $OUTPUT_BASE/exp_2_flip_hv \
    $QUICK_TEST
  
  echo "[2.1] 翻转策略测试完成！"
  echo ""
  
  # 实验 2.2: 测试旋转增强
  echo "[2.2] 测试旋转增强..."
  
  # ±10° 旋转
  python run_semantic_segmentation.py \
    --task $TASK \
    --eval \
    --ckpt_path $CKPT \
    --tta_enable \
    --tta_scales $BEST_SCALES \
    --tta_flips none horizontal \
    --tta_rotations -10 0 10 \
    --tta_resize_method $BEST_RESIZE \
    --tta_fusion mean \
    --output_dir $OUTPUT_BASE/exp_2_rot_10 \
    $QUICK_TEST
  
  # ±5° 旋转
  python run_semantic_segmentation.py \
    --task $TASK \
    --eval \
    --ckpt_path $CKPT \
    --tta_enable \
    --tta_scales $BEST_SCALES \
    --tta_flips none horizontal \
    --tta_rotations -5 0 5 \
    --tta_resize_method $BEST_RESIZE \
    --tta_fusion mean \
    --output_dir $OUTPUT_BASE/exp_2_rot_5 \
    $QUICK_TEST
  
  echo "[2.2] 旋转增强测试完成！"
  echo ""
  echo "阶段 2 完成！"
  echo ""
}

# ==================== 阶段 3: 后处理参数优化 ====================
run_stage3() {
  echo "=========================================="
  echo "阶段 3: 后处理参数优化"
  echo "=========================================="
  
  # 假设阶段 2 最佳配置
  BEST_SCALES="0.75 1.0 1.25"
  BEST_RESIZE="bilinear"
  BEST_FLIPS="none horizontal"
  BEST_ROTS="0"
  
  # 实验 3.1: 阈值调节
  echo "[3.1] 测试不同阈值..."
  for thresh in 0.4 0.45 0.55; do
    exp_name="exp_3_thresh_$(echo $thresh | tr '.' '')"
    echo "  - threshold=$thresh"
    
    python run_semantic_segmentation.py \
      --task $TASK \
      --eval \
      --ckpt_path $CKPT \
      --tta_enable \
      --tta_scales $BEST_SCALES \
      --tta_flips $BEST_FLIPS \
      --tta_rotations $BEST_ROTS \
      --tta_resize_method $BEST_RESIZE \
      --tta_threshold $thresh \
      --tta_fusion mean \
      --output_dir $OUTPUT_BASE/$exp_name \
      $QUICK_TEST
    
    echo "  - $exp_name 完成！"
  done
  echo ""
  
  # 实验 3.2: 形态学后处理
  echo "[3.2] 测试形态学后处理..."
  
  python run_semantic_segmentation.py \
    --task $TASK \
    --eval \
    --ckpt_path $CKPT \
    --tta_enable \
    --tta_scales $BEST_SCALES \
    --tta_flips $BEST_FLIPS \
    --tta_rotations $BEST_ROTS \
    --tta_resize_method $BEST_RESIZE \
    --tta_threshold 0.5 \
    --tta_morphology \
    --tta_fusion mean \
    --output_dir $OUTPUT_BASE/exp_3_morphology \
    $QUICK_TEST
  
  echo "[3.2] 形态学后处理测试完成！"
  echo ""
  
  # 实验 3.3: 小区域去除
  echo "[3.3] 测试小区域去除..."
  for min_area in 0.0005 0.002; do
    exp_name="exp_3_minarea_$(echo $min_area | tr '.' '')"
    echo "  - min_area_ratio=$min_area"
    
    python run_semantic_segmentation.py \
      --task $TASK \
      --eval \
      --ckpt_path $CKPT \
      --tta_enable \
      --tta_scales $BEST_SCALES \
      --tta_flips $BEST_FLIPS \
      --tta_rotations $BEST_ROTS \
      --tta_resize_method $BEST_RESIZE \
      --tta_threshold 0.5 \
      --tta_min_area $min_area \
      --tta_fusion mean \
      --output_dir $OUTPUT_BASE/$exp_name \
      $QUICK_TEST
    
    echo "  - $exp_name 完成！"
  done
  echo ""
  
  echo "阶段 3 完成！"
  echo ""
}

# ==================== 阶段 4: 最优组合验证 ====================
run_stage4() {
  echo "=========================================="
  echo "阶段 4: 最优组合验证"
  echo "=========================================="
  
  # 注意：这里使用示例配置，实际应根据前面阶段的结果调整
  echo "[4.1] 最优组合 (示例配置)..."
  python run_semantic_segmentation.py \
    --task $TASK \
    --eval \
    --ckpt_path $CKPT \
    --tta_enable \
    --tta_scales 0.75 1.0 1.25 \
    --tta_flips none horizontal \
    --tta_rotations 0 \
    --tta_resize_method bilinear \
    --tta_threshold 0.5 \
    --tta_morphology \
    --tta_fusion mean \
    --output_dir $OUTPUT_BASE/exp_4_best_config \
    $QUICK_TEST
  
  echo "[4.2] 最优组合 + weighted_mean..."
  python run_semantic_segmentation.py \
    --task $TASK \
    --eval \
    --ckpt_path $CKPT \
    --tta_enable \
    --tta_scales 0.75 1.0 1.25 \
    --tta_flips none horizontal \
    --tta_rotations 0 \
    --tta_resize_method bilinear \
    --tta_threshold 0.5 \
    --tta_morphology \
    --tta_fusion weighted_mean \
    --output_dir $OUTPUT_BASE/exp_4_best_config_weighted \
    $QUICK_TEST
  
  echo "阶段 4 完成！"
  echo ""
}

# ==================== 快速验证模式 ====================
run_quick() {
  echo "=========================================="
  echo "快速验证模式"
  echo "=========================================="
  
  echo "[Quick] 配置 A: 轻量级 TTA..."
  python run_semantic_segmentation.py \
    --task $TASK \
    --eval \
    --ckpt_path $CKPT \
    --tta_enable \
    --tta_scales 0.75 1.0 1.25 \
    --tta_flips none horizontal \
    --tta_rotations 0 \
    --tta_resize_method bilinear \
    --tta_threshold 0.5 \
    --tta_fusion mean \
    --output_dir $OUTPUT_BASE/quick_config_a \
    --quick_test 50
  
  echo "[Quick] 配置 B: 中等 TTA..."
  python run_semantic_segmentation.py \
    --task $TASK \
    --eval \
    --ckpt_path $CKPT \
    --tta_enable \
    --tta_scales 0.75 1.0 1.25 \
    --tta_flips none horizontal \
    --tta_rotations 0 \
    --tta_resize_method bicubic \
    --tta_threshold 0.5 \
    --tta_morphology \
    --tta_fusion mean \
    --output_dir $OUTPUT_BASE/quick_config_b \
    --quick_test 50
  
  echo "[Quick] 配置 C: 完整 TTA..."
  python run_semantic_segmentation.py \
    --task $TASK \
    --eval \
    --ckpt_path $CKPT \
    --tta_enable \
    --tta_scales 0.75 1.0 1.25 \
    --tta_flips none horizontal vertical \
    --tta_rotations -10 0 10 \
    --tta_resize_method bicubic \
    --tta_threshold 0.5 \
    --tta_morphology \
    --tta_fusion weighted_mean \
    --output_dir $OUTPUT_BASE/quick_config_c \
    --quick_test 50
  
  echo "快速验证模式完成！"
  echo ""
}

# ==================== 主流程 ====================
case $STAGE in
  all)
    echo "执行所有阶段..."
    run_baseline
    run_stage1
    run_stage2
    run_stage3
    run_stage4
    ;;
  1)
    run_baseline
    run_stage1
    ;;
  2)
    run_stage2
    ;;
  3)
    run_stage3
    ;;
  4)
    run_stage4
    ;;
  quick)
    run_quick
    ;;
  *)
    echo "错误: 未知阶段 '$STAGE'"
    echo "用法: $0 [all|1|2|3|4|quick]"
    exit 1
    ;;
esac

echo "=========================================="
echo "所有实验完成！"
echo "结果保存在: $OUTPUT_BASE/"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 查看各实验的 metrics.txt 文件"
echo "2. 对比 IoU 和 Dice 指标"
echo "3. 选择最佳配置用于最终评估"
