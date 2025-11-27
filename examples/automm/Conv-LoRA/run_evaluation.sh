#!/bin/bash

# RLOO 模型评估便捷脚本
# 用法: bash run_evaluation.sh

set -e  # 出错时退出

# 配置
TASK="isic2017"
BASE_CKPT="AutogluonModels/ag-20251126_062717"
RLOO_OUTPUT="outputs/rloo/isic2017/$(date +%Y%m%d%H%M%S)"

echo "=========================================="
echo "RLOO 模型评估"
echo "=========================================="
echo "任务: $TASK"
echo "基础模型: $BASE_CKPT"
echo "RLOO 输出: $RLOO_OUTPUT"
echo "=========================================="
echo ""

# 激活环境
echo "激活 conda 环境: conv-lora"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate conv-lora

# 切换到工作目录
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA

# 运行批量对比
echo ""
echo "开始批量对比评估..."
echo ""

python compare_rloo_models.py \
  --rloo_output_dir "$RLOO_OUTPUT" \
  --base_checkpoint_path "$BASE_CKPT" \
  --task "$TASK"

echo ""
echo "=========================================="
echo "评估完成！"
echo "=========================================="
echo ""
echo "结果保存在: $RLOO_OUTPUT/evaluation_results/"
echo ""
echo "查看详细结果:"
echo "  ls -lh $RLOO_OUTPUT/evaluation_results/"
echo ""

