#!/bin/bash

# S-SAM SVD 微调消融实验批处理脚本
# 用途：系统性对比 Conv-LoRA、GSPO、Adapter 和 SVD 微调的性能贡献

TASK="isic2017"
BASE_PARAMS="--task ${TASK} --rank 3 --expert_num 8 --num_gpus 1"

echo "======================================"
echo "S-SAM SVD 消融实验批处理"
echo "数据集: ${TASK}"
echo "======================================"
echo ""

# Exp-1: Baseline (Conv-LoRA only)
echo "[1/5] 实验 1: Conv-LoRA 基线"
echo "配置: Conv-LoRA (r=3, M=8)"
echo "--------------------------------------"
python3 run_semantic_segmentation.py ${BASE_PARAMS} \
    --output_dir outputs/exp1_baseline
echo "完成！结果保存在 outputs/exp1_baseline/metrics.txt"
echo ""

# Exp-2: Conv-LoRA + GSPO
echo "[2/5] 实验 2: Conv-LoRA + GSPO"
echo "配置: Conv-LoRA + GSPO 强化学习"
echo "--------------------------------------"
python3 run_semantic_segmentation.py ${BASE_PARAMS} \
    --gspo_enable \
    --output_dir outputs/exp2_gspo
echo "完成！结果保存在 outputs/exp2_gspo/metrics.txt"
echo ""

# Exp-3: Conv-LoRA + GSPO + Adapter
echo "[3/5] 实验 3: Conv-LoRA + GSPO + Encoder-Adapter"
echo "配置: Conv-LoRA + GSPO + Encoder-Adapter (dim=64)"
echo "--------------------------------------"
python3 run_semantic_segmentation.py ${BASE_PARAMS} \
    --gspo_enable \
    --adapter_enable \
    --adapter_dim 64 \
    --output_dir outputs/exp3_adapter
echo "完成！结果保存在 outputs/exp3_adapter/metrics.txt"
echo ""

# Exp-4: Conv-LoRA + GSPO + SVD
echo "[4/5] 实验 4: Conv-LoRA + GSPO + SVD 微调"
echo "配置: Conv-LoRA + GSPO + S-SAM SVD (全秩)"
echo "--------------------------------------"
python3 run_semantic_segmentation.py ${BASE_PARAMS} \
    --gspo_enable \
    --svd_enable \
    --output_dir outputs/exp4_svd
echo "完成！结果保存在 outputs/exp4_svd/metrics.txt"
echo ""

# Exp-5: Full architecture (Conv-LoRA + GSPO + Adapter + SVD)
echo "[5/5] 实验 5: 完整架构（方案 A）"
echo "配置: Conv-LoRA + GSPO + Encoder-Adapter + SVD 微调"
echo "--------------------------------------"
python3 run_semantic_segmentation.py ${BASE_PARAMS} \
    --gspo_enable \
    --adapter_enable \
    --adapter_dim 64 \
    --svd_enable \
    --output_dir outputs/exp5_full_ssam
echo "完成！结果保存在 outputs/exp5_full_ssam/metrics.txt"
echo ""

echo "======================================"
echo "所有实验已完成！"
echo "======================================"
echo ""
echo "结果汇总："
echo "--------------------------------------"
for exp in outputs/exp*/metrics.txt; do
    if [ -f "$exp" ]; then
        echo ""
        echo "=== $exp ==="
        cat "$exp"
    fi
done
echo ""
echo "======================================"

