#!/bin/bash
# GSPO + Adapter 训练脚本
# 使用方法: nohup bash run_gspo_adapter_train.sh > outputs/gspo_adapter_phase1-251221.log 2>&1 &

set -e  # 遇到错误立即退出

# ==================== 环境配置 ====================
# 激活 conda 环境（关键步骤！）
# 根据你的环境路径选择正确的 conda.sh 位置
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
elif [ -f "/root/anaconda3/etc/profile.d/conda.sh" ]; then
    source /root/anaconda3/etc/profile.d/conda.sh
elif [ -f "/root/autodl-tmp/envs/conda.sh" ]; then
    source /root/autodl-tmp/envs/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo "Error: Cannot find conda.sh. Please check your conda installation path."
    exit 1
fi

conda activate conv-lora

# 验证环境
echo "Python path: $(which python3)"
echo "Python version: $(python3 --version)"
echo "Checking autogluon.multimodal..."
python3 -c "from autogluon.multimodal import MultiModalPredictor; print('autogluon.multimodal imported successfully!')"

# ==================== Hugging Face 配置 ====================
# 配置 Hugging Face 镜像源（国内加速）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300
export TRANSFORMERS_CACHE=/root/.cache/huggingface/hub
export HF_HOME=/root/.cache/huggingface

# ==================== 训练参数 ====================
TASK="isic2017"
RANK=3
EXPERT_NUM=8
ADAPTER_DIM=64
OUTPUT_DIR="outputs/gspo_adapter_phase1-251221"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Starting GSPO + Adapter Training"
echo "Task: $TASK"
echo "Rank: $RANK"
echo "Expert Num: $EXPERT_NUM"
echo "Adapter Dim: $ADAPTER_DIM"
echo "Output Dir: $OUTPUT_DIR"
echo "========================================"

# ==================== 运行训练 ====================
cd "$(dirname "$0")"  # 切换到脚本所在目录

python3 run_semantic_segmentation.py \
    --task "$TASK" \
    --rank "$RANK" \
    --expert_num "$EXPERT_NUM" \
    --gspo_enable \
    --adapter_enable \
    --adapter_dim "$ADAPTER_DIM" \
    --gspo_adapter_enable \
    --output_dir "$OUTPUT_DIR"

echo "Training completed!"

