#!/bin/bash
# 训练脚本 - 使用 HF-Mirror 解决网络问题
# 使用方法: bash train_with_hf_mirror.sh [其他参数]

# 激活 conda 环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate conv-lora

# 配置 Hugging Face 镜像源（国内加速）
export HF_ENDPOINT=https://hf-mirror.com

# 设置合理的超时时间
export HF_HUB_DOWNLOAD_TIMEOUT=300

# 如果模型已缓存，优先使用本地缓存
export TRANSFORMERS_CACHE=/root/.cache/huggingface/hub
export HF_HOME=/root/.cache/huggingface

# 运行训练脚本，传递所有参数
python3 run_semantic_segmentation.py "$@"


