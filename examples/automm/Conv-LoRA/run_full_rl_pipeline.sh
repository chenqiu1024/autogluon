#!/bin/bash
# Complete RL pipeline with proper environment setup

set -e  # Exit on error

TASK=${1:-isic2017}

echo "================================================================================"
echo "  Conv-LoRA + RL 完整实验流程"
echo "  任务: $TASK"
echo "================================================================================"
echo ""

# Activate conda environment
echo ">>> 激活 conda 环境..."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate conv-lora

# Verify environment
echo ">>> 验证环境..."
python -c "import pandas; import torch; print('✓ Python 环境正确')"
echo "  Python: $(which python)"
echo "  Version: $(python --version)"
echo ""

# Check dataset
echo ">>> 检查数据集..."
if [ ! -f "datasets/$TASK/$TASK/train.csv" ]; then
    echo "ERROR: 数据集不存在，请先运行："
    echo "  python prepare_semantic_segmentation_datasets.py"
    exit 1
fi
echo "✓ 数据集已就绪"
echo ""

# Check for baseline model
echo ">>> 检查基线模型..."
if [ ! -d "outputs_baseline" ]; then
    echo "基线模型不存在，开始训练..."
    echo "预计时间：~1 小时（L20）"
    echo ""
    
    python run_semantic_segmentation.py \
      --task $TASK \
      --rank 3 \
      --expert_num 8 \
      --num_gpus 1 \
      --per_gpu_batch_size 1 \
      --batch_size 4 \
      --output_dir outputs_baseline
    
    echo ""
    echo "✓ 基线模型训练完成"
else
    echo "✓ 使用现有基线模型: outputs_baseline/"
fi
echo ""

# BC Warmstart
echo ">>> 步骤 1: BC 预热..."
echo "预计时间：~7 分钟（L20）"
echo ""

python rl_bc_routing_policy.py \
  --task $TASK \
  --model_path outputs_baseline/ \
  --output_dir bc_warmstart/ \
  --num_epochs 5 \
  --max_samples 2000 \
  --device cuda

echo ""
echo "✓ BC 预热完成: bc_warmstart/routing_bc.pt"
echo ""

# RL Training
echo ">>> 步骤 2: RL 训练..."
echo "预计时间：~30 分钟（L20）"
echo "建议另开终端监控 TensorBoard："
echo "  tensorboard --logdir rl_routing/logs --port 6006 --bind_all"
echo ""

python rl_train_routing_policy.py \
  --task $TASK \
  --output_dir rl_routing/ \
  --warmstart bc_warmstart/routing_bc.pt \
  --num_experts 8 \
  --rank 3 \
  --num_layers 12 \
  --lr 1e-4 \
  --max_steps 6000 \
  --batch_size 12 \
  --kl_coef 0.05 \
  --entropy_coef 0.01 \
  --adapter_l2_coef 0.001 \
  --compute_budget 1e10 \
  --use_lagrangian \
  --ckpt_interval 500 \
  --vis_interval 100 \
  --mixed_precision \
  --device cuda

echo ""
echo "✓ RL 训练完成: rl_routing/checkpoints/final.pt"
echo ""

# Evaluation
echo ">>> 步骤 3: 评估..."
echo "预计时间：~2 分钟"
echo ""

python rl_eval_routing_policy.py \
  --task $TASK \
  --ckpt_path rl_routing/checkpoints/final.pt \
  --output_dir eval_results/ \
  --num_experts 8 \
  --rank 3 \
  --num_layers 12 \
  --save_heatmaps \
  --device cuda

echo ""
echo "================================================================================"
echo "  实验完成！"
echo "================================================================================"
echo ""
echo "结果查看："
echo "  cat eval_results/summary.json"
echo "  ls eval_results/heatmaps/"
echo ""
echo "TensorBoard："
echo "  tensorboard --logdir rl_routing/logs"
echo ""

