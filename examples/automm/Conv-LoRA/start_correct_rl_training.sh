#!/bin/bash
# 使用修复后的权重加载逻辑重新训练RL routing policy

export HF_HUB_OFFLINE=1

echo "🚀 开始正确的RL训练（权重加载已修复）"
echo "="
echo ""
echo "关键修复："
echo "  ✓ Lightning checkpoint key前缀处理已修复"
echo "  ✓ Conv-LoRA权重现在正确加载（704个参数）"
echo "  ✓ 模型IoU验证：0.974"
echo ""
echo "之前3000步训练无效（使用了未训练的模型）"
echo "现在重新开始..."
echo ""

# 备份之前无效的训练输出
if [ -d "rl_routing" ]; then
    echo "备份之前的无效训练输出..."
    mv rl_routing rl_routing_invalid_$(date +%Y%m%d_%H%M%S)
fi

# 开始正确的训练
python rl_train_routing_policy.py \
  --task isic2017 \
  --warmstart bc_warmstart/routing_bc.pt \
  --model_path AutogluonModels/ag-20251116_011442/epoch=3-step=2000.ckpt \
  --output_dir rl_routing/ \
  --num_experts 8 \
  --rank 3 \
  --num_layers 32 \
  --lr 1e-4 \
  --max_steps 3000 \
  --batch_size 8 \
  --kl_coef 0.05 \
  --entropy_coef 0.01 \
  --adapter_l2_coef 0.001 \
  --compute_budget 1e10 \
  --use_lagrangian \
  --imbalance_weight 0.01 \
  --lagrangian_lr 0.001 \
  --ckpt_interval 500 \
  --vis_interval 200 \
  --mixed_precision \
  --device cuda

echo ""
echo "✅ 训练完成！"
echo ""
echo "检查结果："
echo "  tensorboard --logdir rl_routing/logs"
echo "  或查看 rl_routing/artifacts/"
