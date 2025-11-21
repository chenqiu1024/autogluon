#!/bin/bash
# find_latest_model.sh - 自动查找最新训练的 Conv-LoRA 模型

cd "$(dirname "$0")"

echo "=== 查找最新训练的 Conv-LoRA 模型 ==="
echo ""

# 查找最新的模型目录
LATEST_DIR=$(ls -t AutogluonModels/ 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "❌ 没有找到任何模型目录！"
    echo ""
    echo "可能的原因："
    echo "1. 还没有训练过模型"
    echo "2. AutogluonModels/ 目录不存在"
    echo ""
    echo "请先运行训练命令："
    echo "python run_semantic_segmentation.py \\"
    echo "  --task isic2017 \\"
    echo "  --rank 3 \\"
    echo "  --expert_num 8 \\"
    echo "  --output_dir baseline_conv_lora"
    exit 1
fi

MODEL_PATH="AutogluonModels/$LATEST_DIR/model.ckpt"

if [ -f "$MODEL_PATH" ]; then
    echo "✅ 找到最新模型："
    echo ""
    echo "   📁 目录: AutogluonModels/$LATEST_DIR/"
    echo "   📄 文件: model.ckpt"
    echo "   💾 大小: $(du -h $MODEL_PATH | cut -f1)"
    echo "   🕒 时间: $(stat -c %y $MODEL_PATH 2>/dev/null | cut -d'.' -f1 || stat -f %Sm $MODEL_PATH)"
    echo ""
    echo "=== 如何使用此模型 ==="
    echo ""
    echo "1️⃣  评估 Baseline 性能："
    echo ""
    echo "python run_semantic_segmentation.py \\"
    echo "  --task isic2017 \\"
    echo "  --eval \\"
    echo "  --ckpt_path AutogluonModels/$LATEST_DIR \\"
    echo "  --output_dir outputs_baseline"
    echo ""
    echo "2️⃣  用于 RL 训练（Phase 1）："
    echo ""
    echo "python rl_train_routing_policy.py \\"
    echo "  --task isic2017 \\"
    echo "  --model_path $MODEL_PATH \\"
    echo "  --output_dir rl_routing_schemeB \\"
    echo "  --max_steps 5000 \\"
    echo "  --batch_size 4"
    echo ""
    echo "3️⃣  复制到指定目录："
    echo ""
    echo "mkdir -p baseline_conv_lora"
    echo "cp -r AutogluonModels/$LATEST_DIR/* baseline_conv_lora/"
    echo ""
else
    echo "❌ 找到目录但没有 model.ckpt！"
    echo ""
    echo "   目录: AutogluonModels/$LATEST_DIR/"
    echo "   缺失: model.ckpt"
    echo ""
    echo "可能的原因："
    echo "1. 训练还在进行中"
    echo "2. 训练失败或被中断"
    echo ""
    echo "目录内容："
    ls -lh "AutogluonModels/$LATEST_DIR/"
fi

