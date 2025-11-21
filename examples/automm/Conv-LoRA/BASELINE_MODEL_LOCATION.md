# Baseline Conv-LoRA 模型保存位置说明

## 问题

运行 `run_semantic_segmentation.py` 训练 Baseline Conv-LoRA 后，发现 `--output_dir` 指定的目录中只有 `metrics.txt`，没有模型文件。

## 原因

**旧版脚本问题**：`run_semantic_segmentation.py` 在创建 `MultiModalPredictor` 时没有指定 `path` 参数，导致模型保存到默认位置 `AutogluonModels/ag-YYYYMMDD_HHMMSS/`，而不是用户指定的 `--output_dir`。

## 解决方案

### 方案 1：使用修复后的脚本（推荐）

**已修复的代码**（第 102-108 行）：
```python
predictor = MultiModalPredictor(
    problem_type="semantic_segmentation",
    validation_metric=validation_metric,
    eval_metric=validation_metric,
    hyperparameters=hyperparameters,
    label="label",
    path=args.output_dir,  # ← 新增这一行
)
```

**使用方法**：
```bash
cd examples/automm/Conv-LoRA

# 确保使用最新的脚本
git pull  # 或者手动更新 run_semantic_segmentation.py

python run_semantic_segmentation.py \
  --task isic2017 \
  --seed 20251119 \
  --rank 3 \
  --expert_num 8 \
  --output_dir baseline_conv_lora-251119

# 模型会保存到：baseline_conv_lora-251119/model.ckpt
```

### 方案 2：找到旧版脚本保存的模型

如果你已经用旧版脚本训练完成，模型保存在 `AutogluonModels/` 目录：

**查找最新的模型**：
```bash
cd examples/automm/Conv-LoRA

# 方法 1：按时间列出所有模型目录
ls -lt AutogluonModels/

# 方法 2：查找最近 24 小时内创建的 checkpoint
find AutogluonModels/ -name "model.ckpt" -type f -mtime -1

# 方法 3：查看最新的模型目录
ls -t AutogluonModels/ | head -1
```

**示例输出**：
```
AutogluonModels/ag-20251120_041954/model.ckpt
```

**使用找到的模型**：
```bash
# 方法 A：复制到期望的位置
mkdir -p baseline_conv_lora-251119
cp -r AutogluonModels/ag-20251120_041954/* baseline_conv_lora-251119/

# 方法 B：直接使用原路径进行 RL 训练
python rl_train_routing_policy.py \
  --task isic2017 \
  --model_path AutogluonModels/ag-20251120_041954/model.ckpt \
  --output_dir rl_routing_schemeB \
  --max_steps 5000
```

## 完整的模型目录结构

训练完成后，模型目录应包含以下文件：

```
baseline_conv_lora-251119/  (或 AutogluonModels/ag-YYYYMMDD_HHMMSS/)
├── model.ckpt              # 主模型文件 (2.4GB)
├── config.yaml             # 训练配置
├── hparams.yaml            # 超参数
├── data_processors.pkl     # 数据处理器
├── df_preprocessor.pkl     # DataFrame 预处理器
├── eval_metric.pkl         # 评估指标对象
├── assets.json             # 资源清单
├── metrics.txt             # 评估结果（测试集）
└── events.out.tfevents.*   # TensorBoard 日志
```

## 验证模型是否可用

```bash
# 检查模型文件是否存在
ls -lh baseline_conv_lora-251119/model.ckpt

# 或者
ls -lh AutogluonModels/ag-20251120_041954/model.ckpt

# 预期输出：显示约 2.4GB 的文件
# -rw-r--r-- 1 root root 2.4G Nov 20 20:03 model.ckpt
```

## 快速脚本：自动查找最新模型

```bash
#!/bin/bash
# find_latest_model.sh

cd examples/automm/Conv-LoRA

echo "=== 查找最新训练的 Conv-LoRA 模型 ==="
echo ""

# 查找最新的模型目录
LATEST_DIR=$(ls -t AutogluonModels/ | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "❌ 没有找到任何模型！"
    exit 1
fi

MODEL_PATH="AutogluonModels/$LATEST_DIR/model.ckpt"

if [ -f "$MODEL_PATH" ]; then
    echo "✅ 找到最新模型："
    echo "   路径: $MODEL_PATH"
    echo "   大小: $(du -h $MODEL_PATH | cut -f1)"
    echo "   时间: $(stat -c %y $MODEL_PATH | cut -d'.' -f1)"
    echo ""
    echo "使用此模型进行 RL 训练："
    echo "python rl_train_routing_policy.py \\"
    echo "  --task isic2017 \\"
    echo "  --model_path $MODEL_PATH \\"
    echo "  --output_dir rl_routing_schemeB \\"
    echo "  --max_steps 5000"
else
    echo "❌ 找到目录但没有 model.ckpt: $MODEL_PATH"
fi
```

**使用方法**：
```bash
chmod +x find_latest_model.sh
./find_latest_model.sh
```

## 总结

- **新用户**：使用修复后的 `run_semantic_segmentation.py`，模型会保存到 `--output_dir`
- **已完成训练**：在 `AutogluonModels/` 目录中查找模型
- **RL 训练**：使用找到的 `model.ckpt` 路径作为 `--model_path` 参数

---

**更新时间**：2024-11-20  
**相关脚本**：`run_semantic_segmentation.py`  
**相关文档**：`HIERARCHICAL_RL_EXPERIMENT_GUIDE.md`

