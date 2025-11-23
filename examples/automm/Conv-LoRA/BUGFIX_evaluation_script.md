# Bug Fix: Evaluation Script Not Using Layer Masks

## 问题描述

评估脚本 `evaluate_rl_policy.py` 存在**严重的实现bug**：

### 原始错误实现

```python
def evaluate_with_policy(predictor, test_data, policy, device):
    # 生成了layer_masks
    layer_masks, layer_probs, _ = policy(patch_embeddings, deterministic=True)
    
    # ❌ 错误：直接评估，没有使用layer_masks！
    metrics = predictor.evaluate(test_data, metrics=['iou', 'dice'])
    
    # 这相当于baseline评估，导致结果完全一样！
```

**结果**：
- 训练时：Policy选择27-29层（有效）
- 评估时：使用全部32层（bug导致）
- 表现：RL Policy = Baseline（完全相同）

---

## 修复方案

### 正确的实现（已修复）

参考 `conv_lora_env.py` 中训练时使用的per-image评估方式：

```python
def evaluate_with_policy(predictor, test_data, policy, device):
    # 1. 生成layer masks
    layer_masks, _, _ = policy(patch_embeddings, deterministic=True)
    
    # 2. 逐个图像评估（per-image evaluation）
    original_forward = model.forward
    
    for i in range(num_test):
        single_image_data = test_data.iloc[[i]]
        layer_mask = layer_masks[i]  # 该图像的专属mask
        
        # 3. 注入layer_mask到forward方法
        def forward_with_mask(*args, **kwargs):
            kwargs['layer_masks'] = layer_mask.unsqueeze(0)
            return original_forward(*args, **kwargs)
        
        model.forward = forward_with_mask
        
        # 4. 评估该图像
        metrics = predictor.evaluate(single_image_data, metrics=['iou', 'dice'])
        iou_scores.append(metrics['iou'])
        dice_scores.append(metrics['dice'])
    
    # 5. 聚合结果
    avg_iou = np.mean(iou_scores)
    avg_dice = np.mean(dice_scores)
```

### 修复的关键点

1. ✅ **Per-image evaluation**: 每张图像用其专属的layer_mask
2. ✅ **Forward wrapper**: 临时替换forward方法注入layer_masks
3. ✅ **正确传递**: layer_mask通过kwargs传递到底层ConvLoRALinear
4. ✅ **恢复原状**: 评估后恢复original_forward

---

## 修改的文件

```
examples/automm/Conv-LoRA/evaluate_rl_policy.py
  └─ evaluate_with_policy() 函数完全重写
```

---

## 正确的评估命令

### 1. 基础评估（推荐）

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA

# 激活conda环境并评估
bash -c "source /root/miniconda3/etc/profile.d/conda.sh && \
         conda activate conv-lora && \
         export HF_DATASETS_OFFLINE=1 && \
         export TRANSFORMERS_OFFLINE=1 && \
         python evaluate_rl_policy.py \
           --task isic2017 \
           --ckpt_path AutogluonModels/ag-20251123_003958 \
           --policy_path train_rl_layer-700ep-251123/checkpoints/best.pt \
           --output_dir rl_evaluation_fixed \
           --device cuda"
```

### 2. 完整评估（包括随机策略对比）

```bash
python evaluate_rl_policy.py \
  --task isic2017 \
  --ckpt_path AutogluonModels/ag-20251123_003958 \
  --policy_path train_rl_layer-700ep-251123/checkpoints/best.pt \
  --output_dir rl_evaluation_complete \
  --compare_random \
  --num_random_trials 5 \
  --device cuda
```

### 3. 参数说明

| 参数 | 说明 | 必需 |
|------|------|------|
| `--task` | 任务名称 | 否（默认isic2017） |
| `--ckpt_path` | Conv-LoRA模型路径 | ✅ 是 |
| `--policy_path` | RL策略权重路径 | ✅ 是 |
| `--output_dir` | 结果输出目录 | 否（默认rl_evaluation） |
| `--compare_random` | 是否对比随机策略 | 否 |
| `--device` | 计算设备 | 否（默认cuda） |

---

## 预期结果

### 修复前（错误）
```
Baseline (32 layers):   IoU=0.7751, DICE=0.8562
RL Policy (32.0 layers): IoU=0.7751, DICE=0.8562  ❌ 完全相同

Improvements:
  IoU:  +0.00% (+0.0000)  ❌ 没有提升
```

### 修复后（正确）
```
Baseline (32 layers):   IoU=0.7751, DICE=0.8562
RL Policy (27.8 layers): IoU=0.77XX, DICE=0.85XX  ✅ 使用了选择性激活

Improvements:
  IoU:  +X.XX% (+0.00XX)  ✅ 可能有提升/持平/略降
```

**重要**：
- RL Policy现在会显示**27-29层**的平均激活（而非32层）
- IoU/DICE可能略有提升、持平或略降（取决于策略学习效果）
- **关键是证明策略确实在工作**，而不是完全相同的baseline

---

## 评估时间

```
测试集大小: 600张图像
评估方式: Per-image (逐个评估)
预计时间: ~10-15分钟

进度显示:
Evaluating: 100%|██████████| 600/600 [12:34<00:00,  1.26s/it]
```

---

## 输出文件

评估完成后，会在输出目录生成：

```
rl_evaluation_fixed/
├── evaluation_results.json          # 详细结果（JSON）
├── layer_activation_analysis.png    # 层激活频率可视化
└── performance_comparison.png       # 性能对比图（如果--compare_random）
```

### evaluation_results.json 内容示例

```json
{
  "baseline": {
    "iou": 0.7751,
    "dice": 0.8562
  },
  "rl_policy": {
    "metrics": {
      "iou": 0.7755,
      "dice": 0.8565,
      "iou_std": 0.1234,
      "dice_std": 0.0987
    },
    "num_active_layers_mean": 27.8,
    "num_active_layers_std": 2.3,
    "layer_activation_freq": [0.95, 0.92, ..., 0.78]
  },
  "improvements": {
    "iou_absolute": 0.0004,
    "iou_relative_percent": 0.05,
    "dice_absolute": 0.0003,
    "dice_relative_percent": 0.04
  }
}
```

---

## 验证修复成功的标志

✅ **修复成功**的标志：
1. 评估过程显示 `Evaluating: 600/600`（逐个评估）
2. 输出显示平均激活层数**不是32.0**（如27.8）
3. 层激活频率可视化显示**某些层的激活率<1.0**

❌ **仍有问题**的标志：
1. 评估很快完成（<1分钟）
2. 平均激活层数仍然是**32.0**
3. IoU/DICE与baseline**完全相同**（小数点后4位）

---

## 总结

### 根本原因
评估脚本是**未完成的placeholder实现**，注释明确写着：
```python
# For demonstration, evaluate without actually using masks
# In production, this would use the layer_masks
```

### 修复内容
完整实现了per-image evaluation，使用与训练时相同的layer_mask注入机制。

### 影响
- ✅ **训练本身是成功的**（训练日志显示27-29层激活）
- ✅ **修复后可以看到真实效果**
- ⚠️ 性能可能略有提升、持平或略降（取决于训练效果）

---

**最后更新**: 2025-11-23
**修复文件**: evaluate_rl_policy.py
**下一步**: 运行修复后的评估脚本，查看真实结果

