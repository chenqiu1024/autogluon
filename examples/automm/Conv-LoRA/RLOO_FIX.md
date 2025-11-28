# RLOO 训练问题修复说明

## 问题描述
在运行 RLOO 训练时遇到错误：
```
KeyError: '"optim.rloo.k" is not found in the config.'
```

## 原因
RLOO 的配置参数（`optim.rloo.k` 和 `optim.rloo.weight`）没有在 AutoGluon 的默认配置文件中定义。

## 已修复
我已经做了以下修改来解决这个问题：

### 1. 添加了 RLOO 配置到默认配置文件
**文件**: `multimodal/src/autogluon/multimodal/configs/optim/default.yaml`

添加了以下配置：
```yaml
rloo:
  k: 4  # Number of mask samples for RLOO baseline computation
  weight: 1.0  # Weight for RLOO loss component
  structure_weight: 1.0  # Weight for supervised Structure loss component
```

### 2. 更新了 loss utils 以正确读取配置
**文件**: `multimodal/src/autogluon/multimodal/optim/losses/utils.py`

更新了 `get_loss_func` 函数，从配置中读取所有三个 RLOO 参数。

### 3. 简化了训练脚本
**文件**: `examples/automm/Conv-LoRA/run_rloo_sam.py`

移除了命令行参数 `--rloo_k`, `--rloo_weight`, `--structure_weight`，因为这些参数现在使用配置文件中的默认值。

## 现在如何使用

### 方法 1: 使用默认配置（推荐）
```bash
python3 run_rloo_sam.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --output_dir outputs/rloo_v3/251128
```

### 方法 2: 自定义 RLOO 参数
编辑配置文件 `multimodal/src/autogluon/multimodal/configs/optim/default.yaml`：
```yaml
rloo:
  k: 8  # 增加采样数以降低方差
  weight: 1.5  # 增加 RLOO 权重
  structure_weight: 0.5  # 减少监督学习权重
```

## 验证修复
请重新运行您的训练命令：
```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA

python3 run_rloo_sam.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --output_dir outputs/rloo_v3/251128 > rloo_v3-251128.log
```

现在应该不会再出现配置错误了！
