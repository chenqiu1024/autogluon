# 环境配置指南

## 问题诊断

如果看到错误：
```
ModuleNotFoundError: No module named 'pandas'
```

说明你的 conda 环境还需要安装依赖。

---

## 完整环境配置

### 选项 1：使用现有 AutoGluon 环境（推荐）

```bash
# 切换到 AutoGluon 环境（如果已安装）
conda activate autogluon

# 或者你之前创建的 conv-lora 环境
conda activate conv-lora

# 安装缺失的依赖
pip install pandas tensorboard matplotlib tqdm
```

### 选项 2：全新安装（如果环境损坏）

```bash
# 创建新环境
conda create -n conv-lora-rl python=3.10 -y
conda activate conv-lora-rl

# 安装依赖
pip install -U pip setuptools wheel

# 安装 AutoGluon Multimodal
cd /root/autodl-tmp/works/autogluon
pip install -e multimodal/[tests]

# 安装 RL 依赖
pip install tensorboard matplotlib tqdm pandas

# 验证安装
python -c "import pandas; import torch; import autogluon.multimodal; print('✓ All OK')"
```

---

## 快速修复（当前环境）

如果你当前已在某个环境中，只需：

```bash
pip install pandas tensorboard matplotlib tqdm
```

---

## 验证环境

```bash
# 检查关键包
python -c "
import pandas as pd
import torch
import autogluon.multimodal
from autogluon.multimodal.rl.policies.routing_policy import RoutingPolicy
print('✓ 环境配置正确')
print(f'  - PyTorch: {torch.__version__}')
print(f'  - CUDA available: {torch.cuda.is_available()}')
print(f'  - Pandas: {pd.__version__}')
"
```

**预期输出**：
```
✓ 环境配置正确
  - PyTorch: 2.x.x
  - CUDA available: True
  - Pandas: 2.x.x
```

---

## 脚本运行方式修正

根据你遇到的错误，有两种运行方式：

### 方式 1：明确使用 bash（更稳定）

```bash
bash train_baseline_first.sh isic2017 outputs_baseline
```

### 方式 2：./方式（需要正确的shell）

```bash
# 如果看到 "No such file or directory"
# 可能是行尾符问题，转换一下：
dos2unix train_baseline_first.sh  # 如果有这个工具

# 或直接用 bash
bash train_baseline_first.sh isic2017 outputs_baseline
```

---

## 直接运行 Python 脚本（绕过 shell 脚本）

如果 shell 脚本有问题，直接调用 Python：

```bash
python run_semantic_segmentation.py \
  --task isic2017 \
  --rank 3 \
  --expert_num 8 \
  --num_gpus 1 \
  --per_gpu_batch_size 1 \
  --batch_size 4 \
  --output_dir outputs_baseline
```

---

## 常见问题

### Q: "No such file or directory" 但文件存在？

可能原因：
1. **行尾符问题**（Windows vs Unix）
2. **当前目录错误**
3. **Shell 解释器问题**

解决：
```bash
# 方法 1：用 bash 明确运行
bash ./train_baseline_first.sh isic2017 outputs_baseline

# 方法 2：检查行尾符
cat -A train_baseline_first.sh | head -3
# 如果看到 ^M，需要转换：
sed -i 's/\r$//' train_baseline_first.sh
```

### Q: ModuleNotFoundError

解决：
```bash
pip install <缺失的包名>
```

### Q: CUDA out of memory

解决：
```bash
# 减小 batch_size
python run_semantic_segmentation.py ... --per_gpu_batch_size 1 --batch_size 2
```

---

## 推荐的第一条命令（修正后）

**在安装好依赖后**，运行：

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA

# 确保在正确的 conda 环境
conda activate conv-lora  # 或你的环境名

# 安装缺失依赖
pip install pandas tensorboard matplotlib tqdm

# 训练基线（如果没有）
bash train_baseline_first.sh isic2017 outputs_baseline

# 或直接 Python
python run_semantic_segmentation.py \
  --task isic2017 \
  --rank 3 \
  --expert_num 8 \
  --output_dir outputs_baseline
```

---

安装完成后，参考 `START_HERE.md` 继续 RL 实验！

