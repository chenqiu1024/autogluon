# ✅ 问题解决方案

## 问题诊断

你遇到的 "No such file or directory" 和 "ModuleNotFoundError: No module named 'pandas'" 是因为：

**环境不一致**：
- `pip install` 安装到了 `conv-lora` 环境（Python 3.10）
- 但默认 `python` 命令指向 base 环境（Python 3.12）
- 两个环境不同，所以找不到 pandas

## ✅ 解决方案

### 方法 1：确保使用正确的 conda 环境（推荐）

**每次打开终端都要先执行**：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate conv-lora
```

然后再运行任何脚本。

### 方法 2：修改脚本使用正确的 Python

我已经为你创建了修正版脚本（见下方）。

---

## 🎯 正确的第一步命令（复制执行）

```bash
# 1. 激活正确的环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate conv-lora

# 2. 验证环境
python -c "import pandas; import torch; print('✓ 环境正确')"

# 3. 切换到工作目录
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA

# 4. 检查数据集
ls datasets/isic2017/isic2017/train.csv

# 5. 训练基线（如果还没有）- 直接用 python
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

## 🚀 完整实验流程（确保环境正确）

### 一次性执行脚本（推荐）

我为你创建一个包含环境激活的完整脚本：

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA
bash run_full_rl_pipeline.sh isic2017
```

见下方的 `run_full_rl_pipeline.sh`

---

## 关键要点

1. **每次运行前都要激活 conda 环境**：
   ```bash
   source /root/miniconda3/etc/profile.d/conda.sh
   conda activate conv-lora
   ```

2. **或者在你的 `.bashrc` 中添加**：
   ```bash
   echo 'source /root/miniconda3/etc/profile.d/conda.sh' >> ~/.bashrc
   echo 'conda activate conv-lora' >> ~/.bashrc
   source ~/.bashrc
   ```

3. **验证环境的命令**：
   ```bash
   which python  # 应该是 /root/autodl-tmp/envs/conv-lora/bin/python
   python --version  # 应该是 Python 3.10.x
   ```

---

## ⚡ 立即可执行的命令

**假设你已经激活了 conv-lora 环境**，直接运行：

```bash
# BC 预热
python rl_bc_routing_policy.py \
  --task isic2017 \
  --model_path outputs_baseline/ \
  --output_dir bc_warmstart/ \
  --num_epochs 5 \
  --device cuda

# RL 训练
python rl_train_routing_policy.py \
  --task isic2017 \
  --warmstart bc_warmstart/routing_bc.pt \
  --output_dir rl_routing/ \
  --max_steps 6000 \
  --batch_size 12 \
  --device cuda
```

**注意**：如果提示找不到 `outputs_baseline/`，说明你还没训练基线模型，需要先运行步骤 5（见上方）。

