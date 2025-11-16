# 🚀 开始 RL 实验：完整操作指南

## 当前状态

✅ **所有代码已完成集成** - 可以真正运行实验了！  
✅ **BC 数据收集已启用** - Conv-LoRA 会保存中间值
✅ **所有脚本可执行** - 按下面步骤操作即可

---

## 📋 实验准备清单

### 1. 环境检查

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA

# 检查 Python 环境
python --version  # 应该是 3.8+

# 检查 GPU
nvidia-smi  # 确认 L20 可用

# 安装依赖（如果还没装）
pip install tensorboard matplotlib
```

### 2. 数据集检查

```bash
# 检查数据集是否已下载
ls datasets/isic2017/isic2017/

# 如果没有，运行下载
python prepare_semantic_segmentation_datasets.py
```

**预期输出**：
```
datasets/isic2017/isic2017/
├── train.csv
├── val.csv
├── test.csv
├── TrainDataset/
└── TestDataset/
```

---

## 🎯 三步启动实验

### 步骤 0：训练基线 Conv-LoRA（前置要求）

**如果你还没有训练好的 Conv-LoRA 模型**：

```bash
# 完整训练（~1 小时，L20）
python run_semantic_segmentation.py \
  --task isic2017 \
  --rank 3 \
  --expert_num 8 \
  --num_gpus 1 \
  --per_gpu_batch_size 1 \
  --batch_size 4 \
  --output_dir outputs_baseline
```

**或者用快速训练脚本**：
```bash
./train_baseline_first.sh isic2017 outputs_baseline
```

**等待训练完成后，验证**：
```bash
ls outputs_baseline/
# 应该看到 model.ckpt, metrics.txt 等文件
cat outputs_baseline/metrics.txt
# 查看基线 IoU（应该在 0.75-0.80 范围）
```

---

### 步骤 1：BC 预热（~7 分钟，L20）

```bash
python rl_bc_routing_policy.py \
  --task isic2017 \
  --model_path outputs_baseline/ \
  --output_dir bc_warmstart/ \
  --num_epochs 5 \
  --max_samples 2000 \
  --batch_size 32 \
  --lr 1e-3 \
  --device cuda
```

**实时监控（可选）**：
```bash
# 另开终端
tensorboard --logdir bc_warmstart/logs --port 6006 --bind_all
```

**预期输出**：
```
Loading Conv-LoRA model from outputs_baseline/
Found X Conv-LoRA layers
Model config: rank=3, experts=8, layers=X
Collecting Noisy-TopK actions from X layers...
Collected 2000+ BC samples
BC training: 2000+ samples, 5 epochs
BC Epoch 1/5: Loss=1.234, Acc=0.456
...
BC Epoch 5/5: Loss=0.234, Acc=0.876

✓ BC warmstart checkpoint saved to bc_warmstart/routing_bc.pt
  - Trained on 2000+ samples
  - X layers, 8 experts, rank 3
```

**耗时**：约 **7 分钟**（L20）

---

### 步骤 2：RL 训练（~30 分钟，L20）

```bash
python rl_train_routing_policy.py \
  --task isic2017 \
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
  --imbalance_weight 0.01 \
  --lagrangian_lr 0.001 \
  --ckpt_interval 500 \
  --vis_interval 100 \
  --mixed_precision \
  --device cuda
```

**实时监控（强烈推荐）**：
```bash
# 新终端
tensorboard --logdir rl_routing/logs --port 6006 --bind_all
# 访问 AutoDL 提供的 URL 查看训练曲线
```

**关键指标**：
- `train/reward` - 应该逐步上升
- `train/iou` - 目标 > 基线
- `train/flops` - 应围绕 budget 振荡
- `train/dual_alpha` - 应收敛到某个值
- `loss/kl` - 应在 0.01-0.1 范围

**预期输出**：
```
Starting RL training for 6000 steps
Output directory: rl_routing/
Compute budget: 1.00e+10 FLOPs
...
Step 100/6000: reward=0.65, iou=0.76, flops=9.5e9
Step 200/6000: reward=0.68, iou=0.77, flops=9.8e9
...
Checkpoint saved: rl_routing/checkpoints/step_500.pt
...
Training complete. Final checkpoint: rl_routing/checkpoints/final.pt
```

**耗时**：约 **30-35 分钟**（L20）

---

### 步骤 3：评估（~2 分钟）

```bash
python rl_eval_routing_policy.py \
  --task isic2017 \
  --ckpt_path rl_routing/checkpoints/final.pt \
  --output_dir eval_results/ \
  --num_experts 8 \
  --rank 3 \
  --num_layers 12 \
  --save_heatmaps \
  --device cuda
```

**预期输出**：
```
Loading checkpoint from rl_routing/checkpoints/final.pt
Routing policy loaded successfully
Evaluating on 600 images
...
Summary saved to eval_results/summary.json
Evaluation complete
```

**查看结果**：
```bash
cat eval_results/summary.json
# 应显示 avg_iou, avg_flops, expert_usage

# 查看专家使用热力图
ls eval_results/heatmaps/expert_usage.png
```

---

## 🔥 最快验证路径（10 分钟测试）

如果只想快速验证流程：

```bash
# 1. BC 预热（少量数据，2 epochs）
python rl_bc_routing_policy.py \
  --task isic2017 \
  --model_path outputs_baseline/ \
  --output_dir bc_quick/ \
  --num_epochs 2 \
  --max_samples 200 \
  --device cuda

# 2. RL 训练（少量步数）
python rl_train_routing_policy.py \
  --task isic2017 \
  --warmstart bc_quick/routing_bc.pt \
  --output_dir rl_quick/ \
  --max_steps 100 \
  --batch_size 8 \
  --device cuda

# 3. 检查输出
ls rl_quick/checkpoints/
tensorboard --logdir rl_quick/logs
```

**总耗时**：约 **5-10 分钟**

---

## 📊 监控关键指标

### TensorBoard 面板

1. **训练进度**：
   - `train/reward` ↑（越高越好）
   - `train/iou` ↑（目标 > 基线 0.77）
   - `train/flops` ≈ budget（围绕目标振荡）

2. **损失组件**：
   - `loss/kl` - 应稳定在 0.01-0.1
   - `loss/entropy` - 初期高，后期降低
   - `loss/adapter_l2` - 应较小

3. **拉格朗日**：
   - `train/dual_alpha` - 收敛表示约束满足

4. **专家分布**：
   - `train/gates_dist` - 应较均匀（避免全用一个专家）

### 本地文件

```bash
# 训练中间产物
ls rl_routing/artifacts/gates_step_*.png

# Checkpoints
ls rl_routing/checkpoints/step_*.pt
```

---

## 🛠️ 故障排除

### 问题 1：找不到基线模型

**错误**：
```
Error loading model: ...
```

**解决**：
```bash
# 确认路径正确
ls outputs_baseline/

# 如果没有，先训练基线
./train_baseline_first.sh isic2017 outputs_baseline
```

### 问题 2：BC 数据收集失败

**错误**：
```
ERROR: Failed to collect BC data
```

**解决**：
这通常意味着模型加载有问题。检查：
```bash
python -c "from autogluon.multimodal import MultiModalPredictor; p = MultiModalPredictor.load('outputs_baseline/'); print('OK')"
```

### 问题 3：CUDA out of memory

**解决**：
```bash
# 减小 batch_size
python rl_train_routing_policy.py ... --batch_size 4  # 而非 12
```

### 问题 4：TensorBoard 无法访问

**AutoDL 用户**：
1. 在 AutoDL 控制台找"自定义服务"
2. 添加端口 6006
3. 访问提供的 URL

---

## ✅ 验证成功的标志

### BC 预热成功：
- [ ] `bc_warmstart/routing_bc.pt` 文件存在
- [ ] TensorBoard 显示 `bc/loss` 下降
- [ ] `bc/accuracy` 达到 0.7-0.9

### RL 训练成功：
- [ ] `train/reward` 持续上升
- [ ] `train/iou` 超过或接近基线（0.77）
- [ ] `train/dual_alpha` 收敛
- [ ] Checkpoints 正常保存

### 评估成功：
- [ ] `eval_results/summary.json` 包含合理 IoU
- [ ] 专家使用热力图显示多样性
- [ ] 无严重的专家不平衡

---

## 🎓 实验建议

### 第一次运行（验证）：
- 用快速路径（100 steps）验证流程
- 检查所有日志/checkpoint 正常
- **耗时**：~10 分钟

### 第二次运行（小规模）：
- BC 5 epochs + RL 3000 steps
- 观察 reward 收敛趋势
- **耗时**：~25 分钟

### 第三次运行（完整）：
- BC 10 epochs + RL 6000-10000 steps
- 得到可发表的结果
- **耗时**：~50 分钟

---

## 下一步：开始实验！

**立即执行（推荐）**：

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA

# 确保你已有 outputs_baseline/（基线模型）
# 如果没有，先运行：./train_baseline_first.sh isic2017 outputs_baseline

# 然后开始 RL pipeline：

# 步骤 1：BC 预热
python rl_bc_routing_policy.py \
  --task isic2017 \
  --model_path outputs_baseline/ \
  --output_dir bc_warmstart/ \
  --num_epochs 5 \
  --device cuda

# 步骤 2：RL 训练
python rl_train_routing_policy.py \
  --task isic2017 \
  --warmstart bc_warmstart/routing_bc.pt \
  --output_dir rl_routing/ \
  --max_steps 6000 \
  --batch_size 12 \
  --device cuda

# 步骤 3：评估
python rl_eval_routing_policy.py \
  --task isic2017 \
  --ckpt_path rl_routing/checkpoints/final.pt \
  --output_dir eval_results/ \
  --num_experts 8 \
  --rank 3 \
  --device cuda
```

**预计总时间（L20）**：~45-50 分钟

---

## 📞 需要帮助？

查看详细文档：
- `RL_IMPLEMENTATION_GUIDE.md` - 完整技术文档
- `QUICKSTART_RL.md` - 快速入门
- `ARCHITECTURE_DIAGRAM.md` - 架构图解
- `README.md` - 第 5 节（RL 路由）

祝实验顺利！🎉

