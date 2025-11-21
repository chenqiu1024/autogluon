# Conv-LoRA 分层 RL 工具和脚本说明

本文档列出了所有可用的工具脚本及其使用方法。

## 📋 目录

- [模型查找工具](#模型查找工具)
- [训练日志查看工具](#训练日志查看工具)
- [训练脚本](#训练脚本)
- [评估脚本](#评估脚本)
- [实用工具](#实用工具)

---

## 模型查找工具

### `find_latest_model.sh`

**功能**：自动查找最新训练的 Conv-LoRA baseline 模型

**使用方法**：
```bash
cd examples/automm/Conv-LoRA
./find_latest_model.sh
```

**输出示例**：
```
✅ 找到最新模型：
   📁 目录: AutogluonModels/ag-20251120_041954/
   📄 文件: model.ckpt
   💾 大小: 2.4G
   🕒 时间: 2025-11-20 20:03:29

=== 如何使用此模型 ===
[完整的使用命令...]
```

**适用场景**：
- 训练完 baseline 后不知道模型保存在哪里
- 快速获取最新模型的路径用于后续 RL 训练

---

## 训练日志查看工具

### `view_training_logs.py`

**功能**：读取 TensorBoard 日志并生成训练统计摘要和可视化曲线

**使用方法**：
```bash
cd examples/automm/Conv-LoRA

# 查看 Phase 1 (RoutingPolicy) 训练日志
python view_training_logs.py rl_routing_schemeB-251119/logs

# 查看 Phase 2 (LayerPolicy) 训练日志
python view_training_logs.py rl_hier_layer_only/logs

# 查看 Phase 3 (Joint) 训练日志
python view_training_logs.py rl_hier_joint/logs
```

**输出内容**：
1. **统计摘要**：
   - 初始值、最终值、最佳值
   - 均值、标准差
   - 训练过程中的变化量和百分比

2. **可视化曲线**：
   - 保存为 `<output_dir>/training_summary.png`
   - 包含 IoU、Reward、FLOPs、Imbalance、Loss 等指标
   - 自动添加平滑曲线以观察趋势

**依赖**：
```bash
pip install tensorboard matplotlib numpy scipy
```

**适用场景**：
- 快速查看训练是否收敛
- 对比不同实验的训练曲线
- 生成论文/报告用的图表

---

## 训练脚本

### 1. `run_semantic_segmentation.py`

**功能**：训练原始 Conv-LoRA baseline 模型

**使用方法**：
```bash
python run_semantic_segmentation.py \
  --task isic2017 \
  --seed 20251119 \
  --rank 3 \
  --expert_num 8 \
  --num_gpus 1 \
  --per_gpu_batch_size 1 \
  --batch_size 4 \
  --output_dir baseline_conv_lora-251119
```

**关键参数**：
- `--rank`: LoRA 秩（论文中为 3）
- `--expert_num`: 专家数量（论文中为 8）
- `--output_dir`: 模型保存目录（已修复，现在会正确保存到此目录）

**输出**：
- `<output_dir>/model.ckpt`: 训练好的模型
- `<output_dir>/config.yaml`: 训练配置
- `<output_dir>/metrics.txt`: 评估结果

### 2. `rl_train_routing_policy.py`

**功能**：Phase 1 - 训练 RL-based 专家路由策略

**使用方法**：
```bash
python rl_train_routing_policy.py \
  --task isic2017 \
  --model_path AutogluonModels/ag-20251120_041954/model.ckpt \
  --output_dir rl_routing_schemeB-251119 \
  --max_steps 5000 \
  --batch_size 4 \
  --lr 1e-4 \
  --kl_coef 0.05 \
  --entropy_coef 0.01 \
  --adapter_l2_coef 0.001 \
  --compute_budget 1e10 \
  --imbalance_weight 0.01
```

**关键参数**：
- `--model_path`: Baseline Conv-LoRA 模型路径
- `--max_steps`: 训练步数（推荐 5000）
- `--kl_coef`: KL 散度系数（保持接近 Noisy-TopK）
- `--use_lagrangian`: 使用 Lagrangian 约束计算预算

**输出**：
- `<output_dir>/checkpoints/final.pt`: 训练好的 RoutingPolicy
- `<output_dir>/logs/`: TensorBoard 日志
- `<output_dir>/artifacts/`: 专家使用热力图

### 3. `rl_train_hierarchical_policy.py`

**功能**：Phase 2 & 3 - 训练分层 RL（LayerPolicy + RoutingPolicy）

**Phase 2 使用方法**（仅训练 LayerPolicy）：
```bash
python rl_train_hierarchical_policy.py \
  --phase layer \
  --task isic2017 \
  --model_path AutogluonModels/ag-20251120_041954/model.ckpt \
  --routing_ckpt rl_routing_schemeB-251119/checkpoints/final.pt \
  --output_dir rl_hier_layer_only \
  --batch_size 4 \
  --max_steps 5000 \
  --lr_layer 5e-5 \
  --layer_penalty_coef 0.01
```

**Phase 3 使用方法**（联合微调）：
```bash
python rl_train_hierarchical_policy.py \
  --phase joint \
  --task isic2017 \
  --model_path AutogluonModels/ag-20251120_041954/model.ckpt \
  --routing_ckpt rl_routing_schemeB-251119/checkpoints/final.pt \
  --layer_ckpt rl_hier_layer_only/checkpoints/step_3000.pt \
  --output_dir rl_hier_joint \
  --batch_size 4 \
  --max_steps 3000 \
  --lr_layer 5e-5 \
  --lr_routing 5e-5
```

**关键参数**：
- `--phase`: 训练阶段（`routing` / `layer` / `joint`）
- `--routing_ckpt`: Phase 1 训练好的 RoutingPolicy
- `--layer_ckpt`: Phase 2 训练好的 LayerPolicy（仅 Phase 3 需要）
- `--lr_layer` / `--lr_routing`: 两个策略的学习率
- `--layer_penalty_coef`: 层激活惩罚系数（控制稀疏性）

**输出**：
- `<output_dir>/checkpoints/`: LayerPolicy 和/或 RoutingPolicy checkpoints
- `<output_dir>/logs/`: TensorBoard 日志

### 4. `rl_bc_routing_policy.py`

**功能**：使用行为克隆（BC）预训练 RoutingPolicy

**使用方法**：
```bash
python rl_bc_routing_policy.py \
  --task isic2017 \
  --model_path AutogluonModels/ag-20251120_041954/model.ckpt \
  --output_dir bc_warmstart \
  --num_samples 1000 \
  --epochs 50 \
  --batch_size 4
```

**说明**：
- 用 Noisy-TopK 的专家选择作为监督信号
- 可用于 RoutingPolicy 的预热（warmstart）
- 在本实验中为可选步骤（未执行）

---

## 评估脚本

### 1. `eval_hierarchical_policy.py`

**功能**：评估分层 RL 模型（Phase 2 或 Phase 3）

**使用方法**：
```bash
# 评估 Phase 2 模型
python eval_hierarchical_policy.py \
  --task isic2017 \
  --model_path AutogluonModels/ag-20251120_041954/model.ckpt \
  --hier_ckpt rl_hier_layer_only/checkpoints/step_3000.pt \
  --split test \
  --batch_size 4 \
  --output_file hier_layer_eval_test.json

# 评估 Phase 3 模型
python eval_hierarchical_policy.py \
  --task isic2017 \
  --model_path AutogluonModels/ag-20251120_041954/model.ckpt \
  --hier_ckpt rl_hier_joint/checkpoints/step_3000.pt \
  --split test \
  --batch_size 4 \
  --output_file hier_joint_eval_test.json
```

**关键参数**：
- `--hier_ckpt`: 分层 RL checkpoint（包含 LayerPolicy 和 RoutingPolicy）
- `--split`: 评估数据集（`val` / `test`）
- `--output_file`: 结果保存路径（JSON 格式）

**输出**：
```json
{
  "mean_iou": 0.8123,
  "mean_dice": 0.8832,
  "mean_flops": 101234567.89,
  "mean_active_layers": 32.0,
  "num_samples": 600
}
```

### 2. `run_semantic_segmentation.py --eval`

**功能**：评估 baseline Conv-LoRA 模型

**使用方法**：
```bash
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251120_041954 \
  --output_dir outputs_baseline_test
```

**输出**：
- `<output_dir>/metrics.txt`: 评估结果

---

## 实用工具

### TensorBoard

**启动方法**：
```bash
# 查看单个实验
tensorboard --logdir rl_routing_schemeB-251119/logs --port 6006

# 同时对比多个实验
tensorboard --logdir_spec \
  baseline:baseline_conv_lora-251119/logs,\
  phase1:rl_routing_schemeB-251119/logs,\
  phase2:rl_hier_layer_only/logs,\
  phase3:rl_hier_joint/logs \
  --port 6006
```

**访问**：
- 本地：`http://localhost:6006`
- 远程：需要 SSH 端口转发 `ssh -L 6006:localhost:6006 user@server`

### Python 直接读取日志

```python
from tensorboard.backend.event_processing import event_accumulator

# 加载日志
ea = event_accumulator.EventAccumulator("rl_routing_schemeB-251119/logs")
ea.Reload()

# 查看所有指标
print(ea.Tags()['scalars'])

# 读取特定指标
iou_events = ea.Scalars('train/iou')
steps = [e.step for e in iou_events]
values = [e.value for e in iou_events]

# 绘图
import matplotlib.pyplot as plt
plt.plot(steps, values)
plt.xlabel('Step')
plt.ylabel('IoU')
plt.title('Training IoU')
plt.savefig('iou_curve.png')
```

---

## 完整工作流示例

### 1. 从头开始训练

```bash
cd examples/automm/Conv-LoRA

# Step 0: 训练 Baseline
python run_semantic_segmentation.py \
  --task isic2017 \
  --rank 3 \
  --expert_num 8 \
  --output_dir baseline_conv_lora-251119

# Step 1: 找到模型
./find_latest_model.sh

# Step 2: Phase 1 - 训练 RoutingPolicy
python rl_train_routing_policy.py \
  --task isic2017 \
  --model_path AutogluonModels/ag-YYYYMMDD_HHMMSS/model.ckpt \
  --output_dir rl_routing_schemeB-251119 \
  --max_steps 5000

# Step 3: 查看训练日志
python view_training_logs.py rl_routing_schemeB-251119/logs

# Step 4: Phase 2 - 训练 LayerPolicy
python rl_train_hierarchical_policy.py \
  --phase layer \
  --task isic2017 \
  --model_path AutogluonModels/ag-YYYYMMDD_HHMMSS/model.ckpt \
  --routing_ckpt rl_routing_schemeB-251119/checkpoints/final.pt \
  --output_dir rl_hier_layer_only \
  --max_steps 5000

# Step 5: Phase 3 - 联合微调
python rl_train_hierarchical_policy.py \
  --phase joint \
  --task isic2017 \
  --model_path AutogluonModels/ag-YYYYMMDD_HHMMSS/model.ckpt \
  --routing_ckpt rl_routing_schemeB-251119/checkpoints/final.pt \
  --layer_ckpt rl_hier_layer_only/checkpoints/step_3000.pt \
  --output_dir rl_hier_joint \
  --max_steps 3000

# Step 6: 评估所有模型
python eval_hierarchical_policy.py \
  --task isic2017 \
  --model_path AutogluonModels/ag-YYYYMMDD_HHMMSS/model.ckpt \
  --hier_ckpt rl_hier_layer_only/checkpoints/step_3000.pt \
  --split test \
  --output_file phase2_test_results.json
```

### 2. 快速查看现有实验

```bash
cd examples/automm/Conv-LoRA

# 查看所有训练日志
for dir in rl_routing_schemeB-251119 rl_hier_layer_only rl_hier_joint; do
  echo "=== $dir ==="
  python view_training_logs.py $dir/logs
done

# 启动 TensorBoard 对比所有实验
tensorboard --logdir_spec \
  phase1:rl_routing_schemeB-251119/logs,\
  phase2:rl_hier_layer_only/logs,\
  phase3:rl_hier_joint/logs \
  --port 6006
```

---

## 故障排查

### 问题 1：找不到模型文件

**症状**：`--output_dir` 中只有 `metrics.txt`，没有 `model.ckpt`

**解决**：
1. 使用 `./find_latest_model.sh` 查找模型实际位置
2. 模型默认保存在 `AutogluonModels/ag-YYYYMMDD_HHMMSS/`
3. 确保使用最新版本的 `run_semantic_segmentation.py`（已修复保存路径）

### 问题 2：训练日志为空或无法读取

**症状**：`view_training_logs.py` 报错或显示 "No data available"

**解决**：
1. 检查日志目录是否存在：`ls <output_dir>/logs/`
2. 检查 TensorBoard 文件：`ls <output_dir>/logs/events.out.tfevents.*`
3. 确认训练脚本正常运行且未被中断

### 问题 3：TensorBoard 无法连接

**症状**：浏览器无法打开 `http://localhost:6006`

**解决**：
1. 检查端口是否被占用：`lsof -i :6006`
2. 尝试使用其他端口：`tensorboard --logdir ... --port 6007`
3. 如果在远程服务器，确保 SSH 端口转发正确：
   ```bash
   ssh -L 6006:localhost:6006 user@server
   ```

---

## 相关文档

- **完整实验指南**：`HIERARCHICAL_RL_EXPERIMENT_GUIDE.md`
- **快速开始**：`QUICKSTART_RL.md`
- **模型位置说明**：`BASELINE_MODEL_LOCATION.md`
- **原论文理解**：`docs/cursor_paper_understanding-251115.md`

---

**更新时间**：2024-11-20  
**维护者**：AutoGluon Conv-LoRA Team

