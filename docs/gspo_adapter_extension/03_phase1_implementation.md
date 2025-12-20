# 阶段1实现：统一优势加权

## 1. 实现目标

**阶段1的核心思想**：以最小改动验证 GSPO 扩展到 Adapter 的可行性。

```
┌─────────────────────────────────────────────────────────────┐
│                    阶段1：统一优势加权                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  改动范围：                                                   │
│  ✅ AdapterLayer 添加质量追踪 buffer                         │
│  ✅ GSPOConvLoRATrainer 添加 update_adapter_feedback        │
│  ✅ 训练脚本添加 --gspo_adapter_enable 开关                  │
│  ✅ 配置文件添加相应参数                                      │
│                                                              │
│  不改动：                                                     │
│  ❌ 前向传播逻辑（Adapter 输出计算方式不变）                   │
│  ❌ 损失函数（仍使用统一的优势加权损失）                       │
│  ❌ 现有 Conv-LoRA MoE 反馈机制                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 代码修改清单

### 2.1 AdapterLayer 类修改

**文件**: `multimodal/src/autogluon/multimodal/models/adaptation_layers.py`

**新增参数**:

```python
def __init__(
    self, 
    in_features, 
    adapter_dim, 
    scale=1.0, 
    dropout=0.0,
    gspo_enabled=False,        # 🆕 启用 GSPO 质量追踪
    gspo_momentum=0.9,         # 🆕 质量历史动量
    gspo_scale_adaptation=False,  # 🆕 阶段2功能，暂时不使用
):
```

**新增 buffer**（当 `gspo_enabled=True`）:

```python
if gspo_enabled:
    self.register_buffer("quality_history", torch.tensor(0.5))
    self.register_buffer("contribution_score", torch.tensor(1.0))
    self.register_buffer("update_count", torch.tensor(0))
```

**新增方法**:

```python
def update_quality_feedback(self, quality_score: torch.Tensor):
    """
    GSPO: 根据实际分割质量更新质量历史。
    
    由 GSPOConvLoRATrainer 在每个训练步骤后调用。
    """
    if not self.gspo_enabled:
        return
    
    with torch.no_grad():
        quality_value = quality_score.mean().item()
        
        # 动量更新质量历史
        self.quality_history = (
            self.gspo_momentum * self.quality_history +
            (1 - self.gspo_momentum) * quality_value
        )
        
        # 计算贡献分数（阶段2使用）
        self.contribution_score = torch.sigmoid(
            (self.quality_history - 0.5) * 4.0
        )
        
        self.update_count += 1

def get_gspo_stats(self):
    """返回 GSPO 统计信息用于日志记录。"""
    if not self.gspo_enabled:
        return {}
    return {
        "quality_history": self.quality_history.item(),
        "contribution_score": self.contribution_score.item(),
        "update_count": self.update_count.item(),
    }
```

### 2.2 GSPOConvLoRATrainer 类修改

**文件**: `examples/automm/Conv-LoRA/gspo_trainer.py`

**新增初始化参数**:

```python
def __init__(
    self,
    predictor,
    group_size: int = 4,
    # ... 现有参数 ...
    gspo_adapter_enabled: bool = False,  # 🆕
    gspo_adapter_momentum: float = 0.9,   # 🆕
):
    # ...
    self.gspo_adapter_enabled = gspo_adapter_enabled
    self.gspo_adapter_momentum = gspo_adapter_momentum
    self.adapter_quality_log = []  # 🆕 追踪日志
```

**新增方法**:

```python
def update_adapter_feedback(
    self,
    quality_scores: List[torch.Tensor],
    model
):
    """
    GSPO-Adapter: 根据 GSPO 反馈更新 Adapter 质量历史。
    """
    if not self.gspo_adapter_enabled:
        return
    
    # 计算组内平均质量
    avg_quality = torch.stack([q.mean() for q in quality_scores]).mean()
    
    # 收集所有启用 GSPO 的 AdapterLayer
    adapters = []
    for name, module in model.named_modules():
        if hasattr(module, 'update_quality_feedback') and hasattr(module, 'gspo_enabled'):
            if module.gspo_enabled:
                adapters.append((name, module))
    
    # 更新每个 Adapter 的质量反馈
    for name, adapter in adapters:
        adapter.update_quality_feedback(avg_quality)
    
    # 记录日志
    self.adapter_quality_log.append({
        'epoch': self.current_epoch,
        'avg_quality': avg_quality.item(),
        'num_adapters': len(adapters),
    })
```

### 2.3 lit_semantic_seg.py 修改

**文件**: `multimodal/src/autogluon/multimodal/optim/lit_semantic_seg.py`

**在 `_gspo_training_step` 末尾添加**:

```python
# Update expert quality feedback (Conv-LoRA MoE)
if selected_experts_groups:
    # ... 现有代码 ...
    
    # 🆕 Update Adapter quality feedback (GSPO-Adapter extension)
    if hasattr(self.gspo_trainer, 'gspo_adapter_enabled') and self.gspo_trainer.gspo_adapter_enabled:
        self.gspo_trainer.update_adapter_feedback(quality_scores, self.model)
```

### 2.4 配置文件修改

**文件**: `multimodal/src/autogluon/multimodal/configs/optim/default.yaml`

```yaml
gspo:
  # ... 现有参数 ...
  adapter_enabled: False       # 🆕 启用 GSPO-Adapter
  adapter_momentum: 0.9        # 🆕 Adapter 质量历史动量
  adapter_scale_adaptation: False  # 🆕 阶段2功能
```

**文件**: `multimodal/src/autogluon/multimodal/configs/model/default.yaml`

```yaml
sam:
  # ... 现有参数 ...
  adapter_gspo_enabled: False       # 🆕
  adapter_gspo_momentum: 0.9        # 🆕
  adapter_gspo_scale_adaptation: False  # 🆕
```

### 2.5 训练脚本修改

**文件**: `examples/automm/Conv-LoRA/run_semantic_segmentation.py`

**新增命令行参数**:

```python
parser.add_argument("--gspo_adapter_enable", action="store_true", 
                    help="Enable GSPO quality feedback for Encoder Adapters (Phase 1)")
parser.add_argument("--gspo_adapter_momentum", type=float, default=0.9, 
                    help="Momentum for adapter quality history updates")
parser.add_argument("--gspo_adapter_scale_adaptation", action="store_true",
                    help="Enable adaptive scale based on quality history (Phase 2)")
```

**配置传递**:

```python
if args.gspo_adapter_enable:
    if not args.gspo_enable:
        print("Warning: --gspo_adapter_enable requires --gspo_enable. Enabling GSPO automatically.")
        args.gspo_enable = True
    print(f"Enabling GSPO-Adapter extension with momentum={args.gspo_adapter_momentum}")
    hyperparameters.update({
        "optim.gspo.adapter_enabled": True,
        "optim.gspo.adapter_momentum": args.gspo_adapter_momentum,
    })
```

---

## 3. 使用方法

### 3.1 基本用法（阶段1）

```bash
cd examples/automm/Conv-LoRA

python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --gspo_enable \
    --adapter_enable \
    --adapter_dim 64 \
    --gspo_adapter_enable \
    --output_dir outputs/gspo_adapter_phase1
```

### 3.2 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--gspo_adapter_enable` | flag | False | 启用 GSPO-Adapter 质量追踪 |
| `--gspo_adapter_momentum` | float | 0.9 | 质量历史更新动量 |

### 3.3 向后兼容性

**原有命令（不受影响）**:

```bash
# 仅 Conv-LoRA + GSPO，不启用 Adapter
python3 run_semantic_segmentation.py --task isic2017 --gspo_enable

# Conv-LoRA + GSPO + Adapter（Adapter 无 GSPO 反馈）
python3 run_semantic_segmentation.py --task isic2017 --gspo_enable --adapter_enable
```

这些命令的行为与修改前完全一致。

---

## 4. 验证方法

### 4.1 功能验证

```python
# 在训练完成后，检查 Adapter 质量历史是否被更新
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor.load("outputs/gspo_adapter_phase1")
model = predictor._learner.model

# 遍历所有 Adapter，检查 GSPO 统计
for name, module in model.named_modules():
    if hasattr(module, 'get_gspo_stats'):
        stats = module.get_gspo_stats()
        if stats:
            print(f"{name}: {stats}")
```

### 4.2 预期输出

```
model.vision_encoder.layers.0.adapter: {'quality_history': 0.72, 'contribution_score': 0.81, 'update_count': 15000}
model.vision_encoder.layers.1.adapter: {'quality_history': 0.69, 'contribution_score': 0.77, 'update_count': 15000}
...
model.vision_encoder.layers.31.adapter: {'quality_history': 0.75, 'contribution_score': 0.84, 'update_count': 15000}
```

### 4.3 性能验证

比较以下配置的测试集性能：

1. **基线**: `--gspo_enable --adapter_enable`（Adapter 无 GSPO）
2. **阶段1**: `--gspo_enable --adapter_enable --gspo_adapter_enable`

预期：阶段1的性能应该与基线相近或略有提升（因为质量追踪在阶段1不直接影响前向传播）。

---

## 5. 工作原理

### 5.1 信息流

```
┌─────────────────────────────────────────────────────────────────┐
│                     阶段1 信息流                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 前向传播（×G 次）                                            │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │  Image → Vision Encoder → Mask Decoder → Prediction      │ │
│     │           ↓                                              │ │
│     │    Conv-LoRA (空间) + Adapter (通道)                      │ │
│     │           ↓                                              │ │
│     │    [正常计算，无变化]                                      │ │
│     └─────────────────────────────────────────────────────────┘ │
│                                                                  │
│  2. 质量评估                                                     │
│     quality[g] = IoU(pred[g], gt)                                │
│                                                                  │
│  3. 优势计算                                                     │
│     advantage[g] = quality[g] - mean(quality)                    │
│                                                                  │
│  4. 加权损失（同时影响 Conv-LoRA 和 Adapter）                     │
│     loss = Σ sigmoid(advantage) * seg_loss                       │
│                                                                  │
│  5. 反馈更新                                                     │
│     ┌─────────────────────────┐  ┌─────────────────────────┐    │
│     │ Conv-LoRA MoE 反馈      │  │ Adapter 反馈 (🆕)       │    │
│     │ expert_quality_history  │  │ quality_history         │    │
│     │ += momentum * Δq        │  │ += momentum * avg_q     │    │
│     └─────────────────────────┘  └─────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 关键理解

**阶段1 的"统一优势加权"意味着**：

1. **相同的权重**：Conv-LoRA 和 Adapter 接收相同的优势加权梯度
2. **共享质量信号**：两者都根据最终分割质量被更新
3. **质量追踪独立**：虽然梯度共享，但质量历史分别追踪

这种设计的好处是：
- 最小改动，风险低
- 验证 GSPO 框架可以扩展到 Adapter
- 为阶段2的差异化优化打下基础

---

*下一篇：[04_phase2_implementation.md](./04_phase2_implementation.md) - 阶段2实现细节*


