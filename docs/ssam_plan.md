# S-SAM SVD 微调集成方案

## 目标与约束

**目标**：在不破坏现有 Conv-LoRA + GSPO + Encoder-Adapter 架构的前提下，为 SAM Vision Encoder 的 MLP 层引入 S-SAM 的 SVD 微调机制，实现：

- 在"奇异值空间"对 MLP 权重做精细调节
- 与 Conv-LoRA（空间）、Encoder-Adapter（通道）形成三维互补
- 参数增量控制在 0.5M 以内（约 +0.08% SAM 总参数）

**约束**：

- 不引入文本提示适配层（TAL）
- 不使用 Decoder-Adapter
- 保持与现有训练脚本和配置系统的兼容性
- 初始化策略保证训练初期行为接近预训练模型

---

## 1. SVD 微调层的设计与实现

### 1.1 核心原理

对 MLP 的权重矩阵 `W ∈ R^(d_out × d_in)` 进行奇异值分解：

```
W = U Σ V^T
```

引入可训练参数 `A`（缩放）和 `B`（偏移），动态重构权重：

```
W' = U · ReLU(A ⊙ Σ + B) · V^T
```

其中：

- `U`、`V^T` 在初始化时通过 SVD 预计算并冻结
- `Σ` 是奇异值对角矩阵
- `A`、`B` 是可训练参数，形状与 `Σ` 相同
- `⊙` 表示逐元素乘法
- `ReLU` 确保奇异值非负

### 1.2 实现位置

在 [`multimodal/src/autogluon/multimodal/models/adaptation_layers.py`](multimodal/src/autogluon/multimodal/models/adaptation_layers.py) 中新增 `SVDLinear` 类：

```python
class SVDLinear(nn.Module):
    """
    S-SAM style SVD-based fine-tuning layer.
    
    Decomposes a frozen linear layer's weight via SVD and introduces
    learnable scale (A) and bias (B) parameters on singular values.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        base_weight: torch.Tensor,  # 预训练权重
        svd_rank: Optional[int] = None,  # 保留的奇异值数量，None=全部
        init_scale: float = 1.0,
        init_bias: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # SVD 分解预训练权重
        U, S, Vt = torch.linalg.svd(base_weight, full_matrices=False)
        
        # 可选：只保留前 svd_rank 个奇异值（低秩近似）
        if svd_rank is not None and svd_rank < len(S):
            U = U[:, :svd_rank]
            S = S[:svd_rank]
            Vt = Vt[:svd_rank, :]
        
        # 冻结 U 和 V^T
        self.register_buffer('U', U)
        self.register_buffer('Vt', Vt)
        self.register_buffer('S_orig', S)  # 原始奇异值，用于监控
        
        # 可训练参数：缩放 A 和偏移 B
        self.A_scale = nn.Parameter(torch.full_like(S, init_scale))
        self.B_bias = nn.Parameter(torch.full_like(S, init_bias))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 动态重构权重：W' = U @ ReLU(A ⊙ Σ + B) @ V^T
        S_adjusted = torch.relu(self.A_scale * self.S_orig + self.B_bias)
        # 高效计算：x @ W'^T = x @ (V @ diag(S') @ U^T)
        out = x @ self.Vt.T  # (*, in) @ (in, rank) = (*, rank)
        out = out * S_adjusted.unsqueeze(0)  # (*, rank) * (rank,)
        out = out @ self.U.T  # (*, rank) @ (rank, out) = (*, out)
        return out
```

### 1.3 在 SamMLPBlock 中集成

修改 [`multimodal/src/autogluon/multimodal/models/custom_hf_models/modeling_sam_for_conv_lora.py`](multimodal/src/autogluon/multimodal/models/custom_hf_models/modeling_sam_for_conv_lora.py) 中的 `SamMLPBlock`：

**当前结构**：

```python
class SamMLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lin1 = nn.Linear(config.hidden_size, config.mlp_dim)
        self.lin2 = nn.Linear(config.mlp_dim, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]
```

**修改为**：

```python
class SamMLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lin1 = nn.Linear(config.hidden_size, config.mlp_dim)
        self.lin2 = nn.Linear(config.mlp_dim, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]
        
        # S-SAM SVD 微调（可选，config 驱动）
        self.svd_lin1 = None
        self.svd_lin2 = None
        if getattr(config, 'svd_enabled', False):
            svd_rank = getattr(config, 'svd_rank', None)
            self.svd_lin1 = SVDLinear(
                config.hidden_size, config.mlp_dim,
                self.lin1.weight.data.clone(),
                svd_rank=svd_rank
            )
            self.svd_lin2 = SVDLinear(
                config.mlp_dim, config.hidden_size,
                self.lin2.weight.data.clone(),
                svd_rank=svd_rank
            )
            # 冻结原始线性层
            self.lin1.weight.requires_grad = False
            self.lin2.weight.requires_grad = False
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.svd_lin1 is not None:
            # 使用 SVD 微调路径
            hidden_states = self.svd_lin1(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states = self.svd_lin2(hidden_states)
        else:
            # 原始路径
            hidden_states = self.lin1(hidden_states)
            hidden_states = self.act(hidden_states)
            hidden_states = self.lin2(hidden_states)
        return hidden_states
```

---

## 2. 配置与超参数设计

### 2.1 YAML 配置

在 [`multimodal/src/autogluon/multimodal/configs/model/default.yaml`](multimodal/src/autogluon/multimodal/configs/model/default.yaml) 的 `sam` 部分增加：

```yaml
sam:
  checkpoint_name: "facebook/sam-vit-huge"
  # ... 现有配置 ...
  adapter_enabled: False
  adapter_dim: 64
  # S-SAM SVD 微调配置
  svd_enabled: False
  svd_rank: null  # null=使用全部奇异值，或指定数字如 256 做低秩近似
```

### 2.2 训练脚本 CLI 参数

在 [`examples/automm/Conv-LoRA/run_semantic_segmentation.py`](examples/automm/Conv-LoRA/run_semantic_segmentation.py) 中增加：

```python
# S-SAM SVD parameters
parser.add_argument("--svd_enable", action="store_true", 
                   help="Enable S-SAM SVD fine-tuning on MLP layers")
parser.add_argument("--svd_rank", type=int, default=None,
                   help="SVD rank for low-rank approximation (None=full rank)")

# 在 hyperparameters 构造中
if args.svd_enable:
    print(f"Enabling S-SAM SVD fine-tuning with rank={args.svd_rank}")
    hyperparameters.update({
        "model.sam.svd_enabled": True,
        "model.sam.svd_rank": args.svd_rank,
    })
```

---

## 3. SAM 模型加载与配置注入

在 [`multimodal/src/autogluon/multimodal/models/sam.py`](multimodal/src/autogluon/multimodal/models/sam.py) 的 `SAMForSemanticSegmentation.__init__` 中：

**当前逻辑**：

```python
def __init__(self, ..., adapter_enabled: bool = False, adapter_dim: int = 64):
    ...
    self._load_checkpoint(checkpoint_name)
    ...
    if adapter_enabled:
        for layer in self.model.vision_encoder.layers:
            layer.adapter = AdapterLayer(...)
```

**新增 SVD 配置注入**：

```python
def __init__(self, ..., adapter_enabled: bool = False, adapter_dim: int = 64,
             svd_enabled: bool = False, svd_rank: Optional[int] = None):
    ...
    self.svd_enabled = svd_enabled
    self.svd_rank = svd_rank
    
    self._load_checkpoint(checkpoint_name)  # 会调用 from_pretrained，需要传 config
    ...
    # Encoder-Adapter 注入（已有）
    if adapter_enabled:
        for layer in self.model.vision_encoder.layers:
            layer.adapter = AdapterLayer(...)
    
    # SVD 微调已经在 SamMLPBlock.__init__ 中根据 config 自动创建
    # 这里只需确认 config 传递正确
```

**修改 `_load_checkpoint` 方法**：

```python
def _load_checkpoint(self, checkpoint_name):
    if self.pretrained:
        configuration = SamConfig.from_pretrained(checkpoint_name)
        # 注入 SVD 配置到 vision_config
        if hasattr(configuration, 'vision_config'):
            configuration.vision_config.svd_enabled = self.svd_enabled
            configuration.vision_config.svd_rank = self.svd_rank
        
        try:
            self.model = SamModel.from_pretrained(
                checkpoint_name, config=configuration, local_files_only=True
            )
        except:
            self.model = SamModel.from_pretrained(checkpoint_name, config=configuration)
    else:
        configuration = SamConfig(name_or_path=checkpoint_name)
        if hasattr(configuration, 'vision_config'):
            configuration.vision_config.svd_enabled = self.svd_enabled
            configuration.vision_config.svd_rank = self.svd_rank
        self.model = SamModel(configuration)
```

---

## 4. 训练与消融实验设计

### 4.1 基础实验组合

在 isic2017 等数据集上进行以下对比：

1. **Conv-LoRA（基线）**
   ```bash
   python3 run_semantic_segmentation.py --task isic2017 \
       --rank 3 --expert_num 8 --output_dir outputs/baseline
   ```

2. **Conv-LoRA + GSPO**
   ```bash
   python3 run_semantic_segmentation.py --task isic2017 \
       --rank 3 --expert_num 8 --gspo_enable --output_dir outputs/gspo
   ```

3. **Conv-LoRA + GSPO + Encoder-Adapter（当前最佳）**
   ```bash
   python3 run_semantic_segmentation.py --task isic2017 \
       --rank 3 --expert_num 8 --gspo_enable --adapter_enable \
       --adapter_dim 64 --output_dir outputs/hybrid
   ```

4. **Conv-LoRA + GSPO + SVD（新增，验证 SVD 独立效果）**
   ```bash
   python3 run_semantic_segmentation.py --task isic2017 \
       --rank 3 --expert_num 8 --gspo_enable --svd_enable \
       --output_dir outputs/gspo_svd
   ```

5. **完整架构：Conv-LoRA + GSPO + Encoder-Adapter + SVD（方案 A）**
   ```bash
   python3 run_semantic_segmentation.py --task isic2017 \
       --rank 3 --expert_num 8 --gspo_enable --adapter_enable \
       --adapter_dim 64 --svd_enable --output_dir outputs/full_ssam
   ```


### 4.2 关键对比指标

- **IoU / Dice**：主要性能指标
- **参数量**：可训练参数占 SAM 总参数的比例
- **训练时间**：每 epoch 耗时（评估 SVD 动态重构的开销）
- **收敛速度**：达到最佳 Dice 所需的 epoch 数

### 4.3 预期结果

| 配置 | 可训练参数 | 预期 Dice 提升 |

|------|-----------|---------------|

| Conv-LoRA | ~2M (0.3%) | +5-6% |

| Conv-LoRA + GSPO | ~2M (0.3%) | +7-8% |

| Conv-LoRA + GSPO + Adapter | ~4M (0.6%) | +8-10% |

| Conv-LoRA + GSPO + SVD | ~2.5M (0.4%) | +8-9% |

| **完整方案 A** | **~4.5M (0.7%)** | **+10-12%** |

---

## 5. 兼容性与风险检查

### 5.1 前向兼容性

- **默认行为不变**：`svd_enabled=False` 时，`SamMLPBlock` 完全使用原始线性层
- **与 Adapter 无冲突**：SVD 作用在 MLP 内部权重，Adapter 是 MLP 后的并行分支
- **与 Conv-LoRA/GSPO 独立**：Conv-LoRA 在 Attention 路径，SVD 在 MLP 路径

### 5.2 数值稳定性

- **初始化策略**：`A=1, B=0` 保证训练初期 `Σ' ≈ Σ`，权重几乎不变
- **ReLU 保护**：确保奇异值非负，避免权重矩阵退化
- **梯度流通畅**：SVD 分解在初始化时完成，训练时只更新 `A` 和 `B`，梯度路径清晰

### 5.3 计算开销

- **前向传播**：每次需要做 `x @ V^T → * S' → @ U^T`，相比原始 `x @ W^T` 多两次矩阵乘法
- **优化**：可以预计算 `U @ diag(S') @ V^T` 并缓存（但会失去 SVD 的动态性）
- **实测建议**：先不做缓存，观察训练速度，如果瓶颈明显再优化

---

## 6. 实施步骤总结

1. **实现 `SVDLinear` 类**（`adaptation_layers.py`）
2. **修改 `SamMLPBlock`**（`modeling_sam_for_conv_lora.py`）
3. **更新配置文件**（`default.yaml`）
4. **扩展训练脚本**（`run_semantic_segmentation.py`）
5. **修改 SAM 模型加载逻辑**（`sam.py`）
6. **运行消融实验**（5 组对比）
7. **分析结果并调优**（如调整 `svd_rank`、学习率等）

---

## 7. 后续可扩展方向

- **自适应 SVD rank**：根据每层的奇异值分布自动选择保留数量
- **层选择性 SVD**：只在后几层或关键层启用 SVD，进一步减少参数
- **与 LoRA 的统一**：探索 SVD 与 LoRA 的理论联系，设计统一的低秩适配框架