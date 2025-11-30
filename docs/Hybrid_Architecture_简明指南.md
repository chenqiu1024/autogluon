# 混合架构技术文档：Adapter + Conv-LoRA + GSPO 微调 SAM

**简明技术指南**

---

## 🎯 核心思想

本方法结合三种参数高效微调（PEFT）技术，形成互补的混合架构：

| 技术 | 位置 | 优化维度 | 参数量 | 作用 |
|------|------|----------|--------|------|
| **Conv-LoRA** | Attention 层 | 空间 (H×W) | ~8M | 多尺度空间建模 |
| **Adapter** | MLP 层 | 通道 (C) | ~4M | 语义特征适应 |
| **GSPO** | 训练策略 | 专家选择 | 0 | 质量驱动优化 |

**总参数**: 12M (仅占 SAM 总参数的 **1.87%**)

---

## 📐 架构设计

### SAM Vision Layer 的改造

```
原始 SAM Vision Layer:
┌─────────────────────┐
│  LayerNorm1         │
│  Attention          │
│  Residual           │
│  LayerNorm2         │
│  MLP                │
│  Residual           │
└─────────────────────┘

改造后（混合架构）:
┌──────────────────────────────┐
│  LayerNorm1                  │
│  Attention + Conv-LoRA MoE   │ ← 空间优化
│  Residual                    │
│  LayerNorm2                  │
│  MLP + Adapter（并行）       │ ← 通道优化
│  Residual（三路相加）        │
└──────────────────────────────┘
```

### Adapter 加在哪里？

**位置**: Vision Encoder 的每一层（32层）的 MLP 之后

**数量**: 32 个 Adapter（每层一个）

**架构**: 瓶颈结构（Bottleneck）
```python
Input: [B, H, W, 1280]
   ↓
Down: Linear(1280 → 64)    # 降维
   ↓
GELU 激活
   ↓
Up: Linear(64 → 1280)      # 升维（零初始化）
   ↓
Output: [B, H, W, 1280]
```

**连接方式**: 与 MLP 并行，三路残差
```python
output = residual + mlp_output + adapter_output
```

---

## 💡 为什么这样设计有效？

### 1. Conv-LoRA vs Adapter：互补优化

| 对比项 | Conv-LoRA | Adapter |
|--------|-----------|---------|
| **作用域** | Q/K/V 投影 | MLP 输出 |
| **操作类型** | 卷积（空间感知） | 全连接（逐位置） |
| **优化目标** | "哪些位置互相关注" | "特征如何表示" |
| **感受野** | 多尺度（1×1 到 7×7） | 无（逐位置） |
| **医学图像** | 边界精确定位 | 组织语义理解 |

**类比**：
- Conv-LoRA = 调整"空间关系图"（谁和谁说话）
- Adapter = 调整"特征字典"（每个人说什么）

### 2. Vision Transformer 的双路径覆盖

```
ViT Block 有两个关键路径：

路径 1: Attention ← 负责空间信息聚合
        Conv-LoRA 在这里优化 ✓

路径 2: MLP ← 负责特征非线性变换
        Adapter 在这里优化 ✓

结果 → 全面覆盖模型的适应能力
```

### 3. 参数效率的最佳平衡

```
Conv-LoRA (8M)     →  空间适应
   +
Adapter (4M)       →  通道适应
   +
GSPO (0)           →  策略优化
─────────────────────────────────
总计 12M (1.87%)   →  互补收益 > 简单相加
```

---

## 🔧 实现细节

### 1. 代码位置

#### Adapter 实现
```
multimodal/src/autogluon/multimodal/models/
├── adaptation_layers.py           # Adapter 类定义
└── custom_hf_models/
    └── modeling_sam_for_conv_lora.py  # 集成到 SAM
```

#### 配置文件
```
multimodal/src/autogluon/multimodal/configs/model/default.yaml
  sam:
    adapter_enabled: False  # 第 266 行
    adapter_dim: 64         # 第 267 行
```

### 2. 关键代码

#### Adapter 类（adaptation_layers.py）
```python
class AdapterLayer(nn.Module):
    def __init__(self, in_features, adapter_dim, scale=1.0, dropout=0.0):
        super().__init__()
        self.down_proj = nn.Linear(in_features, adapter_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, in_features)
        self.dropout = nn.Dropout(dropout)
        self.scale = scale
        
        # 零初始化：保证训练初期 adapter_out ≈ 0
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        # x: [B, H, W, C]
        h = self.down_proj(x)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.up_proj(h)
        return h * self.scale
```

#### 集成到 SamVisionLayer（modeling_sam_for_conv_lora.py）

**初始化**（第 891-893 行）：
```python
if hasattr(config, "adapter_enabled") and config.adapter_enabled:
    adapter_dim = getattr(config, "adapter_dim", 64)
    self.adapter = AdapterLayer(config.hidden_size, adapter_dim)
```

**前向传播**（第 984-988 行）：
```python
# MLP + Adapter 并行
residual = hidden_states
hidden_states = self.layer_norm2(hidden_states)
mlp_out = self.mlp(hidden_states)

if self.adapter is not None:
    adapter_out = self.adapter(hidden_states)  # 接收相同输入
    hidden_states = residual + mlp_out + adapter_out  # 三路相加
else:
    hidden_states = residual + mlp_out
```

---

## 📊 实验结果

### ISIC2017 皮肤病变分割

| 方法 | IoU | Dice | 可训练参数 |
|------|-----|------|-----------|
| **混合架构** | **0.7787** | **0.8581** | 12M (1.87%) |

**训练配置**：
- 训练轮数：30 epochs
- 学习率：1e-4
- 批次大小：4
- 损失函数：Structure Loss

---

## 🚀 使用方法

### 训练命令

```bash
# 完整混合架构
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --gspo_enable \
    --adapter_enable \
    --adapter_dim 64 \
    --output_dir outputs/hybrid_full
```

### 参数说明

| 参数 | 作用 | 默认值 |
|------|------|--------|
| `--rank` | LoRA 秩 | 3 |
| `--expert_num` | Conv-LoRA 专家数 | 8 |
| `--gspo_enable` | 启用 GSPO | False |
| `--adapter_enable` | 启用 Adapter | False |
| `--adapter_dim` | Adapter 瓶颈维度 | 64 |

---

## 📚 参考文献

### 核心论文

1. **SAM**: Kirillov et al., "Segment Anything" (2023)
   - [Paper](https://arxiv.org/abs/2304.02643) | [Code](https://github.com/facebookresearch/segment-anything)

2. **Conv-LoRA**: Shi et al., "Convolution Meets LoRA" (2024)
   - [Paper](https://arxiv.org/abs/2401.17868)

3. **Medical SAM Adapter**: Wu et al., "Medical SAM Adapter" (2023)
   - [Paper](https://arxiv.org/abs/2304.12620) | [Code](https://github.com/WuJunde/Medical-SAM-Adapter)

4. **Adapter**: Houlsby et al., "Parameter-Efficient Transfer Learning for NLP" (2019)
   - [Paper](https://arxiv.org/abs/1902.00751)

5. **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation" (2021)
   - [Paper](https://arxiv.org/abs/2106.09685) | [Code](https://github.com/microsoft/LoRA)

### 相关资源

- **AutoGluon**: https://github.com/autogluon/autogluon
- **Hugging Face SAM**: https://huggingface.co/facebook/sam-vit-huge
- **PEFT 库**: https://github.com/huggingface/peft

---

## ✅ 关键要点总结

1. **Adapter 位置**: Vision Encoder 每层的 MLP 之后，共 32 个
2. **Adapter 数量**: 32 个（每层一个）
3. **Adapter 结构**: 瓶颈架构（1280 → 64 → 1280）
4. **连接方式**: 与 MLP 并行，三路残差连接
5. **互补性**: Conv-LoRA 优化空间，Adapter 优化通道
6. **参数量**: 单个 ~165K，总计 ~5.3M
7. **初始化**: 上投影零初始化，保证训练稳定

---

**相关文档**:
- 完整技术文档: `Hybrid_Architecture_Adapter_ConvLoRA_GSPO_for_SAM.md`
- 对话历史: `chat- GSPO Conv-LoRA Adapters for SAM.md`

*最后更新: 2025-11-29*

