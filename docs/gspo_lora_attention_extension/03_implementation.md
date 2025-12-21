# 实现细节与代码修改

## 1. 修改的文件列表

| 文件 | 修改内容 |
|------|----------|
| `adaptation_layers.py` | LoRALinear 类添加 GSPO 支持 |
| `modeling_sam_for_conv_lora.py` | SamAttention 等类传递 GSPO 参数 |
| `sam.py` | SAMForSemanticSegmentation 添加 GSPO-LoRA 参数 |
| `gspo_trainer.py` | 添加 LoRA 反馈更新方法 |
| `lit_semantic_seg.py` | 调用 LoRA 反馈更新 |
| `semantic_segmentation.py` | 传递 GSPO-LoRA 参数到 trainer |
| `default.yaml` | 添加 GSPO-LoRA 配置项 |
| `run_semantic_segmentation.py` | 添加 CLI 参数 |

---

## 2. 核心代码修改

### 2.1 LoRALinear 类 (adaptation_layers.py)

**新增参数**：
```python
class LoRALinear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        # GSPO 扩展参数 (NEW)
        gspo_enabled: bool = False,
        gspo_momentum: float = 0.9,
        gspo_scale_adaptation: bool = False,
        **kwargs,
    ):
```

**新增缓冲区**：
```python
if gspo_enabled and r > 0:
    self.register_buffer("quality_history", torch.tensor(0.5))
    self.register_buffer("contribution_score", torch.tensor(1.0))
    self.register_buffer("update_count", torch.tensor(0))
```

**修改的前向传播**：
```python
def forward(self, x: torch.Tensor):
    if self.r > 0 and not self.merged:
        result = F.linear(x, self.T(self.weight), bias=self.bias)
        if self.r > 0:
            lora_delta = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            
            # GSPO: 基于质量历史的动态缩放
            if self.gspo_enabled and self.gspo_scale_adaptation and self.training:
                lora_delta = lora_delta * self.contribution_score
            
            result = result + lora_delta
        return result
    else:
        return F.linear(x, self.T(self.weight), bias=self.bias)
```

**新增方法**：
```python
def update_quality_feedback(self, quality_score: torch.Tensor):
    """
    GSPO: 基于实际性能更新质量历史
    """
    if not self.gspo_enabled or self.r <= 0:
        return
    
    with torch.no_grad():
        if quality_score.numel() > 1:
            quality_score = quality_score.mean()
        quality_value = quality_score.item()
        
        # EMA 更新
        self.quality_history = (
            self.gspo_momentum * self.quality_history +
            (1 - self.gspo_momentum) * quality_value
        )
        
        # 计算贡献分数
        self.contribution_score = torch.sigmoid(
            (self.quality_history - 0.5) * 4.0
        )
        
        self.update_count += 1

def get_gspo_stats(self):
    """返回 GSPO 统计信息用于日志"""
    if not self.gspo_enabled or self.r <= 0:
        return {}
    return {
        "quality_history": self.quality_history.item(),
        "contribution_score": self.contribution_score.item(),
        "update_count": self.update_count.item(),
    }
```

---

### 2.2 SamAttention 类 (modeling_sam_for_conv_lora.py)

**新增参数**：
```python
class SamAttention(nn.Module):
    def __init__(self, config, downsample_rate=None, lora_r=0, lora_alpha=1, lora_dropout=0.0,
                 gspo_lora_enabled=False, gspo_lora_momentum=0.9, gspo_lora_scale_adaptation=False):
```

**传递到 LoRALinear**：
```python
if lora_r > 0:
    self.q_proj = LoRALinear(
        self.hidden_size, self.internal_dim,
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        gspo_enabled=gspo_lora_enabled,
        gspo_momentum=gspo_lora_momentum,
        gspo_scale_adaptation=gspo_lora_scale_adaptation,
    )
    # k_proj, v_proj 类似
```

---

### 2.3 GSPOConvLoRATrainer 类 (gspo_trainer.py)

**新增初始化参数**：
```python
def __init__(
    self,
    predictor,
    ...
    # GSPO-LoRA on Attention 扩展参数 (NEW)
    gspo_lora_attention_enabled: bool = False,
    gspo_lora_attention_momentum: float = 0.9,
):
    ...
    self.gspo_lora_attention_enabled = gspo_lora_attention_enabled
    self.gspo_lora_attention_momentum = gspo_lora_attention_momentum
    self.lora_quality_log = []  # NEW
```

**新增方法**：
```python
def update_lora_attention_feedback(
    self,
    quality_scores: List[torch.Tensor],
    model
):
    """
    GSPO-LoRA 扩展：基于 GSPO 反馈更新 LoRA 层质量历史
    """
    if not self.gspo_lora_attention_enabled:
        return
    
    if not quality_scores:
        return
        
    avg_quality = torch.stack([q.mean() for q in quality_scores]).mean()
    
    # 收集所有启用 GSPO 的 LoRALinear 模块
    lora_layers = []
    for name, module in model.named_modules():
        if (hasattr(module, 'update_quality_feedback') and 
            hasattr(module, 'gspo_enabled') and 
            hasattr(module, 'r') and
            module.gspo_enabled and
            module.r > 0):
            lora_layers.append((name, module))
    
    if not lora_layers:
        return
    
    # 更新每个 LoRA 层
    for name, lora in lora_layers:
        lora.update_quality_feedback(avg_quality)
    
    # 记录日志
    self.lora_quality_log.append({
        'epoch': self.current_epoch,
        'avg_quality': avg_quality.item(),
        'num_lora_layers': len(lora_layers),
    })

def get_lora_stats(self, model) -> Dict:
    """获取所有 LoRA 层的 GSPO 统计信息"""
    stats = {}
    for name, module in model.named_modules():
        if (hasattr(module, 'get_gspo_stats') and 
            hasattr(module, 'r') and 
            hasattr(module, 'gspo_enabled')):
            if module.gspo_enabled and module.r > 0:
                lora_stats = module.get_gspo_stats()
                if lora_stats:
                    stats[name] = lora_stats
    return stats
```

---

### 2.4 lit_semantic_seg.py

**在 _gspo_training_step 中添加调用**：
```python
# 更新 LoRA on Attention 质量反馈 (GSPO-LoRA 扩展)
if hasattr(self.gspo_trainer, 'gspo_lora_attention_enabled') and self.gspo_trainer.gspo_lora_attention_enabled:
    self.gspo_trainer.update_lora_attention_feedback(quality_scores, self.model)
```

---

### 2.5 semantic_segmentation.py

**在 _maybe_create_gspo_trainer 中传递参数**：
```python
# 获取 GSPO-LoRA on Attention 参数 (NEW)
gspo_lora_attention_enabled = getattr(sam_cfg, "gspo_lora_attention_enabled", False) if sam_cfg else False
gspo_lora_attention_momentum = getattr(sam_cfg, "gspo_lora_attention_momentum", 0.9) if sam_cfg else 0.9

return GSPOConvLoRATrainer(
    ...
    # GSPO-LoRA on Attention 扩展 (NEW)
    gspo_lora_attention_enabled=gspo_lora_attention_enabled,
    gspo_lora_attention_momentum=gspo_lora_attention_momentum,
)
```

---

## 3. 配置文件修改

### 3.1 default.yaml

```yaml
sam:
  ...
  # 现有配置
  decoder_attention_lora_r: 0
  decoder_attention_lora_alpha: 1
  decoder_attention_lora_dropout: 0.0
  
  # 新增 GSPO-LoRA 配置
  gspo_lora_attention_enabled: False
  gspo_lora_attention_momentum: 0.9
  gspo_lora_attention_scale_adaptation: False
```

---

## 4. CLI 参数

### 4.1 run_semantic_segmentation.py

```python
# GSPO-LoRA on Attention 参数 (NEW)
parser.add_argument("--gspo_lora_attention_enable", action="store_true",
                    help="Enable GSPO quality feedback for Decoder LoRA on Attention")
parser.add_argument("--gspo_lora_attention_momentum", type=float, default=0.9,
                    help="Momentum for LoRA quality history updates (default: 0.9)")
parser.add_argument("--gspo_lora_attention_scale_adaptation", action="store_true",
                    help="Enable adaptive scaling based on quality for LoRA")
```

---

## 5. 参数传递链路

```
CLI 参数
    ↓
run_semantic_segmentation.py: hyperparameters.update({...})
    ↓
MultiModalPredictor: 读取 hyperparameters
    ↓
semantic_segmentation.py: _maybe_create_gspo_trainer()
    ↓
GSPOConvLoRATrainer: 初始化时接收参数
    ↓
sam.py: SAMForSemanticSegmentation 读取 config
    ↓
modeling_sam_for_conv_lora.py: SamModel → SamMaskDecoder → SamTwoWayTransformer → SamAttention
    ↓
adaptation_layers.py: LoRALinear 初始化时接收 GSPO 参数
```

---

## 6. 向后兼容性

所有新增参数都有默认值：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `gspo_lora_attention_enabled` | `False` | 默认禁用 |
| `gspo_lora_attention_momentum` | `0.9` | 与其他模块一致 |
| `gspo_lora_attention_scale_adaptation` | `False` | 默认不使用自适应缩放 |

**原有行为不受影响**：
- 不指定参数时，LoRA 使用标准固定 scaling
- 只有显式启用时才激活 GSPO 反馈机制

---

*下一篇：[04_configuration_guide.md](./04_configuration_guide.md) - 配置参数说明*

