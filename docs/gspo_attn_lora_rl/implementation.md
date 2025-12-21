## 代码落点（实现视角）

### 1) LoRA on Attention 的实现位置（已有）

`multimodal/src/autogluon/multimodal/models/custom_hf_models/modeling_sam_for_conv_lora.py`

- `SamAttention.__init__`：
  - `lora_r > 0` 时将 `q_proj/k_proj/v_proj` 替换为 `LoRALinear`

### 2) 本次新增：GSPO-AttnLoRA 扩展

#### 2.1 LoRA 层新增 GSPO mode

`multimodal/src/autogluon/multimodal/models/adaptation_layers.py`

- `LoRALinear`
  - 新增 `set_gspo_mode(active, noise_std)`
  - GSPO active 且训练时，可对 LoRA delta 加 `N(0, σ^2)`（阶段 1）

- `GatedLoRALinear`（阶段 2）
  - 只对 LoRA delta 分支乘以 `gate(x)`（base path 不变）
  - `set_gspo_mode(active, gate_noise_std, delta_noise_std)`：GSPO active 时对 gate logits 注入噪声

#### 2.2 在 decoder attention 中启用 gated LoRA（可选）

`multimodal/src/autogluon/multimodal/models/custom_hf_models/modeling_sam_for_conv_lora.py`

- `SamAttention.__init__`：
  - 根据 `decoder_attention_lora_gate_enabled` 选择 `GatedLoRALinear` 或 `LoRALinear`

#### 2.3 将 gate 配置注入到 HF config（可选）

`multimodal/src/autogluon/multimodal/models/sam.py`

- `SAMForSemanticSegmentation.__init__` 新增参数：
  - `decoder_attention_lora_gate_enabled`
  - `decoder_attention_lora_gate_noise_std`
- `_load_checkpoint`：写入 `config.mask_decoder_config.*`

---

## GSPO active step 中“冻结/启用”策略（关键：向后兼容）

`multimodal/src/autogluon/multimodal/optim/lit_semantic_seg.py`

- 在 `_gspo_training_step` 内：
  - 进入 GSPO 时：
    - 对 decoder-attn-LoRA 调用 `set_gspo_mode(active=True, ...)`（注入探索噪声/门控噪声）
    - 若 `optim.gspo.train_decoder_attention_lora=False`（默认），临时将 decoder-attn-LoRA 参数 `requires_grad=False`
  - 退出 GSPO 时：
    - 恢复 `requires_grad` 并关闭 `set_gspo_mode(active=False)`

模块筛选：
- 只匹配 `name` 包含 `mask_decoder` 且 `module` 是 `LoRALinear/GatedLoRALinear`，因此不会影响：
  - encoder 的 Conv‑LoRA
  - encoder adapters
  - decoder adapter（queries adapter）

---

## 配置项（默认值保证兼容）

### `optim` 默认配置

`multimodal/src/autogluon/multimodal/configs/optim/default.yaml`

- `optim.gspo.train_decoder_attention_lora: False`
- `optim.gspo.decoder_attention_lora_noise_std: 0.0`
- `optim.gspo.decoder_attention_lora_gate_noise_std: 0.0`

### `model` 默认配置

`multimodal/src/autogluon/multimodal/configs/model/default.yaml`

- `model.sam.decoder_attention_lora_gate_enabled: False`
- `model.sam.decoder_attention_lora_gate_noise_std: 0.0`

---

## CLI 用法（examples）

脚本：`examples/automm/Conv-LoRA/run_semantic_segmentation.py`

### 1) 让 GSPO 同时训练 decoder attention LoRA（阶段 0 → 开启训练）

```bash
--gspo_enable
--decoder_attn_lora_enable
--gspo_train_decoder_attn_lora
```

### 2) 阶段 1：给 LoRA delta 加探索噪声

```bash
--gspo_decoder_attn_lora_noise_std 0.02
```

### 3) 阶段 2：gated decoder-attn-LoRA（policy）

```bash
--gspo_decoder_attn_lora_gate_enable
--gspo_decoder_attn_lora_gate_noise_std 0.5
```

---

## 调参建议（很简短）

- `decoder_attention_lora_noise_std`：0.01–0.05 起试
- `decoder_attention_lora_gate_noise_std`：0.2–1.0 起试（过大可能不稳）


