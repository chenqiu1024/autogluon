## 代码落点（实现视角）

### 1) Adapter 模块扩展

- `multimodal/src/autogluon/multimodal/models/adaptation_layers.py`
  - `AdapterLayer`
    - 新增 `set_gspo_mode(active, noise_std)`：用于在 GSPO active 时启用探索噪声（阶段 1）
  - `GatedAdapterLayer`
    - 内部包含 `AdapterLayer` + `gate`（`Linear(in_features->1)`）
    - `set_gspo_mode(active, gate_noise_std, adapter_noise_std)`：用于在 GSPO active 时对 gate logits 注入噪声（阶段 2）

### 2) Encoder Adapter 的构建与注入

- `multimodal/src/autogluon/multimodal/models/custom_hf_models/modeling_sam_for_conv_lora.py`
  - `SamVisionLayer.__init__`：
    - `config.encoder_adapter_gate_enabled=True` 时创建 `GatedAdapterLayer`
    - 否则创建 `AdapterLayer`

- `multimodal/src/autogluon/multimodal/models/sam.py`
  - `SAMForSemanticSegmentation.__init__`：
    - 在 vision encoder 的每个 layer 上注入同样类型的 adapter（确保路径一致）

### 3) GSPO active step 时的“冻结/启用”

- `multimodal/src/autogluon/multimodal/optim/lit_semantic_seg.py`
  - `SemanticSegmentationLitModule.__init__` 接收 `gspo_adapter_cfg`
  - `_gspo_training_step` 中：
    - 进入 GSPO 前：对 encoder adapters 调用 `set_gspo_mode(active=True, ...)`
    - 若 `train_encoder_adapter=False`（默认）：临时将 encoder adapter 参数 `requires_grad=False`
    - GSPO 结束后：恢复 `requires_grad` 并 `set_gspo_mode(active=False)`

> 注意：通过 `named_modules()` 的名称匹配只选取 `vision_encoder.layers.*.adapter`，因此不会影响 decoder adapter（`mask_decoder` 路径上的 adapter_queries）。

---

## 配置与开关（向后兼容）

### 默认值（向后兼容）

在 `multimodal/src/autogluon/multimodal/configs/optim/default.yaml`：

- `optim.gspo.train_encoder_adapter: False`
- `optim.gspo.encoder_adapter_noise_std: 0.0`
- `optim.gspo.encoder_adapter_gate_noise_std: 0.0`

在 `multimodal/src/autogluon/multimodal/configs/model/default.yaml`：

- `model.sam.encoder_adapter_gate_enabled: False`
- `model.sam.encoder_adapter_gate_noise_std: 0.0`

因此在默认情况下：
- GSPO active step 会冻结 encoder adapters（如果存在），保证“GSPO 只用于 Conv‑LoRA”；
- 不会引入额外噪声/门控结构变化。

---

## 命令行用法（examples）

入口脚本：`examples/automm/Conv-LoRA/run_semantic_segmentation.py`

### 启用“GSPO 同时训练 encoder adapters”

1) 先启用 adapters（否则不会有 adapter 参数）：

- `--adapter_enable --adapter_dim 64`

2) 再允许 GSPO active 时训练 adapters：

- `--gspo_enable --gspo_train_encoder_adapter`

### 阶段 1：adapter 输出探索噪声

- `--gspo_encoder_adapter_noise_std 0.02`

### 阶段 2：gated adapter（policy）

- `--gspo_encoder_adapter_gate_enable`
- `--gspo_encoder_adapter_gate_noise_std 0.5`

---

## 调参建议（非常简短）

- 阶段 1：
  - `encoder_adapter_noise_std` 从 0.01–0.05 起试
- 阶段 2：
  - `encoder_adapter_gate_noise_std` 从 0.2–1.0 起试
  - gate 噪声太大可能导致训练早期不稳，建议配合较小学习率或更长 warmup


