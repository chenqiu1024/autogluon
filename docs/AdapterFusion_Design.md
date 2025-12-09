# AdapterFusion 设计与实现速览

> 目标：在现有 Conv-LoRA + 标准 Adapter + GSPO 语义分割管线上，引入 AdapterFusion（gating 版），以低风险组合多个 adapter 输出，获得额外性能提升。

---

## 1. 方案概览
- **主干**：SAM 冻结主干 + Conv-LoRA (MoE conv) + 标准 FC Adapter (MLP 并行) + GSPO 训练策略。
- **新增**：AdapterFusion（gating + 线性投影 + LayerNorm + 可选小 FFN + dropout）。
- **放置位置**：与现有 FC Adapter 相同的 MLP 并行插入点，在 residual 路径上融合各 adapter 输出。
- **训练两阶段**：
  - 阶段 A：只训 Fusion（冻结 adapters/LoRA），小 gate 初始值。
  - 阶段 B：解冻 adapters/LoRA，小 lr 联合微调。

---

## 2. 代码改动速览
- 核心实现文件：
  - `multimodal/src/autogluon/multimodal/models/custom_hf_models/modeling_sam_for_conv_lora.py`
    - 新增 `AdapterFusion` 类（gating + proj + LN + optional FFN + dropout）。
    - 在 `SamVisionLayer`：
      - 初始化：读取 fusion 配置，创建 `adapter_fusion`，并注册现有 adapter 输出。
      - 前向：若启用 fusion，则 `fused_out = adapter_fusion([adapter_out])`，并与 `residual + mlp_out` 相加。
  - 配置：
    - `multimodal/src/autogluon/multimodal/configs/model/default.yaml`
      - `adapter_fusion_enabled` / `adapter_fusion_proj_dim` / `adapter_fusion_dropout` / `adapter_fusion_use_ffn` / `adapter_fusion_gate_init`
  - CLI 与超参写入：
    - `examples/automm/Conv-LoRA/run_semantic_segmentation.py`
      - 新增 `--adapter_fusion_*` 参数，并写入 `hyperparameters`。

---

## 3. 模型框架（文字版）
```
SamVisionLayer (×32):
  Attention 路径：
    LayerNorm1 -> QKV (Conv-LoRA/MoE) -> MH-Attn -> Residual

  MLP 路径（并行）：
    LayerNorm2 -> MLP
    Adapter (FC bottleneck, 1280→64→1280, up-proj 零初始化)
    AdapterFusion (gating):
       - 对各 adapter 输出线性投影到 proj_dim
       - gate = sigmoid(g_i), g_i 初始 -4.0
       - 加权和 -> LayerNorm -> (可选 FFN) -> Dropout
    Residual add: residual + mlp_out + fusion_out
```

---

## 4. 关键实现片段（摘录）
> 文件：`modeling_sam_for_conv_lora.py`

- AdapterFusion 结构（简化）：
  - 线性投影到共享维度 -> gate(sigmoid) 加权和 -> LayerNorm -> 可选 FFN -> Dropout
  - gate 初始值 `gate_init=-4.0`（影响极小，训练渐开）
- 集成：
  - 初始化时创建 `self.adapter_fusion`（按配置）
  - 前向中若启用 fusion：`fused_out = self.adapter_fusion([adapter_out])`；否则回退为原有 `residual + mlp_out (+ adapter_out)`

---

## 5. 配置与 CLI
- `default.yaml`（片段）
  - `adapter_fusion_enabled: False`
  - `adapter_fusion_proj_dim: 256`
  - `adapter_fusion_dropout: 0.05`
  - `adapter_fusion_use_ffn: True`
  - `adapter_fusion_gate_init: -4.0`
- CLI（`run_semantic_segmentation.py`）
  - `--adapter_fusion_enable`
  - `--adapter_fusion_proj_dim`
  - `--adapter_fusion_dropout`
  - `--adapter_fusion_gate_init`
  - `--adapter_fusion_disable_ffn`（默认开启 FFN，传此开关可关闭）

---

## 6. 训练流程（建议）
### 阶段 A（Fusion-only warmup）
- 冻结 adapters/LoRA；仅训练 Fusion
- gate_init = -4.0，lr ≈ 1e-4，dropout=0.05
- 5–10 epochs，观察 val Dice/IoU 与 gate 分布

示例命令（ISIC2017）：
```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 --expert_num 8 \
    --gspo_enable \
    --adapter_enable --adapter_dim 64 \
    --adapter_fusion_enable \
    --adapter_fusion_proj_dim 256 \
    --adapter_fusion_dropout 0.05 \
    --adapter_fusion_gate_init -4.0 \
    --output_dir outputs/fusion_gating
```

### 阶段 B（Joint 小 lr 微调）
- 基于阶段 A checkpoint，解冻 adapters/LoRA，lr 降到基线的 0.5~0.1×
- 5–15 epochs，若不稳回滚阶段 A

示例命令（继续阶段 A 产物）：
```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 --expert_num 8 \
    --gspo_enable \
    --adapter_enable --adapter_dim 64 \
    --adapter_fusion_enable \
    --adapter_fusion_proj_dim 256 \
    --adapter_fusion_dropout 0.05 \
    --adapter_fusion_gate_init -4.0 \
    --ckpt_path outputs/fusion_gating \
    --output_dir outputs/fusion_gating_joint \
    --lr 5e-5
```

### 对照与评估
- 对照：Baseline（无 Fusion） / FusionA（阶段A） / FusionB（阶段A+B）
- 每组 ≥3 seeds，记录 Dice/IoU 均值与方差；监控 gate 分布
- Ablation：推理时将单个 gate 置零，评估各 adapter 贡献

---

## 7. 关键超参建议
- proj_dim: 256（大 hidden 可用 512）
- gate_init: -4.0（σ≈0.018）
- fusion dropout: 0.05
- fusion LR: 与基线同级；joint 时 adapter/LoRA lr ≈ 0.1×
- FFN: 开启（proj_dim→4x→proj_dim, GELU），若需极简可关闭

---

## 8. 参考与外部资源
- AdapterFusion 论文与实现：
  - https://aclanthology.org/2021.eacl-main.39.pdf
  - https://github.com/adapter-hub/adapters
  - https://github.com/adapter-hub/AdapterFusion
  - https://docs.adapterhub.ml/adapter_composition.html
  - https://github.com/adapter-hub/adapters/discussions/690
- LoRA 与多 LoRA 融合：
  - https://arxiv.org/abs/2106.09685
  - https://arxiv.org/abs/2307.13269
- 现有文档补充：
  - `docs/AdapterFusion.md`（命令与对照示例）
  - 代码路径见第 2 节

---

## 9. 快速上手清单
- 打开配置：`model.sam.adapter_fusion_enabled: True`（或 CLI 开关）
- 跑阶段 A：只训 Fusion，gate_init=-4.0
- 跑阶段 B：小 lr 联训，基于阶段 A checkpoint
- 评估：Dice/IoU + gate 分布，必要时做 gate 置零 ablation


