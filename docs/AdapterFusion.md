# AdapterFusion（组合 FC + LoRA）

在conv+sam+adapter(FC Bottleneck)加LoRA on Attention dice提升了0.2%， 想再进行AdapterFusion（组合 FC + LoRA），进一步解释AdapterFusion（组合 FC + LoRA）方案在哪里加，以及是否有有效的论文和github等理论支持

### ChatGPT said:

Thought for 10s

好 — 我把 **AdapterFusion（把 FC-bottleneck adapter 与 LoRA 合并/融合）** 的原理、**精确放在哪**、实现/训练流程、工程注意点、以及你可以直接参考的论文与开源实现全给你 — 并标注出最关键的文献/代码，你能立刻开始复现/上手。

我先给结论要点，下面展开细节、可复现步骤和引用。

**结论要点（先看这三句）：**

1. **AdapterFusion 的“融合层”通常放在 transformer 的 residual/adapter 插入点（也就是你已经插入 FC-Bottleneck adapter 的相同位置），而不是直接合并到 Q/K/V 权重上。** [arXiv+1](https://arxiv.org/abs/2005.00247?utm_source=chatgpt.com)
2. **实务上常用两种融合形式**：简单 gating（sigmoid 权重）加权融合；或更复杂的 inter-adapter attention（论文中用 attention 把多个 adapter 输出组合）。两者都已有实现（AdapterFusion paper + AdapterHub 实现）。[ACL Anthology+1](https://aclanthology.org/2021.eacl-main.39.pdf?utm_source=chatgpt.com)
3. **做法建议（工程化）**：先单独训练 FC-adapter 与 LoRA（或用已有 checkpoint），再训练 fusion 层（或用小 lr joint-train）；gating 初始值小（例如 sigmoid(g)=0.01）以保证稳定性。实现参考 AdapterHub/AdapterFusion + LoRA repo。[AdapterHub Documentation+1](https://docs.adapterhub.ml/training.html?utm_source=chatgpt.com)

---

## 1) AdapterFusion 到底“加在哪里” — 精确位置说明

- 你当前 pipeline 有：`encoder (Conv-LoRA on conv layers)` + `SAM decoder` + `FC-Bottleneck adapter`（我理解你把 adapter 插在 decoder 的 residual/MLP 路径或 decoder block 里的某个位置）。
- **AdapterFusion 要放在同一个 adapter 插入点**（也就是 FC-bottleneck 那个位置）：它不会直接改动 attention 的 Q/K/V 矩阵，而是把 **多个 adapter 的输出（FC adapter 的输出、LoRA 的 adapter 输出等）在该 residual 路径上融合**，然后加回到主干（residual add）。这种做法是 AdapterFusion 原论文与实现的设计意图。[ACL Anthology+1](https://aclanthology.org/2021.eacl-main.39.pdf?utm_source=chatgpt.com)

画个简化图（文字版）：

```
x --> (LayerNorm) --> Attention --> ResidualAdd -->  [ ---- adapter fusion 输入 ---- ]
                                                       ↑
            main path output ---------------------------|
adapter outputs:  adapter_FC(x) , adapter_LoRA(x), ...  --> fusion_layer(...) --> fused_out
fused_out 加回 residual path  → 下一层

```

- **重点**：LoRA on attention（你已有）是改 Q/K/V 权重的 low-rank 增量；而 AdapterFusion 把 **adapter 输出层级（通常是 MLP/bottleneck 输出）** 作为输入，进行跨-adapter 融合。两者可以并行共存且互补。[arXiv+1](https://arxiv.org/abs/2106.09685?utm_source=chatgpt.com)

---

## 2) AdapterFusion 的融合方式（两类常见实现）

1. **Gating / Weighted Sum（最简单、工程可行）**
    - 每个 adapter 的输出经线性投影后按权重相加：`y = Σ σ(g_i) * o_i`，其中 `σ(g_i)` 是 sigmoid(gate_i)（常把 gate 的初始值设小，如 g_i≈-4 → σ≈0.018）。这样训练开始时主模型几乎不受影响，fusion 学会时逐步打开。AdapterHub 提供此类实现/示例。[AdapterHub Documentation+1](https://docs.adapterhub.ml/adapter_composition.html?utm_source=chatgpt.com)
2. **Inter-Adapter Attention / Attention-based Fusion（论文方案）**
    - AdapterFusion 原文中提出在不同层用 attention 机制把各 adapter 的隐藏表示相互对齐并融合（paper 图示与实验表明能组合不同任务知识）。这在复杂多任务或需要捕捉 adapter 之间交互时更有效，但实现更复杂、计算开销更大。论文 + AdapterHub 实现里有参考。[ACL Anthology+1](https://aclanthology.org/2021.eacl-main.39.pdf?utm_source=chatgpt.com)

另外，近期也有 LoRA-fusion 专门方案（把多个 LoRA 模块动态组合 / 通过矩阵合并等），相关论文/实现（如 LoraHub / LoRA fusion papers）也证明融合多 LoRA 能带来性能提升。[arXiv+1](https://arxiv.org/abs/2307.13269?utm_source=chatgpt.com)

---

## 3) 训练流程（推荐稳妥做法）

**原则**：先让每个 adapter 学会自己任务，再训练 fusion（两阶段稳定法）；或者小 lr 联合训练（若资源或时间受限）。

**步骤 A（推荐、最稳）**

1. 训练/微调 `FC-Bottleneck adapter`（你原来已有结果）。保存 adapter 权重。[AdapterHub Documentation](https://docs.adapterhub.ml/training.html?utm_source=chatgpt.com)
2. 单独训练 `LoRA on attention`（或已在用 conv-LoRA 的继续），保存权重。[arXiv](https://arxiv.org/abs/2106.09685?utm_source=chatgpt.com)
3. 加入 AdapterFusion 层（gating 或 attention fusion），**冻结原来的 adapter**（或把它们设置为较小 lr），只训练 fusion 层若干轮，让 fusion 学会如何组合已有 adapter 的表示。[AdapterHub Documentation](https://docs.adapterhub.ml/training.html?utm_source=chatgpt.com)
4. 可选：解冻所有 adapter，用非常小的 lr（比如原 lr×0.1）做微调以微调协同效应。监控 val Dice、KL、importance ratios 等。

**步骤 B（快捷）**

- 直接在原模型上新增 fusion 层，joint train fusion + adapters，但用**lower lr**（0.1×）和 gating 初始值小，观察稳定性。若不稳再退回两阶段。[GitHub](https://github.com/adapter-hub/adapters/discussions/690?utm_source=chatgpt.com)

---

## 4) 超参与实现细节（Practical tips）

- **gating 初始化**：`g_gate = -4`（sigmoid≈0.018）或更保守 `6`。这样 fusion 开始时对主干影响微乎其微。[AdapterHub Documentation](https://docs.adapterhub.ml/training.html?utm_source=chatgpt.com)
- **fusion 层容量**：线性投影后再融合（例如把每个 adapter 输出先投到同一低维空间再加权），避免直接在高维做 expensive ops。
- **训练 lr 策略**：fusion 层 lr = baseline_lr; 若 joint train，adapter lr = 0.1×baseline_lr。
- **正则**：加 dropout（0.05）和 weight decay（1e-5）以防过拟合（adapterFusion 给出的经验）。[AdapterHub Documentation](https://docs.adapterhub.ml/training.html?utm_source=chatgpt.com)
- **监控**：除了 Dice，还要监控每个 adapter 的 contribution（sigmoid(g_i) 值）、fusion attention weights（若用 attention fusion）、KL/importance ratio（若你在做 RL 优化）。若某 adapter gate 长期接近0，说明它没贡献或需增大 capacity/重新训练。[GitHub](https://github.com/adapter-hub/adapters/discussions/690?utm_source=chatgpt.com)

---

## 5) 论文 & 开源实现（你可以直接看/clone 并参考）

- **AdapterFusion paper** — Pfeiffer et al., “AdapterFusion: Non-Destructive Task Composition for Transfer Learning”. arXiv / EACL.（原理与 layer-wise fusion 细节）。[arXiv+1](https://arxiv.org/abs/2005.00247?utm_source=chatgpt.com)
    
    Paper: [ACL Anthology](https://aclanthology.org/2021.eacl-main.39.pdf?utm_source=chatgpt.com)
    
- **AdapterHub / AdapterTransformers** — 官方实现、文档与训练指南（包含 AdapterFusion 的用法与注意事项）。非常适合直接复现 gating/attention fusion。
    
    Repo & docs: [GitHub+1](https://github.com/adapter-hub/adapters?utm_source=chatgpt.com)
    
- **AdapterFusion GitHub（示例/实现）** — adapter-hub/AdapterFusion repo（含论文 PDF 与示例）。[GitHub](https://github.com/adapter-hub/AdapterFusion?utm_source=chatgpt.com)
- **LoRA 原始论文 + 实现** — “LoRA: Low-Rank Adaptation of Large Language Models” (Hu et al., ICLR 2022) + Microsoft/LoRA repo。LoRA 是常用的 adapter/low-rank 插件，与你的 LoRA on attention 直接相关。[arXiv+1](https://arxiv.org/abs/2106.09685?utm_source=chatgpt.com)
- **LoRA-fusion / multi-LoRA works** — LoraHub、I-LoRA / LoRA adapter fusion papers 显示多 LoRA 组合可带来好处；也有 2024-2025 年的实证论文讨论 LoRA fusion 在 instruction tuning 的效果。可作为多-LoRA 融合参考。[arXiv+1](https://arxiv.org/abs/2307.13269?utm_source=chatgpt.com)
- **AdapterFusion 在 HuggingFace 上的 collection**（方便找预训练 adapters）也可参考。[Hugging Face](https://huggingface.co/collections/AdapterHub/adapterfusion-adapters?utm_source=chatgpt.com)

---

## 6) 在你当前 conv-LoRA + SAM + FC-adapter pipeline 上的具体建议（一步步可跑）

1. **Baseline**：你当前 Conv-LoRA + GSPO + FC-adapter Dice ≈ 0.8581；加 LoRA on attention 后 +0.2%（你已测）。很合理。
2. **试方案 1（轻量）**：把新训练的 LoRA（attention）和已训练 FC adapter 一起做 **gating fusion**（fusion 层放在 same adapter insertion points），只训练 fusion 层（freeze adapters），观察 Dice。
3. **试方案 2（深度）**：若方案1有效，解冻 adapters 做小 lr 微调 + fusion 训练（5–10 epochs），看是否有进一步提升。
4. **试方案 3（attention fusion）**：若需要更强组合能力（或做多任务），实施 inter-adapter attention fusion（复杂但可能带更大收益）。
5. **评估**：用你已有的 paired bootstrap + 多 seed 判定（见之前我们讨论的统计方法），关注 fusion gate 值与每个 adapter 的贡献。[AdapterHub Documentation+1](https://docs.adapterhub.ml/training.html?utm_source=chatgpt.com)

---

## 7) 预期收益 & 风险

- **预期收益**：AdapterFusion 在 NLP/LLM 和部分 PEFT 场景被证明能把多个 adapter 的优势叠加，现实提升通常在 **0.2%–1.5%（task/数据依赖）**。对你已经有 +0.2% 的 LoRA on attention，相信 AdapterFusion 有潜力再带来几十分点到一百分点的提升，但需多 seed 验证。[ACL Anthology+1](https://aclanthology.org/2021.eacl-main.39.pdf?utm_source=chatgpt.com)
- **风险**：fusion 增加训练复杂度、可能过拟合、可能需要更长训练时间；若 fusion 训练不当（直接 joint 无保守 lr），会造成不稳定。用 gating 初始小 + 两阶段训练能显著降低风险。[GitHub](https://github.com/adapter-hub/adapters/discussions/690?utm_source=chatgpt.com)

---

## 8) 直接可用的资源（即刻行动）

- AdapterFusion paper (read first): [ACL Anthology](https://aclanthology.org/2021.eacl-main.39.pdf?utm_source=chatgpt.com)
- AdapterHub / docs (how-to + code snippets): [AdapterHub Documentation+1](https://docs.adapterhub.ml/adapter_composition.html?utm_source=chatgpt.com)
- LoRA paper + Microsoft LoRA repo: [arXiv+1](https://arxiv.org/abs/2106.09685?utm_source=chatgpt.com)
- LoraHub / LoRA fusion recent work (参考多 LoRA 组合): [arXiv+1](https://arxiv.org/abs/2307.13269?utm_source=chatgpt.com)

---

## 9) 实验配置与命令示例（AdapterFusion gating 版）

### 新增 CLI 开关（已接入 `examples/automm/Conv-LoRA/run_semantic_segmentation.py`）
- `--adapter_fusion_enable`：启用 AdapterFusion（gating）
- `--adapter_fusion_proj_dim`：投影维度，默认 256
- `--adapter_fusion_dropout`：fusion 输出 dropout，默认 0.05
- `--adapter_fusion_gate_init`：gate 初始化，默认 -4.0
- `--adapter_fusion_disable_ffn`：关闭 fusion 内部的可选 FFN（默认开启）

### 推荐超参
- proj_dim=256，dropout=0.05，gate_init=-4.0，use_ffn=True
- 训练 lr 与基线一致；若 joint 不稳，可把 adapter/fusion 相关 lr 调低到 0.5~0.1×

### 命令示例（ISIC2017）
```bash
# Baseline（无 Fusion，Conv-LoRA + Adapter + GSPO）
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 --expert_num 8 \
    --gspo_enable \
    --adapter_enable --adapter_dim 64 \
    --output_dir outputs/baseline

# FusionA：启用 AdapterFusion（gating）
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

# 阶段B（继续在阶段A checkpoint 上 joint 微调，小 lr 解冻）
# 手动设置较小 lr（示例：基线 1e-4 -> 5e-5），并允许 adapter/LoRA 训练
# 先把阶段A产出的 checkpoint 路径作为 ckpt_path 继续训练（示例假设输出在 outputs/fusion_gating）
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

### 对照与分析
- 对照：Baseline（无 fusion） vs FusionA（开启 fusion）。每个配置跑 ≥3 seeds，记录 Dice/IoU 均值与方差。
- Ablation：训练好 fusion 后查看 gate 分布；推理时可将单个 gate 置零以评估各 adapter 贡献。

---

如果你愿意，我可以下一步直接帮你做其中一项（任选其一）：

A. **给出 AdapterFusion（gating）在你 SAM 代码里的 pseudo-code / patch（PyTorch）**，方便你直接插入并跑试验；

B. **写一个两阶段训练脚本（shell + HF/AdapterHub 命令）**，包含 freeze、fusion training、unfreeze 微调步骤；

C. **做一个小的超参表（fusion lr、gate init、dropout、是否解冻）并按优先级排序，直接给你可跑的 sweep**。

回一个字母选项（A/B/C）我就马上把可直接运行的代码/脚本给你。

### You said:

是再加一个AdapterFusion放在transformer 的 residual/adapter 插入点，还是把原来的FC-bottleneck adapter 与 LoRA去掉，只在transformer 的 residual/adapter 插入点加AdapterFusion

### ChatGPT said:

短答先给结论：**不要把已有的 FC-bottleneck adapter 与 LoRA 一刀切去掉。先把 AdapterFusion *加上去*（放在你现有 adapter 的插入点，把各 adapter 的输出融合），按两阶段训练流程去试——这既能保留已学到的有用表征，又能让 fusion 学会如何组合/筛选，风险最低、可解释性最佳。**

下面是完整理由、精确放置位置、实现/训练流程、超参建议、以及什么时候才考虑“移除原 adapter 做纯 AdapterFusion”的判定标准。

---

## 1) 为什么先**加上去**而不是先全部替换？

- **保留已学知识**：你当前的 FC-bottleneck 已经带来 +0.7%（或在某处有收益），LoRA on attention 也有 +0.2%。直接删除会丢掉这些已有收益，回炉重训风险高。
- **更低风险的实验性路径**：AdapterFusion 设计初衷是“组合不同 adapter”，就是为并存场景服务，使用 gating 可以逐步打开新路径，稳定性更好。
- **可做因果分析**：加上 fusion 后可以观测每个 adapter 的 gate 值来判断哪个 adapter 真有用，便于后续决定是否移除某个 adapter（数据驱动）。
- **工程成本更低**：实现 fusion 在原 adapter 插入点最自然，少改主干结构，便于回滚和对照实验。

结论：**先加、再评估、再决定删或保留。**

---

## 2) 精确放置位置（exact）

- **放在 transformer 的 residual / adapter 插入点**（也就是你现在插入 FC-bottleneck adapter 的同一位置）。
- 输入是该层的 hidden state hhh。各 adapter（FC adapter 输出 oFC(h)o_{FC}(h)oFC(h)、LoRA adapter 输出 oLoRA(h)o_{LoRA}(h)oLoRA(h)、可能还有 conv-lora 的输出 if it produces residuals）分别产生向量，AdapterFusion 将这些向量融合成单一 `fused_out`，然后做 residual add 回主干：
    
    ```
    fused = AdapterFusion([o_FC, o_LoRA, ...])
    output = main_path_output + fused
    
    ```
    

不要把 fusion 替换成直接在 Q/K/V 合并那类改动——那是不同语义（attention-weight vs adapter residual）。

---

## 3) 推荐的 AdapterFusion 结构（实用版/gating）

- **每个 adapter 输出先过一个线性投影到 shared dim d_f（例如 256）**
- **每个分支有一个 gate scalar g_i（可训练）**，融合为：
    
    ```
    scores = sigmoid(g)  # elementwise per adapter or per-head scalar
    fused = sum_i scores[i] * proj_i(o_i)
    fused = LayerNorm(fused)
    fused = FeedForward(fused)  # optional small FFN
    
    ```
    
- gating init：`g_i = -4.0`（sigmoid ≈ 0.018）——即训练初期影响很小。
- optional: **per-channel gate** (vector) if you want fine granularity, but scalar per-adapter is simplest and robust.

---

## 4) 两阶段训练流程（强烈推荐）

**阶段 A（最稳） — Fusion warmup**

- Freeze existing adapter weights (FC + LoRA + conv-loRA).
- Initialize fusion gating to -4. Train only fusion params (projections, gates, fusion FFN).
- LR_fusion = 1e-4 (或与你 adapter lr 同量级)，epochs = small (e.g. 5–10) 根据 val 观察。

**阶段 B — Joint fine-tune（可选）**

- Unfreeze adapters, but set adapter_lr = 0.1 × base_lr；fusion_lr keep as is or reduce slightly.
- Train a few more epochs (5–15) 监控稳定性。若不稳可回滚到阶段A checkpoint。

**好处**：阶段A能让 fusion学会组合而不破坏已学权重；阶段B再细调协同效应。

---

## 5) 关键超参（直接可用）

- shared proj dim `d_f = 256`（若hidden dim很大，可用 256 或 512）
- gate init `g_init = -4.0`
- fusion LR `= 1e-4`（AdamW, weight_decay=1e-6）
- adapter LR when unfreezing `= base_lr * 0.1`（例如 base_lr=1e-4 → adapter_lr=1e-5）
- dropout on fused_out `= 0.05`
- FFN in fusion: one hidden layer, hidden = 4*d_f, GELU activation（optional）
- epochs stageA: 5–10; stageB: 5–15（视val变化调整）
- seeds: ≥3 seeds per config；使用 paired bootstrap 判定效果

---

## 6) 监控指标（你必须看）

- **Val Dice_mean & Dice_std**（核心）
- **Gate values σ(g_i)**：若某 gate ≈0长期不变，说明对应 adapter没被利用
- **Per-image diff distribution**（paired bootstrap CI）
- **Training stability**：loss曲线、KL term（如果有RL优化）、importance ratios
- **Adapter contribution time series**（观察 gate 从-4→逐步打开的路径）

---

## 7) 何时考虑删除原 adapter（把 adapter 换成只有 fusion）

做两步 AB 判断后再决定：

- 若 fusion 学习后 **某 adapter gate长期 ≈0** 或者取消该 adapter并重训后 **性能不降甚至更好** → 可安全删除它。
- 若 fusion 学习后保留所有 adapter 的 gate 都有显著值，说明它们都是互补的，保留组合。
- 仅当你需要降低参数量/推理成本时，才做“替换实验”（把某些 adapter 移除，保留 fusion-only），并且**先在小规模上做 ablation**再全量跑。

---

## 8) 简短 PyTorch 风格 pseudo-code（fusion module）

```python
import torch.nn as nn
import torch

class AdapterFusion(nn.Module):
    def __init__(self, input_dim, proj_dim=256, use_ffn=True):
        super().__init__()
        self.n_adapters = None  # set at init_adapter_registration
        self.proj_dim = proj_dim
        self.proj_layers = nn.ModuleList()  # one per adapter
        self.gates = nn.ParameterList()     # one scalar param per adapter
        self.layernorm = nn.LayerNorm(proj_dim)
        self.use_ffn = use_ffn
        if use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(proj_dim, proj_dim*4),
                nn.GELU(),
                nn.Linear(proj_dim*4, proj_dim)
            )
        self.dropout = nn.Dropout(0.05)

    def register_adapter(self, adapter_output_dim):
        # Call this for each adapter at model setup time
        self.proj_layers.append(nn.Linear(adapter_output_dim, self.proj_dim))
        g = nn.Parameter(torch.tensor(-4.0))  # init gate small
        self.gates.append(g)
        self.n_adapters = len(self.proj_layers)

    def forward(self, adapter_outputs: List[Tensor]):
        # adapter_outputs: list of tensors [B, seq, dim_i]
        assert len(adapter_outputs) == self.n_adapters
        projs = []
        for out, proj in zip(adapter_outputs, self.proj_layers):
            p = proj(out)  # -> [B, seq, proj_dim]
            projs.append(p)
        gates = torch.stack([torch.sigmoid(g) for g in self.gates])  # [n]
        # broadcast gates to [n, 1, 1] then weight sum
        stack = torch.stack(projs, dim=0)  # [n, B, seq, proj_dim]
        weighted = (gates[:, None, None, None] * stack).sum(dim=0)  # [B, seq, proj_dim]
        x = self.layernorm(weighted)
        if self.use_ffn:
            x = self.ffn(x)
        x = self.dropout(x)
        return x

```

在每个 adapter 插入点，把各 adapter 的输出收集后调用 `AdapterFusion.forward()`，然后 `residual += fused`.

---

## 9) 风险/回滚策略

- 若在 stageA val Dice 未提升或下降：立即停止并检查 gate distribution（可能 fusion架构/ proj_dim不合适或需要更长训练）。可尝试增大 proj_dim 或增加训练步。
- 若 stageB joint 微调时发散：立刻回滚到 stageA checkpoint，并降低 adapter_lr 或在 joint 训练里增加梯度裁剪。

---

## 10) 实验对照建议（必做）

- Baseline (no fusion) — current conv-LoRA + FC adapter + LoRA(on attention)
- Fusion1 — Fusion added, freeze adapters, train fusion only
- Fusion2 — Fusion added, unfreeze adapters with adapter_lr = 0.1×base_lr
- Ablation — After Fusion2 converged：zero out each gate in turn and measure delta (per-adapter ablation)
- Optional Replacement — If ablation shows no loss removing adapter X, try full retrain with adapter X removed.

每个配置跑 3 seeds，做 paired bootstrap CI 判定。

---

如果你想我现在做哪一项帮你准备：

A) 生成你项目代码里具体 patch（把 AdapterFusion 插到你现有 adapter 插入点的 pseudo-patch，含注册 adapter 输出）

B) 给出一份 Slurm/experiment yaml（stageA + stageB + ablation configs，含 seed & metrics）

C) 都要（A + B）

回一个字母我立刻把相应内容发给你。