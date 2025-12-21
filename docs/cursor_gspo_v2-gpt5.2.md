# Untitled Chat
_Exported on 12/20/2025 at 23:47:52 PST from Cursor (2.2.43)_

---

**User**

从@docs/cursor_gspo_adapter_convlora_followingups-251130.md 和@docs/cursor_gspo_adapter_convlora-251129.md 以及 examples/automm/Conv-LoRA下的代码了解之前的上下文。然后研究并回答：在当前的模型和训练方案设计中，被GSPO强化学习微调的是哪一部分？

---

**Cursor**

我先快速把两份设计文档和 `examples/automm/Conv-LoRA` 下面的关键训练入口/模型封装代码读一遍，梳理“监督微调 vs GSPO 强化学习微调”的参数解冻/优化器参数组设置；然后给出结论：**GSPO 实际更新的是哪些参数（基座、Adapter/LoRA、价值头/奖励头等）**，并用代码位置佐证。

我已经定位到两条关键线索：文档里明确写了 **“Conv-LoRA（Attention 的 qkv 线性层，含 MoE-Conv + GSPO）”**，说明 GSPO 是挂在 Conv-LoRA 这条 PEFT 分支上的；接下来我会从 `gspo_trainer.py` / `run_semantic_segmentation.py` 把“哪些参数进 optimizer、哪些冻结”抠出来，给你一个可被代码验证的结论。

### 结论：GSPO 强化学习微调的是哪一部分？
在你当前的模型与训练方案里，**被 GSPO“强化学习式”微调的核心对象是 Conv-LoRA 的 MoE 门控/专家选择机制（MoEGate）**，它位于 **SAM Vision Encoder 的 Attention q/k/v 投影所注入的 Conv-LoRA 分支**；换句话说，GSPO在优化“选哪个（些）卷积专家”的策略，并通过组内优势加权让这部分获得更强/更弱的梯度信号。

- 文档层面已经把 GSPO 明确绑定在 **Vision Encoder 的 Conv‑LoRA（Attention qkv）** 上：  
```160:168:/Users/domqiu/.cursor/worktrees/autogluon/uyu/docs/cursor_gspo_adapter_convlora_followingups-251130.md
SAM Vision Encoder（ViT 主干，冻结）
  ├─ Conv-LoRA（Attention 中 qkv 线性层，含 MoE-Conv + GSPO）
  └─ Encoder-Adapter（可选，插在每个 SamVisionLayer 的 MLP 之后）
...
SAM Mask Decoder（Two-Way Transformer）
...
```

### 代码证据：GSPO具体更新哪些参数？
在 `ConvLoRALinear` 中，**基座线性层权重被显式冻结**，训练（也包括 GSPO 阶段）更新的是 Conv‑LoRA 分支参数：`lora_A/lora_B`、MoE gate 的 `w_gate/w_noise`，以及 `lora_moe_experts` 的卷积专家参数。  
```692:718:/Users/domqiu/.cursor/worktrees/autogluon/uyu/multimodal/src/autogluon/multimodal/models/adaptation_layers.py
# Actual trainable parameters
if r > 0:
    # Sec. 3: Low-rank LoRA factors around a frozen base linear
    self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
    self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
    self.scaling = self.lora_alpha / self.r
    # Freezing the pre-trained weight matrix
    self.weight.requires_grad = False

    # Sec. 3.2: MoE-Conv experts with Noisy-TopK gating
    # GSPO: Pass GSPO parameters to MoEGate
    topk = 1
    self.lora_moe_gating = MoEGate(
        M=conv_lora_expert_num, 
        d=self.r, 
        K=topk,
        gspo_enabled=gspo_enabled,
        gspo_group_size=gspo_group_size,
        gspo_quality_momentum=gspo_quality_momentum
    )
    self.lora_moe_experts = nn.ModuleList([])
```

`MoEGate` 的**可训练参数**就是门控网络参数（相当于“策略网络”）：  
```823:886:/Users/domqiu/.cursor/worktrees/autogluon/uyu/multimodal/src/autogluon/multimodal/models/adaptation_layers.py
self.w_gate = nn.Parameter(torch.zeros(d, M), requires_grad=True)
self.w_noise = nn.Parameter(torch.zeros(d, M), requires_grad=True)
...
# GSPO: Add quality-aware bias to encourage high-performing experts
if self.gspo_enabled and self.training:
    quality_bias = self._compute_quality_bias()
    clean_logits = clean_logits + quality_bias.unsqueeze(0)
...
top_k_indices = top_indices[:, : self.k]
...
return gates, loss, top_k_indices
```

另外，GSPO还会通过 `update_quality_history` **更新门控里的“专家质量历史”缓冲区**（这是“强化学习反馈环”，但不是可学习参数）：  
```979:1004:/Users/domqiu/.cursor/worktrees/autogluon/uyu/multimodal/src/autogluon/multimodal/models/adaptation_layers.py
def update_quality_history(self, selected_experts, quality_scores):
    if not self.gspo_enabled or not hasattr(self, 'expert_quality_history'):
        return

    with torch.no_grad():
        ...
        self.expert_quality_history[expert_id] = (
            self.gspo_quality_momentum * current_quality +
            (1 - self.gspo_quality_momentum) * quality
        )
        self.expert_usage_count[expert_id] += 1
```

### 训练流程层面：GSPO怎么“只针对这块”起作用？
Lightning 的语义分割训练 step 会在 warmup 后切到 GSPO：**同一张图像做 G 次前向，算质量→优势→加权 loss**，然后反传；再把质量反馈写回 MoEGate 的历史缓冲。  
```313:413:/Users/domqiu/.cursor/worktrees/autogluon/uyu/multimodal/src/autogluon/multimodal/optim/lit_semantic_seg.py
use_gspo = (
    self.gspo_trainer is not None and 
    self.gspo_trainer.is_gspo_active(self.current_epoch)
)

if use_gspo:
    loss, metrics, selected_experts = self._gspo_training_step(batch)
...
loss, metrics, selected_experts_groups = self.gspo_trainer.gspo_group_training_step(...)
...
self.gspo_trainer.update_expert_feedback(
    selected_experts_groups, quality_scores, self.model
)
```

### 一句话回答你问的“哪一部分”
**GSPO 强化学习微调的是：Conv‑LoRA 的 MoE 门控（MoEGate：`w_gate/w_noise` + 质量历史驱动的 bias）及其所驱动的 Conv‑LoRA 专家分支；它作用位置是 SAM Vision Encoder 的 Attention q/k/v 注入点。**  
Adapter（Encoder/Decoder Adapter）是并行的 PEFT 分支，是否训练由你开关决定，但 **“GSPO 这套 RL 机制”本身是为 Conv‑LoRA 的 MoE 选择策略服务的**。

```plaintext
SAM Vision Encoder（ViT 主干，冻结）
  ├─ Conv-LoRA（Attention 中 qkv 线性层，含 MoE-Conv + GSPO）
  └─ Encoder-Adapter（可选，插在每个 SamVisionLayer 的 MLP 之后）
...
SAM Mask Decoder（Two-Way Transformer）
...
```

```python
# Actual trainable parameters
if r > 0:
    # Sec. 3: Low-rank LoRA factors around a frozen base linear
    self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
    self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
    self.scaling = self.lora_alpha / self.r
    # Freezing the pre-trained weight matrix
    self.weight.requires_grad = False

    # Sec. 3.2: MoE-Conv experts with Noisy-TopK gating
    # GSPO: Pass GSPO parameters to MoEGate
    topk = 1
    self.lora_moe_gating = MoEGate(
        M=conv_lora_expert_num, 
        d=self.r, 
        K=topk,
        gspo_enabled=gspo_enabled,
        gspo_group_size=gspo_group_size,
        gspo_quality_momentum=gspo_quality_momentum
    )
    self.lora_moe_experts = nn.ModuleList([])
```

```python
self.w_gate = nn.Parameter(torch.zeros(d, M), requires_grad=True)
self.w_noise = nn.Parameter(torch.zeros(d, M), requires_grad=True)
...
# GSPO: Add quality-aware bias to encourage high-performing experts
if self.gspo_enabled and self.training:
    quality_bias = self._compute_quality_bias()
    clean_logits = clean_logits + quality_bias.unsqueeze(0)
...
top_k_indices = top_indices[:, : self.k]
...
return gates, loss, top_k_indices
```

```python
def update_quality_history(self, selected_experts, quality_scores):
    if not self.gspo_enabled or not hasattr(self, 'expert_quality_history'):
        return

    with torch.no_grad():
        ...
        self.expert_quality_history[expert_id] = (
            self.gspo_quality_momentum * current_quality +
            (1 - self.gspo_quality_momentum) * quality
        )
        self.expert_usage_count[expert_id] += 1
```

```python
use_gspo = (
    self.gspo_trainer is not None and 
    self.gspo_trainer.is_gspo_active(self.current_epoch)
)

if use_gspo:
    loss, metrics, selected_experts = self._gspo_training_step(batch)
...
loss, metrics, selected_experts_groups = self.gspo_trainer.gspo_group_training_step(...)
...
self.gspo_trainer.update_expert_feedback(
    selected_experts_groups, quality_scores, self.model
)
```

---

**User**

本项目中加入了一类FC Adapter，它是加在哪里的？

---

**Cursor**

我会直接从代码里把 FC Adapter 的“注入位置”和“前向计算位置”两处都定位出来：一处是在构建 `SAMForSemanticSegmentation` 时把 `AdapterLayer` 挂到哪些层上，另一处是在每个 block 的 forward 里它到底加在 MLP 的哪里。接着用代码片段给出结论。

### FC Adapter 加在了哪里？
本项目里的 **FC Adapter（`AdapterLayer`）**主要加在两处（按你现在的实现/开关）：

- **加在 SAM 的 Vision Encoder（ViT）每一层的 MLP 路径上（核心）**：也就是每个 `SamVisionLayer` 里，先算 `mlp_out = self.mlp(layer_norm2(hidden_states))`，再并行算 `adapter_out = self.adapter(hidden_states)`，最后做三路残差相加：

```1035:1044:/Users/domqiu/.cursor/worktrees/autogluon/uyu/multimodal/src/autogluon/multimodal/models/custom_hf_models/modeling_sam_for_conv_lora.py
# MLP Block + Parallel Adapter
residual = hidden_states
hidden_states = self.layer_norm2(hidden_states)
mlp_out = self.mlp(hidden_states)

if self.adapter is not None:
    adapter_out = self.adapter(hidden_states)
    hidden_states = residual + mlp_out + adapter_out
else:
    hidden_states = residual + mlp_out
```

同时，模型构建时会把这个 adapter 挂到 **vision_encoder 的每个 layer** 上（因此是“每层一个”）：

```349:355:/Users/domqiu/.cursor/worktrees/autogluon/uyu/multimodal/src/autogluon/multimodal/models/custom_hf_models/modeling_sam_for_conv_lora.py
# Inject Adapters into vision encoder if enabled
if self.adapter_enabled:
    logger.info(f"Injecting Adapters with dim={self.adapter_dim} into Vision Encoder")
    for layer in self.model.vision_encoder.layers:
        # Manually create and assign adapter since config didn't have it at init time
        layer.adapter = AdapterLayer(self.config.hidden_size, self.adapter_dim)
```

- **（可选）加在 SAM Mask Decoder 的 Two-Way Transformer 的 queries MLP 路径上**：开启 `decoder_adapter_enabled` 时，会创建 `adapter_queries`，并在 decoder 的 MLP 后把 `adapter_out` 加到 `queries` 上（也是并行加法）：

```335:379:/Users/domqiu/.cursor/worktrees/autogluon/uyu/multimodal/src/autogluon/multimodal/models/custom_hf_models/modeling_sam_for_conv_lora.py
decoder_adapter_enabled = getattr(config, "decoder_adapter_enabled", False)
decoder_adapter_dim = getattr(config, "decoder_adapter_dim", getattr(config, "adapter_dim", 64))
if decoder_adapter_enabled:
    self.adapter_queries = AdapterLayer(self.hidden_size, decoder_adapter_dim)
...
adapter_out = self.adapter_queries(queries)
queries = queries + mlp_out + adapter_out
```

总结一句：**FC Adapter 的默认/主要位置是 Vision Encoder 每层的 MLP 之后（与 MLP 并行，三路残差相加）；另外还支持（可选）在 Mask Decoder 的 queries MLP 之后再加一组 FC Adapter。**

```python
# MLP Block + Parallel Adapter
residual = hidden_states
hidden_states = self.layer_norm2(hidden_states)
mlp_out = self.mlp(hidden_states)

if self.adapter is not None:
    adapter_out = self.adapter(hidden_states)
    hidden_states = residual + mlp_out + adapter_out
else:
    hidden_states = residual + mlp_out
```

```python
# Inject Adapters into vision encoder if enabled
if self.adapter_enabled:
    logger.info(f"Injecting Adapters with dim={self.adapter_dim} into Vision Encoder")
    for layer in self.model.vision_encoder.layers:
        # Manually create and assign adapter since config didn't have it at init time
        layer.adapter = AdapterLayer(self.config.hidden_size, self.adapter_dim)
```

```python
decoder_adapter_enabled = getattr(config, "decoder_adapter_enabled", False)
decoder_adapter_dim = getattr(config, "decoder_adapter_dim", getattr(config, "adapter_dim", 64))
if decoder_adapter_enabled:
    self.adapter_queries = AdapterLayer(self.hidden_size, decoder_adapter_dim)
...
adapter_out = self.adapter_queries(queries)
queries = queries + mlp_out + adapter_out
```

---

**User**

请研究如何使GSPO强化学习不仅仅用来微调 Conv-LoRA的MoE 还同时微调 FC Adapter？可能带来的好处是什么？

---

**Cursor**

我会先澄清一个关键点：**当前 GSPO 的“组内优势加权 loss”其实已经会对所有可训练参数回传梯度**（包括 FC Adapter），只是“质量历史反馈”目前只写回到 `MoEGate`。接着我给你几种把 GSPO 显式扩展到 Adapter 的可行设计（从轻量到激进），以及各自可能收益与风险。

### 先澄清：现在的 GSPO 其实“已经会更新 FC Adapter”，但方式不同
你们当前的 GSPO 训练步是**用组内质量（IoU/Dice 等）算 advantage，然后对分割损失做加权**，最后对 `total_loss` 反传：

- 这意味着：**只要 FC Adapter 参数在 optimizer 里且 `requires_grad=True`，它就会被 GSPO 这套加权 loss 间接“强化”更新**（因为梯度从 `pred_masks` 反传会经过 Adapter 分支）。  
- 但同时也确实存在你说的“仅仅用来微调 Conv‑LoRA 的 MoE”：目前只有 **MoEGate 有“质量历史 feedback”机制**（`update_quality_history` 更新 buffer 并影响 gate logits）；Adapter 没有类似的“策略/动作 + 反馈回路”。

你要做的，是把 GSPO 从“只对 MoE 路由有显式 RL feedback”，扩展到“对 Adapter 也有显式的策略优化/反馈”。

---

### 目标：让 GSPO 同时“微调 FC Adapter”的几种可行设计（从易到难）

#### 方案 A（最小改动，立刻能用）：让 Adapter 参与 GSPO 的“组内探索”
GSPO 的组采样目前靠 **dropout 的随机性**制造多次 forward 的差异。`AdapterLayer` 本身支持 `dropout` 和可学习 `scale`：

- **做法**：把 Adapter 的 dropout 打开（如 0.1），保证同一张图像在 G 次 forward 中 Adapter 路径会产生差异；GSPO 的 advantage-weighted loss 会自然偏向“产生更高质量预测”的那些 Adapter 随机路径，从而**对 Adapter 参数形成更强的质量驱动更新**。
- **局限**：这是“隐式”RL（本质还是加权监督损失），没有把 Adapter 的某个决策变量当作 policy 来学。

**可能收益**
- 更稳：几乎不改架构，只是增强 exploration。
- 让 Adapter 的更新更“质量导向”，减少它在早期对主干表征的破坏。

---

#### 方案 B（推荐，最像你现在的 MoE-GSPO）：把 Adapter 做成“可选择的策略”（Adapter-MoE），用 GSPO 优化选择
把每层的 FC Adapter 从“单个 bottleneck”升级为“多个 Adapter 专家”，再加一个轻量 gate：

- **动作/策略**：对每个样本（或每个 batch）在第 i 层从 \(M\) 个 Adapter 专家里选 Top‑K（K 可设为 `gspo_group_size` 或固定 1）。
- **奖励/质量**：沿用你们已有的 shaped_quality（IoU/Dice + 边界/平滑 shaping）。
- **优化**：
  - gate 参数（类似 `MoEGate.w_gate/w_noise`）用 advantage-weighted 的梯度 + 质量历史 bias（完全照搬 MoE gate 的套路）。
  - 被选中的 Adapter 专家参数也会通过反传更新。

**需要改哪些地方（高层）**
- `AdapterLayer` → `MoEAdapterLayer(M experts + gate)`（新增一个“AdapterGate”，可以比 `MoEGate` 更简单，比如 token-mean 特征 → linear logits）。
- `SamVisionLayer` 的 MLP 残差处，从 `adapter_out = adapter(x)` 变为 `adapter_out = moe_adapter(x)`（返回 `selected_adapters` 供 GSPO 记录）。
- GSPO 的 `update_expert_feedback` 目前只找 `hasattr(module,'update_quality_history')` 的模块；如果你的 AdapterGate 也实现这个接口，就能复用现有反馈路径。

**可能收益（最有潜力）**
- **更强的条件化/自适应能力**：不同图像（边界复杂 vs 纹理复杂）可选择不同 Adapter 专家，等价于“通道维度的 MoE”。
- **更好地和 Conv‑LoRA 的空间 MoE 协同**：一个管空间（Conv‑LoRA experts），一个管通道（Adapter experts），GSPO 统一用质量信号做 credit assignment。
- **减少过拟合**：MoE 形式的稀疏激活可降低“所有样本都把同一套 Adapter 权重推向某种偏置”的风险。

**主要代价/风险**
- 计算和显存上升（虽然比再做一次大模型 forward 小很多，但 gate+多专家仍有开销）。
- 训练更不稳定：需要 entropy/负载均衡（类似 MoE 的 CV² loss）防止 Adapter 专家塌缩。

---

#### 方案 C（把“Adapter 的强度”当作 policy）：用 GSPO 优化每层 Adapter 的门控/缩放分布
你们的 `AdapterLayer` 有可学习标量 `scale`。可以把它升级为**输入条件化的门控**（例如 `scale = sigmoid(w·pool(x))`），并引入探索噪声：

- **动作**：每层对每个样本采样一个 \(z_i\in[0,1]\)（Adapter 开关/强度），或采样离散 on/off（Hard Concrete / Gumbel-Sigmoid）。
- **GSPO**：同一张图像做 G 次 rollout，采不同的 \(z\)，用 advantage 让“产生更高质量”的 \(z\) 对应的门控参数获得更大梯度。
- **优点**：不需要多 Adapter 专家，参数增量很小。
- **缺点**：如果做离散采样，方差会更大、调参更难；做连续采样则更像“带噪声的加权监督”。

**可能收益**
- 自动学到“哪些层的 Adapter 更重要”（层级选择）。
- 在小数据医疗分割里常见：浅层可能不需要太大改动，深层更需要语义适配——门控能自动发现这种分配。

---

### 总结：你想要的“GSPO 同时微调 Adapter”，最建议怎么走
- **短期可验证**：先用方案 A（开启 Adapter dropout），确认“GSPO-on 时 Adapter 的学习确实更质量导向/更稳”。
- **中期最有价值**：做方案 B（Adapter‑MoE + 复用 `update_quality_history`），把“显式 RL feedback”扩展到通道适配这条分支。
- **长期更精细**：方案 C（层级门控/强度策略），解决“Adapter 应该在哪些层、以多大强度介入”的结构搜索问题。

---

### 可能带来的好处（从机制角度，而不是拍脑袋）
- **更好的 credit assignment**：GSPO 的 advantage 把“好预测”对应的梯度放大，能让 Adapter 不再只是“跟着 supervised loss 走”，而是更直接对齐 IoU/Dice 等最终质量信号。
- **更强的空间-通道协同**：Conv‑LoRA（空间 MoE）+ Adapter（通道适配）由同一质量信号统一调度，减少两条 PEFT 分支互相打架。
- **更鲁棒/更不易过拟合**：通过组内对比与反馈，抑制那些“训练 loss 降了但质量指标没升/边界变差”的更新方向（尤其在医学分割很关键）。
- **可能提升细粒度结构**：你们已有的 reward shaping（boundary/smooth/thin）如果同时驱动 Adapter 的策略，会更有希望改善边界、细丝状结构等“纯 CE/structure loss 难对齐”的部分。

如果你希望我把方案 B/C 具体落到“该新增哪些配置项、哪些模块要返回什么字段、`update_expert_feedback` 怎么复用、需要哪些额外正则/损失”这种可直接开工的设计稿，你让我要“出一份实现级方案（含伪代码/接口）”，我就按你们当前代码结构写到可以直接分支开发的粒度。

---

**User**

好的，我想要GSPO可以同时微调Conv-LoRA的MoE以及Vision Encoder 每层 MLP 后的Encoder Adapter（暂时不要包括decoder adapter)，请按照你说的上述路线设计并实现各阶段方案，并注意要通过加入可选开关参数使其兼容已有实现，即当不开启相应选项时，GSPO仍然只用于微调Conv-LoRA。为避免文档太多太乱，请在docs目录下创建子目录专门用于存放为此次设计而撰写的各种设计和说明文档，文件和目录名请用英文，内容用中文。设计文档应包含模型设计框图和必要的公式，清晰地说明强化学习是用在哪里、如何应用的（即在你的实际实现里，损失/奖励函数的形式是什么、其如何与GSPO的公式相对应），等等。

---

**Cursor**

明白了。我会按“逐步增强、默认不改行为”的思路推进：先实现**阶段 1（最小改动）**让 GSPO 的组内探索/加权机制确实能驱动 Encoder Adapter 学到东西；再实现**阶段 2（显式把 Adapter 引入 GSPO 决策/反馈回路）**，并都用开关参数控制，默认仍保持“GSPO 只作用于 Conv‑LoRA MoE”。同时我会在 `docs/` 下新建一个英文子目录，集中放本次的中文设计说明、框图与公式映射。

Nice，GSPO 的创建点就在 `SemanticSegmentationLearner._maybe_create_gspo_trainer()`，而 Adapter 已经有单测覆盖 `SamVisionLayer` 的注入/前向。这意味着我们可以把“GSPO-Adapter 扩展”集中在 learner + lit module + adapter 模块三处改动，并用现有单测框架补齐开关行为。接下来我会打开 learner 相关片段，看它如何从 config 取值并把 trainer 传给 `SemanticSegmentationLitModule`。

我先把“GSPO 训练器如何被创建/注入到 LitModule”的链路打通了：`SemanticSegmentationLearner._maybe_create_gspo_trainer()` → `SemanticSegmentationLitModule(gspo_trainer=...)`。接下来我会开始落地阶段1/2：先扩展 Adapter 模块以支持“GSPO 模式下的探索噪声/门控策略”，再在 LitModule 里按开关在 GSPO step 期间启用/冻结 encoder adapters，保证默认行为完全不变。

```worktree-python
# Modified based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal



def identity(x):
    return x


class AdapterLayer(nn.Module):
    """
    Standard Bottleneck Adapter Layer.
    Structure: Linear(d, r) -> Activation -> Linear(r, d) -> Scale
    """
    def __init__(self, in_features, adapter_dim, scale=1.0, dropout=0.0):
        super().__init__()
        self.down_proj = nn.Linear(in_features, adapter_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, in_features)
        self.scale = nn.Parameter(torch.tensor(scale))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # GSPO adapter extension (default off; activated by training loop via set_gspo_mode)
        self._gspo_active = False
        self._gspo_noise_std = 0.0
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        down = self.down_proj(x)
        act = self.activation(down)
        up = self.up_proj(act)
        # Optional GSPO exploration noise (only when explicitly enabled)
        if self._gspo_active and self.training and self._gspo_noise_std and self._gspo_noise_std > 0:
            up = up + torch.randn_like(up) * self._gspo_noise_std
        return self.dropout(up) * self.scale

    def set_gspo_mode(self, active: bool, noise_std: Optional[float] = None):
        """
        Enable/disable GSPO-specific behavior for this Adapter.
        This is intentionally a no-op unless called by the training loop.
        """
        self._gspo_active = bool(active)
        if noise_std is not None:
            self._gspo_noise_std = float(noise_std)


class GatedAdapterLayer(nn.Module):
    """
    Encoder Adapter with an input-conditioned gate (policy-like scalar per sample).

    Design goal:
    - When GSPO is active: add stochasticity to the gate logits for exploration across group rollouts.
    - When GSPO is inactive: use deterministic gate (no sampling noise).

    Input formats supported:
    - [B, H, W, C] (SAM vision tokens)
    - [B, L, C] (generic tokens)
    """

    def __init__(
        self,
        in_features: int,
        adapter_dim: int,
        scale: float = 1.0,
        dropout: float = 0.0,
        gate_noise_std: float = 0.0,
    ):
        super().__init__()
        self.adapter = AdapterLayer(in_features=in_features, adapter_dim=adapter_dim, scale=scale, dropout=dropout)
        self.gate = nn.Linear(in_features, 1)
        # Initialize gate to near-open (sigmoid(0)=0.5); training can learn stronger/ weaker gating.
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        self._gspo_active = False
        self._gate_noise_std = float(gate_noise_std)

    def set_gspo_mode(self, active: bool, gate_noise_std: Optional[float] = None, adapter_noise_std: Optional[float] = None):
        self._gspo_active = bool(active)
        if gate_noise_std is not None:
            self._gate_noise_std = float(gate_noise_std)
        # Propagate adapter exploration noise if requested
        self.adapter.set_gspo_mode(active=active, noise_std=adapter_noise_std)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            # [B,H,W,C] -> [B,C]
            return x.mean(dim=(1, 2))
        if x.dim() == 3:
            # [B,L,C] -> [B,C]
            return x.mean(dim=1)
        raise ValueError(f"Unsupported input shape for GatedAdapterLayer: {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self._pool(x)
        logits = self.gate(pooled)  # [B,1]
        if self._gspo_active and self.training and self._gate_noise_std and self._gate_noise_std > 0:
            logits = logits + torch.randn_like(logits) * self._gate_noise_std
        gate = torch.sigmoid(logits)  # [B,1]

        out = self.adapter(x)
        # Broadcast gate to match out shape
        while gate.dim() < out.dim():
            gate = gate.unsqueeze(1)
        return out * gate



class LoRALayer:
    """
    Abstract LoRA Layer.

    Parameters
    ----------
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = identity
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class IA3LoRALinear(nn.Linear, LoRALayer):
    """
    LoRA (low-rank adaptation) followed by (IA)^3 (weight rescaling) incorporated in a Linear Layer. Weights of Linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r=8,
        lora_alpha=8,
        lora_dropout: float = 0.0,
        fan_in_fan_out=False,
        merge_weights=False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        # In essence the $b$ parameter of LoRA.
        self.lora_b = nn.Parameter(torch.ones(out_features, 1))
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        self.fan_in_fan_out = fan_in_fan_out
        self.weight.requires_grad = False

        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.T(self.weight), bias=self.bias)
        if self.r > 0:
            result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

        hidden = result * self.lora_b.flatten()
        return hidden

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data /= self.lora_b.flatten()
                self.weight.data -= self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        # def T(w):
        #     return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += self.T(self.lora_B @ self.lora_A) * self.scaling
                self.weight.data *= self.lora_b.flatten()
            self.merged = True
        return hidden

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class IA3Linear(nn.Linear, LoRALayer):
    """
    (IA)^3 incorporated in a Linear Layer. Weights of Linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    scaling_rank
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Liu, Haokun and Tam, Derek and Muqeeth, Mohammed and Mohta, Jay and Huang, Tenghao and Bansal, Mohit and Raffel, Colin,
    "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning", 2022
    https://arxiv.org/pdf/2205.05638.pdf
    2. Code: https://github.com/r-three/t-few
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        merge_weights: False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self, r=4, lora_alpha=4, lora_dropout=0.0, merge_weights=merge_weights
        )  # Default arguments, only
        # In essence the $b$ parameter of LoRA.
        self.lora_b = nn.Parameter(torch.ones(out_features, 1))
        self.weight.requires_grad = False

    def forward(self, x: torch.Tensor):
        hidden = F.linear(x, self.weight, self.bias)
        hidden = hidden * self.lora_b.flatten()
        return hidden

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data /= self.lora_b.flatten()
            self.merged = False

    def eval(self):
        # def T(w):
        #     return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data *= self.lora_b.flatten()
            self.merged = True
        return hidden

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class LoRALinear(nn.Linear, LoRALayer):
    """
    LoRA incorporated in Linear Layer. Weights of linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        # def T(w):
        #     return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, self.T(self.weight), bias=self.bias)


class LoRAEmbedding(nn.Embedding, LoRALayer):
    """
    LoRA incorporated in Embedding Layer. Weights of embedding layer are set to be frozen per default.

    Parameters
    ----------
    num_embeddings
        size of the dictionary of embeddings.
    embedding_dim
         the size of each embedding vector.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0, merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(
                    x,
                    self.lora_A.T,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (after_A @ self.lora_B.T) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class LoRAMergedLinear(nn.Linear, LoRALayer):
    """
    LoRA where single nn.Linear represents more than one layer (used in some implementations of attention query/key/value projections). Weights of linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing
    r
        rank r of the low-rank decomposition
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout rate for LoRA
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out)
    merge_weights
        Merging weights during inference to reduce latency

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, "The length of enable_lora must divide out_features"
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(-1, self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
            self.merged = False

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data += self.zero_pad(T(delta_w * self.scaling))
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(
                    after_A.transpose(-2, -1), self.lora_B.unsqueeze(-1), groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result


class LoRAConv2d(nn.Conv2d, LoRALayer):
    """
    LoRA incorporated in 2d-Convolutional Layer. Weights of convolutional layer are set to be frozen per default.

    Parameters
    ----------
    in_channels
         Number of channels in the input image.
    out_channels
        Number of channels produced by the convolution.
    kernel_size
        Size of the convolving kernel.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Adding dropout to LoRA.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert type(kernel_size) is int
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * kernel_size, in_channels * kernel_size)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_channels * kernel_size, r * kernel_size)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = False

    def eval(self):
        nn.Conv2d.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.conv2d(
                x,
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return nn.Conv2d.forward(self, x)


class ConvLoRALinear(nn.Linear, LoRALayer):
    """
    Conv-LoRA incorporated in Linear Layer. Weights of linear layer are set to be frozen per default.

    Paper alignment (Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model, ICLR 2024):
    - Sec. 3 (Method): Replace standard LoRA's linear residual with a spatial, convolutional expert branch (MoE-Conv).
    - Sec. 3.2 (MoE-Conv & Gating): Noisy-TopK gating over GAP features, plus load-balancing auxiliary loss.
    - Sec. 3.3 (Integration): Inject at attention projections (q/k/v) of the vision encoder.
    - Sec. 4 (Training): Return moe_loss for aggregation with task loss.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.
    conv_lora_expert_num
        The number of experts in MoE-Conv.

    References
    ----------
    1. Zihan Zhong, Zhiqiang Tang, Tong He, Haoyang Fang, Chun Yuan,
    "Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model", 2024
    https://arxiv.org/abs/2401.17868
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        conv_lora_expert_num: Optional[int] = None,
        gspo_enabled: bool = False,
        gspo_group_size: int = 3,
        gspo_quality_momentum: float = 0.9,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.gspo_enabled = gspo_enabled
        # Actual trainable parameters
        if r > 0:
            # Sec. 3: Low-rank LoRA factors around a frozen base linear
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            # Sec. 3.2: MoE-Conv experts with Noisy-TopK gating
            # GSPO: Pass GSPO parameters to MoEGate
            topk = 1
            self.lora_moe_gating = MoEGate(
                M=conv_lora_expert_num, 
                d=self.r, 
                K=topk,
                gspo_enabled=gspo_enabled,
                gspo_group_size=gspo_group_size,
                gspo_quality_momentum=gspo_quality_momentum
            )
            self.lora_moe_experts = nn.ModuleList([])
            self.upsample_ratios = list(range(1, conv_lora_expert_num + 1))
            for upsample_ratio in self.upsample_ratios:
                expert = nn.Conv2d(in_channels=r, out_channels=r, kernel_size=3, stride=1, padding=1, bias=True)
                expert.bias.data.zero_()
                self.lora_moe_experts.append(nn.Sequential(expert, nn.GELU()))
            self.num_experts = conv_lora_expert_num
            self.multiply_by_gates = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def forward(self, x: torch.Tensor):
        # Base frozen path + residual (LoRA principle)
        result = F.linear(x, self.T(self.weight), bias=self.bias)
        moe_loss = 0
        selected_experts = None
        
        if self.r > 0:
            # 1) Project to rank-r LoRA space
            lora_res = self.lora_dropout(x) @ self.lora_A.T
            dim = lora_res.dim()
            if dim == 3:
                B, L, C = lora_res.size()
                H = W = int(math.sqrt(L))
                lora_res = lora_res.reshape(B, H, W, C)
            else:
                H, W = lora_res.size()[1:3]

            # Calculate the gating values.
            # 2) Sec. 3.2: Compute Noisy-TopK gates over spatial features; also returns auxiliary moe_loss.
            lora_res = lora_res.permute(0, 3, 1, 2).contiguous()
            
            # GSPO: Receive selected experts info
            if self.gspo_enabled:
                gates, moe_loss, selected_experts = self.lora_moe_gating(lora_res)
            else:
                gate_output = self.lora_moe_gating(lora_res)
                if len(gate_output) == 3:
                    gates, moe_loss, selected_experts = gate_output
                else:
                    gates, moe_loss = gate_output

            # Distribute data samples to experts.
            # 3) Route tokens to experts (SparseDispatcher)
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(lora_res)
            expert_outputs = []
            for i in range(self.num_experts):
                if len(expert_inputs[i]) == 0:
                    continue
                upsample_ratio = self.upsample_ratios[i]
                cur_res = expert_inputs[i]
                # Per-expert resolution change to diversify receptive fields (paper Sec. 3.2)
                if upsample_ratio != 1:
                    cur_res = F.interpolate(cur_res, scale_factor=upsample_ratio, mode="bicubic")
                cur_res = self.lora_moe_experts[i](cur_res)
                if upsample_ratio != 1:
                    cur_res = F.interpolate(cur_res, size=(int(H), int(W)), mode="bicubic")
                expert_outputs.append(cur_res)

            # Combine data samples after processing by each expert.
            # 4) Combine expert outputs (weighted by gates) back to the batch layout
            temp_lora_res = dispatcher.combine(expert_outputs, multiply_by_gates=self.multiply_by_gates)
            lora_res = lora_res + temp_lora_res

            lora_res = lora_res.permute(0, 2, 3, 1).contiguous()
            if dim == 3:
                lora_res = lora_res.reshape(B, L, C)
            # 5) Project back to output dim with LoRA scale (final residual add)
            result += (lora_res @ self.lora_B.T) * self.scaling

        # Return 3 values if GSPO is enabled, 2 values otherwise for backward compatibility
        if self.gspo_enabled and selected_experts is not None:
            return result, moe_loss, selected_experts
        else:
            return result, moe_loss


class MoEGate(nn.Module):
    def __init__(self, d, M=4, K=1, noisy_gating=True, gspo_enabled=False, gspo_group_size=3, gspo_quality_momentum=0.9):
        """Constructor
        Args:
            d: input channel dimensionality.
            M: the number of experts.
            K: the number of chosen experts for each forward pass.
            gspo_enabled: whether to enable GSPO (Group Sequence Policy Optimization) enhancements.
            gspo_group_size: number of experts to select when GSPO is enabled (overrides K).
            gspo_quality_momentum: momentum coefficient for expert quality history updates.
        """
        super(MoEGate, self).__init__()
        self.M = M
        self.gspo_enabled = gspo_enabled
        self.gspo_group_size = gspo_group_size
        self.gspo_quality_momentum = gspo_quality_momentum
        
        # GSPO: Use group_size as K when enabled, otherwise use original K
        self.k = gspo_group_size if gspo_enabled else K
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # global average pooling

        self.noisy_gating = noisy_gating

        self.w_gate = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(d, M), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        
        # GSPO: Expert quality tracking
        if gspo_enabled:
            self.register_buffer("expert_quality_history", torch.zeros(M))
            self.register_buffer("expert_usage_count", torch.zeros(M))
        
        assert self.k <= self.M

    def forward(self, feats, loss_coef=1e-2, noise_epsilon=1e-2):
        """
        Paper alignment (Sec. 3.2):
        - GAP over spatial features -> linear logits (+ noise during training) -> TopK expert selection
        - Softmax over TopK logits yields gates; compute importance/load stats and CV^2-based load-balancing loss
        
        GSPO enhancement:
        - Quality-aware bias added to logits based on expert historical performance
        - Supports multi-expert selection (TopK > 1) for group-level optimization
        """
        batch_size = feats.shape[0]

        feats_S = self.gap(feats).view(batch_size, -1)

        clean_logits = feats_S @ self.w_gate
        
        # GSPO: Add quality-aware bias to encourage high-performing experts
        if self.gspo_enabled and self.training:
            quality_bias = self._compute_quality_bias()
            clean_logits = clean_logits + quality_bias.unsqueeze(0)
        
        if self.noisy_gating and self.training:
            raw_noise_stddev = feats_S @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.M), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True).float()
        gates = zeros.scatter(1, top_k_indices, top_k_gates).to(logits.dtype)

        if self.noisy_gating and self.k < self.M and self.training:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)

        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        return gates, loss, top_k_indices

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    
    def _compute_quality_bias(self):
        """
        GSPO: Compute quality-aware bias for expert selection.
        
        This method balances exploitation (favoring high-quality experts) and
        exploration (giving chances to less-used experts).
        
        Returns:
            bias: Tensor of shape [M] to be added to logits
        """
        if not hasattr(self, 'expert_quality_history'):
            return torch.zeros(self.M, device=self.w_gate.device)
        
        # Normalize quality history to compute exploitation bias
        quality_history = self.expert_quality_history
        if quality_history.sum() > 0:
            quality_bias = (quality_history - quality_history.mean()) / (quality_history.std() + 1e-8)
        else:
            quality_bias = torch.zeros_like(quality_history)
        
        # Compute exploration bonus for less-used experts
        usage_count = self.expert_usage_count
        if usage_count.sum() > 0:
            usage_normalized = usage_count / (usage_count.sum() + 1e-8)
            exploration_bonus = 1.0 - usage_normalized
        else:
            exploration_bonus = torch.ones_like(usage_count)
        
        # Combine: 80% exploitation + 20% exploration
        combined_bias = 0.8 * quality_bias + 0.2 * exploration_bonus
        
        return combined_bias
    
    def update_quality_history(self, selected_experts, quality_scores):
        """
        GSPO: Update expert quality history based on actual performance.
        
        This provides the feedback loop for quality-aware gating.
        
        Args:
            selected_experts: Tensor of shape [batch, k] containing selected expert indices
            quality_scores: Tensor of shape [batch] containing quality metrics (e.g., IoU, DICE)
        """
        if not self.gspo_enabled or not hasattr(self, 'expert_quality_history'):
            return
        
        with torch.no_grad():
            for b in range(selected_experts.shape[0]):
                quality = quality_scores[b].item() if quality_scores[b].numel() == 1 else quality_scores[b].mean().item()
                for k in range(selected_experts.shape[1]):
                    expert_id = selected_experts[b, k].item()
                    # Momentum update
                    current_quality = self.expert_quality_history[expert_id]
                    self.expert_quality_history[expert_id] = (
                        self.gspo_quality_momentum * current_quality +
                        (1 - self.gspo_quality_momentum) * quality
                    )
                    self.expert_usage_count[expert_id] += 1


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.

    References
    ----------
    1. Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean,
    "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", 2017
    https://arxiv.org/abs/1701.06538
    2. Code: https://github.com/davidmrau/mixture-of-experts/blob/master/moe.py
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)  # [bs * num_of chosen experts, dim]
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            # stitched = stitched.mul(self._nonzero_gates)
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(-1).unsqueeze(-1))
        # zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size()[1],
            expert_out[-1].size()[2],
            expert_out[-1].size()[3],
            requires_grad=True,
            device=stitched.device,
        )
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

```

```worktree-python
*** Begin Patch
*** Update File: multimodal/src/autogluon/multimodal/models/custom_hf_models/modeling_sam_for_conv_lora.py
@@
-from ..adaptation_layers import ConvLoRALinear, AdapterLayer
+from ..adaptation_layers import ConvLoRALinear, AdapterLayer, GatedAdapterLayer
@@
 class SamVisionLayer(nn.Module):
@@
         # Adapter Integration
         self.adapter = None
         if hasattr(config, "adapter_enabled") and config.adapter_enabled:
             adapter_dim = getattr(config, "adapter_dim", 64)
-            self.adapter = AdapterLayer(config.hidden_size, adapter_dim)
+            # Optional GSPO adapter policy (gated adapter) for encoder only
+            encoder_gate_enabled = getattr(config, "encoder_adapter_gate_enabled", False)
+            encoder_gate_noise_std = getattr(config, "encoder_adapter_gate_noise_std", 0.0)
+            if encoder_gate_enabled:
+                self.adapter = GatedAdapterLayer(
+                    in_features=config.hidden_size,
+                    adapter_dim=adapter_dim,
+                    gate_noise_std=encoder_gate_noise_std,
+                )
+            else:
+                self.adapter = AdapterLayer(config.hidden_size, adapter_dim)
*** End Patch
```

```worktree-python
# coding=utf-8
# Copyright 2023 The Meta AI Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch SAM model."""

import collections
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.sam.configuration_sam import (
    SamConfig,
    SamMaskDecoderConfig,
    SamPromptEncoderConfig,
    SamVisionConfig,
)
from transformers.utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from ..adaptation_layers import AdapterLayer, GatedAdapterLayer, LoRALinear

logger = logging.get_logger(__name__)


SAM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/sam-vit-huge",
    "facebook/sam-vit-large",
    "facebook/sam-vit-base",
    # See all SAM models at https://huggingface.co/models?filter=sam
]


@dataclass
class SamVisionEncoderOutput(ModelOutput):
    """
    Base class for sam vision model's outputs that also contains image embeddings obtained by applying the projection
    layer to the pooler_output.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    moe_loss: Optional[torch.FloatTensor] = None


@dataclass
class SamImageSegmentationOutput(ModelOutput):
    """
    Base class for Segment-Anything model's output

    Args:
        iou_scores (`torch.FloatTensor` of shape `(batch_size, num_masks)`):
            The iou scores of the predicted masks.
        pred_masks (`torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`):
            The predicted low resolutions masks. Needs to be post-processed by the processor
        vision_hidden_states  (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the vision model at the output of each layer plus the optional initial embedding outputs.
        vision_attentions  (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        mask_decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    iou_scores: torch.FloatTensor = None
    pred_masks: torch.FloatTensor = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = None
    mask_decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    vision_moe_loss: Optional[torch.FloatTensor] = None


class SamPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        embeddings = self.projection(pixel_values).permute(0, 2, 3, 1)
        return embeddings


class SamMLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lin1 = nn.Linear(config.hidden_size, config.mlp_dim)
        self.lin2 = nn.Linear(config.mlp_dim, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.lin1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.lin2(hidden_states)
        return hidden_states


# Copied from transformers.models.convnext.modeling_convnext.ConvNextLayerNorm with ConvNext->Sam
class SamLayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SamAttention(nn.Module):
    """
    SAM's attention layer that allows for downscaling the size of the embedding after projection to queries, keys, and
    values.
    
    Supports optional LoRA (Low-Rank Adaptation) on Q/K/V projection layers for parameter-efficient fine-tuning.
    """

    def __init__(self, config, downsample_rate=None, lora_r=0, lora_alpha=1, lora_dropout=0.0):
        """
        Parameters
        ----------
        config : SamMaskDecoderConfig
            Configuration for the attention layer
        downsample_rate : int, optional
            Downsample rate for attention (default: from config)
        lora_r : int, default=0
            LoRA rank. If 0, LoRA is disabled (default behavior for backward compatibility)
        lora_alpha : int, default=1
            LoRA scaling factor
        lora_dropout : float, default=0.0
            Dropout probability for LoRA
        """
        super().__init__()
        self.hidden_size = config.hidden_size

        downsample_rate = config.attention_downsample_rate if downsample_rate is None else downsample_rate

        self.internal_dim = config.hidden_size // downsample_rate
        self.num_attention_heads = config.num_attention_heads
        if self.internal_dim % config.num_attention_heads != 0:
            raise ValueError("num_attention_heads must divide hidden_size.")

        # Q/K/V projections: Use LoRALinear if lora_r > 0, otherwise use standard nn.Linear
        if lora_r > 0:
            self.q_proj = LoRALinear(
                self.hidden_size, self.internal_dim,
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )
            self.k_proj = LoRALinear(
                self.hidden_size, self.internal_dim,
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )
            self.v_proj = LoRALinear(
                self.hidden_size, self.internal_dim,
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.internal_dim)
            self.k_proj = nn.Linear(self.hidden_size, self.internal_dim)
            self.v_proj = nn.Linear(self.hidden_size, self.internal_dim)
        
        # Output projection: standard Linear (LoRA not applied here per design)
        self.out_proj = nn.Linear(self.internal_dim, self.hidden_size)

    def _separate_heads(self, hidden_states: Tensor, num_attention_heads: int) -> Tensor:
        batch, point_batch_size, n_tokens, channel = hidden_states.shape
        c_per_head = channel // num_attention_heads
        hidden_states = hidden_states.reshape(batch * point_batch_size, n_tokens, num_attention_heads, c_per_head)
        return hidden_states.transpose(1, 2)

    def _recombine_heads(self, hidden_states: Tensor, point_batch_size: int) -> Tensor:
        batch, n_heads, n_tokens, c_per_head = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states.reshape(batch // point_batch_size, point_batch_size, n_tokens, n_heads * c_per_head)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_similarity: Tensor = None) -> Tensor:
        # Input projections
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        point_batch_size = query.shape[1]
        # Separate into heads
        query = self._separate_heads(query, self.num_attention_heads)
        key = self._separate_heads(key, self.num_attention_heads)
        value = self._separate_heads(value, self.num_attention_heads)

        # SamAttention
        _, _, _, c_per_head = query.shape
        attn = query @ key.permute(0, 1, 3, 2)  # batch_size * point_batch_size  x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        if attention_similarity is not None:
            attn = attn + attention_similarity
            attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ value
        out = self._recombine_heads(out, point_batch_size)
        out = self.out_proj(out)

        return out


class SamTwoWayAttentionBlock(nn.Module):
    def __init__(self, config, attention_downsample_rate: int = 2, skip_first_layer_pe: bool = False,
                 lora_r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0.0):
        """
        A transformer block with four layers:
            (1) self-attention of sparse inputs (2) cross attention of sparse inputs -> dense inputs (3) mlp block on
            sparse inputs (4) cross attention of dense inputs -> sparse inputs

        Arguments:
            config (`SamMaskDecoderConfig`):
                The configuration file used to instantiate the block
            attention_downsample_rate (*optional*, int, defaults to 2):
                The downsample ratio of the block used to reduce the inner dim of the attention.
            skip_first_layer_pe (*optional*, bool, defaults to `False`):
                Whether or not to skip the addition of the query_point_embedding on the first layer.
            lora_r (int, default=0):
                LoRA rank for attention layers. If 0, LoRA is disabled.
            lora_alpha (int, default=1):
                LoRA scaling factor
            lora_dropout (float, default=0.0):
                Dropout probability for LoRA
        """
        super().__init__()

        self.hidden_size = config.hidden_size
        self.layer_norm_eps = config.layer_norm_eps

        self.self_attn = SamAttention(config, downsample_rate=1, 
                                      lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.cross_attn_token_to_image = SamAttention(config, downsample_rate=attention_downsample_rate,
                                                      lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.mlp = SamMLPBlock(config)
        self.layer_norm3 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        
        # Decoder MLP-Adapter on queries (optional, config-driven)
        self.adapter_queries = None
        # Prefer explicit decoder-specific flags, but fall back to generic adapter flags if provided
        decoder_adapter_enabled = getattr(config, "decoder_adapter_enabled", False)
        decoder_adapter_dim = getattr(config, "decoder_adapter_dim", getattr(config, "adapter_dim", 64))
        if decoder_adapter_enabled:
            self.adapter_queries = AdapterLayer(self.hidden_size, decoder_adapter_dim)

        self.layer_norm4 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.cross_attn_image_to_token = SamAttention(config, downsample_rate=attention_downsample_rate,
                                                      lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_point_embedding: Tensor,
        key_point_embedding: Tensor,
        attention_similarity: Tensor,
        output_attentions: bool = False,
    ):
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(query=queries, key=queries, value=queries)
        else:
            query = queries + query_point_embedding
            attn_out = self.self_attn(query=query, key=query, value=queries)
            queries = queries + attn_out
        queries = self.layer_norm1(queries)

        # Cross attention block, tokens attending to image embedding
        query = queries + query_point_embedding
        key = keys + key_point_embedding

        attn_out = self.cross_attn_token_to_image(
            query=query, key=key, value=keys, attention_similarity=attention_similarity
        )
        queries = queries + attn_out

        queries = self.layer_norm2(queries)

        # MLP block + optional parallel Adapter on queries
        mlp_out = self.mlp(queries)
        if self.adapter_queries is not None:
            adapter_out = self.adapter_queries(queries)
            queries = queries + mlp_out + adapter_out
        else:
            queries = queries + mlp_out
        queries = self.layer_norm3(queries)

        # Cross attention block, image embedding attending to tokens
        query = queries + query_point_embedding
        key = keys + key_point_embedding

        attn_out = self.cross_attn_image_to_token(query=key, key=query, value=queries)
        keys = keys + attn_out

        keys = self.layer_norm4(keys)

        outputs = (queries, keys)

        if output_attentions:
            outputs = outputs + (attn_out,)
        else:
            outputs = outputs + (None,)

        return outputs


class SamTwoWayTransformer(nn.Module):
    def __init__(self, config: SamMaskDecoderConfig, lora_r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0.0):
        super().__init__()
        self.config = config

        self.num_hidden_layers = config.num_hidden_layers
        self.layers = nn.ModuleList()

        for i in range(self.num_hidden_layers):
            self.layers.append(SamTwoWayAttentionBlock(config, skip_first_layer_pe=(i == 0),
                                                       lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout))

        self.final_attn_token_to_image = SamAttention(config, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.layer_norm_final_attn = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        point_embeddings: Tensor,
        image_embeddings: Tensor,
        image_positional_embeddings: Tensor,
        attention_similarity: Tensor,
        target_embedding=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        all_attentions = ()

        if image_embeddings is None:
            raise ValueError("You have to specify an image_embedding")

        image_embeddings = image_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)
        image_positional_embeddings = image_positional_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)

        # Prepare queries
        queries = point_embeddings
        keys = image_embeddings

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            if target_embedding is not None:
                queries += target_embedding

            queries, keys, attention_outputs = layer(
                queries=queries,
                keys=keys,
                query_point_embedding=point_embeddings,
                key_point_embedding=image_positional_embeddings,
                attention_similarity=attention_similarity,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_attentions = all_attentions + (attention_outputs,)

        # Apply the final attenion layer from the points to the image
        query = queries + point_embeddings
        key = keys + image_positional_embeddings

        attn_out = self.final_attn_token_to_image(query=query, key=key, value=keys)

        queries = queries + attn_out
        queries = self.layer_norm_final_attn(queries)
        return queries, keys, all_attentions


class SamFeedForward(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.activation = nn.ReLU()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        self.sigmoid_output = sigmoid_output

    def forward(self, hidden_states):
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.activation(hidden_states)
        for layer in self.layers:
            hidden_states = self.activation(layer(hidden_states))

        hidden_states = self.proj_out(hidden_states)
        if self.sigmoid_output:
            hidden_states = F.sigmoid(hidden_states)
        return hidden_states


class SamMaskDecoder(nn.Module):
    def __init__(self, config: SamMaskDecoderConfig, lora_r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0.0):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.num_multimask_outputs = config.num_multimask_outputs
        self.num_mask_tokens = config.num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, self.hidden_size)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.hidden_size)

        self.transformer = SamTwoWayTransformer(config, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        # should we create a new class for this?
        self.upscale_conv1 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(self.hidden_size // 4, self.hidden_size // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = SamLayerNorm(self.hidden_size // 4, data_format="channels_first")
        self.activation = nn.GELU()

        mlps_list = []
        for _ in range(self.num_mask_tokens):
            mlps_list += [SamFeedForward(self.hidden_size, self.hidden_size, self.hidden_size // 8, 3)]
        self.output_hypernetworks_mlps = nn.ModuleList(mlps_list)

        self.iou_prediction_head = SamFeedForward(
            self.hidden_size, config.iou_head_hidden_dim, self.num_mask_tokens, config.iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        output_attentions: Optional[bool] = None,
        attention_similarity: torch.Tensor = None,
        target_embedding: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (`torch.Tensor`):
                the embeddings from the image encoder
            image_positional_embedding (`torch.Tensor`):
                positional encoding with the shape of image_embeddings
            sparse_prompt_embeddings (`torch.Tensor`):
                The embeddings of the points and boxes
            dense_prompt_embeddings (`torch.Tensor`):
                the embeddings of the mask inputs
            multimask_output (bool):
                Whether to return multiple masks or a single mask.
            output_attentions (bool, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        if sparse_prompt_embeddings.sum().item() != 0:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.iou_token.weight.dtype)

        # Expand per-image data in batch direction to be per-point
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat(point_batch_size, 1, 1, 1)
        image_positional_embeddings = image_positional_embeddings.repeat(point_batch_size, 1, 1, 1)

        # Run the transformer, image_positional_embedding are consumed
        point_embedding, image_embeddings, attentions = self.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )

        upscaled_embedding = self.upscale_conv1(image_embeddings)
        upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)

        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
        masks = (hyper_in @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, height, width)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]
        iou_pred = iou_pred[:, :, mask_slice]

        outputs = (masks, iou_pred)

        if output_attentions:
            outputs = outputs + (attentions,)
        else:
            outputs = outputs + (None,)

        return outputs


class SamPositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = config.hidden_size // 2
        self.register_buffer("positional_embedding", self.scale * torch.randn((2, config.num_pos_feats)))

    def forward(self, input_coords, input_shape=None):
        """Positionally encode points that are normalized to [0,1]."""
        coordinates = input_coords.clone()

        if input_shape is not None:
            coordinates[:, :, :, 0] = coordinates[:, :, :, 0] / input_shape[1]
            coordinates[:, :, :, 1] = coordinates[:, :, :, 1] / input_shape[0]

        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coordinates = 2 * coordinates - 1
        coordinates = coordinates.to(self.positional_embedding.dtype)
        coordinates = coordinates @ self.positional_embedding
        coordinates = 2 * np.pi * coordinates
        # outputs d_1 x ... x d_n x channel shape
        return torch.cat([torch.sin(coordinates), torch.cos(coordinates)], dim=-1)


class SamMaskEmbedding(nn.Module):
    def __init__(self, config: SamPromptEncoderConfig):
        super().__init__()
        self.mask_input_channels = config.mask_input_channels // 4
        self.activation = ACT2FN[config.hidden_act]
        self.conv1 = nn.Conv2d(1, self.mask_input_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.mask_input_channels, config.mask_input_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(config.mask_input_channels, config.hidden_size, kernel_size=1)
        self.layer_norm1 = SamLayerNorm(
            self.mask_input_channels, eps=config.layer_norm_eps, data_format="channels_first"
        )
        self.layer_norm2 = SamLayerNorm(
            self.mask_input_channels * 4, eps=config.layer_norm_eps, data_format="channels_first"
        )

    def forward(self, masks):
        hidden_states = self.conv1(masks)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.activation(hidden_states)
        dense_embeddings = self.conv3(hidden_states)
        return dense_embeddings


class SamPromptEncoder(nn.Module):
    def __init__(self, config: SamPromptEncoderConfig, shared_patch_embedding):
        super().__init__()
        self.shared_embedding = shared_patch_embedding
        self.mask_embed = SamMaskEmbedding(config)
        self.no_mask_embed = nn.Embedding(1, config.hidden_size)

        self.image_embedding_size = (config.image_embedding_size, config.image_embedding_size)
        self.input_image_size = config.image_size

        self.point_embed = nn.ModuleList(
            [nn.Embedding(1, config.hidden_size) for i in range(config.num_point_embeddings)]
        )
        self.hidden_size = config.hidden_size
        self.not_a_point_embed = nn.Embedding(1, config.hidden_size)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            target_point_shape = (points.shape[0], points.shape[1], 1, points.shape[-1])
            target_labels_shape = (points.shape[0], points.shape[1], 1)
            padding_point = torch.zeros(target_point_shape, device=points.device)
            padding_label = -torch.ones(target_labels_shape, device=labels.device)
            points = torch.cat([points, padding_point], dim=2)
            labels = torch.cat([labels, padding_label], dim=2)
        input_shape = (self.input_image_size, self.input_image_size)
        point_embedding = self.shared_embedding(points, input_shape)

        # torch.where and expanding the labels tensor is required by the ONNX export
        point_embedding = torch.where(labels[..., None] == -1, self.not_a_point_embed.weight, point_embedding)

        # This is required for the ONNX export. The dtype, device need to be explicitly
        # specified as otherwise torch.onnx.export interprets as double
        point_embedding = torch.where(
            labels[..., None] != -10,
            point_embedding,
            torch.tensor(0.0, dtype=point_embedding.dtype, device=point_embedding.device),
        )

        point_embedding = torch.where(
            (labels == 0)[:, :, :, None],
            point_embedding + self.point_embed[0].weight[None, None, :, :],
            point_embedding,
        )

        point_embedding = torch.where(
            (labels == 1)[:, :, :, None],
            point_embedding + self.point_embed[1].weight[None, None, :, :],
            point_embedding,
        )

        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        batch_size, nb_boxes = boxes.shape[:2]
        coords = boxes.reshape(batch_size, nb_boxes, 2, 2)
        input_shape = (self.input_image_size, self.input_image_size)
        corner_embedding = self.shared_embedding(coords, input_shape)
        corner_embedding[:, :, 0, :] += self.point_embed[2].weight
        corner_embedding[:, :, 1, :] += self.point_embed[3].weight
        return corner_embedding

    def forward(
        self,
        input_points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        input_labels: Optional[torch.Tensor],
        input_boxes: Optional[torch.Tensor],
        input_masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Args:
            points (`torch.Tensor`, *optional*):
                point coordinates and labels to embed.
            boxes (`torch.Tensor`, *optional*):
                boxes to embed
            masks (`torch.Tensor`, *optional*):
                masks to embed
        """
        sparse_embeddings = None
        batch_size = 1
        target_device = self.shared_embedding.positional_embedding.device
        if input_points is not None:
            batch_size, point_batch_size = input_points.shape[:2]
            if input_labels is None:
                raise ValueError("If points are provided, labels must also be provided.")
            point_embeddings = self._embed_points(input_points, input_labels, pad=(input_boxes is None))
            sparse_embeddings = point_embeddings
        if input_boxes is not None:
            batch_size = input_boxes.shape[0]
            box_embeddings = self._embed_boxes(input_boxes)
            if sparse_embeddings is None:
                sparse_embeddings = box_embeddings
            else:
                sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=2)
        if input_masks is not None:
            dense_embeddings = self.mask_embed(input_masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch_size, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        if sparse_embeddings is None:
            sparse_embeddings = torch.zeros((batch_size, 1, 1, self.hidden_size), device=target_device)

        return sparse_embeddings, dense_embeddings


class SamVisionAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(self, config, window_size):
        super().__init__()
        input_size = (
            (config.image_size // config.patch_size, config.image_size // config.patch_size)
            if window_size == 0
            else (window_size, window_size)
        )

        self.num_attention_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.scale = head_dim**-0.5
        self.dropout = config.attention_dropout

        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.use_rel_pos = config.use_rel_pos
        if self.use_rel_pos:
            if input_size is None:
                raise ValueError("Input size must be provided if using relative positional encoding.")

            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.

        Args:
            q_size (int):
                size of the query.
            k_size (int):
                size of key k.
            rel_pos (`torch.Tensor`):
                relative position embeddings (L, channel).

        Returns:
            Extracted positional embeddings according to relative positions.
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)

        # Scale the coords with short length if shapes for q and k are different.
        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.long()]

    def add_decomposed_rel_pos(
        self,
        attn: torch.Tensor,
        query: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

        Args:
            attn (`torch.Tensor`):
                attention map.
            query (`torch.Tensor`):
                query q in the attention layer with shape (batch_size, query_height * query_width, channel).
            rel_pos_h (`torch.Tensor`):
                relative position embeddings (Lh, channel) for height axis.
            rel_pos_w (`torch.Tensor`):
                relative position embeddings (Lw, channel) for width axis.
            q_size (tuple):
                spatial sequence size of query q with (query_height, query_width).
            k_size (tuple):
                spatial sequence size of key k with (key_height, key_width).

        Returns:
            attn (`torch.Tensor`):
                attention map with added relative positional embeddings.
        """
        query_height, query_width = q_size
        key_height, key_width = k_size
        relative_position_height = self.get_rel_pos(query_height, key_height, rel_pos_h)
        relative_position_width = self.get_rel_pos(query_width, key_width, rel_pos_w)

        batch_size, _, dim = query.shape
        reshaped_query = query.reshape(batch_size, query_height, query_width, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", reshaped_query, relative_position_height)
        rel_w = torch.einsum("bhwc,wkc->bhwk", reshaped_query, relative_position_width)
        attn = attn.reshape(batch_size, query_height, query_width, key_height, key_width)
        attn = attn + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        attn = attn.reshape(batch_size, query_height * query_width, key_height * key_width)
        return attn

    def forward(self, hidden_states: torch.Tensor, output_attentions=False, output_moe_loss=False) -> torch.Tensor:
        batch_size, height, width, _ = hidden_states.shape
        # qkv with shape (3, batch_size, nHead, height * width, channel)
        ##### modify here for conv-lora
        # Paper alignment:
        # - Sec. 3.3 (Integration): Inject Conv-LoRA at attention projections (q/k/v).
        #   The adapted linear layer (possibly Conv-LoRA) may return (tensor, moe_loss, selected_experts) when output_moe_loss=True.
        qkv = self.qkv(hidden_states)
        if output_moe_loss:
            # Handle both 2-value and 3-value returns for backward compatibility
            if isinstance(qkv, tuple) and len(qkv) == 3:
                qkv, moe_loss, selected_experts = qkv
            else:
                qkv, moe_loss = qkv
        qkv = qkv.reshape(batch_size, height * width, 3, self.num_attention_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (batch_size * nHead, height * width, channel)
        query, key, value = qkv.reshape(3, batch_size * self.num_attention_heads, height * width, -1).unbind(0)

        attn_weights = (query * self.scale) @ key.transpose(-2, -1)

        if self.use_rel_pos:
            attn_weights = self.add_decomposed_rel_pos(
                attn_weights, query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = (attn_probs @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)

        attn_output = self.proj(attn_output)

        if output_attentions:
            outputs = (attn_output, attn_weights)
        else:
            outputs = (attn_output, None)
        if output_moe_loss:
            # Sec. 4: propagate MoE load-balancing loss upward for aggregation
            outputs += (moe_loss,)
        return outputs


class SamVisionLayer(nn.Module):
    def __init__(self, config, window_size):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = SamVisionAttention(config, window_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SamMLPBlock(config)
        self.window_size = window_size
        
        # Adapter Integration
        self.adapter = None
        if hasattr(config, "adapter_enabled") and config.adapter_enabled:
            adapter_dim = getattr(config, "adapter_dim", 64)
            # Optional GSPO encoder adapter policy (gated adapter). Decoder adapters are handled elsewhere.
            encoder_gate_enabled = getattr(config, "encoder_adapter_gate_enabled", False)
            encoder_gate_noise_std = getattr(config, "encoder_adapter_gate_noise_std", 0.0)
            if encoder_gate_enabled:
                self.adapter = GatedAdapterLayer(
                    in_features=config.hidden_size,
                    adapter_dim=adapter_dim,
                    gate_noise_std=encoder_gate_noise_std,
                )
            else:
                self.adapter = AdapterLayer(config.hidden_size, adapter_dim)


    def window_partition(self, hidden_states: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
        Partition into non-overlapping windows with padding if needed.
            hidden_states (tensor): input tokens with [batch_size, height, width, channel]. window_size (int): window
            size.

        Returns:
            windows: windows after partition with [batch_size * num_windows, window_size, window_size, channel].
            (pad_height, pad_width): padded height and width before partition
        """
        batch_size, height, width, channel = hidden_states.shape

        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_w, 0, pad_h))
        pad_height, pad_width = height + pad_h, width + pad_w

        hidden_states = hidden_states.reshape(
            batch_size, pad_height // window_size, window_size, pad_width // window_size, window_size, channel
        )
        windows = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, window_size, window_size, channel)
        return windows, (pad_height, pad_width)

    def window_unpartition(
        self, windows: torch.Tensor, window_size: int, padding_shape: Tuple[int, int], original_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Args:
        Window unpartition into original sequences and removing padding.
            hidden_states (tensor):
                input tokens with [batch_size * num_windows, window_size, window_size, channel].
            window_size (int):
                window size.
            padding_shape (Tuple):
                padded height and width (pad_height, pad_width).
            original_shape (Tuple): original height and width (height, width) before padding.

        Returns:
            hidden_states: unpartitioned sequences with [batch_size, height, width, channel].
        """
        pad_height, pad_width = padding_shape
        height, width = original_shape
        batch_size = windows.shape[0] // (pad_height * pad_width // window_size // window_size)
        hidden_states = windows.reshape(
            batch_size, pad_height // window_size, pad_width // window_size, window_size, window_size, -1
        )
        hidden_states = (
            hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(batch_size, pad_height, pad_width, -1)
        )

        hidden_states = hidden_states[:, :height, :width, :].contiguous()
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
        output_moe_loss: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        # Window partition
        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, padding_shape = self.window_partition(hidden_states, self.window_size)

        ### modify here
        attn_outputs = self.attn(
            hidden_states=hidden_states, output_attentions=output_attentions, output_moe_loss=output_moe_loss
        )
        if output_moe_loss:
            hidden_states, attn_weights, moe_loss = attn_outputs
        else:
            hidden_states, attn_weights = attn_outputs
        # Reverse window partition
        if self.window_size > 0:
            hidden_states = self.window_unpartition(hidden_states, self.window_size, padding_shape, (height, width))


        hidden_states = residual + hidden_states
        
        # MLP Block + Parallel Adapter
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        mlp_out = self.mlp(hidden_states)
        
        if self.adapter is not None:
            adapter_out = self.adapter(hidden_states)
            hidden_states = residual + mlp_out + adapter_out
        else:
            hidden_states = residual + mlp_out


        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        else:
            outputs += (None,)  # keep attn_weights on the same place
        if output_moe_loss:
            # Sec. 4: return per-layer MoE loss to be summed in encoder
            outputs += (moe_loss,)

        return outputs


class SamVisionNeck(nn.Module):
    def __init__(self, config: SamVisionConfig):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv2d(config.hidden_size, config.output_channels, kernel_size=1, bias=False)
        self.layer_norm1 = SamLayerNorm(config.output_channels, data_format="channels_first")
        self.conv2 = nn.Conv2d(config.output_channels, config.output_channels, kernel_size=3, padding=1, bias=False)
        self.layer_norm2 = SamLayerNorm(config.output_channels, data_format="channels_first")

    def forward(self, hidden_states):
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.layer_norm1(hidden_states)

        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        return hidden_states


class SamVisionEncoder(nn.Module):
    def __init__(self, config: SamVisionConfig):
        super().__init__()
        self.config = config
        self.image_size = config.image_size

        self.patch_embed = SamPatchEmbeddings(config)

        self.pos_embed = None
        if config.use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    config.image_size // config.patch_size,
                    config.image_size // config.patch_size,
                    config.hidden_size,
                )
            )

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = SamVisionLayer(
                config,
                window_size=config.window_size if i not in config.global_attn_indexes else 0,
            )
            self.layers.append(layer)

        self.neck = SamVisionNeck(config)

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.patch_embed

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_moe_loss: Optional[bool] = None,  # MoE loss for Conv-LoRA
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SamVisionEncoderOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_moe_loss = 0.0 if output_moe_loss else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, output_attentions=output_attentions, output_moe_loss=output_moe_loss
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if output_moe_loss:
                # Accumulate MoE load-balancing losses across layers (Sec. 4)
                all_moe_loss = all_moe_loss + layer_outputs[2]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.neck(hidden_states)

        if not return_dict:
            outputs = (
                hidden_states,
                all_hidden_states,
                all_self_attentions,
                all_moe_loss,
            )
            # if output_hidden_states:
            #     outputs = outputs + (all_hidden_states,)
            # if output_attentions:
            #     outputs = outputs + (all_self_attentions,)
            # if output_moe_loss:
            #     outputs = outputs + (all_moe_loss,)
            return outputs

        return SamVisionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            moe_loss=all_moe_loss,  # modify here
        )


class SamPreTrainedModel(PreTrainedModel):
    config_class = SamConfig
    base_model_prefix = "sam"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


SAM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SamConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


SAM_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`SamProcessor`]. See [`SamProcessor.__call__`] for
            details.
        input_points (`torch.FloatTensor` of shape `(batch_size, num_points, 2)`):
            Input 2D spatial points, this is used by the prompt encoder to encode the prompt. Generally yields to much
            better results. The points can be obtained by passing a list of list of list to the processor that will
            create corresponding `torch` tensors of dimension 4. The first dimension is the image batch size, the
            second dimension is the point batch size (i.e. how many segmentation masks do we want the model to predict
            per input point), the third dimension is the number of points per segmentation mask (it is possible to pass
            multiple points for a single mask), and the last dimension is the x (vertical) and y (horizontal)
            coordinates of the point. If a different number of points is passed either for each image, or for each
            mask, the processor will create "PAD" points that will correspond to the (0, 0) coordinate, and the
            computation of the embedding will be skipped for these points using the labels.
        input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points)`):
            Input labels for the points, this is used by the prompt encoder to encode the prompt. According to the
            official implementation, there are 3 types of labels

            - `1`: the point is a point that contains the object of interest
            - `0`: the point is a point that does not contain the object of interest
            - `-1`: the point corresponds to the background

            We added the label:

            - `-10`: the point is a padding point, thus should be ignored by the prompt encoder

            The padding labels should be automatically done by the processor.
        input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes, 4)`):
            Input boxes for the points, this is used by the prompt encoder to encode the prompt. Generally yields to
            much better generated masks. The boxes can be obtained by passing a list of list of list to the processor,
            that will generate a `torch` tensor, with each dimension corresponding respectively to the image batch
            size, the number of boxes per image and the coordinates of the top left and bottom right point of the box.
            In the order (`x1`, `y1`, `x2`, `y2`):

            - `x1`: the x coordinate of the top left point of the input box
            - `y1`: the y coordinate of the top left point of the input box
            - `x2`: the x coordinate of the bottom right point of the input box
            - `y2`: the y coordinate of the bottom right point of the input box

        input_masks (`torch.FloatTensor` of shape `(batch_size, image_size, image_size)`):
            SAM model also accepts segmentation masks as input. The mask will be embedded by the prompt encoder to
            generate a corresponding embedding, that will be fed later on to the mask decoder. These masks needs to be
            manually fed by the user, and they need to be of shape (`batch_size`, `image_size`, `image_size`).

        image_embeddings (`torch.FloatTensor` of shape `(batch_size, output_channels, window_size, window_size)`):
            Image embeddings, this is used by the mask decder to generate masks and iou scores. For more memory
            efficient computation, users can first retrieve the image embeddings using the `get_image_embeddings`
            method, and then feed them to the `forward` method instead of feeding the `pixel_values`.
        multimask_output (`bool`, *optional*):
            In the original implementation and paper, the model always outputs 3 masks per image (or per point / per
            bounding box if relevant). However, it is possible to just output a single mask, that corresponds to the
            "best" mask, by specifying `multimask_output=False`.
        attention_similarity (`torch.FloatTensor`, *optional*):
            Attention similarity tensor, to be provided to the mask decoder for target-guided attention in case the
            model is used for personalization as introduced in [PerSAM](https://arxiv.org/abs/2305.03048).
        target_embedding (`torch.FloatTensor`, *optional*):
            Embedding of the target concept, to be provided to the mask decoder for target-semantic prompting in case
            the model is used for personalization as introduced in [PerSAM](https://arxiv.org/abs/2305.03048).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "Segment Anything Model (SAM) for generating segmentation masks, given an input image and ",
    " optional 2D location and bounding boxes.",
    SAM_START_DOCSTRING,
)
class SamModel(SamPreTrainedModel):
    _tied_weights_keys = ["prompt_encoder.shared_embedding.positional_embedding"]

    def __init__(self, config):
        super().__init__(config)
        self.shared_image_embedding = SamPositionalEmbedding(config.vision_config)

        self.vision_encoder = SamVisionEncoder(config.vision_config)
        self.prompt_encoder = SamPromptEncoder(config.prompt_encoder_config, self.shared_image_embedding)
        
        # Extract decoder attention LoRA parameters from config (if available)
        decoder_attn_lora_r = getattr(config.mask_decoder_config, 'decoder_attention_lora_r', 0)
        decoder_attn_lora_alpha = getattr(config.mask_decoder_config, 'decoder_attention_lora_alpha', 1)
        decoder_attn_lora_dropout = getattr(config.mask_decoder_config, 'decoder_attention_lora_dropout', 0.0)
        
        self.mask_decoder = SamMaskDecoder(
            config.mask_decoder_config,
            lora_r=decoder_attn_lora_r,
            lora_alpha=decoder_attn_lora_alpha,
            lora_dropout=decoder_attn_lora_dropout
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.vision_encoder.get_input_embeddings()

    def get_image_wide_positional_embeddings(self):
        size = self.config.prompt_encoder_config.image_embedding_size
        target_device = self.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width

    @torch.no_grad()
    def get_image_embeddings(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns the image embeddings by passing the pixel values through the vision encoder.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Input pixel values
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        vision_output = self.vision_encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeddings = vision_output[0]
        return image_embeddings

    @torch.no_grad()
    def get_prompt_embeddings(
        self,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
    ):
        r"""
        Returns the prompt embeddings by passing the input points, labels, boxes and masks through the prompt encoder.

        Args:
            input_points (`torch.FloatTensor` of shape `(batch_size, point_batch_size, num_points_per_image, 2)`):
                Optional input points for the prompt encoder. The padding of the point is automatically done by the
                processor. `point_batch_size` refers to the number of masks that we want the model to predict per
                point. The model will output `point_batch_size` times 3 masks in total.
            input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points_per_image)`):
                Optional input labels for the prompt encoder. The padding of the labels is automatically done by the
                processor, or can be fed by the user.
            input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes_per_image, 4)`):
                Optional input boxes for the prompt encoder. The padding of the boxes is automatically done by the
                processor. users can also pass manually the input boxes.
            input_masks (`torch.LongTensor` of shape `(batch_size, image_size, image_size)`):
                Optional input masks for the prompt encoder.
        """
        prompt_output = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
        return prompt_output

    @add_start_docstrings_to_model_forward(SAM_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        multimask_output: bool = True,
        attention_similarity: Optional[torch.FloatTensor] = None,
        target_embedding: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_moe_loss: Optional[bool] = None,  # MoE loss for Conv-LoRA
        return_dict=None,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        r"""
        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoModel, AutoProcessor

        >>> model = AutoModel.from_pretrained("facebook/sam-vit-base")
        >>> processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")

        >>> img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-car.png"
        >>> raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        >>> input_points = [[[400, 650]]]  # 2D location of a window on the car
        >>> inputs = processor(images=raw_image, input_points=input_points, return_tensors="pt")

        >>> # Get segmentation mask
        >>> outputs = model(**inputs)

        >>> # Postprocess masks
        >>> masks = processor.post_process_masks(
        ...     outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        ... )
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and image_embeddings is None:
            raise ValueError("Either pixel_values or image_embeddings must be provided.")

        if pixel_values is not None and image_embeddings is not None:
            raise ValueError("Only one of pixel_values and image_embeddings can be provided.")

        if input_points is not None and len(input_points.shape) != 4:
            raise ValueError(
                "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.",
                " got {}.".format(input_points.shape),
            )
        if input_boxes is not None and len(input_boxes.shape) != 3:
            raise ValueError(
                "The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.",
                " got {}.".format(input_boxes.shape),
            )
        if input_points is not None and input_boxes is not None:
            point_batch_size = input_points.shape[1]
            box_batch_size = input_boxes.shape[1]
            if point_batch_size != box_batch_size:
                raise ValueError(
                    "You should provide as many bounding boxes as input points per box. Got {} and {}.".format(
                        point_batch_size, box_batch_size
                    )
                )

        image_positional_embeddings = self.get_image_wide_positional_embeddings()
        # repeat with batch size
        batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        vision_attentions = None
        vision_hidden_states = None
        vision_moe_loss = None

        if pixel_values is not None:
            vision_outputs = self.vision_encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_moe_loss=output_moe_loss,
                return_dict=return_dict,
            )
            image_embeddings = vision_outputs[0]

            if output_hidden_states:
                vision_hidden_states = vision_outputs[1]
            if output_attentions:
                # vision_attentions = vision_outputs[-1]
                vision_attentions = vision_outputs[2]
            if output_moe_loss:
                vision_moe_loss = vision_outputs[-1]

        if input_points is not None and input_labels is None:
            input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

        if input_points is not None and image_embeddings.shape[0] != input_points.shape[0]:
            raise ValueError(
                "The batch size of the image embeddings and the input points must be the same. ",
                "Got {} and {} respectively.".format(image_embeddings.shape[0], input_points.shape[0]),
                " if you want to pass multiple points for the same image, make sure that you passed ",
                " input_points of shape (batch_size, point_batch_size, num_points_per_image, 3) and ",
                " input_labels of shape (batch_size, point_batch_size, num_points_per_image)",
            )

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )

        low_res_masks, iou_predictions, mask_decoder_attentions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )

        if not return_dict:
            # output = (iou_predictions, low_res_masks)
            # if output_hidden_states:
            #     output = output + (vision_hidden_states,)

            # if output_attentions:
            #     output = output + (vision_attentions, mask_decoder_attentions)
            output = (
                iou_predictions,
                low_res_masks,
                vision_hidden_states,
                vision_attentions,
                mask_decoder_attentions,
                vision_moe_loss,
            )
            return output

        return SamImageSegmentationOutput(
            iou_scores=iou_predictions,
            pred_masks=low_res_masks,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
            mask_decoder_attentions=mask_decoder_attentions,
            vision_moe_loss=vision_moe_loss,
        )

```

```worktree-python
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import SamConfig

from ..constants import CLASS_LABEL, CLASS_LOGITS, COLUMN, IMAGE, IMAGE_VALID_NUM, LABEL, LOGITS, MASK_LABEL, MOE_LOSS
from .adaptation_layers import ConvLoRALinear, AdapterLayer, GatedAdapterLayer
from .custom_hf_models.modeling_sam_for_conv_lora import SamImageSegmentationOutput, SamModel
from .utils import assign_layer_ids, freeze_model_layers, image_mean_std

logger = logging.getLogger(__name__)


def multi_class_mask_decoder_forward(
    self,
    image_embeddings: torch.Tensor,
    image_positional_embeddings: torch.Tensor,
    sparse_prompt_embeddings: torch.Tensor,
    dense_prompt_embeddings: torch.Tensor,
    multimask_output: bool,
    output_attentions: Optional[bool] = None,
    attention_similarity: torch.Tensor = None,
    target_embedding: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Modify the forward method of SamMaskDecoder for multi-class semantic segmentation
    based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/modeling_sam.py#L468.

    Args:
        image_embeddings (`torch.Tensor`):
            the embeddings from the image encoder
        image_positional_embedding (`torch.Tensor`):
            positional encoding with the shape of image_embeddings
        sparse_prompt_embeddings (`torch.Tensor`):
            The embeddings of the points and boxes
        dense_prompt_embeddings (`torch.Tensor`):
            the embeddings of the mask inputs
        multimask_output (bool):
            Whether to return multiple masks or a single mask.
        output_attentions (bool, *optional*):
            Whether or not to return the attentions tensors of all attention layers.
    """
    batch_size, num_channels, height, width = image_embeddings.shape
    point_batch_size = sparse_prompt_embeddings.shape[1]
    # Concatenate output tokens
    output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
    output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

    if sparse_prompt_embeddings.sum().item() != 0:
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
    else:
        tokens = output_tokens
    point_embeddings = tokens.to(self.iou_token.weight.dtype)

    # Expand per-image data in batch direction to be per-point
    image_embeddings = image_embeddings + dense_prompt_embeddings
    image_embeddings = image_embeddings.repeat(point_batch_size, 1, 1, 1)
    image_positional_embeddings = image_positional_embeddings.repeat(point_batch_size, 1, 1, 1)

    # Run the transformer, image_positional_embedding are consumed
    point_embedding, image_embeddings, attentions = self.transformer(
        point_embeddings=point_embeddings,
        image_embeddings=image_embeddings,
        image_positional_embeddings=image_positional_embeddings,
        attention_similarity=attention_similarity,
        target_embedding=target_embedding,
        output_attentions=output_attentions,
    )
    iou_token_out = point_embedding[:, :, 0, :]
    mask_tokens_out = point_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

    # Upscale mask embeddings and predict masks using the mask tokens
    image_embeddings = image_embeddings.transpose(2, 3).reshape(
        batch_size * point_batch_size, num_channels, height, width
    )

    upscaled_embedding = self.upscale_conv1(image_embeddings)
    upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
    upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

    ################ Modify the original code. We aim at returning a single mask for each object.
    ################ Original logic is to return multiple masks.
    ################ So we use an MLP network to process all the mask proposals instead of multiple networks.
    # hyper_in_list = []
    # for i in range(self.num_mask_tokens):
    #     current_mlp = self.output_hypernetworks_mlps[i]
    #     hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
    # hyper_in = torch.stack(hyper_in_list, dim=2)
    hyper_in = self.output_hypernetworks_mlps[0](mask_tokens_out)
    ################

    _, num_channels, height, width = upscaled_embedding.shape
    upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
    masks = (hyper_in @ upscaled_embedding).reshape(
        batch_size, point_batch_size, -1, height, width
    )  # bs, 1, num_tokens, h, w

    ################ New added class prediction logic.
    class_predictions = self.output_classifier_mlps(mask_tokens_out).reshape(
        batch_size, point_batch_size, -1, self.num_classes + 1
    )
    ################

    # Generate mask quality predictions
    iou_pred = self.iou_prediction_head(iou_token_out)

    outputs = (masks, iou_pred)

    if output_attentions:
        outputs = outputs + (attentions,)
    else:
        outputs = outputs + (None,)

    return outputs + (class_predictions,)


def multi_class_sam_model_forward(
    self,
    pixel_values: Optional[torch.FloatTensor] = None,
    input_points: Optional[torch.FloatTensor] = None,
    input_labels: Optional[torch.LongTensor] = None,
    input_boxes: Optional[torch.FloatTensor] = None,
    input_masks: Optional[torch.LongTensor] = None,
    image_embeddings: Optional[torch.FloatTensor] = None,
    multimask_output: bool = True,
    attention_similarity: Optional[torch.FloatTensor] = None,
    target_embedding: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_moe_loss: Optional[bool] = None,  # MoE loss for Conv-LoRA
    return_dict=None,
    **kwargs,
) -> List[Dict[str, torch.Tensor]]:
    r"""
    Modify the forward method of SamModel for multi-class semantic segmentation
    based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/modeling_sam.py#L1279.

    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if pixel_values is None and image_embeddings is None:
        raise ValueError("Either pixel_values or image_embeddings must be provided.")

    if pixel_values is not None and image_embeddings is not None:
        raise ValueError("Only one of pixel_values and image_embeddings can be provided.")

    if input_points is not None and len(input_points.shape) != 4:
        raise ValueError(
            "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.",
            " got {}.".format(input_points.shape),
        )
    if input_boxes is not None and len(input_boxes.shape) != 3:
        raise ValueError(
            "The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.",
            " got {}.".format(input_boxes.shape),
        )
    if input_points is not None and input_boxes is not None:
        point_batch_size = input_points.shape[1]
        box_batch_size = input_boxes.shape[1]
        if point_batch_size != box_batch_size:
            raise ValueError(
                "You should provide as many bounding boxes as input points per box. Got {} and {}.".format(
                    point_batch_size, box_batch_size
                )
            )

    image_positional_embeddings = self.get_image_wide_positional_embeddings()
    # repeat with batch size
    batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]
    image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

    vision_attentions = None
    vision_hidden_states = None
    vision_moe_loss = None

    if pixel_values is not None:
        vision_outputs = self.vision_encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_moe_loss=output_moe_loss,
            return_dict=return_dict,
        )
        image_embeddings = vision_outputs[0]

        if output_hidden_states:
            vision_hidden_states = vision_outputs[1]
        if output_attentions:
            # vision_attentions = vision_outputs[-1]
            vision_attentions = vision_outputs[2]
        if output_moe_loss:
            vision_moe_loss = vision_outputs[-1]

    if input_points is not None and input_labels is None:
        input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

    if input_points is not None and image_embeddings.shape[0] != input_points.shape[0]:
        raise ValueError(
            "The batch size of the image embeddings and the input points must be the same. ",
            "Got {} and {} respectively.".format(image_embeddings.shape[0], input_points.shape[0]),
            " if you want to pass multiple points for the same image, make sure that you passed ",
            " input_points of shape (batch_size, point_batch_size, num_points_per_image, 3) and ",
            " input_labels of shape (batch_size, point_batch_size, num_points_per_image)",
        )

    sparse_embeddings, dense_embeddings = self.prompt_encoder(
        input_points=input_points,
        input_labels=input_labels,
        input_boxes=input_boxes,
        input_masks=input_masks,
    )

    low_res_masks, iou_predictions, mask_decoder_attentions, class_predictions = self.mask_decoder(
        image_embeddings=image_embeddings,
        image_positional_embeddings=image_positional_embeddings,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
        attention_similarity=attention_similarity,
        target_embedding=target_embedding,
        output_attentions=output_attentions,
    )

    if not return_dict:
        output = (
            iou_predictions,
            low_res_masks,
            vision_hidden_states,
            vision_attentions,
            mask_decoder_attentions,
            vision_moe_loss,
        )
        return output

    return (
        SamImageSegmentationOutput(
            iou_scores=iou_predictions,
            pred_masks=low_res_masks,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
            mask_decoder_attentions=mask_decoder_attentions,
            vision_moe_loss=vision_moe_loss,
        ),
        ################ New added. Return class predictions as well.
        class_predictions,
        ################
    )


class SAMForSemanticSegmentation(nn.Module):
    """
    Support SAM for semantic segmentation.
    Refer to https://huggingface.co/docs/transformers/main/model_doc/sam
    Paper alignment (Conv-LoRA, ICLR 2024):
    - Sec. 4 (Training): When Conv-LoRA is active, expose/aggregate MoE load-balancing loss produced inside
      attention projections (q/k/v). This module toggles and collects that loss when trainable Conv-LoRA exists.
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str,
        num_classes: int = 1,
        pretrained: Optional[bool] = True,
        frozen_layers: Optional[list] = None,
        num_mask_tokens: int = 1,
        image_norm: Optional[str] = None,
        adapter_enabled: bool = False,
        adapter_dim: int = 64,
        decoder_attention_lora_r: int = 0,
        decoder_attention_lora_alpha: int = 1,
        decoder_attention_lora_dropout: float = 0.0,
        decoder_adapter_enabled: bool = False,
        decoder_adapter_dim: int = 64,
    ):
        """
        Load a pretrained Segment Anything Model (SAM).

        Parameters
        ----------
        prefix
            The prefix of the SAMForSemanticSegmentation model.
        checkpoint_name
            Name of the SAM checkpoint.
        num_classes
            The number of classes
        pretrained
            Whether using the pretrained SAM models. If pretrained=True, download the pretrained model.
        frozen_layers
            A list of substrings of frozen layers' names.
        num_mask_tokens
            The number of mask proposals.
        image_norm
            How to normalize an image. We now support:
            - inception
                Normalize image by IMAGENET_INCEPTION_MEAN and IMAGENET_INCEPTION_STD from timm
            - imagenet
                Normalize image by IMAGENET_DEFAULT_MEAN and IMAGENET_DEFAULT_STD from timm
            - clip
                Normalize image by mean (0.48145466, 0.4578275, 0.40821073) and
                std (0.26862954, 0.26130258, 0.27577711), used for CLIP.
        adapter_enabled
            Whether to enable standard Adapter modules in the vision encoder.
        adapter_dim
            The dimension of the adapter bottleneck.
        decoder_attention_lora_r
            LoRA rank for decoder attention layers. If 0 (default), LoRA is disabled.
            Recommended value: 8 for parameter-efficient fine-tuning.
        decoder_attention_lora_alpha
            LoRA scaling factor for decoder attention. Default: 1.
        decoder_attention_lora_dropout
            Dropout probability for LoRA in decoder attention. Default: 0.0.
        """

        super().__init__()
        self.prefix = prefix
        self.pretrained = pretrained
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes
        self.frozen_layers = frozen_layers
        self.decoder_attention_lora_r = decoder_attention_lora_r
        self.decoder_attention_lora_alpha = decoder_attention_lora_alpha
        self.decoder_attention_lora_dropout = decoder_attention_lora_dropout

        self.device = None
        self.name_to_id = {}

        # Store adapter configs
        self.adapter_enabled = adapter_enabled
        self.adapter_dim = adapter_dim
        self.decoder_adapter_enabled = decoder_adapter_enabled
        self.decoder_adapter_dim = decoder_adapter_dim

        self._load_checkpoint(checkpoint_name)

        freeze_model_layers(self.model, self.frozen_layers)

        self.image_size = self.model.vision_encoder.image_size
        self.config = self.model.config
        self.image_mean, self.image_std = image_mean_std(image_norm)

        # Inject Adapters into vision encoder if enabled
        if self.adapter_enabled:
            logger.info(f"Injecting Adapters with dim={self.adapter_dim} into Vision Encoder")
            encoder_gate_enabled = getattr(self.config, "encoder_adapter_gate_enabled", False)
            encoder_gate_noise_std = getattr(self.config, "encoder_adapter_gate_noise_std", 0.0)
            for layer in self.model.vision_encoder.layers:
                # Manually create and assign adapter since config didn't have it at init time
                if encoder_gate_enabled:
                    layer.adapter = GatedAdapterLayer(
                        in_features=self.config.hidden_size,
                        adapter_dim=self.adapter_dim,
                        gate_noise_std=encoder_gate_noise_std,
                    )
                else:
                    layer.adapter = AdapterLayer(self.config.hidden_size, self.adapter_dim)


        self.model.mask_decoder.num_mask_tokens = num_mask_tokens
        mask_token_data = self.model.mask_decoder.mask_tokens.weight.data[0]
        self.model.mask_decoder.mask_tokens = nn.Embedding(num_mask_tokens, self.model.mask_decoder.hidden_size)
        for i in range(num_mask_tokens):
            self.model.mask_decoder.mask_tokens.weight.data[i] = mask_token_data
        hyper_mlps = self.model.mask_decoder.output_hypernetworks_mlps[0]
        self.model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList([hyper_mlps])
        if num_classes > 1:
            self.model.mask_decoder.num_classes = num_classes
            self.model.mask_decoder.output_classifier_mlps = nn.Linear(
                self.model.mask_decoder.hidden_size, num_classes + 1
            )

            mask_decoder_forward = multi_class_mask_decoder_forward.__get__(
                self.model.mask_decoder, self.model.mask_decoder.__class__
            )
            setattr(self.model.mask_decoder, "forward", mask_decoder_forward)

            sam_model_forward = multi_class_sam_model_forward.__get__(self.model, self.model.__class__)
            setattr(self.model, "forward", sam_model_forward)

        # for Conv-LoRA
        self.output_moe_loss = False

    def _load_checkpoint(self, checkpoint_name):
        if self.pretrained:
            # Load configuration and inject decoder attention LoRA parameters
            try:
                config = SamConfig.from_pretrained(checkpoint_name, local_files_only=True)
            except:
                config = SamConfig.from_pretrained(checkpoint_name)
            
            # Propagate decoder adapter settings into mask decoder config (if present)
            if hasattr(config, "mask_decoder_config") and config.mask_decoder_config is not None:
                config.mask_decoder_config.decoder_adapter_enabled = self.decoder_adapter_enabled
                config.mask_decoder_config.decoder_adapter_dim = self.decoder_adapter_dim

            # Inject LoRA parameters into mask decoder config
            config.mask_decoder_config.decoder_attention_lora_r = self.decoder_attention_lora_r
            config.mask_decoder_config.decoder_attention_lora_alpha = self.decoder_attention_lora_alpha
            config.mask_decoder_config.decoder_attention_lora_dropout = self.decoder_attention_lora_dropout
            
            # Try to load from local cache first to avoid network issues
            try:
                self.model = SamModel.from_pretrained(checkpoint_name, config=config, local_files_only=True)
                logger.info(f"Loaded model from local cache: {checkpoint_name}")
            except Exception as e:
                logger.info(f"Local cache not found, downloading from Hugging Face: {checkpoint_name}")
                self.model = SamModel.from_pretrained(checkpoint_name, config=config)
            
            if self.decoder_attention_lora_r > 0:
                logger.info(f"Decoder Attention LoRA enabled: r={self.decoder_attention_lora_r}, "
                          f"alpha={self.decoder_attention_lora_alpha}, dropout={self.decoder_attention_lora_dropout}")
        else:
            config = SamConfig(name_or_path=checkpoint_name)
            if hasattr(config, "mask_decoder_config") and config.mask_decoder_config is not None:
                config.mask_decoder_config.decoder_adapter_enabled = self.decoder_adapter_enabled
                config.mask_decoder_config.decoder_adapter_dim = self.decoder_adapter_dim

            config.mask_decoder_config.decoder_attention_lora_r = self.decoder_attention_lora_r
            config.mask_decoder_config.decoder_attention_lora_alpha = self.decoder_attention_lora_alpha
            config.mask_decoder_config.decoder_attention_lora_dropout = self.decoder_attention_lora_dropout
            self.model = SamModel(config)

    def save(self, save_path: str = "./"):
        self.model.save_pretrained(save_path)
        logger.info(f"Model weights for {self.prefix} is saved to {save_path}.")

    @property
    def image_key(self):
        return f"{self.prefix}_{IMAGE}"

    @property
    def image_valid_num_key(self):
        return f"{self.prefix}_{IMAGE_VALID_NUM}"

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    @property
    def image_column_prefix(self):
        return f"{self.image_key}_{COLUMN}"

    @property
    def image_feature_dim(self):
        return self.model.num_features

    @property
    def mask_label_key(self):
        return f"{self.prefix}_{MASK_LABEL}"

    @property
    def class_label_key(self):
        return f"{self.prefix}_{CLASS_LABEL}"

    @property
    def box_key(self):
        return f"{self.prefix}_box"

    @property
    def image_path_key(self):
        return f"{self.prefix}_image_path"

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, ConvLoRALinear):
                # If Conv-LoRA is present and trainable, enable returning moe_loss from the attention blocks (Sec. 4)
                self.output_moe_loss = True
                return self

        return self

    def forward(
        self,
        batch,
    ):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with mask predictions.
        """
        # binary
        if self.num_classes == 1:
            rets = self.model(
                batch[self.image_key],
                input_boxes=batch.get(self.box_key, None),
                multimask_output=False,
                output_moe_loss=self.output_moe_loss,
            )
            pred_masks = rets.pred_masks[:, 0, :, :, :]
            pred_masks = F.interpolate(
                pred_masks, (self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
            if self.training:
                rets_dict = {self.prefix: {LOGITS: pred_masks}}
            else:
                rets_dict = {self.prefix: {LOGITS: pred_masks, LABEL: batch[self.label_key]}}
        # multi-class
        else:
            rets = self.model(
                batch[self.image_key],
                input_boxes=batch.get(self.box_key, None),
                multimask_output=False,
                output_moe_loss=self.output_moe_loss,
            )
            rets, class_predictions = rets
            pred_masks = rets.pred_masks[:, 0, :, :, :]
            pred_classes = class_predictions[:, 0, :, :]
            pred_masks = F.interpolate(
                pred_masks, (self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
            if self.training:
                rets_dict = {self.prefix: {LOGITS: pred_masks, CLASS_LOGITS: pred_classes}}
            else:
                rets_dict = {
                    self.prefix: {
                        LOGITS: pred_masks,
                        CLASS_LOGITS: pred_classes,
                        LABEL: batch[self.label_key],
                    }
                }
        if self.output_moe_loss:
            rets_dict[self.prefix].update({MOE_LOSS: rets.vision_moe_loss})
        return rets_dict

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        In the AutoModel scenario, this function may not always return the correct result.
        Thus, you can use "print(json.dumps(name_to_id, indent=2))" to manually check whether
        the layer ids are reasonable.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = "model"
        pre_encoder_patterns = (
            "vision_encoder",
            "prompt_encoder",
        )
        post_encoder_patterns = ("mask_decoder",)

        names = [n for n, _ in self.named_parameters()]

        name_to_id, names = assign_layer_ids(
            names=names,
            pre_encoder_patterns=pre_encoder_patterns,
            post_encoder_patterns=post_encoder_patterns,
            model_pre=model_prefix,
        )
        if len(names) > 0:
            logger.debug(f"outer layers are treated as head: {names}")
        for n in names:
            assert n not in name_to_id
            name_to_id[n] = 0

        return name_to_id

```

```worktree-yaml
model:
  names:
  categorical_mlp:
    hidden_size: 64
    activation: "leaky_relu"
    num_layers: 1
    dropout: 0.1
    normalization: "layer_norm"
    data_types:
      - "categorical"

  numerical_mlp:
    hidden_size: 128
    activation: "leaky_relu"
    num_layers: 1
    dropout: 0.1
    normalization: "layer_norm"
    token_dim: 8
    embedding_arch:
    data_types:
      - "numerical"
    merge: "concat"

  hf_text:
    checkpoint_name: "google/electra-base-discriminator"
    gradient_checkpointing: False
    pooling_mode: 'cls'  # The pooling mode, can be 'cls' or 'mean'
    data_types:
      - "text"
    tokenizer_name: "hf_auto"
    use_fast: True  # Use a fast Rust-based tokenizer if it is supported for a given model. If a fast tokenizer is not available for a given model, a normal Python-based tokenizer is returned instead.
    max_text_len: 512  # If None or <=0, then use the max length of pretrained models.
    insert_sep: True
    low_cpu_mem_usage: False
    text_segment_num: 2
    stochastic_chunk: False
    text_aug_detect_length: 10                # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.1            # augmentation magnitude randomly drawn from [0, text_trivial_aug_maxscale]
    text_train_augment_types:        # specify augmentation space manually, will randomly select one from the following and identity
      # - "random_swap(0.05)"          # less than 0.1 based on eda paper
      # - "random_delete(0.05)"        # less than 0.1 based on eda paper
      # - "syn_replacement(0.05)"  # less than 0.1 based on eda paper
      # - "insert_punc(0.05)"

  ner_text:
    checkpoint_name: "bert-base-cased"
    max_text_len: 512
    gradient_checkpointing: False
    low_cpu_mem_usage: False
    data_types:
      - "text_ner"
    tokenizer_name: "hf_auto"
    insert_sep: False
    text_segment_num: 2
    stochastic_chunk: False
    special_tags: 
      - X # CLS, SEP, and non-first tokens of a word will be labelled as X
      - O # Outside of a named entity

  document_transformer:
    checkpoint_name: "microsoft/layoutlmv3-base" # document foundation models
    gradient_checkpointing: False
    pooling_mode: 'cls'  # The pooling mode, can be 'cls' or 'mean'
    data_types:
      - "document"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: 224
    tokenizer_name: "hf_auto"
    max_text_len: 512  # If None or <=0, then use the max length of pretrained models.
    insert_sep: True
    low_cpu_mem_usage: False
    text_segment_num: 2
    stochastic_chunk: False
    text_aug_detect_length: 10                # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.0            # augmentation magnitude randomly drawn from [0, text_trivial_aug_maxscale]

  t_few:
    checkpoint_name: "t5-small" #"bigscience/T0_3B"
    gradient_checkpointing: False
    data_types:
      - "text"
    tokenizer_name: "hf_auto"
    length_norm: 1.0 # Normalizes length to adjust for length bias in target template
    unlikely_loss: 1.0 # Adds loss term that lowers probability of incorrect outputs
    mc_loss: 1.0 # Adds multiple choice cross entropy loss
    max_text_len: 512  # If None or <=0, then use the max length of pretrained models.
    text_segment_num: 2
    insert_sep: True
    low_cpu_mem_usage: False
    stochastic_chunk: False
    text_aug_detect_length: 10                # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.0            # augmentation magnititude randomly drawn from [0, text_trivial_aug_maxscale]
    text_train_augment_types:

  timm_image:
    checkpoint_name: "swin_base_patch4_window7_224"
    mix_choice: "all_logits"
    data_types:
      - "image"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: null
    image_chan_num: 3
    use_learnable_image: False
    max_image_num_per_column: 1

  mmdet_image:
    checkpoint_name: "yolov3_mobilenetv2_8xb24-320-300e_coco"
    config_file: ""
    data_types:
      - "image"
    max_img_num_per_col: 1
    output_bbox_format: "xyxy"  # now support xyxy or xywh, for bbox format details see https://keras.io/api/keras_cv/bounding_box/formats/
    frozen_layers: null
    coco_root: null

  mmocr_text_detection:
    checkpoint_name: "TextSnake"
    data_types:
      - "image"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: 224
    max_img_num_per_col: 2

  mmocr_text_recognition:
    checkpoint_name: "ABINet"
    data_types:
      - "image"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: 224
    max_img_num_per_col: 2

  clip:
    checkpoint_name: "openai/clip-vit-base-patch32"
    data_types:
      - "image"
      - "text"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "clip"
    image_size: 224
    image_chan_num: 3
    use_learnable_image: False
    max_image_num_per_column: 1
    tokenizer_name: "clip"
    max_text_len: 77  # The maximum possible length.
    insert_sep: False
    text_segment_num: 1
    stochastic_chunk: False
    text_aug_detect_length: 10                     # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.0                      # scale randomly drawn from [0, text_trivial_aug_maxscale]
    text_train_augment_types:        # specify augmentation space manually, will randomly select one from the following and identity
      # - "random_swap(0.05)"          # less than 0.1 based on eda paper
      # - "random_delete(0.05)"        # less than 0.1 based on eda paper
      # - "syn_replacement(0.05)"  # less than 0.1 based on eda paper
      # - "insert_punc(0.05)"

  fusion_mlp:
    aux_loss_weight:
    adapt_in_features: "max"
    hidden_sizes:
      - 128
    activation: "leaky_relu"
    dropout: 0.1
    normalization: "layer_norm"
    data_types:

  fusion_ner:
    weight:
    adapt_in_features: "max"
    hidden_sizes:
      - 128
    activation: "leaky_relu"
    drop_rate: 0.1
    normalization: "layer_norm"
    data_types:

  fusion_transformer:
    aux_loss_weight:
    hidden_size: 192
    num_blocks: 3
    attention_num_heads: 8
    adapt_in_features: "max"
    attention_dropout: 0.2
    residual_dropout: 0.0
    ffn_dropout: 0.1
    ffn_hidden_size: 192
    normalization: "layer_norm"
    ffn_activation: "geglu"
    head_activation: "relu"
    data_types:
    additive_attention: False # Whether to use lightweight additive attention, can be True, False or "auto"
    share_qv_weights: False # Whether to share weight for query and value, can be True, False or "auto"

  ft_transformer:
    data_types:
      - "categorical"
      - "numerical"
    embedding_arch:
      - 'linear'
    token_dim: 192
    hidden_size: 192
    num_blocks: 3
    attention_num_heads: 8
    attention_dropout: 0.2
    residual_dropout: 0.0
    ffn_dropout: 0.1
    ffn_hidden_size: 192
    ffn_activation: "geglu"
    head_activation: "relu"
    normalization: "layer_norm"
    merge: "concat"
    requires_all_dtypes: False
    additive_attention: False # Whether to use lightweight additive attention, can be True, False or "auto"
    share_qv_weights: False # Whether to share weight for query and value, can be True, False or "auto"
    pooling_mode: "cls"
    checkpoint_name: null

  sam:
    checkpoint_name: "facebook/sam-vit-huge"
    image_norm: "imagenet"
    data_types:
      - "semantic_segmentation_img"
    train_transforms:
      - "random_horizontal_flip"
    val_transforms: []
    img_transforms:
      - "resize_to_square"
    gt_transforms:
      - "resize_gt_to_square"
    max_img_num_per_col: 1
    frozen_layers: ["mask_decoder.iou_prediction_head", "prompt_encoder"]
    num_mask_tokens: 1
    ignore_label: 255
    adapter_enabled: False
    adapter_dim: 64
    # GSPO-Adapter extension (encoder only; kept here to build the correct adapter module type)
    encoder_adapter_gate_enabled: False
    encoder_adapter_gate_noise_std: 0.0
    decoder_attention_lora_r: 0
    decoder_attention_lora_alpha: 1
    decoder_attention_lora_dropout: 0.0
    decoder_adapter_enabled: False
    decoder_adapter_dim: 64

  meta_transformer:
    data_types:
      - "image"
      - "text"
      - "categorical"
      - "numerical"
    checkpoint_path: null
    model_version: "base"
    requires_all_dtypes: False
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: 224
    image_chan_num: 3
    use_learnable_image: False
    max_image_num_per_column: 1
    tokenizer_name: "hf_auto"
    max_text_len: 512  # If None or <=0, then use the max length of pretrained models.
    insert_sep: True
    text_segment_num: 2
    stochastic_chunk: False
    text_aug_detect_length: 10     # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.1  # augmentation magnitude randomly drawn from [0, text_trivial_aug_maxscale]
    text_train_augment_types:
    merge: "concat"

```

```worktree-yaml
optim:
  optim_type: "adamw"
  lr: 1.0e-4
  weight_decay: 0.001
  lr_choice: "layerwise_decay"
  lr_decay: 0.9
  lr_schedule: "cosine_decay"
  max_epochs: 20
  max_steps: -1
  warmup_steps: 0.1
  end_lr: 0
  lr_mult: 1 # multiply lr for downstream heads
  patience: 10
  val_check_interval: 0.5 # Check validation score a fraction of epoch if float and every n steps if integer with n < total step in epoch.
  check_val_every_n_epoch: 1 # Check validation score every n epochs.
  skip_final_val: False # Flag to skip the last validation
  gradient_clip_val: 1
  gradient_clip_algorithm: "norm"
  track_grad_norm: -1 # Whether to check gradient norm. We can set it to 2 to check for gradient norm.
  log_every_n_steps: 10
  label_smoothing: 0
  top_k: 3
  top_k_average_method:
    "greedy_soup" # We support averaging method described in https://arxiv.org/pdf/2203.05482.pdf.
    # Currently support "uniform_soup", "greedy_soup", and "best".
  peft: null # Can be 'bit_fit' (only finetune bias), 'norm_fit' (finetune the normalization terms + bias terms), lora (LoRA Adaptations only), lora_bias (LoRA Adaptation + bit_fit), lora_norm (LoRA Adaptation + norm_fit), or null
  lora:
    module_filter: null # Specify which module (if any) to adapt(e.g. ".*EncDecAttention|.*DenseReluDense"). Default consider all modules in a model (i.e. empty filter).
    filter: # Specify which layer in a module to adapt. Default fine-tune only query and value attention weights, recommended in https://arxiv.org/abs/2106.09685
      - "query"
      - "value"
      - "^q$" # The filter can also be a regex. We will use re.match(filter, name) to determine if LoRA should be applied.
      - "^v$"
      - "^k$"
      - "^o$"
    r: 8
    alpha: 8
    conv_lora_expert_num: 8 # default setting for Conv-LoRA
    gspo_enabled: False # Enable GSPO (Group Sequence Policy Optimization) enhancements
    gspo_group_size: 3 # Number of experts to select in GSPO mode (TopK)
    gspo_quality_momentum: 0.9 # Momentum coefficient for expert quality history updates
    gspo_warmup_epochs: 5 # Number of epochs before enabling full GSPO training
    gspo_contrastive_weight: 0.1 # Weight for contrastive loss in GSPO training
  loss_func:
    "auto" # The replaced loss for regression. Can only support loss function in torch.nn.
    # example
    # "BCEWithLogitsLoss" or "nn.BCEWithLogitsloss"
  focal_loss:
    alpha: null
    gamma: 2.0
    reduction: "mean"
  mask2former_loss:
    loss_cross_entropy_weight: 10.0
    loss_mask_weight: 5.0
    loss_dice_weight: 5.0
  extra_trainable_params: []
  cross_modal_align: null
  cross_modal_align_weight: 0
  automatic_optimization: True
  lemda:
    turn_on: False
    arch_type: "mlp_vae"
    z_dim: 8
    num_layers: 6
    kld_weight: 0.1
    mse_weight: 0.1
    adv_weight: 1.0e-4
    consist_weight: 0.01
    consist_threshold: 0.5
    lr: 1.0e-4
    optim_type: "adamw"
    weight_decay: 1.0e-5
  gspo:
    lambda_smooth: 0.1
    lambda_boundary: 0.3
    w_boundary: 0.3
    w_smooth: 0.1
    w_thin: 0.05
    # GSPO-Adapter extension (encoder adapters only; decoder adapters excluded by design)
    # When False (default), encoder adapters will be frozen during GSPO-active epochs/steps.
    train_encoder_adapter: False
    # Stage 1: add small exploration noise on adapter outputs during GSPO rollouts
    encoder_adapter_noise_std: 0.0
    # Stage 2: gated adapter policy exploration noise (applies only if model.sam.encoder_adapter_gate_enabled=True)
    encoder_adapter_gate_noise_std: 0.0

```

```worktree-python
import importlib.util
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.special import softmax

from autogluon.core.metrics import Scorer

from ..constants import LABEL, LOGITS, SEMANTIC_MASK, SEMANTIC_SEGMENTATION_IMG
from ..optim import SemanticSegmentationLitModule, get_loss_func, get_norm_layer_param_names, get_peft_param_names
from ..optim.metrics.semantic_seg_metrics import Balanced_Error_Rate_Pred as Balanced_Error_Rate
from ..optim.metrics.semantic_seg_metrics import Binary_IoU_Pred as Binary_IoU
from ..optim.metrics.semantic_seg_metrics import Binary_DICE_Pred as Binary_DICE
from ..optim.metrics.semantic_seg_metrics import COD_METRICS_NAMES_Pred as COD_METRICS_NAMES
from ..optim.metrics.semantic_seg_metrics import Multiclass_IoU_Pred as Multiclass_IoU
from ..optim.metrics.semantic_seg_metrics import Multiclass_DICE_Pred as Multiclass_DICE
from ..utils import extract_from_output, setup_save_path
from ..utils.tta_utils import TTAPredictor, dice_coefficient, iou_score
from .base import BaseLearner

logger = logging.getLogger(__name__)

from ..constants import BER, DICE, EM, FM, IOU, MAE, SEMANTIC_SEGMENTATION, SM


class SemanticSegmentationLearner(BaseLearner):
    def __init__(
        self,
        label: Optional[str] = None,
        problem_type: Optional[str] = SEMANTIC_SEGMENTATION,
        presets: Optional[str] = None,
        eval_metric: Optional[Union[str, Scorer]] = "iou",
        hyperparameters: Optional[dict] = None,
        path: Optional[str] = None,
        verbosity: Optional[int] = 2,
        num_classes: Optional[int] = None,  # TODO: can we infer this from data?
        warn_if_exist: Optional[bool] = True,
        enable_progress_bar: Optional[bool] = None,
        pretrained: Optional[bool] = True,
        validation_metric: Optional[str] = "iou",
        sample_data_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            label=label,
            problem_type=problem_type,
            presets=presets,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            path=path,
            verbosity=verbosity,
            warn_if_exist=warn_if_exist,
            enable_progress_bar=enable_progress_bar,
            pretrained=pretrained,
            validation_metric=validation_metric,
        )
        self._output_shape = num_classes
        self._sample_data_path = sample_data_path
        self._tta_predictor = None  # Will be initialized when TTA is enabled

        if self._sample_data_path is not None:
            infer_output_shape = self.get_semantic_segmentation_class_num(self._sample_data_path)
            if num_classes is not None:
                assert (
                    num_classes == infer_output_shape
                ), f"The provided number of classes '{num_classes}' and the inferred class number {infer_output_shape}' from the sample data should be consistent."
            else:
                self._output_shape = infer_output_shape

        # Training-time box prompt configuration (used when constructing LitModule)
        self._train_box_prompt_cfg = {
            "mode": "off",       # off / gt / noisy / mix / predict
            "p_no": 0.0,
            "p_gt": 0.0,
            "p_noisy": 0.0,
            "noise_frac": 0.12,
        }
        # BBoxPromptPredictor instance for "predict" mode (set via external config)
        self._train_bbox_predictor = None

    def get_semantic_segmentation_class_num(self, sample_data_path):
        """
        Get the number of classes for given data.

        Parameters
        ----------
            sample_data_path
                This is used for automatically inference num_classes of semantic segmentation dataset.
                Could be an image directory, image file or pd.DataFrame.
        Returns
        -------
            The number of classes.
        """
        if isinstance(sample_data_path, str):
            if os.path.isdir(sample_data_path):
                mask_files = os.listdir(sample_data_path)
                num_classes = []
                for mask_file in mask_files:
                    per_num_classes = self.get_semantic_segmentation_class_num(
                        os.path.join(sample_data_path, mask_file)
                    )
                    num_classes.append(per_num_classes)
                return max(num_classes)
            else:
                mask = Image.open(sample_data_path)
                mode = mask.mode

                if mode == "1":
                    return 1
                classes = np.unique(mask)
                if mode == "L" and np.array_equal(classes, np.array([0, 255])):
                    return 1

                return max(classes).item() + 1  # include background

        elif isinstance(sample_data_path, pd.DataFrame):
            num_classes = []
            for idx in range(sample_data_path.shape[0]):
                row = sample_data_path.iloc[idx]
                mask_file = row[self._label_column]
                per_num_classes = self.get_semantic_segmentation_class_num(mask_file)
                num_classes.append(per_num_classes)
            return max(num_classes)

    def infer_output_shape(self):
        if self._output_shape is None:
            self._output_shape = self.get_semantic_segmentation_class_num(self._train_data)

    @staticmethod
    def get_peft_param_names_per_run(model, config):
        peft_param_names = None
        peft = config.optim.peft
        if peft:
            norm_param_names = get_norm_layer_param_names(model)
            peft_param_names = get_peft_param_names(
                norm_param_names,
                peft=peft,
                extra_params=config.optim.extra_trainable_params,
            )
        return peft_param_names

    def get_loss_func_per_run(self, config, mixup_active=None):
        loss_func = get_loss_func(
            problem_type=self._problem_type,
            loss_func_name=config.optim.loss_func,
            config=config.optim,
            num_classes=self._output_shape,
        )
        return loss_func, None

    def evaluate_semantic_segmentation(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = False,
    ):
        """
        Evaluate semantic segmentation on a test dataset based on "torchmetrics".

        Parameters
        ----------
        data
            A dataframe, containing the same columns as the training data.
            Or a str, that is a path of the annotation file for detection.
        metrics
            Metrics used for evaluation.
        return_pred
            Whether to return the prediction result of each row.
        realtime
            Whether to do realtime inference, which is efficient for small data (default None).
            If provided None, we would infer it on based on the data modalities
            and sample number.
        """
        
        # If TTA is enabled, use TTA evaluation
        if self._tta_predictor is not None:
            # Use cache settings if configured
            cache_dir = getattr(self, '_tta_cache_dir', None)
            resume_from_cache = getattr(self, '_tta_resume_from_cache', True)
            return self._evaluate_with_tta(
                data, metrics, return_pred, 
                cache_dir=cache_dir,
                resume_from_cache=resume_from_cache
            )

        def get_metric_predict(
            metric_name: str,
            num_classes: Optional[int] = None,
        ):
            """
            Obtain a torchmerics.Metric from its name.
            Define a customized metric function in case that torchmetrics doesn't support some metric.

            Parameters
            ----------
            metric_name
                Name of metric.
            num_classes
                Number of classes.
            is_matching
                Whether is matching.
            problem_type
                Type of problem, e.g., binary and multiclass.

            Returns
            -------
            torchmetrics.Metric
                A torchmetrics.Metric object.
            custom_metric_func
                A customized metric function.
            """
            if metric_name == BER:
                return Balanced_Error_Rate()
            elif metric_name in [SM, EM, FM, MAE]:
                return COD_METRICS_NAMES[metric_name]
            elif metric_name == IOU:
                if num_classes == 1:
                    return Binary_IoU()
                else:
                    return Multiclass_IoU(num_classes=num_classes)
            elif metric_name == DICE:
                if num_classes == 1:
                    return Binary_DICE()
                else:
                    return Multiclass_DICE(num_classes=num_classes)
            else:
                raise ValueError(f"Unknown metric {metric_name}")

        outputs = self.predict_per_run(
            data=data,
            realtime=realtime,
            requires_label=False,
        )

        if self._output_shape == 1:
            logits = extract_from_output(ret_type=LOGITS, outputs=outputs, as_ndarray=False)
        else:
            logits = extract_from_output(ret_type=SEMANTIC_MASK, outputs=outputs, as_ndarray=False)
        y_pred = logits.float()
        y_true = [ele[LABEL] for ele in outputs]
        y_true = torch.cat(y_true)

        assert len(y_true) == len(y_pred)

        results = {}
        if isinstance(metrics, str):
            metrics = [metrics]
        for per_metric_name in metrics:
            per_metric = get_metric_predict(metric_name=per_metric_name.lower(), num_classes=self._output_shape)
            for y_p, y_t in zip(y_pred, y_true):
                per_metric.update(y_p.unsqueeze(0), y_t.unsqueeze(0))
            score = per_metric.compute()
            results[per_metric_name] = score.item()

        if return_pred:
            return results, outputs
        else:
            return results
    
    def _evaluate_with_tta(
        self,
        data: Union[pd.DataFrame, dict, list],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        cache_dir: Optional[str] = None,
        resume_from_cache: bool = True,
    ):
        """
        Evaluate using Test-Time Augmentation with checkpoint support.
        
        This method performs TTA for each image and evaluates the results.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, dict, list]
            Input data containing image and label paths
        metrics : Optional[Union[str, List[str]]]
            Metrics to compute
        return_pred : Optional[bool]
            Whether to return predictions
        cache_dir : Optional[str]
            Directory to cache TTA predictions for resume (default: None, no caching)
        resume_from_cache : bool
            Whether to resume from cache if exists (default: True)
        """
        logger.info(f"Evaluating with TTA ({len(self._tta_predictor.transforms)} augmentations per image)...")
        
        def get_metric_predict(metric_name: str, num_classes: Optional[int] = None):
            """Get metric function."""
            if metric_name == BER:
                return Balanced_Error_Rate()
            elif metric_name in [SM, EM, FM, MAE]:
                return COD_METRICS_NAMES[metric_name]
            elif metric_name == IOU:
                if num_classes == 1:
                    return Binary_IoU()
                else:
                    return Multiclass_IoU(num_classes=num_classes)
            elif metric_name == DICE:
                if num_classes == 1:
                    return Binary_DICE()
                else:
                    return Multiclass_DICE(num_classes=num_classes)
            else:
                raise ValueError(f"Unknown metric {metric_name}")
        
        # Get model and preprocessor
        model = self._model
        model.eval()
        
        # Ensure model is on the correct device (GPU if available)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            if next(model.parameters()).device.type != 'cuda':
                logger.info("Moving model to GPU for TTA inference...")
                model = model.cuda()
        else:
            device = torch.device('cpu')
            logger.warning("GPU not available, using CPU for TTA (will be slow)")
        
        logger.info(f"TTA inference device: {device}")
        
        # ============================================================
        # 方案 4: Sanity Check - 提前发现问题
        # ============================================================
        logger.info("🔍 Running sanity checks before full evaluation...")
        
        # Sanity Check 1: 验证数据
        if len(data) == 0:
            raise ValueError("❌ No data to evaluate!")
        logger.info(f"  ✓ Total images to process: {len(data)}")
        logger.info(f"  ✓ DataFrame columns: {data.columns.tolist()}")
        
        # Sanity Check 2: 检查设备和模型
        logger.info(f"  ✓ Model device: {device}")
        logger.info(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            logger.info(f"  ✓ GPU memory allocated: {allocated_gb:.2f} GB")
        
        # Sanity Check 3: 测试单个样本的 TTA pipeline
        logger.info(f"  Testing TTA pipeline on first image...")
        test_row = data.iloc[0]
        test_img_path = test_row['image']
        
        # Get data processors for image preprocessing
        from torchvision import transforms as T
        
        # Create preprocessing pipeline similar to model's preprocessing
        # Based on SAM's image preprocessing
        def preprocess_image_for_sam(image_np):
            """Preprocess image for SAM model."""
            from PIL import Image as PILImage
            
            # Ensure correct format
            if len(image_np.shape) == 2:
                image_np = np.stack([image_np] * 3, axis=2)
            
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            
            # Convert to PIL
            img_pil = PILImage.fromarray(image_np.astype(np.uint8))
            
            # Resize to model's expected input size (typically 1024x1024 for SAM)
            target_size = model.image_size
            img_resized = img_pil.resize((target_size, target_size), PILImage.BILINEAR)
            
            # Convert to tensor and normalize
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            
            # Close PIL objects
            img_pil.close()
            img_resized.close()
            
            # Apply model's normalization (SAM uses specific mean/std)
            mean = np.array(model.image_mean).reshape(1, 1, 3)
            std = np.array(model.image_std).reshape(1, 1, 3)
            img_array = (img_array - mean) / std
            
            # Convert to tensor (C, H, W)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().contiguous()
            
            # Free intermediate array
            del img_array
            
            return img_tensor
        
        # Create a prediction function for TTA
        def predict_fn(
            image: np.ndarray,
            box_prompt: Optional[np.ndarray] = None,
            box_ref_shape: Optional[Tuple[int, int]] = None,
        ) -> np.ndarray:
            """
            Prediction function for a single image used by TTA.
            
            Parameters
            ----------
            image : np.ndarray
                Input image (H, W, C) in range [0, 255] or [0, 1]
            
            Returns
            -------
            prob : np.ndarray
                Probability map (H, W) for binary or (C, H, W) for multi-class
            """
            # Preprocess image
            img_tensor = preprocess_image_for_sam(image)
            
            # Add batch dimension and move to device
            img_tensor = img_tensor.unsqueeze(0).to(device, non_blocking=True)

            box_tensor = None
            if box_prompt is not None:
                # box_prompt stays in ORIGINAL image coords (no TTA transform)
                ref_h, ref_w = box_ref_shape if box_ref_shape is not None else image.shape[:2]
                scale_x = model.image_size / float(ref_w)
                scale_y = model.image_size / float(ref_h)
                box_scaled = box_prompt.astype(np.float32).copy()
                box_scaled[0::2] *= scale_x
                box_scaled[1::2] *= scale_y
                box_scaled = np.clip(box_scaled, 0, model.image_size - 1)
                box_tensor = torch.from_numpy(box_scaled).to(device).unsqueeze(0).unsqueeze(0)  # (1,1,4)
            
            def forward_prob(optional_box_tensor: Optional[torch.Tensor]) -> np.ndarray:
                # Create batch dict for model
                # Need to provide a dummy label for model's forward pass
                batch = {
                    model.prefix + "_image": img_tensor,
                    model.prefix + "_label": torch.zeros(
                        (1, model.image_size, model.image_size), dtype=torch.long, device=device
                    ),
                }
                if optional_box_tensor is not None:
                    batch[model.box_key] = optional_box_tensor
                outputs = model(batch)

                logits = outputs[model.prefix][LOGITS]
                if self._output_shape == 1:
                    # Binary segmentation - remove channel dim
                    prob = torch.sigmoid(logits[0, 0]).cpu().numpy().copy()
                else:
                    # Multi-class segmentation
                    prob = torch.softmax(logits[0], dim=0).cpu().numpy().copy()

                # Free GPU memory immediately and aggressively
                del batch, outputs, logits
                return prob

            # Forward pass
            with torch.no_grad():
                mode = getattr(self, "_tta_box_prompt_mode", "off")
                if mode == "off" or box_tensor is None:
                    # No box prompt
                    prob = forward_prob(None)
                elif mode == "replace":
                    # Use only box-prompted prediction
                    prob = forward_prob(box_tensor)
                elif mode == "add":
                    # Fuse base prediction + box-prompted prediction (treat box as an extra inference view)
                    prob_base = forward_prob(None)
                    prob_box = forward_prob(box_tensor)
                    prob = ((prob_base + prob_box) / 2.0).astype(np.float32, copy=False)
                    del prob_base, prob_box
                else:
                    raise ValueError(f"Unknown tta_box_prompt_mode: {mode}")

                # Free GPU memory immediately and aggressively
                del img_tensor
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for GPU to finish

            return prob
        
        # Complete Sanity Check 3: Test TTA on first image
        try:
            test_img_pil = Image.open(test_img_path)
            test_img = np.array(test_img_pil)
            test_img_pil.close()
            
            if len(test_img.shape) == 2:
                test_img = np.stack([test_img] * 3, axis=2)
            
            logger.info(f"    Image shape: {test_img.shape}")
            
            # Test TTA prediction
            test_pred_mask, test_pred_prob = self._tta_predictor.predict_with_tta(
                test_img, predict_fn, return_probs=True
            )
            
            logger.info(f"    Output shape: {test_pred_prob.shape}")
            
            # Verify output shape matches input
            assert test_pred_prob.shape[:2] == test_img.shape[:2], \
                f"❌ Output shape mismatch! Expected {test_img.shape[:2]}, got {test_pred_prob.shape[:2]}"
            
            logger.info(f"  ✓ TTA pipeline test PASSED!")
            
            # Clean up test data
            del test_img, test_pred_mask, test_pred_prob
            
        except Exception as e:
            logger.error(f"❌ Sanity check FAILED! Error: {e}")
            logger.error("Please fix the issue before running full evaluation.")
            raise RuntimeError(f"TTA Sanity Check Failed: {e}") from e
        
        logger.info("✅ All sanity checks passed! Starting full evaluation...\n")
        
        # ============================================================
        # 方案 2: 断点续传机制
        # ============================================================
        import os
        import pickle
        
        # Setup cache directory
        processed_indices = set()
        all_preds = []
        all_labels = []
        cache_file = None
        processed_indices_file = None
        
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "tta_predictions.pkl")
            processed_indices_file = os.path.join(cache_dir, "processed_indices.txt")
            
            # Try to load cached data
            if resume_from_cache and os.path.exists(cache_file):
                try:
                    logger.info(f"📂 Loading cached predictions from {cache_file}")
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                        all_preds = cache_data['preds']
                        all_labels = cache_data['labels']
                        processed_indices = set(cache_data['indices'])
                    logger.info(f"✅ Loaded {len(all_preds)} cached predictions (resuming from checkpoint)")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load cache: {e}. Starting fresh.")
                    processed_indices = set()
                    all_preds = []
                    all_labels = []
            else:
                if cache_dir:
                    logger.info(f"💾 Caching enabled. Predictions will be saved to {cache_dir}")
        
        # Initialize metrics early so we can stream updates and avoid storing all predictions
        metric_objects = {}
        if isinstance(metrics, str):
            metrics = [metrics]
        for per_metric_name in metrics:
            metric_objects[per_metric_name] = get_metric_predict(
                metric_name=per_metric_name.lower(),
                num_classes=self._output_shape
            )

        # Determine whether to stream metrics (no caching => no large tensors kept)
        streaming_metrics = cache_dir is None

        # Process each image with TTA
        import time
        import gc
        start_time = time.time()
        total_images = len(data)

        for idx, row in data.iterrows():
            # ⭐ 跳过已处理的样本（断点续传）
            if idx in processed_indices:
                logger.info(f"⏭️  Skipping image {idx+1}/{total_images} (already processed)")
                continue

            img_start = time.time()

            # Load image and label
            image_path = row['image']
            label_path = row['label'] if 'label' in row else None

            # Load image with proper resource management
            img_pil = Image.open(image_path)
            image = np.array(img_pil)
            img_pil.close()  # Release file handle

            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=2)  # Convert grayscale to RGB

            # Load label if exists
            label = None
            gt_box = None
            if label_path:
                label_pil = Image.open(label_path)
                label = np.array(label_pil)
                label_pil.close()  # Release file handle

                if len(label.shape) == 3:
                    label = label[:, :, 0]  # Take first channel if RGB

                # Binarize label: convert [0, 255] to [0, 1]
                # This matches the preprocessing in semantic_seg_metrics.py
                if self._output_shape == 1:  # Binary segmentation
                    label = (label > 128).astype(np.int64)

                if not streaming_metrics:
                    all_labels.append(torch.from_numpy(label))

                # Compute GT bounding box (single box per sample, original coords)
                ys, xs = np.where(label > 0)
                if xs.size > 0 and ys.size > 0:
                    gt_box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

            # If predicted boxes are provided in DataFrame, override gt_box
            if {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}.issubset(row.index):
                vals = [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]]
                if not any([v is None or (isinstance(v, float) and np.isnan(v)) for v in vals]):
                    gt_box = np.array(vals, dtype=np.float32)

            # Predict with TTA
            if self._output_shape == 1:
                # Binary segmentation
                pred_mask, pred_prob = self._tta_predictor.predict_with_tta(
                    image, predict_fn, return_probs=True, box_prompt=gt_box, box_ref_shape=image.shape[:2]
                )
                pred_tensor = torch.from_numpy(pred_prob)
                if not streaming_metrics:
                    # Store only when caching/resume is enabled
                    all_preds.append(pred_tensor)
                # Free memory
                del pred_mask, pred_prob
            else:
                # Multi-class segmentation
                _, pred_prob = self._tta_predictor.predict_with_tta(
                    image, predict_fn, return_probs=True, box_prompt=gt_box, box_ref_shape=image.shape[:2]
                )
                pred_tensor = torch.from_numpy(pred_prob)
                if not streaming_metrics:
                    all_preds.append(pred_tensor)
                # Free memory
                del pred_prob

            # Stream metric updates to avoid holding all predictions in memory
            if streaming_metrics:
                y_p = pred_tensor.float()
                if label is None:
                    raise ValueError("Labels are required for evaluation")
                y_t = torch.from_numpy(label)
                y_t = y_t.long() if y_t.dtype != torch.long else y_t

                # Resize prediction to match label size if needed
                if y_p.shape[-2:] != y_t.shape[-2:]:
                    target_size = y_t.shape[-2:]
                    y_p = F.interpolate(
                        y_p.unsqueeze(0),
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)

                for metric_name, metric_obj in metric_objects.items():
                    metric_obj.update(y_p.unsqueeze(0), y_t.unsqueeze(0))

            # Free image and label memory
            del image
            if label is not None:
                del label

            # ⭐ 标记为已处理（断点续传）
            processed_indices.add(idx)

            img_time = time.time() - img_start

            if idx == 0:
                logger.info(f"First image processed in {img_time:.2f}s (includes warmup)")

            # ⭐ 定期保存缓存（每 10 张图或最后一张）
            num_processed = len(processed_indices)
            if (not streaming_metrics) and cache_file is not None and (num_processed % 10 == 0 or num_processed == total_images):
                try:
                    logger.info(f"💾 Saving checkpoint ({num_processed}/{total_images} images)")
                    with open(cache_file, 'wb') as f:
                        pickle.dump({
                            'preds': all_preds,
                            'labels': all_labels,
                            'indices': list(processed_indices)
                        }, f)

                    # Save processed indices as text file (for easy inspection)
                    with open(processed_indices_file, 'w') as f:
                        f.write('\n'.join(map(str, sorted(processed_indices))))

                    logger.info(f"✅ Checkpoint saved successfully")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to save checkpoint: {e}")

            # More aggressive garbage collection to prevent memory leaks
            if (idx + 1) % 5 == 0:  # Every 5 images instead of 10
                gc.collect()  # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear GPU cache

            # Progress logging
            if (idx + 1) % 10 == 0 or num_processed == total_images:
                elapsed = time.time() - start_time
                avg_time = elapsed / num_processed if num_processed > 0 else 0
                remaining = total_images - num_processed
                eta = avg_time * remaining if remaining > 0 else 0
                logger.info(
                    f"Processed {num_processed}/{total_images} images with TTA "
                    f"(avg: {avg_time:.2f}s/img, ETA: {eta/60:.1f}min)"
                )

                # ⭐ 中间指标：在评估过程中周期性输出当前已处理样本的指标
                # 说明：
                # - 仅在 streaming_metrics=True（即未开启 cache_dir，走低内存流式评估）时启用
                # - 使用 torchmetrics.compute() 不会清空内部状态，因此可以安全多次调用
                if streaming_metrics:
                    try:
                        intermediate_results = {}
                        for metric_name, metric_obj in metric_objects.items():
                            score = metric_obj.compute()
                            # score 可能是 tensor 或数值，这里统一为 float 便于日志展示
                            if hasattr(score, "item"):
                                score = score.item()
                            intermediate_results[metric_name] = float(score)
                        if intermediate_results:
                            logger.info(
                                f"Intermediate TTA metrics after {num_processed} images: "
                                f"{intermediate_results}"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to compute intermediate metrics: {e}")
        
        # Final garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Finished processing all {len(data)} images, computing metrics...")

        results = {}
        if streaming_metrics:
            # Metrics were updated on the fly; just compute
            for metric_name, metric_obj in metric_objects.items():
                score = metric_obj.compute()
                results[metric_name] = score.item()
        else:
            # Check that we have predictions and labels
            if len(all_labels) == 0:
                raise ValueError("Labels are required for evaluation")
            
            if len(all_preds) != len(all_labels):
                raise ValueError(f"Number of predictions ({len(all_preds)}) != number of labels ({len(all_labels)})")

            # Process each image individually (can't stack due to different sizes)
            for y_p, y_t in zip(all_preds, all_labels):
                # Ensure predictions and labels are on CPU and have matching sizes
                y_p = y_p.float()
                y_t = y_t.long() if y_t.dtype != torch.long else y_t
                
                # Resize prediction to match label size if needed
                if y_p.shape[-2:] != y_t.shape[-2:]:
                    target_size = y_t.shape[-2:]
                    y_p = F.interpolate(
                        y_p.unsqueeze(0),  # Add batch dim
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)  # Remove batch dim
                
                # Update all metrics with this image
                for metric_name, metric_obj in metric_objects.items():
                    metric_obj.update(y_p.unsqueeze(0), y_t.unsqueeze(0))

            # Compute final scores
            for metric_name, metric_obj in metric_objects.items():
                score = metric_obj.compute()
                results[metric_name] = score.item()

        logger.info(f"TTA evaluation completed: {results}")
        
        # ⭐ 清理缓存文件（评估成功完成后）
        if cache_file is not None and os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                if processed_indices_file and os.path.exists(processed_indices_file):
                    os.remove(processed_indices_file)
                logger.info(f"🗑️  Cleaned up cache files (evaluation completed successfully)")
            except Exception as e:
                logger.warning(f"⚠️  Failed to clean up cache: {e}")
        
        # Free memory
        del all_preds, all_labels, metric_objects
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if return_pred:
            logger.warning("return_pred=True not fully supported with TTA (memory optimization)")
            return results, []
        else:
            return results

    def get_litmodule_per_run(
        self,
        model=None,
        model_postprocess_fn=None,
        peft_param_names=None,
        optim_kwargs=None,
        distillation_kwargs=None,
        is_train=True,
    ):
        gspo_trainer = None
        if is_train:
            gspo_trainer = self._maybe_create_gspo_trainer()
        if is_train:
            return SemanticSegmentationLitModule(
                model=model,
                model_postprocess_fn=model_postprocess_fn,
                trainable_param_names=peft_param_names,
                gspo_trainer=gspo_trainer,
                gspo_adapter_cfg=self._get_gspo_adapter_cfg() if is_train else None,
                train_box_prompt_cfg=self._train_box_prompt_cfg,
                train_bbox_predictor=self._train_bbox_predictor,
                **optim_kwargs,
            )
        else:
            return SemanticSegmentationLitModule(
                model=self._model,
                model_postprocess_fn=self._model_postprocess_fn,
                train_box_prompt_cfg=self._train_box_prompt_cfg,
                train_bbox_predictor=self._train_bbox_predictor,
                **optim_kwargs,
            )

    def _maybe_create_gspo_trainer(self):
        """
        Auto-create GSPO trainer if the config enables GSPO.
        The trainer implementation lives in examples/automm/Conv-LoRA/gspo_trainer.py.
        """
        if self._config is None:
            return None

    def _get_gspo_adapter_cfg(self) -> Dict:
        """
        GSPO-Adapter extension config (encoder adapters only).

        Defaults are chosen to be fully backward-compatible:
        - train_encoder_adapter: False => encoder adapters are frozen during GSPO-active steps
        - noise stds default to 0.0 => no extra exploration injected
        """
        cfg = {
            "train_encoder_adapter": False,
            "encoder_adapter_noise_std": 0.0,
            "encoder_adapter_gate_noise_std": 0.0,
        }
        if self._config is None:
            return cfg
        gspo_cfg = getattr(self._config.optim, "gspo", None)
        if gspo_cfg is None:
            return cfg
        cfg["train_encoder_adapter"] = bool(getattr(gspo_cfg, "train_encoder_adapter", False))
        cfg["encoder_adapter_noise_std"] = float(getattr(gspo_cfg, "encoder_adapter_noise_std", 0.0))
        cfg["encoder_adapter_gate_noise_std"] = float(getattr(gspo_cfg, "encoder_adapter_gate_noise_std", 0.0))
        return cfg

        lora_cfg = getattr(self._config.optim, "lora", None)
        if not lora_cfg or not getattr(lora_cfg, "gspo_enabled", False):
            return None

        gspo_cfg = getattr(self._config.optim, "gspo", None)
        lambda_smooth = getattr(gspo_cfg, "lambda_smooth", 0.1) if gspo_cfg else 0.1
        lambda_boundary = getattr(gspo_cfg, "lambda_boundary", 0.3) if gspo_cfg else 0.3
        w_boundary = getattr(gspo_cfg, "w_boundary", 0.3) if gspo_cfg else 0.3
        w_smooth = getattr(gspo_cfg, "w_smooth", 0.1) if gspo_cfg else 0.1
        w_thin = getattr(gspo_cfg, "w_thin", 0.05) if gspo_cfg else 0.05

        # Dynamically load GSPOConvLoRATrainer from examples path.
        examples_root = Path(__file__).resolve().parents[3] / "examples" / "automm" / "Conv-LoRA" / "gspo_trainer.py"
        if not examples_root.exists():
            logger.warning("GSPO is enabled but gspo_trainer.py not found at %s; GSPO will be skipped.", examples_root)
            return None

        spec = importlib.util.spec_from_file_location("autogluon_multimodal.gspo_trainer", examples_root)
        if spec is None or spec.loader is None:
            logger.warning("Failed to load GSPO trainer spec from %s; GSPO will be skipped.", examples_root)
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        GSPOConvLoRATrainer = getattr(module, "GSPOConvLoRATrainer", None)
        if GSPOConvLoRATrainer is None:
            logger.warning("GSPOConvLoRATrainer not found in %s; GSPO will be skipped.", examples_root)
            return None

        try:
            return GSPOConvLoRATrainer(
                predictor=None,  # Not required by current implementation
                group_size=getattr(lora_cfg, "gspo_group_size", 3),
                warmup_epochs=getattr(lora_cfg, "gspo_warmup_epochs", 5),
                contrastive_weight=getattr(lora_cfg, "gspo_contrastive_weight", 0.1),
                quality_metric="iou",
                advantage_temperature=5.0,
                lambda_smooth=lambda_smooth,
                lambda_boundary=lambda_boundary,
                w_boundary=w_boundary,
                w_smooth=w_smooth,
                w_thin=w_thin,
            )
        except Exception as e:
            logger.warning("Failed to instantiate GSPOConvLoRATrainer: %s; GSPO will be skipped.", e)
            return None

    def on_predict_start(self, data: pd.DataFrame):
        data = self.data_to_df(data=data)
        if self._output_shape is None:  # for zero-shot evaluation/prediction
            self._output_shape = self.get_semantic_segmentation_class_num(data)
        self.ensure_predict_ready()
        return data
    
    def enable_tta(
        self,
        scales: List[float] = [0.75, 1.0, 1.25],
        flips: List[str] = ["none", "horizontal"],
        rotations: List[float] = [0],
        fusion_method: str = "mean",
        scale_weights: Optional[Dict[float, float]] = None,
        threshold: float = 0.5,
        min_area_ratio: float = 0.001,
        use_morphology: bool = False,
        resize_method: str = "bilinear",
        cache_dir: Optional[str] = None,
        resume_from_cache: bool = True,
        box_prompt_mode: str = "off",
    ):
        """
        Enable Test-Time Augmentation (TTA) for inference.
        
        Parameters
        ----------
        scales : List[float]
            List of scale factors for multi-scale testing.
            Recommended: [0.75, 1.0, 1.25] for fast (6 inferences).
            Or [0.5, 0.75, 1.0, 1.25, 1.5] for better results.
        flips : List[str]
            List of flip types: "none", "horizontal", "vertical".
            Recommended: ["none", "horizontal"] for most cases.
        rotations : List[float]
            List of rotation angles in degrees.
            Recommended: [0] for fast, or [-10, 0, 10] for better (18 inferences).
        fusion_method : str
            Method to fuse predictions: "mean" or "weighted_mean".
        scale_weights : Optional[Dict[float, float]]
            Weights for each scale when using weighted_mean.
        threshold : float
            Threshold for binary segmentation (can be tuned on validation set).
        min_area_ratio : float
            Remove connected components with area < min_area_ratio * image_area.
        use_morphology : bool
            Whether to apply morphological closing for smoothing.
        resize_method : str
            Resize interpolation method: "bilinear" or "bicubic".
            Recommended: "bilinear" for speed, "bicubic" for quality.
        cache_dir : Optional[str]
            Directory to cache TTA predictions for resume (default: None, no caching).
            Useful for long-running evaluations that might be interrupted.
        resume_from_cache : bool
            Whether to resume from cache if exists (default: True).
        
        Examples
        --------
        >>> # Basic TTA (6 inferences)
        >>> predictor.enable_tta()
        
        >>> # Advanced TTA (18 inferences)
        >>> predictor.enable_tta(
        ...     scales=[0.75, 1.0, 1.25],
        ...     flips=["none", "horizontal"],
        ...     rotations=[-10, 0, 10],
        ...     fusion_method="weighted_mean",
        ...     resize_method="bicubic"
        ... )
        """
        self._tta_predictor = TTAPredictor(
            scales=scales,
            flips=flips,
            rotations=rotations,
            fusion_method=fusion_method,
            scale_weights=scale_weights,
            threshold=threshold,
            min_area_ratio=min_area_ratio,
            use_morphology=use_morphology,
            resize_method=resize_method,
        )
        
        # Store cache configuration
        self._tta_cache_dir = cache_dir
        self._tta_resume_from_cache = resume_from_cache
        self._tta_box_prompt_mode = box_prompt_mode
        
        logger.info(f"TTA enabled with {len(self._tta_predictor.transforms)} augmentations")
        if cache_dir:
            logger.info(f"TTA caching enabled: {cache_dir} (resume={resume_from_cache})")
    
    def disable_tta(self):
        """Disable Test-Time Augmentation."""
        self._tta_predictor = None
        logger.info("TTA disabled")

    def evaluate(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = False,
        **kwargs,
    ):
        """
        Evaluate model on a test dataset.

        Parameters
        ----------
        data
            A dataframe, containing the same columns as the training data.
            Or a str, that is a path of the annotation file for detection.
        metrics
            A list of metric names to report.
            If None, we only return the score for the stored `_eval_metric_name`.
        return_pred
            Whether to return the prediction result of each row.
        realtime
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        A dictionary with the metric names and their corresponding scores.
        Optionally return a dataframe of prediction results.
        """
        data = self.on_predict_start(data)
        return self.evaluate_semantic_segmentation(data, metrics, realtime)

    def predict(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        realtime: Optional[bool] = False,
        save_results: Optional[bool] = None,
        **kwargs,
    ):
        """
        Predict values for the label column of new data.

        Parameters
        ----------
        data
            The data to make predictions for. Should contain same column names as training data and
            follow same format (except for the `label` column).
        realtime
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.
        save_results
            Whether to save the prediction results.

        Returns
        -------
        Array of predictions, one corresponding to each row in given dataset.
        When save_results is True, the output is a pandas dataframe containing the path of the predicted mask file for each input image.
        Otherwise, the output will have shape (#samples, height, width).
        """
        data = self.on_predict_start(data)
        if self._output_shape == 1:
            ret_type = LOGITS
        else:
            ret_type = SEMANTIC_MASK

        outputs = self.predict_per_run(
            data=data,
            realtime=realtime,
            requires_label=False,
        )

        logits = self.post_process_prediction(data, outputs, ret_type)

        pred = []
        for logit in logits:
            logit = logit.numpy()
            if ret_type == SEMANTIC_MASK:
                pred.append(logit.argmax(axis=1))
            else:
                pred.append((logit > 0.5).squeeze(axis=1))

        if save_results:
            self._save_path = setup_save_path(
                old_save_path=self._save_path,
                warn_if_exist=False,
            )
            pred = self.save_segmentation_result(
                pred=pred,
                data=data,
                result_path=self._save_path,
            )

        return pred

        # if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
        #     # TODO
        #     pred = self._as_pandas(data=data, to_be_converted=pred)

    def predict_proba(
        self,
        data: Union[pd.DataFrame, dict, list],
        as_pandas: Optional[bool] = None,
        as_multiclass: Optional[bool] = True,
        realtime: Optional[bool] = False,
        **kwargs,
    ):
        """
        Predict probabilities class probabilities rather than class labels.
        This is only for the classification. Calling it for regression will throw an exception.

        Parameters
        ----------
        data
            The data to make predictions for. Should contain same column names as training data and
              follow same format (except for the `label` column).
        as_pandas
            Whether to return the output as a pandas DataFrame(Series) (True) or numpy array (False).
        as_multiclass
            Whether to return the probability of all labels or
            just return the probability of the positive class for binary classification problems.
        realtime
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        Array of predicted class-probabilities, corresponding to each row in the given data.
        The output will always have shape (#samples, #classes, height, width).
        """
        assert (self._output_shape == 1 and as_multiclass == False) or (
            self._output_shape > 1 and as_multiclass == True
        )
        data = self.on_predict_start(data)

        outputs = self.predict_per_run(
            data=data,
            realtime=realtime,
            requires_label=False,
        )

        if as_multiclass:
            ret_type = SEMANTIC_MASK
        else:
            ret_type = LOGITS

        logits = self.post_process_prediction(data, outputs, ret_type)

        prob = []
        for logit in logits:
            logit = logit.numpy()
            if ret_type == SEMANTIC_MASK:
                prob.append(softmax(logit, axis=1))
            else:
                prob.append(logit)

        return prob

    def extract_embedding(
        self,
        data: Union[pd.DataFrame, dict, list],
        as_tensor: Optional[bool] = False,
        as_pandas: Optional[bool] = False,
        realtime: Optional[bool] = False,
        **kwargs,
    ):
        raise NotImplementedError("Semantic segmentation doesn't support calling `extract_embedding` yet.")

    def save_segmentation_result(self, pred: Iterable, data: Union[pd.DataFrame, Dict], result_path: str):
        """
        Saving segmentation results in pd.DataFrame format (per image)

        Parameters
        ----------
        pred
            List containing segmentation results for one image
        data
            Pandas data frame or dict containing the image information to be tested
        result_path
            Path to save result
        Returns
        -------
        The paths of the segmentation results as pandas DataFrame
        """

        def show_mask(mask, ax):
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)

        image_column_name = self.get_image_column_name(data)
        if isinstance(data, dict):
            image_names = data[image_column_name]
        else:
            image_names = data[image_column_name].to_list()
        results = []

        mask_path = os.path.join(result_path, "masks")
        txt_path = os.path.join(result_path, "result.txt")
        os.makedirs(mask_path, exist_ok=True)
        for image_pred, image_name in zip(pred, image_names):
            if self._output_shape == 1:
                mask = Image.fromarray(image_pred.squeeze(axis=0))
                per_mask_path = os.path.join(mask_path, os.path.basename(image_name))
                mask.save(per_mask_path)
            else:
                masks = []
                classes = np.unique(image_pred)
                for class_id in classes:
                    if class_id == 0:  # bg
                        continue
                    masks.append(image_pred == class_id)

                for mask in masks:
                    show_mask(mask, plt.gca())
                mask_name = ""
                for i in os.path.basename(image_name).split(".")[:-1]:
                    mask_name += i
                per_mask_path = os.path.join(mask_path, os.path.basename(image_name))
                plt.axis("off")
                plt.savefig(per_mask_path, bbox_inches="tight", dpi=300, pad_inches=0.0)

            results.append([image_name, per_mask_path])

        result_df = pd.DataFrame(results, columns=["image", "mask"])
        result_df.to_csv(txt_path, index=False)
        return result_df

    def post_process_prediction(self, data, outputs, ret_type):
        """
        Post-process segmentation results to match the size of original input images.

        Parameters
        ----------
        data
            Pandas data frame or dict containing the image information.
        outputs
            A list of segmentation output results.
        ret_type
            What kind of information to extract from model outputs.

        Returns
        -------
        A list of the post-processed segmentation results.
        """
        logits = [ele[ret_type] for ele in outputs]
        image_column_name = self.get_image_column_name(data)
        for idx in range(data.shape[0]):
            ori_image_size = Image.open(data[image_column_name][idx]).size  # width, height
            logits[idx] = F.interpolate(
                logits[idx].float(), (ori_image_size[1], ori_image_size[0]), mode="bilinear", align_corners=False
            )
        return logits

    def get_image_column_name(self, data: pd.DataFrame):
        if self.column_types is None:
            column_names = list(data.columns)
            if self._label_column in column_names:
                column_names.remove(self._label_column)
            assert (
                len(column_names) == 1
            ), f"More than one image columns {column_names} exist in the data. Make sure to provide data with one image column."
            return column_names[0]
        else:
            for k, v in self.column_types.items():
                if v == SEMANTIC_SEGMENTATION_IMG:
                    return k
        return None

```

```worktree-python
*** Begin Patch
*** Update File: multimodal/src/autogluon/multimodal/optim/lit_semantic_seg.py
@@
 class SemanticSegmentationLitModule(LitModule):
-    def __init__(self, *args, gspo_trainer=None, train_box_prompt_cfg=None, train_bbox_predictor=None, **kwargs):
+    def __init__(
+        self,
+        *args,
+        gspo_trainer=None,
+        gspo_adapter_cfg=None,
+        train_box_prompt_cfg=None,
+        train_bbox_predictor=None,
+        **kwargs,
+    ):
         super().__init__(*args, **kwargs)
         self.gspo_trainer = gspo_trainer
+        # GSPO-Adapter extension config (encoder adapters only). When None, keep fully backward-compatible behavior.
+        self.gspo_adapter_cfg = gspo_adapter_cfg or {
+            "train_encoder_adapter": False,
+            "encoder_adapter_noise_std": 0.0,
+            "encoder_adapter_gate_noise_std": 0.0,
+        }
@@
     def _gspo_training_step(self, batch):
@@
-        images = batch[self.model.image_key] if hasattr(self.model, 'image_key') else batch['image']
+        images = batch[self.model.image_key] if hasattr(self.model, "image_key") else batch["image"]
         labels = batch[self.model.label_key]
+
+        # GSPO-Adapter: activate per-step behavior (exploration/noise) and optionally freeze encoder adapters
+        adapter_prev_states = None
+        self._set_encoder_adapters_gspo_mode(
+            active=True,
+            adapter_noise_std=float(self.gspo_adapter_cfg.get("encoder_adapter_noise_std", 0.0)),
+            gate_noise_std=float(self.gspo_adapter_cfg.get("encoder_adapter_gate_noise_std", 0.0)),
+        )
+        if not bool(self.gspo_adapter_cfg.get("train_encoder_adapter", False)):
+            adapter_prev_states = self._set_encoder_adapter_requires_grad(trainable=False)
         
         # Define forward function for GSPO
         def forward_fn(images):
             batch_copy = batch.copy()
             if hasattr(self.model, 'image_key'):
                 batch_copy[self.model.image_key] = images
             else:
                 batch_copy['image'] = images
             output = run_model(self.model, batch_copy)
             
             # Extract predictions and MOE info
             logits = output[self.model.prefix][LOGITS]
             moe_loss = output[self.model.prefix].get(MOE_LOSS, 0)
             
             # Extract selected experts if available
             selected_experts = None
-            if hasattr(output[self.model.prefix], 'selected_experts'):
-                selected_experts = output[self.model.prefix]['selected_experts']
+            if isinstance(output.get(self.model.prefix, None), dict) and "selected_experts" in output[self.model.prefix]:
+                selected_experts = output[self.model.prefix]["selected_experts"]
             
             return logits, moe_loss, selected_experts
@@
-        # Run GSPO group training
-        loss, metrics, selected_experts_groups = self.gspo_trainer.gspo_group_training_step(
-            images=images,
-            masks_gt=labels,
-            forward_fn=forward_fn,
-            loss_fn=loss_fn,
-        )
+        try:
+            # Run GSPO group training
+            loss, metrics, selected_experts_groups = self.gspo_trainer.gspo_group_training_step(
+                images=images,
+                masks_gt=labels,
+                forward_fn=forward_fn,
+                loss_fn=loss_fn,
+            )
+        finally:
+            # Restore adapter grad states and disable GSPO mode
+            if adapter_prev_states is not None:
+                self._restore_requires_grad(adapter_prev_states)
+            self._set_encoder_adapters_gspo_mode(active=False)
@@
         return loss, metrics, selected_experts_groups
+
+    def _iter_encoder_adapter_named_modules(self):
+        """
+        Yield (name, module) for encoder adapters only (exclude decoder adapters).
+        We match by module name path to avoid touching decoder adapter_queries.
+        """
+        for name, module in self.model.named_modules():
+            if "vision_encoder.layers" in name and name.endswith(".adapter"):
+                yield name, module
+
+    def _set_encoder_adapters_gspo_mode(self, active: bool, adapter_noise_std: float = 0.0, gate_noise_std: float = 0.0):
+        for _, m in self._iter_encoder_adapter_named_modules():
+            if hasattr(m, "set_gspo_mode"):
+                # GatedAdapterLayer.set_gspo_mode(active, gate_noise_std, adapter_noise_std)
+                try:
+                    m.set_gspo_mode(active=active, gate_noise_std=gate_noise_std, adapter_noise_std=adapter_noise_std)
+                except TypeError:
+                    # AdapterLayer.set_gspo_mode(active, noise_std)
+                    m.set_gspo_mode(active=active, noise_std=adapter_noise_std)
+
+    def _set_encoder_adapter_requires_grad(self, trainable: bool):
+        """
+        Set requires_grad for encoder adapter parameters. Returns a dict for restoration.
+        """
+        prev = {}
+        for name, m in self._iter_encoder_adapter_named_modules():
+            for p_name, p in m.named_parameters(recurse=True):
+                key = f"{name}.{p_name}"
+                prev[key] = p.requires_grad
+                p.requires_grad = bool(trainable)
+        return prev
+
+    def _restore_requires_grad(self, prev: dict):
+        for name, m in self._iter_encoder_adapter_named_modules():
+            for p_name, p in m.named_parameters(recurse=True):
+                key = f"{name}.{p_name}"
+                if key in prev:
+                    p.requires_grad = prev[key]
*** End Patch
```

```worktree-python
import logging
from typing import Callable, Dict, Optional

import torch
import torchmetrics
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss

from ..constants import CLASS_LOGITS, LOGITS, MOE_LOSS, SEMANTIC_MASK, WEIGHT
from ..models.utils import run_model
from .lit_module import LitModule
from .metrics.semantic_seg_metrics import Multiclass_IoU

logger = logging.getLogger(__name__)


class SemanticSegmentationLitModule(LitModule):
    """
    Control the loops for training, evaluation, and prediction. This module is independent of
    the model definition. This class inherits from the Pytorch Lightning's LightningModule:
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    
    GSPO enhancement:
    - Supports group-level training with quality-aware feedback
    - Integrates contrastive loss for better expert selection
    """
    
    def __init__(
        self,
        *args,
        gspo_trainer=None,
        gspo_adapter_cfg=None,
        train_box_prompt_cfg=None,
        train_bbox_predictor=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gspo_trainer = gspo_trainer
        # GSPO-Adapter extension config (encoder adapters only).
        # Defaults keep backward compatibility: encoder adapters are NOT trained during GSPO-active steps.
        self.gspo_adapter_cfg = gspo_adapter_cfg or {
            "train_encoder_adapter": False,
            "encoder_adapter_noise_std": 0.0,
            "encoder_adapter_gate_noise_std": 0.0,
        }
        # Training-time box prompt config (dict with keys: mode, p_no, p_gt, p_noisy, noise_frac, bbox_predictor)
        self.train_box_prompt_cfg = train_box_prompt_cfg or {
            "mode": "off",
            "p_no": 0.0,
            "p_gt": 0.0,
            "p_noisy": 0.0,
            "noise_frac": 0.12,
        }
        # BBoxPromptPredictor instance for "predict" mode
        self.train_bbox_predictor = train_bbox_predictor

    def _compute_loss(self, output: Dict, label: torch.Tensor, **kwargs):
        loss = 0
        for _, per_output in output.items():
            weight = per_output[WEIGHT] if WEIGHT in per_output else 1
            if isinstance(self.loss_func, Mask2FormerLoss):
                mask_labels = [mask_labels.to(per_output[LOGITS]) for mask_labels in kwargs["mask_labels"]]
                dict_loss = self.loss_func(
                    masks_queries_logits=per_output[LOGITS],  # bs, num_mask_tokens, height, width
                    class_queries_logits=per_output[CLASS_LOGITS],  # bs, num_mask_tokens, num_classes
                    mask_labels=mask_labels,
                    class_labels=kwargs["class_labels"],
                )
                for v in dict_loss.values():
                    loss += v
            else:
                loss += (
                    self.loss_func(
                        input=per_output[LOGITS],
                        target=label,
                    )
                    * weight
                )
            # MoE loss
            if MOE_LOSS in per_output:
                loss += per_output[MOE_LOSS]

        return loss

    def _compute_metric_score(
        self,
        metric: torchmetrics.Metric,
        custom_metric_func: Callable,
        logits: torch.Tensor,
        label: torch.Tensor,
        **kwargs,
    ):
        if isinstance(metric, Multiclass_IoU):
            metric.update(kwargs["semantic_masks"], label)
        else:
            metric.update(logits.float(), label)

    def _shared_step(
        self,
        batch: Dict,
    ):
        label = batch[self.model.label_key]

        # -------- Train-time box prompt injection (configurable) --------
        # 控制项：self.train_box_prompt_cfg = {
        #   mode: off / gt / noisy / mix / predict,
        #   p_no, p_gt, p_noisy: 仅在 mode=mix 时生效并会归一化,
        #   noise_frac: 噪声扰动比例（相对 bbox 宽高）
        # }
        # mode=predict: 使用 self.train_bbox_predictor 预测框作为 box prompt
        if self.training:
            cfg = self.train_box_prompt_cfg or {}
            mode = cfg.get("mode", "off")
            box_mode = None
            if mode != "off":
                p_no = max(float(cfg.get("p_no", 0.0)), 0.0)
                p_gt = max(float(cfg.get("p_gt", 0.0)), 0.0)
                p_noisy = max(float(cfg.get("p_noisy", 0.0)), 0.0)
                noise_frac = float(cfg.get("noise_frac", 0.12))

                if mode == "mix":
                    total = p_no + p_gt + p_noisy
                    if total > 0:
                        p_no, p_gt, p_noisy = [x / total for x in (p_no, p_gt, p_noisy)]
                    r = torch.rand(1).item()
                    if r < p_no:
                        box_mode = "off"
                    elif r < p_no + p_gt:
                        box_mode = "gt"
                    else:
                        box_mode = "noisy"
                elif mode == "predict":
                    box_mode = "predict"
                else:
                    box_mode = mode

            def _compute_gt_boxes(mask: torch.Tensor) -> torch.Tensor:
                """mask: (B,H,W) or (B,1,H,W) int/long -> boxes (B,4) in pixel coords."""
                if mask.dim() == 4 and mask.shape[1] == 1:
                    mask = mask[:, 0, :, :]
                b, h, w = mask.shape
                boxes = torch.zeros((b, 4), device=mask.device, dtype=torch.float32)
                for i in range(b):
                    ys, xs = torch.nonzero(mask[i] > 0, as_tuple=True)
                    if xs.numel() == 0:
                        continue
                    x1, x2 = xs.min(), xs.max()
                    y1, y2 = ys.min(), ys.max()
                    boxes[i] = torch.tensor([x1, y1, x2, y2], device=mask.device, dtype=torch.float32)
                return boxes

            def _jitter_boxes(boxes: torch.Tensor, h: int, w: int, frac: float = 0.12) -> torch.Tensor:
                """Uniform jitter relative to box size."""
                jittered = boxes.clone()
                for i in range(jittered.shape[0]):
                    x1, y1, x2, y2 = jittered[i]
                    if x2 <= x1 or y2 <= y1:
                        continue
                    bw = (x2 - x1).clamp(min=1.0)
                    bh = (y2 - y1).clamp(min=1.0)
                    dx = bw * frac
                    dy = bh * frac
                    rx1 = (torch.rand(1, device=boxes.device) * 2 - 1) * dx
                    ry1 = (torch.rand(1, device=boxes.device) * 2 - 1) * dy
                    rx2 = (torch.rand(1, device=boxes.device) * 2 - 1) * dx
                    ry2 = (torch.rand(1, device=boxes.device) * 2 - 1) * dy
                    x1n = torch.clamp(x1 + rx1, 0, w - 1)
                    y1n = torch.clamp(y1 + ry1, 0, h - 1)
                    x2n = torch.clamp(x2 + rx2, 0, w - 1)
                    y2n = torch.clamp(y2 + ry2, 0, h - 1)
                    jittered[i] = torch.stack(
                        [torch.min(x1n, x2n), torch.min(y1n, y2n), torch.max(x1n, x2n), torch.max(y1n, y2n)]
                    ).squeeze()
                return jittered

            def _predict_boxes_from_images(image_paths: list, target_h: int, target_w: int) -> torch.Tensor:
                """Use BBoxPromptPredictor to predict boxes from image paths.
                
                Args:
                    image_paths: List of image file paths
                    target_h: Target image height (model input size)
                    target_w: Target image width (model input size)
                
                Returns:
                    boxes: (B, 4) tensor in pixel coords, scaled to target_h x target_w
                """
                import numpy as np
                from PIL import Image
                
                if self.train_bbox_predictor is None:
                    raise ValueError("train_bbox_predictor is required for mode='predict'")
                
                b = len(image_paths)
                boxes = torch.zeros((b, 4), device=label.device, dtype=torch.float32)
                
                for i, img_path in enumerate(image_paths):
                    if not img_path:
                        continue
                    try:
                        # predict() returns bbox in original image pixels: (x1, y1, x2, y2)
                        bbox_px = self.train_bbox_predictor.predict(img_path)
                        if bbox_px is None:
                            continue
                        
                        # Get original image size to scale bbox to target size
                        with Image.open(img_path) as img:
                            orig_w, orig_h = img.size
                        
                        # Scale bbox from original coords to target coords
                        scale_x = target_w / orig_w
                        scale_y = target_h / orig_h
                        x1 = bbox_px[0] * scale_x
                        y1 = bbox_px[1] * scale_y
                        x2 = bbox_px[2] * scale_x
                        y2 = bbox_px[3] * scale_y
                        
                        # Clamp to valid range
                        x1 = max(0, min(x1, target_w - 1))
                        y1 = max(0, min(y1, target_h - 1))
                        x2 = max(0, min(x2, target_w - 1))
                        y2 = max(0, min(y2, target_h - 1))
                        
                        # Ensure x1 < x2 and y1 < y2
                        x1, x2 = min(x1, x2), max(x1, x2)
                        y1, y2 = min(y1, y2), max(y1, y2)
                        
                        boxes[i] = torch.tensor([x1, y1, x2, y2], device=label.device, dtype=torch.float32)
                    except Exception as e:
                        logger.warning(f"Failed to predict bbox for {img_path}: {e}")
                        continue
                
                return boxes

            if box_mode in ["gt", "noisy"]:
                gt_boxes = _compute_gt_boxes(label)
                h, w = label.shape[-2], label.shape[-1]
                if box_mode == "noisy":
                    gt_boxes = _jitter_boxes(gt_boxes, h=h, w=w, frac=noise_frac)
                if not (gt_boxes.sum(dim=1) == 0).all():
                    batch[self.model.box_key] = gt_boxes.unsqueeze(1)  # (B,1,4)
            elif box_mode == "predict":
                # Get image paths from batch
                image_path_key = self.model.image_path_key
                image_paths = batch.get(image_path_key, [])
                if image_paths:
                    h, w = label.shape[-2], label.shape[-1]
                    pred_boxes = _predict_boxes_from_images(image_paths, target_h=h, target_w=w)
                    if not (pred_boxes.sum(dim=1) == 0).all():
                        batch[self.model.box_key] = pred_boxes.unsqueeze(1)  # (B,1,4)
        # prepare_targets
        output = run_model(self.model, batch)
        if isinstance(self.loss_func, Mask2FormerLoss):
            loss = self._compute_loss(
                output=output,
                label=label,
                mask_labels=batch[self.model.mask_label_key],
                class_labels=batch[self.model.class_label_key],
            )
        else:
            loss = self._compute_loss(
                output=output,
                label=label,
            )

        return output, loss

    def validation_step(self, batch, batch_idx, **kwargs):
        """
        Per validation step. This function is registered by LightningModule.
        Refer to https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-loop

        Parameters
        ----------
        batch
            A dictionary containing the mini-batch data, including both input data and
            ground-truth labels. The mini-batch data are passed to each individual model,
            which indexes its required input data by keys with its model prefix. The
            ground-truth labels are used here to compute the validation loss and metric.
            The validation metric is used for top k model selection and early stopping.
        batch_idx
            Index of mini-batch.
        """
        output, loss = self._shared_step(batch)
        if self.model_postprocess_fn:
            output = self.model_postprocess_fn(output)
        # By default, on_step=False and on_epoch=True
        self.log("val_loss", loss)
        if isinstance(self.loss_func, Mask2FormerLoss):
            self._compute_metric_score(
                metric=self.validation_metric,
                custom_metric_func=self.custom_metric_func,
                logits=output[self.model.prefix][LOGITS],
                label=batch[self.model.label_key],
                semantic_masks=output[self.model.prefix][SEMANTIC_MASK],
            )
        else:
            self._compute_metric_score(
                metric=self.validation_metric,
                custom_metric_func=self.custom_metric_func,
                logits=output[self.model.prefix][LOGITS],
                label=batch[self.model.label_key],
            )

        self.log(
            self.validation_metric_name,
            self.validation_metric,
            on_step=False,
            on_epoch=True,
        )
    
    def training_step(self, batch, batch_idx):
        """
        Per training step with GSPO enhancement.
        
        If GSPO is enabled and warmed up, use group-level optimization.
        Otherwise, fall back to standard training.
        
        Parameters
        ----------
        batch
            A dictionary containing the mini-batch data
        batch_idx
            Index of mini-batch
            
        Returns
        -------
        Average loss of the mini-batch data
        """
        # Check if GSPO should be used
        use_gspo = (
            self.gspo_trainer is not None and 
            self.gspo_trainer.is_gspo_active(self.current_epoch)
        )
        
        if use_gspo:
            # GSPO-enhanced training
            loss, metrics, selected_experts = self._gspo_training_step(batch)
            
            # Log GSPO-specific metrics
            for key, value in metrics.items():
                self.log(f"train_{key}", value, on_step=True, on_epoch=True)
        else:
            # Standard training (same as parent class)
            output, loss = self._shared_step(batch)
            selected_experts = None
        
        # Handle manual optimization if needed
        if not self.automatic_optimization:
            if self.hparams.use_aug_optim:
                optimizer, aug_optimizer = self.optimizers()
            else:
                optimizer = self.optimizers()
                aug_optimizer = None

            lr_scheduler = self.lr_schedulers()
            loss = loss / self.hparams.accumulate_grad_batches
            self.manual_backward(loss)

            if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0 or self.trainer.is_last_batch:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                if aug_optimizer is not None:
                    aug_optimizer.step()
                    aug_optimizer.zero_grad()
        
        self.log("train_loss", loss)
        return loss
    
    def _gspo_training_step(self, batch):
        """
        GSPO group-level training step.
        
        This method generates multiple predictions per image and uses
        group-level advantage functions to weight the losses.
        """
        images = batch[self.model.image_key] if hasattr(self.model, 'image_key') else batch['image']
        labels = batch[self.model.label_key]
        
        # Define forward function for GSPO
        def forward_fn(images):
            batch_copy = batch.copy()
            if hasattr(self.model, 'image_key'):
                batch_copy[self.model.image_key] = images
            else:
                batch_copy['image'] = images
            output = run_model(self.model, batch_copy)
            
            # Extract predictions and MOE info
            logits = output[self.model.prefix][LOGITS]
            moe_loss = output[self.model.prefix].get(MOE_LOSS, 0)
            
            # Extract selected experts if available
            selected_experts = None
            if hasattr(output[self.model.prefix], 'selected_experts'):
                selected_experts = output[self.model.prefix]['selected_experts']
            
            return logits, moe_loss, selected_experts
        
        # Define loss function for GSPO
        def loss_fn(predictions, targets):
            if isinstance(self.loss_func, Mask2FormerLoss):
                # Handle Mask2Former case
                return self.loss_func(predictions, targets)
            else:
                return self.loss_func(input=predictions, target=targets)
        
        # Run GSPO group training
        loss, metrics, selected_experts_groups = self.gspo_trainer.gspo_group_training_step(
            images=images,
            masks_gt=labels,
            forward_fn=forward_fn,
            loss_fn=loss_fn,
        )
        
        # Update expert quality feedback
        if selected_experts_groups:
            quality_scores = [
                self.gspo_trainer.compute_segmentation_quality(
                    pred, labels, self.gspo_trainer.quality_metric
                )
                for pred in [forward_fn(images)[0] for _ in range(self.gspo_trainer.group_size)]
            ]
            self.gspo_trainer.update_expert_feedback(
                selected_experts_groups, quality_scores, self.model
            )
        
        return loss, metrics, selected_experts_groups

```

```worktree-python
import logging
from typing import Callable, Dict, Optional

import torch
import torchmetrics
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss

from ..constants import CLASS_LOGITS, LOGITS, MOE_LOSS, SEMANTIC_MASK, WEIGHT
from ..models.utils import run_model
from .lit_module import LitModule
from .metrics.semantic_seg_metrics import Multiclass_IoU

logger = logging.getLogger(__name__)


class SemanticSegmentationLitModule(LitModule):
    """
    Control the loops for training, evaluation, and prediction. This module is independent of
    the model definition. This class inherits from the Pytorch Lightning's LightningModule:
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    
    GSPO enhancement:
    - Supports group-level training with quality-aware feedback
    - Integrates contrastive loss for better expert selection
    """
    
    def __init__(
        self,
        *args,
        gspo_trainer=None,
        gspo_adapter_cfg=None,
        train_box_prompt_cfg=None,
        train_bbox_predictor=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gspo_trainer = gspo_trainer
        # GSPO-Adapter extension config (encoder adapters only).
        # Defaults keep backward compatibility: encoder adapters are NOT trained during GSPO-active steps.
        self.gspo_adapter_cfg = gspo_adapter_cfg or {
            "train_encoder_adapter": False,
            "encoder_adapter_noise_std": 0.0,
            "encoder_adapter_gate_noise_std": 0.0,
        }
        # Training-time box prompt config (dict with keys: mode, p_no, p_gt, p_noisy, noise_frac, bbox_predictor)
        self.train_box_prompt_cfg = train_box_prompt_cfg or {
            "mode": "off",
            "p_no": 0.0,
            "p_gt": 0.0,
            "p_noisy": 0.0,
            "noise_frac": 0.12,
        }
        # BBoxPromptPredictor instance for "predict" mode
        self.train_bbox_predictor = train_bbox_predictor

    def _compute_loss(self, output: Dict, label: torch.Tensor, **kwargs):
        loss = 0
        for _, per_output in output.items():
            weight = per_output[WEIGHT] if WEIGHT in per_output else 1
            if isinstance(self.loss_func, Mask2FormerLoss):
                mask_labels = [mask_labels.to(per_output[LOGITS]) for mask_labels in kwargs["mask_labels"]]
                dict_loss = self.loss_func(
                    masks_queries_logits=per_output[LOGITS],  # bs, num_mask_tokens, height, width
                    class_queries_logits=per_output[CLASS_LOGITS],  # bs, num_mask_tokens, num_classes
                    mask_labels=mask_labels,
                    class_labels=kwargs["class_labels"],
                )
                for v in dict_loss.values():
                    loss += v
            else:
                loss += (
                    self.loss_func(
                        input=per_output[LOGITS],
                        target=label,
                    )
                    * weight
                )
            # MoE loss
            if MOE_LOSS in per_output:
                loss += per_output[MOE_LOSS]

        return loss

    def _compute_metric_score(
        self,
        metric: torchmetrics.Metric,
        custom_metric_func: Callable,
        logits: torch.Tensor,
        label: torch.Tensor,
        **kwargs,
    ):
        if isinstance(metric, Multiclass_IoU):
            metric.update(kwargs["semantic_masks"], label)
        else:
            metric.update(logits.float(), label)

    def _shared_step(
        self,
        batch: Dict,
    ):
        label = batch[self.model.label_key]

        # -------- Train-time box prompt injection (configurable) --------
        # 控制项：self.train_box_prompt_cfg = {
        #   mode: off / gt / noisy / mix / predict,
        #   p_no, p_gt, p_noisy: 仅在 mode=mix 时生效并会归一化,
        #   noise_frac: 噪声扰动比例（相对 bbox 宽高）
        # }
        # mode=predict: 使用 self.train_bbox_predictor 预测框作为 box prompt
        if self.training:
            cfg = self.train_box_prompt_cfg or {}
            mode = cfg.get("mode", "off")
            box_mode = None
            if mode != "off":
                p_no = max(float(cfg.get("p_no", 0.0)), 0.0)
                p_gt = max(float(cfg.get("p_gt", 0.0)), 0.0)
                p_noisy = max(float(cfg.get("p_noisy", 0.0)), 0.0)
                noise_frac = float(cfg.get("noise_frac", 0.12))

                if mode == "mix":
                    total = p_no + p_gt + p_noisy
                    if total > 0:
                        p_no, p_gt, p_noisy = [x / total for x in (p_no, p_gt, p_noisy)]
                    r = torch.rand(1).item()
                    if r < p_no:
                        box_mode = "off"
                    elif r < p_no + p_gt:
                        box_mode = "gt"
                    else:
                        box_mode = "noisy"
                elif mode == "predict":
                    box_mode = "predict"
                else:
                    box_mode = mode

            def _compute_gt_boxes(mask: torch.Tensor) -> torch.Tensor:
                """mask: (B,H,W) or (B,1,H,W) int/long -> boxes (B,4) in pixel coords."""
                if mask.dim() == 4 and mask.shape[1] == 1:
                    mask = mask[:, 0, :, :]
                b, h, w = mask.shape
                boxes = torch.zeros((b, 4), device=mask.device, dtype=torch.float32)
                for i in range(b):
                    ys, xs = torch.nonzero(mask[i] > 0, as_tuple=True)
                    if xs.numel() == 0:
                        continue
                    x1, x2 = xs.min(), xs.max()
                    y1, y2 = ys.min(), ys.max()
                    boxes[i] = torch.tensor([x1, y1, x2, y2], device=mask.device, dtype=torch.float32)
                return boxes

            def _jitter_boxes(boxes: torch.Tensor, h: int, w: int, frac: float = 0.12) -> torch.Tensor:
                """Uniform jitter relative to box size."""
                jittered = boxes.clone()
                for i in range(jittered.shape[0]):
                    x1, y1, x2, y2 = jittered[i]
                    if x2 <= x1 or y2 <= y1:
                        continue
                    bw = (x2 - x1).clamp(min=1.0)
                    bh = (y2 - y1).clamp(min=1.0)
                    dx = bw * frac
                    dy = bh * frac
                    rx1 = (torch.rand(1, device=boxes.device) * 2 - 1) * dx
                    ry1 = (torch.rand(1, device=boxes.device) * 2 - 1) * dy
                    rx2 = (torch.rand(1, device=boxes.device) * 2 - 1) * dx
                    ry2 = (torch.rand(1, device=boxes.device) * 2 - 1) * dy
                    x1n = torch.clamp(x1 + rx1, 0, w - 1)
                    y1n = torch.clamp(y1 + ry1, 0, h - 1)
                    x2n = torch.clamp(x2 + rx2, 0, w - 1)
                    y2n = torch.clamp(y2 + ry2, 0, h - 1)
                    jittered[i] = torch.stack(
                        [torch.min(x1n, x2n), torch.min(y1n, y2n), torch.max(x1n, x2n), torch.max(y1n, y2n)]
                    ).squeeze()
                return jittered

            def _predict_boxes_from_images(image_paths: list, target_h: int, target_w: int) -> torch.Tensor:
                """Use BBoxPromptPredictor to predict boxes from image paths.
                
                Args:
                    image_paths: List of image file paths
                    target_h: Target image height (model input size)
                    target_w: Target image width (model input size)
                
                Returns:
                    boxes: (B, 4) tensor in pixel coords, scaled to target_h x target_w
                """
                import numpy as np
                from PIL import Image
                
                if self.train_bbox_predictor is None:
                    raise ValueError("train_bbox_predictor is required for mode='predict'")
                
                b = len(image_paths)
                boxes = torch.zeros((b, 4), device=label.device, dtype=torch.float32)
                
                for i, img_path in enumerate(image_paths):
                    if not img_path:
                        continue
                    try:
                        # predict() returns bbox in original image pixels: (x1, y1, x2, y2)
                        bbox_px = self.train_bbox_predictor.predict(img_path)
                        if bbox_px is None:
                            continue
                        
                        # Get original image size to scale bbox to target size
                        with Image.open(img_path) as img:
                            orig_w, orig_h = img.size
                        
                        # Scale bbox from original coords to target coords
                        scale_x = target_w / orig_w
                        scale_y = target_h / orig_h
                        x1 = bbox_px[0] * scale_x
                        y1 = bbox_px[1] * scale_y
                        x2 = bbox_px[2] * scale_x
                        y2 = bbox_px[3] * scale_y
                        
                        # Clamp to valid range
                        x1 = max(0, min(x1, target_w - 1))
                        y1 = max(0, min(y1, target_h - 1))
                        x2 = max(0, min(x2, target_w - 1))
                        y2 = max(0, min(y2, target_h - 1))
                        
                        # Ensure x1 < x2 and y1 < y2
                        x1, x2 = min(x1, x2), max(x1, x2)
                        y1, y2 = min(y1, y2), max(y1, y2)
                        
                        boxes[i] = torch.tensor([x1, y1, x2, y2], device=label.device, dtype=torch.float32)
                    except Exception as e:
                        logger.warning(f"Failed to predict bbox for {img_path}: {e}")
                        continue
                
                return boxes

            if box_mode in ["gt", "noisy"]:
                gt_boxes = _compute_gt_boxes(label)
                h, w = label.shape[-2], label.shape[-1]
                if box_mode == "noisy":
                    gt_boxes = _jitter_boxes(gt_boxes, h=h, w=w, frac=noise_frac)
                if not (gt_boxes.sum(dim=1) == 0).all():
                    batch[self.model.box_key] = gt_boxes.unsqueeze(1)  # (B,1,4)
            elif box_mode == "predict":
                # Get image paths from batch
                image_path_key = self.model.image_path_key
                image_paths = batch.get(image_path_key, [])
                if image_paths:
                    h, w = label.shape[-2], label.shape[-1]
                    pred_boxes = _predict_boxes_from_images(image_paths, target_h=h, target_w=w)
                    if not (pred_boxes.sum(dim=1) == 0).all():
                        batch[self.model.box_key] = pred_boxes.unsqueeze(1)  # (B,1,4)
        # prepare_targets
        output = run_model(self.model, batch)
        if isinstance(self.loss_func, Mask2FormerLoss):
            loss = self._compute_loss(
                output=output,
                label=label,
                mask_labels=batch[self.model.mask_label_key],
                class_labels=batch[self.model.class_label_key],
            )
        else:
            loss = self._compute_loss(
                output=output,
                label=label,
            )

        return output, loss

    def validation_step(self, batch, batch_idx, **kwargs):
        """
        Per validation step. This function is registered by LightningModule.
        Refer to https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-loop

        Parameters
        ----------
        batch
            A dictionary containing the mini-batch data, including both input data and
            ground-truth labels. The mini-batch data are passed to each individual model,
            which indexes its required input data by keys with its model prefix. The
            ground-truth labels are used here to compute the validation loss and metric.
            The validation metric is used for top k model selection and early stopping.
        batch_idx
            Index of mini-batch.
        """
        output, loss = self._shared_step(batch)
        if self.model_postprocess_fn:
            output = self.model_postprocess_fn(output)
        # By default, on_step=False and on_epoch=True
        self.log("val_loss", loss)
        if isinstance(self.loss_func, Mask2FormerLoss):
            self._compute_metric_score(
                metric=self.validation_metric,
                custom_metric_func=self.custom_metric_func,
                logits=output[self.model.prefix][LOGITS],
                label=batch[self.model.label_key],
                semantic_masks=output[self.model.prefix][SEMANTIC_MASK],
            )
        else:
            self._compute_metric_score(
                metric=self.validation_metric,
                custom_metric_func=self.custom_metric_func,
                logits=output[self.model.prefix][LOGITS],
                label=batch[self.model.label_key],
            )

        self.log(
            self.validation_metric_name,
            self.validation_metric,
            on_step=False,
            on_epoch=True,
        )
    
    def training_step(self, batch, batch_idx):
        """
        Per training step with GSPO enhancement.
        
        If GSPO is enabled and warmed up, use group-level optimization.
        Otherwise, fall back to standard training.
        
        Parameters
        ----------
        batch
            A dictionary containing the mini-batch data
        batch_idx
            Index of mini-batch
            
        Returns
        -------
        Average loss of the mini-batch data
        """
        # Check if GSPO should be used
        use_gspo = (
            self.gspo_trainer is not None and 
            self.gspo_trainer.is_gspo_active(self.current_epoch)
        )
        
        if use_gspo:
            # GSPO-enhanced training
            loss, metrics, selected_experts = self._gspo_training_step(batch)
            
            # Log GSPO-specific metrics
            for key, value in metrics.items():
                self.log(f"train_{key}", value, on_step=True, on_epoch=True)
        else:
            # Standard training (same as parent class)
            output, loss = self._shared_step(batch)
            selected_experts = None
        
        # Handle manual optimization if needed
        if not self.automatic_optimization:
            if self.hparams.use_aug_optim:
                optimizer, aug_optimizer = self.optimizers()
            else:
                optimizer = self.optimizers()
                aug_optimizer = None

            lr_scheduler = self.lr_schedulers()
            loss = loss / self.hparams.accumulate_grad_batches
            self.manual_backward(loss)

            if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0 or self.trainer.is_last_batch:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                if aug_optimizer is not None:
                    aug_optimizer.step()
                    aug_optimizer.zero_grad()
        
        self.log("train_loss", loss)
        return loss
    
    def _gspo_training_step(self, batch):
        """
        GSPO group-level training step.
        
        This method generates multiple predictions per image and uses
        group-level advantage functions to weight the losses.
        """
        images = batch[self.model.image_key] if hasattr(self.model, "image_key") else batch["image"]
        labels = batch[self.model.label_key]

        # GSPO-Adapter extension:
        # - Enable GSPO mode on encoder adapters (adds exploration noise / gate sampling if configured)
        # - Optionally freeze encoder adapters during GSPO-active steps (default behavior)
        prev_requires_grad = None
        self._set_encoder_adapters_gspo_mode(
            active=True,
            adapter_noise_std=float(self.gspo_adapter_cfg.get("encoder_adapter_noise_std", 0.0)),
            gate_noise_std=float(self.gspo_adapter_cfg.get("encoder_adapter_gate_noise_std", 0.0)),
        )
        if not bool(self.gspo_adapter_cfg.get("train_encoder_adapter", False)):
            prev_requires_grad = self._set_encoder_adapter_requires_grad(trainable=False)
        
        # Define forward function for GSPO
        def forward_fn(images):
            batch_copy = batch.copy()
            if hasattr(self.model, 'image_key'):
                batch_copy[self.model.image_key] = images
            else:
                batch_copy['image'] = images
            output = run_model(self.model, batch_copy)
            
            # Extract predictions and MOE info
            logits = output[self.model.prefix][LOGITS]
            moe_loss = output[self.model.prefix].get(MOE_LOSS, 0)
            
            # Extract selected experts if available
            selected_experts = None
            if isinstance(output.get(self.model.prefix, None), dict) and "selected_experts" in output[self.model.prefix]:
                selected_experts = output[self.model.prefix]["selected_experts"]
            
            return logits, moe_loss, selected_experts
        
        # Define loss function for GSPO
        def loss_fn(predictions, targets):
            if isinstance(self.loss_func, Mask2FormerLoss):
                # Handle Mask2Former case
                return self.loss_func(predictions, targets)
            else:
                return self.loss_func(input=predictions, target=targets)
        
        try:
            # Run GSPO group training
            loss, metrics, selected_experts_groups = self.gspo_trainer.gspo_group_training_step(
                images=images,
                masks_gt=labels,
                forward_fn=forward_fn,
                loss_fn=loss_fn,
            )
        finally:
            # Restore adapter grad states and disable GSPO mode
            if prev_requires_grad is not None:
                self._restore_requires_grad(prev_requires_grad)
            self._set_encoder_adapters_gspo_mode(active=False)
        
        # Update expert quality feedback
        if selected_experts_groups:
            quality_scores = [
                self.gspo_trainer.compute_segmentation_quality(
                    pred, labels, self.gspo_trainer.quality_metric
                )
                for pred in [forward_fn(images)[0] for _ in range(self.gspo_trainer.group_size)]
            ]
            self.gspo_trainer.update_expert_feedback(
                selected_experts_groups, quality_scores, self.model
            )
        
        return loss, metrics, selected_experts_groups

    def _iter_encoder_adapter_named_modules(self):
        """
        Yield (name, module) for encoder adapters only (exclude decoder adapters).
        We match by module name path to avoid touching decoder adapter modules.
        """
        for name, module in self.model.named_modules():
            if "vision_encoder.layers" in name and name.endswith(".adapter"):
                yield name, module

    def _set_encoder_adapters_gspo_mode(self, active: bool, adapter_noise_std: float = 0.0, gate_noise_std: float = 0.0):
        for _, m in self._iter_encoder_adapter_named_modules():
            if hasattr(m, "set_gspo_mode"):
                # GatedAdapterLayer.set_gspo_mode(active, gate_noise_std, adapter_noise_std)
                try:
                    m.set_gspo_mode(active=active, gate_noise_std=gate_noise_std, adapter_noise_std=adapter_noise_std)
                except TypeError:
                    # AdapterLayer.set_gspo_mode(active, noise_std)
                    m.set_gspo_mode(active=active, noise_std=adapter_noise_std)

    def _set_encoder_adapter_requires_grad(self, trainable: bool) -> Dict[str, bool]:
        """
        Set requires_grad for encoder adapter parameters. Returns a dict for restoration.
        """
        prev: Dict[str, bool] = {}
        for name, m in self._iter_encoder_adapter_named_modules():
            for p_name, p in m.named_parameters(recurse=True):
                key = f"{name}.{p_name}"
                prev[key] = p.requires_grad
                p.requires_grad = bool(trainable)
        return prev

    def _restore_requires_grad(self, prev: Dict[str, bool]):
        for name, m in self._iter_encoder_adapter_named_modules():
            for p_name, p in m.named_parameters(recurse=True):
                key = f"{name}.{p_name}"
                if key in prev:
                    p.requires_grad = prev[key]

```

```worktree-python
import argparse
import os

import numpy as np
import pandas as pd
import torch

from autogluon.multimodal import MultiModalPredictor
from bbox_prompt_model import BBoxPromptPredictor


def get_default_training_setting(dataset_name):
    validation_metric = "iou"
    loss = "structure_loss"
    max_epoch = 30
    lr = 1e-4

    if dataset_name == "SBU-shadow":
        validation_metric = "ber"
        loss = "balanced_bce"
        max_epoch = 10

    elif dataset_name == "polyp":
        validation_metric = "sm"

    elif dataset_name == "camo_sem_seg":
        validation_metric = "sm"
        max_epoch = 20

    elif dataset_name == "road_segmentation":
        validation_metric = "iou"
        max_epoch = 20
        lr = 3e-4

    elif dataset_name == "leaf_disease_segmentation":
        validation_metric = "iou"
        lr = 3e-4

    return validation_metric, loss, max_epoch, lr


def expand_path(df, dataset_dir):
    for col in ["image", "label"]:
        df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script support converting voc format xmls to coco format json")
    parser.add_argument(
        "--task",
        type=str,
        default="leaf_disease_segmentation",
        choices=["polyp", "leaf_disease_segmentation", "camo_sem_seg", "isic2017", "road_segmentation", "SBU-shadow"],
    )
    parser.add_argument("--seed", type=int, default=42686693)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--expert_num", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--ckpt_path", type=str, default="outputs", help="Checkpoint path.")
    parser.add_argument("--per_gpu_batch_size", type=int, default=1, help="The batch size for each GPU.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="The effective batch size. If batch_size > per_gpu_batch_size * num_gpus, gradient accumulation would be used.",
    )
    parser.add_argument("--eval", action="store_true")
    
    # GSPO parameters
    parser.add_argument("--gspo_enable", action="store_true", help="Enable GSPO (Group Sequence Policy Optimization) training")
    parser.add_argument("--gspo_group_size", type=int, default=4, help="Number of predictions per group in GSPO")
    parser.add_argument("--gspo_warmup_epochs", type=int, default=5, help="Number of epochs before enabling GSPO")
    parser.add_argument("--gspo_contrastive_weight", type=float, default=0.1, help="Weight for contrastive loss in GSPO")
    parser.add_argument("--gspo_quality_momentum", type=float, default=0.9, help="Momentum for expert quality history")
    # GSPO reward / loss shaping hyper-parameters (可选，带默认值)
    parser.add_argument("--lambda_smooth", type=float, default=0.1, help="Smoothness loss weight (λ_smooth)")
    parser.add_argument("--lambda_boundary", type=float, default=0.3, help="Boundary loss weight (λ_boundary)")
    parser.add_argument("--w_boundary", type=float, default=0.3, help="Boundary reward weight (w_boundary)")
    parser.add_argument("--w_smooth", type=float, default=0.1, help="Smoothness reward weight (w_smooth)")
    parser.add_argument("--w_thin", type=float, default=0.05, help="Thin-structure reward weight (w_thin)")
    
    # Adapter parameters
    parser.add_argument("--adapter_enable", action="store_true", help="Enable standard Adapter modules")
    parser.add_argument("--adapter_dim", type=int, default=64, help="Dimension of the adapter bottleneck")

    # GSPO-Adapter extension (encoder adapters only)
    parser.add_argument(
        "--gspo_train_encoder_adapter",
        action="store_true",
        help="When GSPO is active, also train Vision Encoder adapters (default: False => adapters frozen during GSPO steps).",
    )
    parser.add_argument(
        "--gspo_encoder_adapter_noise_std",
        type=float,
        default=0.0,
        help="Stage 1: exploration noise std added to encoder adapter outputs during GSPO rollouts (default: 0.0).",
    )
    parser.add_argument(
        "--gspo_encoder_adapter_gate_enable",
        action="store_true",
        help="Stage 2: use gated encoder adapters (policy-like per-sample scalar gate).",
    )
    parser.add_argument(
        "--gspo_encoder_adapter_gate_noise_std",
        type=float,
        default=0.0,
        help="Stage 2: exploration noise std added to gated-adapter gate logits during GSPO rollouts (default: 0.0).",
    )
    
    # Decoder Attention LoRA parameters (LoRA on Attention)
    parser.add_argument("--decoder_attn_lora_enable", action="store_true", 
                        help="Enable LoRA on decoder attention Q/K/V projections")
    parser.add_argument("--decoder_attn_lora_r", type=int, default=8,
                        help="LoRA rank for decoder attention (default: 8)")
    parser.add_argument("--decoder_attn_lora_alpha", type=int, default=8,
                        help="LoRA alpha for decoder attention (default: 8, same as rank)")
    parser.add_argument("--decoder_attn_lora_dropout", type=float, default=0.0,
                        help="LoRA dropout for decoder attention (default: 0.0)")
    
    # TTA (Test-Time Augmentation) parameters
    parser.add_argument("--tta_enable", action="store_true", help="Enable Test-Time Augmentation for evaluation")
    parser.add_argument("--tta_scales", type=float, nargs="+", default=[0.75, 1.0, 1.25], 
                        help="Scale factors for multi-scale TTA (default: [0.75, 1.0, 1.25]). "
                             "Recommended: [0.75, 1.0, 1.25] or [0.8, 1.0, 1.2]")
    parser.add_argument("--tta_flips", type=str, nargs="+", default=["none", "horizontal"],
                        choices=["none", "horizontal", "vertical"],
                        help="Flip types for TTA (default: ['none', 'horizontal'])")
    parser.add_argument("--tta_rotations", type=float, nargs="+", default=[0],
                        help="Rotation angles in degrees for TTA (default: [0]). "
                             "Recommended: [0] for fast, or [-10, 0, 10] for better")
    parser.add_argument("--tta_fusion", type=str, default="mean", choices=["mean", "weighted_mean"],
                        help="Fusion method for TTA predictions (default: 'mean')")
    parser.add_argument("--tta_resize_method", type=str, default="bilinear", choices=["bilinear", "bicubic"],
                        help="Resize interpolation method for multi-scale TTA (default: 'bilinear'). "
                             "bilinear: faster, bicubic: higher quality")
    parser.add_argument("--tta_threshold", type=float, default=0.5,
                        help="Threshold for binary segmentation in TTA (default: 0.5)")
    parser.add_argument("--tta_min_area", type=float, default=0.001,
                        help="Remove components with area < tta_min_area * image_area (default: 0.001)")
    parser.add_argument("--tta_morphology", action="store_true",
                        help="Apply morphological closing in TTA post-processing")
    parser.add_argument("--tta_cache_dir", type=str, default=None,
                        help="Directory to cache TTA predictions for resume (default: None, no caching)")
    parser.add_argument("--tta_no_resume", action="store_true",
                        help="Disable resume from cache (start fresh even if cache exists)")
    parser.add_argument("--tta_box_prompt_mode", type=str, default="off",
                        choices=["off", "add", "replace"],
                        help="Use GT box as SAM box prompt (off/add/replace); box not transformed with TTA")
    # BBox prompt configuration
    parser.add_argument("--bbox_prompt_source", type=str, default="none",
                        choices=["none", "gt", "predict"],
                        help="Box prompt source: none (no prompt), gt (from mask), predict (from bbox model).")
    parser.add_argument("--bbox_model_ckpt", type=str, default=None,
                        help="Checkpoint for bbox predictor when bbox_prompt_source=predict.")
    parser.add_argument("--bbox_model_image_size", type=int, default=320,
                        help="Input size for bbox predictor inference.")
    parser.add_argument("--bbox_preds_csv", type=str, default=None,
                        help="Optional CSV with columns image,bbox_x1,bbox_y1,bbox_x2,bbox_y2 to attach prompts.")
    # Training-time box prompt mixing (default off)
    parser.add_argument("--train_box_prompt_mode", type=str, default="off",
                        choices=["off", "gt", "noisy", "mix", "predict"],
                        help="Training-time box prompt injection: off/gt/noisy/mix/predict (default off). "
                             "predict: use bbox predictor model to generate box prompts from images.")
    parser.add_argument("--train_box_prob_no_prompt", type=float, default=0.2,
                        help="When mode=mix, probability of no prompt.")
    parser.add_argument("--train_box_prob_gt", type=float, default=0.3,
                        help="When mode=mix, probability of GT box prompt.")
    parser.add_argument("--train_box_prob_noisy", type=float, default=0.5,
                        help="When mode=mix, probability of noisy GT box prompt.")
    parser.add_argument("--train_box_noise_frac", type=float, default=0.12,
                        help="Noisy box jitter fraction relative to box size (e.g., 0.12 => ±12%).")
    parser.add_argument("--train_bbox_model_ckpt", type=str, default=None,
                        help="Checkpoint for bbox predictor when train_box_prompt_mode=predict.")
    parser.add_argument("--train_bbox_model_image_size", type=int, default=320,
                        help="Input size for bbox predictor during training (when train_box_prompt_mode=predict).")
    
    # Quick test / Debug parameters
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: only process first 5 images for quick validation")
    parser.add_argument("--quick_test", type=int, default=None,
                        help="Quick test mode: only process first N images")
    
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: auto based on task, typically 1e-4 or 3e-4)")
    
    # Decoder Adapter parameters
    parser.add_argument("--decoder_adapter_enable", action="store_true", help="Enable MLP-Adapter modules in the mask decoder")
    parser.add_argument("--decoder_adapter_dim", type=int, default=64, help="Dimension of the decoder adapter bottleneck")
    args = parser.parse_args()

    dataset_name = args.task
    dataset_dir = os.path.join(f"datasets/{dataset_name}", dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare dataframes
    train_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"train.csv")), dataset_dir)
    val_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"val.csv")), dataset_dir)

    # get the validation metric
    validation_metric, loss, max_epoch, lr = get_default_training_setting(dataset_name)
    
    # Override learning rate if user specified it via CLI
    if args.lr is not None:
        lr = args.lr
        print(f"Using custom learning rate: {lr}")

    hyperparameters = {}
    hyperparameters.update(
        {
            "optim.lora.r": args.rank,
            # Paper alignment (Conv-LoRA):
            # - optim.peft: choose Conv-LoRA adaptation (Sec. 3)
            # - optim.lora.r: LoRA rank r (Sec. 3)
            # - optim.lora.conv_lora_expert_num: number of MoE-Conv experts M (Sec. 3.2)
            "optim.peft": "conv_lora",
            "optim.lora.conv_lora_expert_num": args.expert_num,
            "env.num_gpus": args.num_gpus,
            "optim.loss_func": loss,
            "optim.max_epochs": max_epoch,
            "optim.lr": lr,
            "env.per_gpu_batch_size": args.per_gpu_batch_size,
            "env.batch_size": args.batch_size,
        }
    )
    
    # GSPO configuration
    if args.gspo_enable:
        print(f"Enabling GSPO with group_size={args.gspo_group_size}, warmup_epochs={args.gspo_warmup_epochs}")
        hyperparameters.update({
            "optim.lora.gspo_enabled": True,
            "optim.lora.gspo_group_size": args.gspo_group_size,
            "optim.lora.gspo_quality_momentum": args.gspo_quality_momentum,
            "optim.lora.gspo_warmup_epochs": args.gspo_warmup_epochs,
            "optim.lora.gspo_contrastive_weight": args.gspo_contrastive_weight,
            # GSPO reward / loss shaping
            "optim.gspo.lambda_smooth": args.lambda_smooth,
            "optim.gspo.lambda_boundary": args.lambda_boundary,
            "optim.gspo.w_boundary": args.w_boundary,
            "optim.gspo.w_smooth": args.w_smooth,
            "optim.gspo.w_thin": args.w_thin,
            # GSPO-Adapter extension (encoder adapters only)
            "optim.gspo.train_encoder_adapter": args.gspo_train_encoder_adapter,
            "optim.gspo.encoder_adapter_noise_std": args.gspo_encoder_adapter_noise_std,
            "optim.gspo.encoder_adapter_gate_noise_std": args.gspo_encoder_adapter_gate_noise_std,
        })

    # Adapter configuration
    if args.adapter_enable:
        print(f"Enabling Standard Encoder Adapters with dim={args.adapter_dim}")
        hyperparameters.update({
            "model.sam.adapter_enabled": True,
            "model.sam.adapter_dim": args.adapter_dim,
        })

    # Stage 2: gated encoder adapter (policy-like). This only changes model structure if adapters are enabled.
    if args.gspo_encoder_adapter_gate_enable:
        print(
            f"Enabling Gated Encoder Adapters (policy) with gate_noise_std={args.gspo_encoder_adapter_gate_noise_std}"
        )
        hyperparameters.update(
            {
                "model.sam.encoder_adapter_gate_enabled": True,
                "model.sam.encoder_adapter_gate_noise_std": args.gspo_encoder_adapter_gate_noise_std,
            }
        )
    
    # Decoder Attention LoRA configuration
    if args.decoder_attn_lora_enable:
        print(f"Enabling Decoder Attention LoRA: r={args.decoder_attn_lora_r}, "
              f"alpha={args.decoder_attn_lora_alpha}, dropout={args.decoder_attn_lora_dropout}")
        hyperparameters.update({
            "model.sam.decoder_attention_lora_r": args.decoder_attn_lora_r,
            "model.sam.decoder_attention_lora_alpha": args.decoder_attn_lora_alpha,
            "model.sam.decoder_attention_lora_dropout": args.decoder_attn_lora_dropout,
        })
        
    # Decoder Adapter configuration
    if args.decoder_adapter_enable:
        print(f"Enabling Decoder MLP-Adapters with dim={args.decoder_adapter_dim}")
        hyperparameters.update({
            "model.sam.decoder_adapter_enabled": True,
            "model.sam.decoder_adapter_dim": args.decoder_adapter_dim,
        })

    if args.eval:  # load a checkpoint for evaluation
        predictor = MultiModalPredictor.load(args.ckpt_path)
    else:  # training
        predictor = MultiModalPredictor(
            problem_type="semantic_segmentation",
            validation_metric=validation_metric,
            eval_metric=validation_metric,
            hyperparameters=hyperparameters,
            label="label",
        )

        # Configure training-time box prompt mixing
        predictor._learner._train_box_prompt_cfg = {
            "mode": args.train_box_prompt_mode,
            "p_no": args.train_box_prob_no_prompt,
            "p_gt": args.train_box_prob_gt,
            "p_noisy": args.train_box_prob_noisy,
            "noise_frac": args.train_box_noise_frac,
        }

        # Setup bbox predictor for training if mode=predict
        if args.train_box_prompt_mode == "predict":
            if not args.train_bbox_model_ckpt:
                raise ValueError("train_box_prompt_mode=predict requires --train_bbox_model_ckpt.")
            train_bbox_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_bbox_predictor = BBoxPromptPredictor(
                ckpt_path=args.train_bbox_model_ckpt,
                device=train_bbox_device,
                image_size=args.train_bbox_model_image_size,
            )
            predictor._learner._train_bbox_predictor = train_bbox_predictor
            print(f"\n[Info] Training-time bbox predictor loaded from: {args.train_bbox_model_ckpt}")
            print(f"       Input size: {args.train_bbox_model_image_size}\n")

        # 打印当前实验的模型保存目录，便于后续通过 AutogluonModels 路径追溯到具体实验日志
        try:
            # MultiModalPredictor 会在内部根据时间戳生成类似 AutogluonModels/ag-YYYYMMDD_HHMMSS 的目录
            model_path = predictor.path
        except AttributeError:
            # 兼容极端情况：如果未来接口变化，没有 path 属性，则显式提示
            model_path = None

        if model_path is not None:
            print("\n========================================")
            print("Training MultiModalPredictor")
            print(f"  Task        : {dataset_name}")
            print(f"  Save path   : {model_path}")
            print("  (You can use this directory name to link back to the training log.)")
            print("========================================\n")
        else:
            print("\n[Warning] MultiModalPredictor has no 'path' attribute. "
                  "Model save directory cannot be printed.\n")

        predictor.fit(train_data=train_df, tuning_data=val_df, seed=args.seed)

    # Enable TTA if requested
    if args.tta_enable:
        print(f"\n{'='*60}")
        print(f"Enabling Test-Time Augmentation (TTA)")
        print(f"  Scales: {args.tta_scales}")
        print(f"  Flips: {args.tta_flips}")
        print(f"  Rotations: {args.tta_rotations}")
        print(f"  Fusion: {args.tta_fusion}")
        print(f"  Resize method: {args.tta_resize_method}")
        print(f"  Total augmentations: {len(args.tta_scales) * len(args.tta_flips) * len(args.tta_rotations)}")
        print(f"{'='*60}\n")
        
        predictor.enable_tta(
            scales=args.tta_scales,
            flips=args.tta_flips,
            rotations=args.tta_rotations,
            fusion_method=args.tta_fusion,
            resize_method=args.tta_resize_method,
            threshold=args.tta_threshold,
            min_area_ratio=args.tta_min_area,
            use_morphology=args.tta_morphology,
            cache_dir=args.tta_cache_dir,
            resume_from_cache=not args.tta_no_resume,
            box_prompt_mode=args.tta_box_prompt_mode,
        )

    # Optional bbox predictor for predicted box prompts
    bbox_predictor = None
    if args.bbox_prompt_source == "predict":
        if not args.bbox_model_ckpt and not args.bbox_preds_csv:
            raise ValueError("bbox_prompt_source=predict requires --bbox_model_ckpt or --bbox_preds_csv.")
        if args.bbox_model_ckpt:
            bbox_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            bbox_predictor = BBoxPromptPredictor(
                ckpt_path=args.bbox_model_ckpt,
                device=bbox_device,
                image_size=args.bbox_model_image_size,
            )
        if args.tta_box_prompt_mode == "off":
            print("[Warning] bbox_prompt_source=predict set but tta_box_prompt_mode=off; box prompts will be ignored.")

    def attach_pred_boxes(df: pd.DataFrame) -> pd.DataFrame:
        """Attach predicted or precomputed boxes to dataframe."""
        df = df.copy()
        if args.bbox_preds_csv:
            preds = pd.read_csv(args.bbox_preds_csv)
            merged = df.merge(preds, on="image", how="left")
            if merged[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].isna().all().all():
                preds["image_basename"] = preds["image"].apply(lambda p: os.path.basename(p))
                merged["image_basename"] = merged["image"].apply(lambda p: os.path.basename(p))
                merged = merged.merge(
                    preds.drop(columns=["image"]), on="image_basename", how="left", suffixes=("", "_pred")
                )
                merged.drop(columns=["image_basename"], inplace=True)
                for col in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
                    pred_col = f"{col}_pred"
                    if pred_col in merged.columns:
                        merged[col] = merged[col].fillna(merged[pred_col])
                        merged.drop(columns=[pred_col], inplace=True)
            df = merged
        elif bbox_predictor is not None:
            boxes = []
            for _, r in df.iterrows():
                box = bbox_predictor.predict(r["image"])
                boxes.append(box)
            boxes = np.stack(boxes, axis=0)
            df["bbox_x1"] = boxes[:, 0]
            df["bbox_y1"] = boxes[:, 1]
            df["bbox_x2"] = boxes[:, 2]
            df["bbox_y2"] = boxes[:, 3]
        return df

    # evaluation
    metric_file = os.path.join(args.output_dir, "metrics.txt")
    f = open(metric_file, "a")
    if dataset_name in ["isic2017", "SBU-shadow", "road_segmentation", "leaf_disease_segmentation"]:
        test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test.csv")), dataset_dir)
        
        # Apply quick test / debug mode (方案 1: 快速验证模式)
        original_size = len(test_df)
        if args.debug:
            test_df = test_df.head(5)
            print(f"\n🔍 DEBUG MODE: Processing only first 5 images (out of {original_size})\n")
        elif args.quick_test is not None:
            test_df = test_df.head(args.quick_test)
            print(f"\n🔍 QUICK TEST MODE: Processing only first {args.quick_test} images (out of {original_size})\n")
        
        if dataset_name == "SBU-shadow":
            eval_metrics = ["ber"]
        else:
            eval_metrics = ["iou", "dice"]  # Evaluate both IoU and DICE

        # Attach predicted bbox prompts if requested
        if args.bbox_prompt_source == "predict" and (args.bbox_preds_csv or bbox_predictor):
            test_df = attach_pred_boxes(test_df)

        res = predictor.evaluate(test_df, metrics=eval_metrics)
        print(f"Evaluation results for test dataset {dataset_name}: ", res)
        f.write(f"Evaluation results for test dataset {dataset_name}: {res} \n")
    elif dataset_name in ["polyp", "camo_sem_seg"]:
        if dataset_name == "polyp":
            test_datasets = ["CVC-ClinicDB", "Kvasir"]
        elif dataset_name == "camo_sem_seg":
            test_datasets = ["CAMO"]
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}.")
        for per_dataset in test_datasets:
            test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test_{per_dataset}.csv")), dataset_dir)
            
            # Apply quick test / debug mode (方案 1: 快速验证模式)
            original_size = len(test_df)
            if args.debug:
                test_df = test_df.head(5)
                print(f"\n🔍 DEBUG MODE: Processing only first 5 images (out of {original_size}) for {per_dataset}\n")
            elif args.quick_test is not None:
                test_df = test_df.head(args.quick_test)
                print(f"\n🔍 QUICK TEST MODE: Processing only first {args.quick_test} images (out of {original_size}) for {per_dataset}\n")
            
            if args.bbox_prompt_source == "predict" and (args.bbox_preds_csv or bbox_predictor):
                test_df = attach_pred_boxes(test_df)

            res = predictor.evaluate(test_df, metrics=["sm", "fm", "em", "mae"])
            print(f"Evaluation results for test dataset {per_dataset}: ", res)
            f.write(f"Evaluation results for test dataset {per_dataset}: {res} \n")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}.")

    f.close()

```

```worktree-python

import torch
import torch.nn as nn
from transformers import SamConfig
from autogluon.multimodal.models.custom_hf_models.modeling_sam_for_conv_lora import SamVisionLayer
from autogluon.multimodal.models.adaptation_layers import AdapterLayer, GatedAdapterLayer

def test_sam_vision_layer_with_adapter():
    # 1. Setup Config
    config = SamConfig()
    config.hidden_size = 768
    config.num_attention_heads = 12
    config.patch_size = 16
    config.image_size = 1024
    config.adapter_enabled = True
    config.adapter_dim = 32
    
    # 2. Instantiate Layer
    layer = SamVisionLayer(config, window_size=0)
    
    # 3. Verify Adapter Existence
    assert hasattr(layer, "adapter")
    assert isinstance(layer.adapter, AdapterLayer)
    assert layer.adapter.down_proj.out_features == 32
    
    # 4. Verify Forward Pass
    batch_size = 2
    seq_len = 16 * 16
    hidden_dim = 768
    x = torch.randn(batch_size, 16, 16, hidden_dim) # [B, H, W, C]
    
    output = layer(x)
    assert output[0].shape == (batch_size, 16, 16, hidden_dim)
    
    # 5. Verify Trainable Parameters
    adapter_params = list(layer.adapter.parameters())
    assert len(adapter_params) > 0
    for p in adapter_params:
        assert p.requires_grad


def test_sam_vision_layer_with_gated_adapter():
    # 1. Setup Config
    config = SamConfig()
    config.hidden_size = 768
    config.num_attention_heads = 12
    config.patch_size = 16
    config.image_size = 1024
    config.adapter_enabled = True
    config.adapter_dim = 32
    config.encoder_adapter_gate_enabled = True
    config.encoder_adapter_gate_noise_std = 0.5

    # 2. Instantiate Layer
    layer = SamVisionLayer(config, window_size=0)

    # 3. Verify gated adapter
    assert layer.adapter is not None
    assert isinstance(layer.adapter, GatedAdapterLayer)

    # 4. Verify forward pass shape
    x = torch.randn(2, 16, 16, 768)
    out = layer(x)
    assert out[0].shape == (2, 16, 16, 768)

    # 5. Verify GSPO mode toggling doesn't error
    layer.adapter.set_gspo_mode(active=True, gate_noise_std=0.1, adapter_noise_std=0.0)
    layer.adapter.set_gspo_mode(active=False)

def test_sam_vision_layer_without_adapter():
    # 1. Setup Config
    config = SamConfig()
    config.hidden_size = 768
    config.adapter_enabled = False
    
    # 2. Instantiate Layer
    layer = SamVisionLayer(config, window_size=0)
    
    # 3. Verify Adapter Absence
    assert layer.adapter is None
    
    # 4. Verify Forward Pass
    batch_size = 2
    hidden_dim = 768
    x = torch.randn(batch_size, 16, 16, hidden_dim)
    
    output = layer(x)
    assert output[0].shape == (batch_size, 16, 16, hidden_dim)

if __name__ == "__main__":
    test_sam_vision_layer_with_adapter()
    test_sam_vision_layer_without_adapter()
    print("All tests passed!")

```

```markdown
## 目录说明（GSPO + Conv-LoRA + Encoder Adapter 强化学习扩展）

本子目录用于集中存放本次“让 GSPO 同时作用于 Conv-LoRA 的 MoE 与 Vision Encoder 的 FC Adapter（仅 Encoder Adapter，不含 Decoder Adapter）”的设计与实现说明文档。

- `design.md`：总体设计（阶段 1 / 阶段 2）、模型结构框图、公式与实现映射、损失/奖励定义
- `implementation.md`：代码落点、配置/命令行参数说明、向后兼容策略、常见排错


```

```markdown
## 目标

在现有 **Conv-LoRA + GSPO** 方案基础上，让 GSPO 在 **Vision Encoder 每层 MLP 后的 FC Adapter（Encoder Adapter）** 上也产生“强化学习式”的优化效果，并且通过开关保持向后兼容：

- **默认行为不变**：当不开启本扩展选项时，GSPO 仍按原逻辑用于 Conv‑LoRA（Adapter 在 GSPO step 中默认冻结，避免被 GSPO 的组级加权间接影响）。
- **开启扩展**：GSPO active 时同时训练 encoder adapters，并可选启用两阶段增强：
  - **阶段 1（最小改动）**：在 GSPO rollouts 中对 encoder adapter 输出注入探索噪声，增强组内多样性与 credit assignment。
  - **阶段 2（显式策略）**：将 encoder adapter 替换为 *gated adapter*，引入输入条件化 gate（policy-like），并在 GSPO rollouts 中对 gate logits 注入探索噪声。

> 本次明确 **不包含 decoder adapter**（mask decoder 里的 adapter_queries 不参与本扩展）。

---

## 总体结构与插入位置

Encoder 的每个 ViT block（`SamVisionLayer`）里，Adapter 位于 **MLP 路径**，与 MLP 并行，外部残差相加：

```text
SamVisionLayer (×N layers)

  residual
    │
    ├─ Attention(...)  ← Conv-LoRA 注入在 q/k/v 线性投影内部（MoE-Conv）
    │
    └─ MLP path:
        LN2(x)
          ├─ MLP(x)      -> mlp_out
          └─ Adapter(x)  -> adapter_out   (FC bottleneck, encoder adapter)
        y = residual + mlp_out + adapter_out
```

阶段 2 时，Adapter 变为 gated adapter：

```text
adapter_out = AdapterCore(x) * gate(x)
gate(x) = sigmoid( w · pool(x) + ε ),   ε ~ N(0, σ_gate^2)  (仅 GSPO rollouts 时注入)
```

---

## GSPO 的公式（与实现对应）

在每个训练 step，对每张图像生成 \(G\) 次 rollouts（利用 dropout/噪声产生差异），得到 \(G\) 组预测：

- 预测：\(\hat{y}^{(g)} = f_\theta(x; \xi_g)\)，其中 \(\xi_g\) 表示 rollout 噪声（dropout、adapter 噪声、gate 噪声等）
- 质量（奖励）：\(R^{(g)} = \text{Quality}(\hat{y}^{(g)}, y)\)
  - 实现中使用 IoU/Dice，并加入边界/平滑/细结构 shaping 得到 `shaped_quality`
- baseline：\(\bar{R} = \frac{1}{G}\sum_{g=1}^G R^{(g)}\)
- advantage：\(A^{(g)} = R^{(g)} - \bar{R}\)
- 权重（温度 \(\tau\)）：\(w^{(g)} = \sigma(\tau \cdot A^{(g)})\)

最终优化目标（实现中的主要项）：

\[
\mathcal{L}
= \frac{1}{G}\sum_{g=1}^G w^{(g)} \cdot \mathcal{L}_{seg}(\hat{y}^{(g)}, y)
 + \lambda_{moe}\mathcal{L}_{moe}
 + \lambda_{ctr}\mathcal{L}_{ctr}
\]

其中：
- \(\mathcal{L}_{seg}\)：分割损失（并叠加 smooth/boundary 约束）
- \(\mathcal{L}_{moe}\)：MoE load-balancing 辅助损失（Conv‑LoRA gate）
- \(\mathcal{L}_{ctr}\)：组内对比损失（鼓励高质量预测聚合）

> 这是一种“GSPO inspired”的可微 surrogate：通过 advantage 形成 **组内相对质量** 的权重，放大/缩小对应 rollout 的梯度贡献，达到类似策略优化的 credit assignment 效果。

---

## 阶段 0（基线，向后兼容）

当不启用本扩展（`optim.gspo.train_encoder_adapter=False`）时：

- Encoder Adapter 仍可存在（`model.sam.adapter_enabled=True`），但在 **GSPO active 的 step** 中会被临时冻结（`requires_grad=False`），确保“GSPO 只用于 Conv‑LoRA”这一语义成立。
- 在 GSPO warmup 阶段（GSPO 未激活）仍按普通监督训练逻辑运行。

---

## 阶段 1：Encoder Adapter 探索噪声（最小改动）

### 设计

在 GSPO active 的 rollouts 期间，对每层 encoder adapter 输出加入零均值噪声：

\[
\Delta h = \text{Adapter}(x) + \epsilon,\quad \epsilon \sim \mathcal{N}(0,\sigma_{adp}^2)
\]

噪声只在 GSPO active 且训练模式下生效，用于：
- 增强 rollouts 多样性（同一输入得到不同预测）
- 强化 advantage 权重对“更好预测路径”的梯度分配

### 预期收益

- 更强的组内探索：让 Adapter 不再在 \(G\) 次 rollouts 中“几乎相同”，从而使 GSPO 的优势加权更有效。
- 更稳的 credit assignment：帮助模型把“质量提升”更明确地归因到 Adapter 分支的改动。

---

## 阶段 2：Gated Encoder Adapter（显式策略）

### 设计

将每层 encoder adapter 替换为 gated adapter：

\[
\Delta h = \text{AdapterCore}(x)\cdot g(x),\quad g(x)=\sigma(\text{MLP}_g(\text{pool}(x)) + \epsilon)
\]

其中：
- \(\text{pool}(x)\) 为 token 的均值池化（对 `[B,H,W,C]` 做 H/W 平均）
- \(\epsilon \sim \mathcal{N}(0,\sigma_{gate}^2)\) 仅在 GSPO active rollouts 中注入

### 直观解释

- gate 是“每个样本、每层一个可学习开关/强度”，是一个轻量 policy。
- GSPO 的 rollouts 会对 gate 进行不同采样（通过 \(\epsilon\)），得到不同预测质量；advantage-weighted loss 会促使 gate 网络朝“更高质量的门控策略”收敛。

### 预期收益

- 自适应层级分配：让模型学会“哪些层的 adapter 应该更强/更弱地介入”。
- 更强的空间-通道协同：Conv‑LoRA（空间 MoE）与 gated adapter（通道强度策略）同时受质量信号驱动。

---

## 不包含 Decoder Adapter 的原因

Decoder adapter 直接作用于 mask decoder 的 queries 分支，属于解码阶段适配；本次扩展聚焦于 **encoder representation** 的 GSPO credit assignment，以避免同时改动两个阶段导致归因混乱与调参复杂。


```

```markdown
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


```

### ✅ 已实现：GSPO 同时微调 Conv-LoRA MoE + Encoder Adapter（不含 decoder adapter），且默认完全兼容旧行为

#### 1) **向后兼容策略（默认不改）**
- 在 **GSPO active** 的 step 中，若你不显式开启本扩展：`optim.gspo.train_encoder_adapter=False`（默认）
  - **Vision Encoder 的 adapters 会被临时冻结**（`requires_grad=False`）
  - 因此“GSPO 仍然只用于 Conv‑LoRA”这件事在实现语义上成立

对应实现：`SemanticSegmentationLitModule._gspo_training_step()` 中对 encoder adapters 做冻结/恢复（只匹配 `vision_encoder.layers.*.adapter`，不会碰 decoder adapter）。

#### 2) **阶段 1（最小改动）：Adapter 输出探索噪声**
- 在 GSPO rollouts 中，可给 encoder adapter 输出注入探索噪声（增强组内多样性）：
  - 开关：`--gspo_train_encoder_adapter`
  - 噪声：`--gspo_encoder_adapter_noise_std <float>`

实现：`AdapterLayer.set_gspo_mode(active=True, noise_std=...)` + forward 里在 GSPO active 时加 `N(0,σ²)`。

#### 3) **阶段 2（显式策略）：Gated Encoder Adapter（policy-like gate）**
- 可选把 encoder adapter 替换为 gated adapter，引入 per-sample gate，并在 GSPO rollouts 中对 gate logits 注入噪声（显式“策略探索”）：
  - `--gspo_encoder_adapter_gate_enable`
  - `--gspo_encoder_adapter_gate_noise_std <float>`

实现：新增 `GatedAdapterLayer`（`AdapterLayer` + `Linear(C→1)` gate），仅在 GSPO active 时对 gate logits 加噪声。

---

### 🧩 你可以怎么用（examples 入口）
脚本：`examples/automm/Conv-LoRA/run_semantic_segmentation.py`

- **默认（兼容旧行为）**：GSPO 只用于 Conv‑LoRA  
  - 不加 `--gspo_train_encoder_adapter` 即可（即便你开启了 `--adapter_enable`，GSPO active step 也会冻结 encoder adapters）

- **开启“GSPO 同时训练 Conv‑LoRA + Encoder Adapter”**（阶段 1）：

```bash
python3 run_semantic_segmentation.py \
  --task isic2017 \
  --rank 3 --expert_num 8 \
  --adapter_enable --adapter_dim 64 \
  --gspo_enable \
  --gspo_train_encoder_adapter \
  --gspo_encoder_adapter_noise_std 0.02 \
  --output_dir outputs/gspo_with_encoder_adapter_stage1
```

- **阶段 2（gated adapter policy）**：

```bash
python3 run_semantic_segmentation.py \
  --task isic2017 \
  --rank 3 --expert_num 8 \
  --adapter_enable --adapter_dim 64 \
  --gspo_enable \
  --gspo_train_encoder_adapter \
  --gspo_encoder_adapter_gate_enable \
  --gspo_encoder_adapter_gate_noise_std 0.5 \
  --output_dir outputs/gspo_with_encoder_adapter_stage2
```

---

### 📚 文档（英文目录名，中文内容）
已新增目录：`docs/gspo_adapter_rl/`
- `docs/gspo_adapter_rl/design.md`：阶段 0/1/2 的结构框图、公式、奖励/损失与实现映射
- `docs/gspo_adapter_rl/implementation.md`：代码落点、开关说明、兼容策略与调参建议
- `docs/gspo_adapter_rl/README.md`：索引

---

### 🔧 主要代码改动位置（便于你 review）
- **Adapter 扩展**：`multimodal/src/autogluon/multimodal/models/adaptation_layers.py`
  - `AdapterLayer` 增加 `set_gspo_mode()` 和 GSPO active 噪声
  - 新增 `GatedAdapterLayer`
- **Encoder adapter 构建（可选 gated）**：
  - `multimodal/src/autogluon/multimodal/models/custom_hf_models/modeling_sam_for_conv_lora.py`
  - `multimodal/src/autogluon/multimodal/models/sam.py`
- **GSPO active step 中冻结/启用/噪声开关**：
  - `multimodal/src/autogluon/multimodal/optim/lit_semantic_seg.py`
- **配置/CLI**：
  - `multimodal/src/autogluon/multimodal/configs/optim/default.yaml`
  - `multimodal/src/autogluon/multimodal/configs/model/default.yaml`
  - `examples/automm/Conv-LoRA/run_semantic_segmentation.py`
- **单测补充**：`tests/unittests/models/test_sam_adapter.py`（新增 gated adapter 覆盖）

> 说明：我在本环境里运行 `pytest` 会触发 numpy/torch 的 macOS 环境崩溃（非代码问题），但已用 `py_compile` 对所有改动文件做了语法编译校验通过；你本机/CI 环境应能正常跑 pytest。

```shellscript
python3 run_semantic_segmentation.py \
  --task isic2017 \
  --rank 3 --expert_num 8 \
  --adapter_enable --adapter_dim 64 \
  --gspo_enable \
  --gspo_train_encoder_adapter \
  --gspo_encoder_adapter_noise_std 0.02 \
  --output_dir outputs/gspo_with_encoder_adapter_stage1
```

```shellscript
python3 run_semantic_segmentation.py \
  --task isic2017 \
  --rank 3 --expert_num 8 \
  --adapter_enable --adapter_dim 64 \
  --gspo_enable \
  --gspo_train_encoder_adapter \
  --gspo_encoder_adapter_gate_enable \
  --gspo_encoder_adapter_gate_noise_std 0.5 \
  --output_dir outputs/gspo_with_encoder_adapter_stage2
```

---

**User**

以上实验还在执行中。现在我想进一步把LoRA on Attention的部分也纳入GSPO强化学习的微调里，请先研究本项目中的LoRA on Attention的部分（可以参考docs/中的对话记录Markdown文档），然后思考应如何基于现有基础，实现对LoRA on Attention的GSPO强化学习微调，最后完成设计和实现。附加要求与前面一样：为避免文档太多太乱，请在docs目录下创建子目录专门用于存放为此次设计而撰写的各种设计和说明文档，文件和目录名请用英文，内容用中文。设计文档应包含模型设计框图和必要的公式，清晰地说明强化学习是用在哪里、如何应用的（即在你的实际实现里，损失/奖励函数的形式是什么、其如何与GSPO的公式相对应），等等。另外请注意不要动任何.pdf文件！

---

**Cursor**

我会先把“LoRA on Attention”在本项目里到底指哪一段（vision encoder 还是 mask decoder 的 Q/K/V 投影）定位清楚：读 `docs/` 里相关对话记录 + 在代码里搜 `decoder_attention_lora_* / LoRALinear` 的实际注入点。然后按和前面一致的兼容路线实现：默认 GSPO 仍只优化 Conv‑LoRA；开启新开关后，GSPO active step 会**同时训练 Attention‑LoRA**，并可选在 GSPO rollouts 中加入探索噪声/门控以强化“RL 式 credit assignment”。接着补齐独立文档子目录（英文名、中文内容），并保证不触碰任何 `.pdf`。

Aha，LoRA on Attention 在这个项目里明确指的是 **Mask Decoder 的 SamAttention 里 q_proj/k_proj/v_proj 使用 `LoRALinear`**（而不是 encoder 的 Conv‑LoRA）。下一步我会把它按和前面一致的兼容策略接入 GSPO：默认 GSPO active step 冻结 decoder-attn-LoRA；开启开关后才让 GSPO 同时优化它，并在 GSPO rollouts 中可选对 LoRA 分支注入探索噪声（对应“RL 式”组内 credit assignment）。然后补齐独立文档子目录，且不会触碰任何 `.pdf`。

```worktree-python
# Modified based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal



def identity(x):
    return x


class AdapterLayer(nn.Module):
    """
    Standard Bottleneck Adapter Layer.
    Structure: Linear(d, r) -> Activation -> Linear(r, d) -> Scale
    """
    def __init__(self, in_features, adapter_dim, scale=1.0, dropout=0.0):
        super().__init__()
        self.down_proj = nn.Linear(in_features, adapter_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, in_features)
        self.scale = nn.Parameter(torch.tensor(scale))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # GSPO adapter extension (default off; activated by training loop via set_gspo_mode)
        self._gspo_active = False
        self._gspo_noise_std = 0.0
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        down = self.down_proj(x)
        act = self.activation(down)
        up = self.up_proj(act)
        # Optional GSPO exploration noise (only when explicitly enabled)
        if self._gspo_active and self.training and self._gspo_noise_std and self._gspo_noise_std > 0:
            up = up + torch.randn_like(up) * self._gspo_noise_std
        return self.dropout(up) * self.scale

    def set_gspo_mode(self, active: bool, noise_std: Optional[float] = None):
        """
        Enable/disable GSPO-specific behavior for this Adapter.
        This is intentionally a no-op unless called by the training loop.
        """
        self._gspo_active = bool(active)
        if noise_std is not None:
            self._gspo_noise_std = float(noise_std)


class GatedAdapterLayer(nn.Module):
    """
    Encoder Adapter with an input-conditioned gate (policy-like scalar per sample).

    Design goal:
    - When GSPO is active: add stochasticity to the gate logits for exploration across group rollouts.
    - When GSPO is inactive: use deterministic gate (no sampling noise).

    Input formats supported:
    - [B, H, W, C] (SAM vision tokens)
    - [B, L, C] (generic tokens)
    """

    def __init__(
        self,
        in_features: int,
        adapter_dim: int,
        scale: float = 1.0,
        dropout: float = 0.0,
        gate_noise_std: float = 0.0,
    ):
        super().__init__()
        self.adapter = AdapterLayer(in_features=in_features, adapter_dim=adapter_dim, scale=scale, dropout=dropout)
        self.gate = nn.Linear(in_features, 1)
        # Initialize gate to near-open (sigmoid(0)=0.5); training can learn stronger/ weaker gating.
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        self._gspo_active = False
        self._gate_noise_std = float(gate_noise_std)

    def set_gspo_mode(self, active: bool, gate_noise_std: Optional[float] = None, adapter_noise_std: Optional[float] = None):
        self._gspo_active = bool(active)
        if gate_noise_std is not None:
            self._gate_noise_std = float(gate_noise_std)
        # Propagate adapter exploration noise if requested
        self.adapter.set_gspo_mode(active=active, noise_std=adapter_noise_std)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            # [B,H,W,C] -> [B,C]
            return x.mean(dim=(1, 2))
        if x.dim() == 3:
            # [B,L,C] -> [B,C]
            return x.mean(dim=1)
        raise ValueError(f"Unsupported input shape for GatedAdapterLayer: {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self._pool(x)
        logits = self.gate(pooled)  # [B,1]
        if self._gspo_active and self.training and self._gate_noise_std and self._gate_noise_std > 0:
            logits = logits + torch.randn_like(logits) * self._gate_noise_std
        gate = torch.sigmoid(logits)  # [B,1]

        out = self.adapter(x)
        # Broadcast gate to match out shape
        while gate.dim() < out.dim():
            gate = gate.unsqueeze(1)
        return out * gate



class LoRALayer:
    """
    Abstract LoRA Layer.

    Parameters
    ----------
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = identity
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class IA3LoRALinear(nn.Linear, LoRALayer):
    """
    LoRA (low-rank adaptation) followed by (IA)^3 (weight rescaling) incorporated in a Linear Layer. Weights of Linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r=8,
        lora_alpha=8,
        lora_dropout: float = 0.0,
        fan_in_fan_out=False,
        merge_weights=False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        # In essence the $b$ parameter of LoRA.
        self.lora_b = nn.Parameter(torch.ones(out_features, 1))
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        self.fan_in_fan_out = fan_in_fan_out
        self.weight.requires_grad = False

        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.T(self.weight), bias=self.bias)
        if self.r > 0:
            result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

        hidden = result * self.lora_b.flatten()
        return hidden

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data /= self.lora_b.flatten()
                self.weight.data -= self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        # def T(w):
        #     return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += self.T(self.lora_B @ self.lora_A) * self.scaling
                self.weight.data *= self.lora_b.flatten()
            self.merged = True
        return hidden

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class IA3Linear(nn.Linear, LoRALayer):
    """
    (IA)^3 incorporated in a Linear Layer. Weights of Linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    scaling_rank
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Liu, Haokun and Tam, Derek and Muqeeth, Mohammed and Mohta, Jay and Huang, Tenghao and Bansal, Mohit and Raffel, Colin,
    "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning", 2022
    https://arxiv.org/pdf/2205.05638.pdf
    2. Code: https://github.com/r-three/t-few
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        merge_weights: False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self, r=4, lora_alpha=4, lora_dropout=0.0, merge_weights=merge_weights
        )  # Default arguments, only
        # In essence the $b$ parameter of LoRA.
        self.lora_b = nn.Parameter(torch.ones(out_features, 1))
        self.weight.requires_grad = False

    def forward(self, x: torch.Tensor):
        hidden = F.linear(x, self.weight, self.bias)
        hidden = hidden * self.lora_b.flatten()
        return hidden

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data /= self.lora_b.flatten()
            self.merged = False

    def eval(self):
        # def T(w):
        #     return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data *= self.lora_b.flatten()
            self.merged = True
        return hidden

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class LoRALinear(nn.Linear, LoRALayer):
    """
    LoRA incorporated in Linear Layer. Weights of linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # GSPO attention-LoRA extension (default off; activated by training loop via set_gspo_mode)
        self._gspo_active = False
        self._gspo_noise_std = 0.0
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        # def T(w):
        #     return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.T(self.weight), bias=self.bias)
            if self.r > 0:
                delta = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
                if self._gspo_active and self.training and self._gspo_noise_std and self._gspo_noise_std > 0:
                    delta = delta + torch.randn_like(delta) * self._gspo_noise_std
                result += delta
            return result
        else:
            return F.linear(x, self.T(self.weight), bias=self.bias)

    def set_gspo_mode(self, active: bool, noise_std: Optional[float] = None):
        """
        Enable/disable GSPO-specific behavior for this LoRA layer.
        No-op unless invoked by the GSPO training loop.
        """
        self._gspo_active = bool(active)
        if noise_std is not None:
            self._gspo_noise_std = float(noise_std)


class GatedLoRALinear(nn.Linear, LoRALayer):
    """
    LoRA linear layer with an input-conditioned gate that scales ONLY the LoRA delta branch.

    This is designed for GSPO rollouts:
    - When GSPO is active: add stochasticity to gate logits (exploration) and optionally to delta (fine-grained exploration).
    - When GSPO is inactive: deterministic gate (no sampling noise), behaves like standard LoRA with a learned per-sample scale.

    Supports x shapes:
    - [B, P, T, C] (SAM mask decoder attention format)
    - [B, T, C]
    - [B, C]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        gate_noise_std: float = 0.0,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.fan_in_fan_out = fan_in_fan_out

        # Gate network (policy-like scalar per sample)
        self.gate = nn.Linear(in_features, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        self._gspo_active = False
        self._gate_noise_std = float(gate_noise_std)
        self._delta_noise_std = 0.0

        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def set_gspo_mode(
        self,
        active: bool,
        gate_noise_std: Optional[float] = None,
        delta_noise_std: Optional[float] = None,
    ):
        self._gspo_active = bool(active)
        if gate_noise_std is not None:
            self._gate_noise_std = float(gate_noise_std)
        if delta_noise_std is not None:
            self._delta_noise_std = float(delta_noise_std)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            # [B,P,T,C] -> [B,C]
            return x.mean(dim=(1, 2))
        if x.dim() == 3:
            # [B,T,C] -> [B,C]
            return x.mean(dim=1)
        if x.dim() == 2:
            # [B,C]
            return x
        raise ValueError(f"Unsupported input shape for GatedLoRALinear: {tuple(x.shape)}")

    def forward(self, x: torch.Tensor):
        base = F.linear(x, self.T(self.weight), bias=self.bias)
        if self.r <= 0 or self.merged:
            return base

        delta = (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

        # Gate per sample
        pooled = self._pool(x)
        logits = self.gate(pooled)  # [B,1]
        if self._gspo_active and self.training and self._gate_noise_std and self._gate_noise_std > 0:
            logits = logits + torch.randn_like(logits) * self._gate_noise_std
        gate = torch.sigmoid(logits)  # [B,1]
        while gate.dim() < delta.dim():
            gate = gate.unsqueeze(1)

        if self._gspo_active and self.training and self._delta_noise_std and self._delta_noise_std > 0:
            delta = delta + torch.randn_like(delta) * self._delta_noise_std

        return base + delta * gate


class LoRAEmbedding(nn.Embedding, LoRALayer):
    """
    LoRA incorporated in Embedding Layer. Weights of embedding layer are set to be frozen per default.

    Parameters
    ----------
    num_embeddings
        size of the dictionary of embeddings.
    embedding_dim
         the size of each embedding vector.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0, merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(
                    x,
                    self.lora_A.T,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (after_A @ self.lora_B.T) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class LoRAMergedLinear(nn.Linear, LoRALayer):
    """
    LoRA where single nn.Linear represents more than one layer (used in some implementations of attention query/key/value projections). Weights of linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing
    r
        rank r of the low-rank decomposition
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout rate for LoRA
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out)
    merge_weights
        Merging weights during inference to reduce latency

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, "The length of enable_lora must divide out_features"
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(-1, self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
            self.merged = False

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data += self.zero_pad(T(delta_w * self.scaling))
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(
                    after_A.transpose(-2, -1), self.lora_B.unsqueeze(-1), groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result


class LoRAConv2d(nn.Conv2d, LoRALayer):
    """
    LoRA incorporated in 2d-Convolutional Layer. Weights of convolutional layer are set to be frozen per default.

    Parameters
    ----------
    in_channels
         Number of channels in the input image.
    out_channels
        Number of channels produced by the convolution.
    kernel_size
        Size of the convolving kernel.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Adding dropout to LoRA.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert type(kernel_size) is int
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * kernel_size, in_channels * kernel_size)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_channels * kernel_size, r * kernel_size)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = False

    def eval(self):
        nn.Conv2d.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.conv2d(
                x,
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return nn.Conv2d.forward(self, x)


class ConvLoRALinear(nn.Linear, LoRALayer):
    """
    Conv-LoRA incorporated in Linear Layer. Weights of linear layer are set to be frozen per default.

    Paper alignment (Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model, ICLR 2024):
    - Sec. 3 (Method): Replace standard LoRA's linear residual with a spatial, convolutional expert branch (MoE-Conv).
    - Sec. 3.2 (MoE-Conv & Gating): Noisy-TopK gating over GAP features, plus load-balancing auxiliary loss.
    - Sec. 3.3 (Integration): Inject at attention projections (q/k/v) of the vision encoder.
    - Sec. 4 (Training): Return moe_loss for aggregation with task loss.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.
    conv_lora_expert_num
        The number of experts in MoE-Conv.

    References
    ----------
    1. Zihan Zhong, Zhiqiang Tang, Tong He, Haoyang Fang, Chun Yuan,
    "Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model", 2024
    https://arxiv.org/abs/2401.17868
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        conv_lora_expert_num: Optional[int] = None,
        gspo_enabled: bool = False,
        gspo_group_size: int = 3,
        gspo_quality_momentum: float = 0.9,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        self.gspo_enabled = gspo_enabled
        # Actual trainable parameters
        if r > 0:
            # Sec. 3: Low-rank LoRA factors around a frozen base linear
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            # Sec. 3.2: MoE-Conv experts with Noisy-TopK gating
            # GSPO: Pass GSPO parameters to MoEGate
            topk = 1
            self.lora_moe_gating = MoEGate(
                M=conv_lora_expert_num, 
                d=self.r, 
                K=topk,
                gspo_enabled=gspo_enabled,
                gspo_group_size=gspo_group_size,
                gspo_quality_momentum=gspo_quality_momentum
            )
            self.lora_moe_experts = nn.ModuleList([])
            self.upsample_ratios = list(range(1, conv_lora_expert_num + 1))
            for upsample_ratio in self.upsample_ratios:
                expert = nn.Conv2d(in_channels=r, out_channels=r, kernel_size=3, stride=1, padding=1, bias=True)
                expert.bias.data.zero_()
                self.lora_moe_experts.append(nn.Sequential(expert, nn.GELU()))
            self.num_experts = conv_lora_expert_num
            self.multiply_by_gates = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def forward(self, x: torch.Tensor):
        # Base frozen path + residual (LoRA principle)
        result = F.linear(x, self.T(self.weight), bias=self.bias)
        moe_loss = 0
        selected_experts = None
        
        if self.r > 0:
            # 1) Project to rank-r LoRA space
            lora_res = self.lora_dropout(x) @ self.lora_A.T
            dim = lora_res.dim()
            if dim == 3:
                B, L, C = lora_res.size()
                H = W = int(math.sqrt(L))
                lora_res = lora_res.reshape(B, H, W, C)
            else:
                H, W = lora_res.size()[1:3]

            # Calculate the gating values.
            # 2) Sec. 3.2: Compute Noisy-TopK gates over spatial features; also returns auxiliary moe_loss.
            lora_res = lora_res.permute(0, 3, 1, 2).contiguous()
            
            # GSPO: Receive selected experts info
            if self.gspo_enabled:
                gates, moe_loss, selected_experts = self.lora_moe_gating(lora_res)
            else:
                gate_output = self.lora_moe_gating(lora_res)
                if len(gate_output) == 3:
                    gates, moe_loss, selected_experts = gate_output
                else:
                    gates, moe_loss = gate_output

            # Distribute data samples to experts.
            # 3) Route tokens to experts (SparseDispatcher)
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(lora_res)
            expert_outputs = []
            for i in range(self.num_experts):
                if len(expert_inputs[i]) == 0:
                    continue
                upsample_ratio = self.upsample_ratios[i]
                cur_res = expert_inputs[i]
                # Per-expert resolution change to diversify receptive fields (paper Sec. 3.2)
                if upsample_ratio != 1:
                    cur_res = F.interpolate(cur_res, scale_factor=upsample_ratio, mode="bicubic")
                cur_res = self.lora_moe_experts[i](cur_res)
                if upsample_ratio != 1:
                    cur_res = F.interpolate(cur_res, size=(int(H), int(W)), mode="bicubic")
                expert_outputs.append(cur_res)

            # Combine data samples after processing by each expert.
            # 4) Combine expert outputs (weighted by gates) back to the batch layout
            temp_lora_res = dispatcher.combine(expert_outputs, multiply_by_gates=self.multiply_by_gates)
            lora_res = lora_res + temp_lora_res

            lora_res = lora_res.permute(0, 2, 3, 1).contiguous()
            if dim == 3:
                lora_res = lora_res.reshape(B, L, C)
            # 5) Project back to output dim with LoRA scale (final residual add)
            result += (lora_res @ self.lora_B.T) * self.scaling

        # Return 3 values if GSPO is enabled, 2 values otherwise for backward compatibility
        if self.gspo_enabled and selected_experts is not None:
            return result, moe_loss, selected_experts
        else:
            return result, moe_loss


class MoEGate(nn.Module):
    def __init__(self, d, M=4, K=1, noisy_gating=True, gspo_enabled=False, gspo_group_size=3, gspo_quality_momentum=0.9):
        """Constructor
        Args:
            d: input channel dimensionality.
            M: the number of experts.
            K: the number of chosen experts for each forward pass.
            gspo_enabled: whether to enable GSPO (Group Sequence Policy Optimization) enhancements.
            gspo_group_size: number of experts to select when GSPO is enabled (overrides K).
            gspo_quality_momentum: momentum coefficient for expert quality history updates.
        """
        super(MoEGate, self).__init__()
        self.M = M
        self.gspo_enabled = gspo_enabled
        self.gspo_group_size = gspo_group_size
        self.gspo_quality_momentum = gspo_quality_momentum
        
        # GSPO: Use group_size as K when enabled, otherwise use original K
        self.k = gspo_group_size if gspo_enabled else K
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # global average pooling

        self.noisy_gating = noisy_gating

        self.w_gate = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(d, M), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        
        # GSPO: Expert quality tracking
        if gspo_enabled:
            self.register_buffer("expert_quality_history", torch.zeros(M))
            self.register_buffer("expert_usage_count", torch.zeros(M))
        
        assert self.k <= self.M

    def forward(self, feats, loss_coef=1e-2, noise_epsilon=1e-2):
        """
        Paper alignment (Sec. 3.2):
        - GAP over spatial features -> linear logits (+ noise during training) -> TopK expert selection
        - Softmax over TopK logits yields gates; compute importance/load stats and CV^2-based load-balancing loss
        
        GSPO enhancement:
        - Quality-aware bias added to logits based on expert historical performance
        - Supports multi-expert selection (TopK > 1) for group-level optimization
        """
        batch_size = feats.shape[0]

        feats_S = self.gap(feats).view(batch_size, -1)

        clean_logits = feats_S @ self.w_gate
        
        # GSPO: Add quality-aware bias to encourage high-performing experts
        if self.gspo_enabled and self.training:
            quality_bias = self._compute_quality_bias()
            clean_logits = clean_logits + quality_bias.unsqueeze(0)
        
        if self.noisy_gating and self.training:
            raw_noise_stddev = feats_S @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.M), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True).float()
        gates = zeros.scatter(1, top_k_indices, top_k_gates).to(logits.dtype)

        if self.noisy_gating and self.k < self.M and self.training:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)

        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        return gates, loss, top_k_indices

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    
    def _compute_quality_bias(self):
        """
        GSPO: Compute quality-aware bias for expert selection.
        
        This method balances exploitation (favoring high-quality experts) and
        exploration (giving chances to less-used experts).
        
        Returns:
            bias: Tensor of shape [M] to be added to logits
        """
        if not hasattr(self, 'expert_quality_history'):
            return torch.zeros(self.M, device=self.w_gate.device)
        
        # Normalize quality history to compute exploitation bias
        quality_history = self.expert_quality_history
        if quality_history.sum() > 0:
            quality_bias = (quality_history - quality_history.mean()) / (quality_history.std() + 1e-8)
        else:
            quality_bias = torch.zeros_like(quality_history)
        
        # Compute exploration bonus for less-used experts
        usage_count = self.expert_usage_count
        if usage_count.sum() > 0:
            usage_normalized = usage_count / (usage_count.sum() + 1e-8)
            exploration_bonus = 1.0 - usage_normalized
        else:
            exploration_bonus = torch.ones_like(usage_count)
        
        # Combine: 80% exploitation + 20% exploration
        combined_bias = 0.8 * quality_bias + 0.2 * exploration_bonus
        
        return combined_bias
    
    def update_quality_history(self, selected_experts, quality_scores):
        """
        GSPO: Update expert quality history based on actual performance.
        
        This provides the feedback loop for quality-aware gating.
        
        Args:
            selected_experts: Tensor of shape [batch, k] containing selected expert indices
            quality_scores: Tensor of shape [batch] containing quality metrics (e.g., IoU, DICE)
        """
        if not self.gspo_enabled or not hasattr(self, 'expert_quality_history'):
            return
        
        with torch.no_grad():
            for b in range(selected_experts.shape[0]):
                quality = quality_scores[b].item() if quality_scores[b].numel() == 1 else quality_scores[b].mean().item()
                for k in range(selected_experts.shape[1]):
                    expert_id = selected_experts[b, k].item()
                    # Momentum update
                    current_quality = self.expert_quality_history[expert_id]
                    self.expert_quality_history[expert_id] = (
                        self.gspo_quality_momentum * current_quality +
                        (1 - self.gspo_quality_momentum) * quality
                    )
                    self.expert_usage_count[expert_id] += 1


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.

    References
    ----------
    1. Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean,
    "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", 2017
    https://arxiv.org/abs/1701.06538
    2. Code: https://github.com/davidmrau/mixture-of-experts/blob/master/moe.py
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)  # [bs * num_of chosen experts, dim]
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            # stitched = stitched.mul(self._nonzero_gates)
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(-1).unsqueeze(-1))
        # zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size()[1],
            expert_out[-1].size()[2],
            expert_out[-1].size()[3],
            requires_grad=True,
            device=stitched.device,
        )
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

```

```worktree-python
# coding=utf-8
# Copyright 2023 The Meta AI Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch SAM model."""

import collections
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.sam.configuration_sam import (
    SamConfig,
    SamMaskDecoderConfig,
    SamPromptEncoderConfig,
    SamVisionConfig,
)
from transformers.utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from ..adaptation_layers import AdapterLayer, GatedAdapterLayer, GatedLoRALinear, LoRALinear

logger = logging.get_logger(__name__)


SAM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/sam-vit-huge",
    "facebook/sam-vit-large",
    "facebook/sam-vit-base",
    # See all SAM models at https://huggingface.co/models?filter=sam
]


@dataclass
class SamVisionEncoderOutput(ModelOutput):
    """
    Base class for sam vision model's outputs that also contains image embeddings obtained by applying the projection
    layer to the pooler_output.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    moe_loss: Optional[torch.FloatTensor] = None


@dataclass
class SamImageSegmentationOutput(ModelOutput):
    """
    Base class for Segment-Anything model's output

    Args:
        iou_scores (`torch.FloatTensor` of shape `(batch_size, num_masks)`):
            The iou scores of the predicted masks.
        pred_masks (`torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`):
            The predicted low resolutions masks. Needs to be post-processed by the processor
        vision_hidden_states  (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the vision model at the output of each layer plus the optional initial embedding outputs.
        vision_attentions  (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        mask_decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    iou_scores: torch.FloatTensor = None
    pred_masks: torch.FloatTensor = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = None
    mask_decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    vision_moe_loss: Optional[torch.FloatTensor] = None


class SamPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        embeddings = self.projection(pixel_values).permute(0, 2, 3, 1)
        return embeddings


class SamMLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lin1 = nn.Linear(config.hidden_size, config.mlp_dim)
        self.lin2 = nn.Linear(config.mlp_dim, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.lin1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.lin2(hidden_states)
        return hidden_states


# Copied from transformers.models.convnext.modeling_convnext.ConvNextLayerNorm with ConvNext->Sam
class SamLayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SamAttention(nn.Module):
    """
    SAM's attention layer that allows for downscaling the size of the embedding after projection to queries, keys, and
    values.
    
    Supports optional LoRA (Low-Rank Adaptation) on Q/K/V projection layers for parameter-efficient fine-tuning.
    """

    def __init__(self, config, downsample_rate=None, lora_r=0, lora_alpha=1, lora_dropout=0.0):
        """
        Parameters
        ----------
        config : SamMaskDecoderConfig
            Configuration for the attention layer
        downsample_rate : int, optional
            Downsample rate for attention (default: from config)
        lora_r : int, default=0
            LoRA rank. If 0, LoRA is disabled (default behavior for backward compatibility)
        lora_alpha : int, default=1
            LoRA scaling factor
        lora_dropout : float, default=0.0
            Dropout probability for LoRA
        """
        super().__init__()
        self.hidden_size = config.hidden_size

        downsample_rate = config.attention_downsample_rate if downsample_rate is None else downsample_rate

        self.internal_dim = config.hidden_size // downsample_rate
        self.num_attention_heads = config.num_attention_heads
        if self.internal_dim % config.num_attention_heads != 0:
            raise ValueError("num_attention_heads must divide hidden_size.")

        # Q/K/V projections: Use LoRA if lora_r > 0, otherwise use standard nn.Linear
        # GSPO-AttnLoRA (stage 2): optionally use gated LoRA to create policy-like per-sample scaling of the LoRA delta.
        lora_gate_enabled = getattr(config, "decoder_attention_lora_gate_enabled", False)
        lora_gate_noise_std = getattr(config, "decoder_attention_lora_gate_noise_std", 0.0)
        if lora_r > 0:
            LinearCls = GatedLoRALinear if lora_gate_enabled else LoRALinear
            self.q_proj = LinearCls(
                self.hidden_size,
                self.internal_dim,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                gate_noise_std=lora_gate_noise_std,
            )
            self.k_proj = LinearCls(
                self.hidden_size,
                self.internal_dim,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                gate_noise_std=lora_gate_noise_std,
            )
            self.v_proj = LinearCls(
                self.hidden_size,
                self.internal_dim,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                gate_noise_std=lora_gate_noise_std,
            )
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.internal_dim)
            self.k_proj = nn.Linear(self.hidden_size, self.internal_dim)
            self.v_proj = nn.Linear(self.hidden_size, self.internal_dim)
        
        # Output projection: standard Linear (LoRA not applied here per design)
        self.out_proj = nn.Linear(self.internal_dim, self.hidden_size)

    def _separate_heads(self, hidden_states: Tensor, num_attention_heads: int) -> Tensor:
        batch, point_batch_size, n_tokens, channel = hidden_states.shape
        c_per_head = channel // num_attention_heads
        hidden_states = hidden_states.reshape(batch * point_batch_size, n_tokens, num_attention_heads, c_per_head)
        return hidden_states.transpose(1, 2)

    def _recombine_heads(self, hidden_states: Tensor, point_batch_size: int) -> Tensor:
        batch, n_heads, n_tokens, c_per_head = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states.reshape(batch // point_batch_size, point_batch_size, n_tokens, n_heads * c_per_head)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_similarity: Tensor = None) -> Tensor:
        # Input projections
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        point_batch_size = query.shape[1]
        # Separate into heads
        query = self._separate_heads(query, self.num_attention_heads)
        key = self._separate_heads(key, self.num_attention_heads)
        value = self._separate_heads(value, self.num_attention_heads)

        # SamAttention
        _, _, _, c_per_head = query.shape
        attn = query @ key.permute(0, 1, 3, 2)  # batch_size * point_batch_size  x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        if attention_similarity is not None:
            attn = attn + attention_similarity
            attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ value
        out = self._recombine_heads(out, point_batch_size)
        out = self.out_proj(out)

        return out


class SamTwoWayAttentionBlock(nn.Module):
    def __init__(self, config, attention_downsample_rate: int = 2, skip_first_layer_pe: bool = False,
                 lora_r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0.0):
        """
        A transformer block with four layers:
            (1) self-attention of sparse inputs (2) cross attention of sparse inputs -> dense inputs (3) mlp block on
            sparse inputs (4) cross attention of dense inputs -> sparse inputs

        Arguments:
            config (`SamMaskDecoderConfig`):
                The configuration file used to instantiate the block
            attention_downsample_rate (*optional*, int, defaults to 2):
                The downsample ratio of the block used to reduce the inner dim of the attention.
            skip_first_layer_pe (*optional*, bool, defaults to `False`):
                Whether or not to skip the addition of the query_point_embedding on the first layer.
            lora_r (int, default=0):
                LoRA rank for attention layers. If 0, LoRA is disabled.
            lora_alpha (int, default=1):
                LoRA scaling factor
            lora_dropout (float, default=0.0):
                Dropout probability for LoRA
        """
        super().__init__()

        self.hidden_size = config.hidden_size
        self.layer_norm_eps = config.layer_norm_eps

        self.self_attn = SamAttention(config, downsample_rate=1, 
                                      lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.cross_attn_token_to_image = SamAttention(config, downsample_rate=attention_downsample_rate,
                                                      lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.mlp = SamMLPBlock(config)
        self.layer_norm3 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        
        # Decoder MLP-Adapter on queries (optional, config-driven)
        self.adapter_queries = None
        # Prefer explicit decoder-specific flags, but fall back to generic adapter flags if provided
        decoder_adapter_enabled = getattr(config, "decoder_adapter_enabled", False)
        decoder_adapter_dim = getattr(config, "decoder_adapter_dim", getattr(config, "adapter_dim", 64))
        if decoder_adapter_enabled:
            self.adapter_queries = AdapterLayer(self.hidden_size, decoder_adapter_dim)

        self.layer_norm4 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.cross_attn_image_to_token = SamAttention(config, downsample_rate=attention_downsample_rate,
                                                      lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_point_embedding: Tensor,
        key_point_embedding: Tensor,
        attention_similarity: Tensor,
        output_attentions: bool = False,
    ):
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(query=queries, key=queries, value=queries)
        else:
            query = queries + query_point_embedding
            attn_out = self.self_attn(query=query, key=query, value=queries)
            queries = queries + attn_out
        queries = self.layer_norm1(queries)

        # Cross attention block, tokens attending to image embedding
        query = queries + query_point_embedding
        key = keys + key_point_embedding

        attn_out = self.cross_attn_token_to_image(
            query=query, key=key, value=keys, attention_similarity=attention_similarity
        )
        queries = queries + attn_out

        queries = self.layer_norm2(queries)

        # MLP block + optional parallel Adapter on queries
        mlp_out = self.mlp(queries)
        if self.adapter_queries is not None:
            adapter_out = self.adapter_queries(queries)
            queries = queries + mlp_out + adapter_out
        else:
            queries = queries + mlp_out
        queries = self.layer_norm3(queries)

        # Cross attention block, image embedding attending to tokens
        query = queries + query_point_embedding
        key = keys + key_point_embedding

        attn_out = self.cross_attn_image_to_token(query=key, key=query, value=queries)
        keys = keys + attn_out

        keys = self.layer_norm4(keys)

        outputs = (queries, keys)

        if output_attentions:
            outputs = outputs + (attn_out,)
        else:
            outputs = outputs + (None,)

        return outputs


class SamTwoWayTransformer(nn.Module):
    def __init__(self, config: SamMaskDecoderConfig, lora_r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0.0):
        super().__init__()
        self.config = config

        self.num_hidden_layers = config.num_hidden_layers
        self.layers = nn.ModuleList()

        for i in range(self.num_hidden_layers):
            self.layers.append(SamTwoWayAttentionBlock(config, skip_first_layer_pe=(i == 0),
                                                       lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout))

        self.final_attn_token_to_image = SamAttention(config, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.layer_norm_final_attn = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        point_embeddings: Tensor,
        image_embeddings: Tensor,
        image_positional_embeddings: Tensor,
        attention_similarity: Tensor,
        target_embedding=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        all_attentions = ()

        if image_embeddings is None:
            raise ValueError("You have to specify an image_embedding")

        image_embeddings = image_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)
        image_positional_embeddings = image_positional_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)

        # Prepare queries
        queries = point_embeddings
        keys = image_embeddings

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            if target_embedding is not None:
                queries += target_embedding

            queries, keys, attention_outputs = layer(
                queries=queries,
                keys=keys,
                query_point_embedding=point_embeddings,
                key_point_embedding=image_positional_embeddings,
                attention_similarity=attention_similarity,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_attentions = all_attentions + (attention_outputs,)

        # Apply the final attenion layer from the points to the image
        query = queries + point_embeddings
        key = keys + image_positional_embeddings

        attn_out = self.final_attn_token_to_image(query=query, key=key, value=keys)

        queries = queries + attn_out
        queries = self.layer_norm_final_attn(queries)
        return queries, keys, all_attentions


class SamFeedForward(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.activation = nn.ReLU()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        self.sigmoid_output = sigmoid_output

    def forward(self, hidden_states):
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.activation(hidden_states)
        for layer in self.layers:
            hidden_states = self.activation(layer(hidden_states))

        hidden_states = self.proj_out(hidden_states)
        if self.sigmoid_output:
            hidden_states = F.sigmoid(hidden_states)
        return hidden_states


class SamMaskDecoder(nn.Module):
    def __init__(self, config: SamMaskDecoderConfig, lora_r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0.0):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.num_multimask_outputs = config.num_multimask_outputs
        self.num_mask_tokens = config.num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, self.hidden_size)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.hidden_size)

        self.transformer = SamTwoWayTransformer(config, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        # should we create a new class for this?
        self.upscale_conv1 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(self.hidden_size // 4, self.hidden_size // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = SamLayerNorm(self.hidden_size // 4, data_format="channels_first")
        self.activation = nn.GELU()

        mlps_list = []
        for _ in range(self.num_mask_tokens):
            mlps_list += [SamFeedForward(self.hidden_size, self.hidden_size, self.hidden_size // 8, 3)]
        self.output_hypernetworks_mlps = nn.ModuleList(mlps_list)

        self.iou_prediction_head = SamFeedForward(
            self.hidden_size, config.iou_head_hidden_dim, self.num_mask_tokens, config.iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        output_attentions: Optional[bool] = None,
        attention_similarity: torch.Tensor = None,
        target_embedding: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (`torch.Tensor`):
                the embeddings from the image encoder
            image_positional_embedding (`torch.Tensor`):
                positional encoding with the shape of image_embeddings
            sparse_prompt_embeddings (`torch.Tensor`):
                The embeddings of the points and boxes
            dense_prompt_embeddings (`torch.Tensor`):
                the embeddings of the mask inputs
            multimask_output (bool):
                Whether to return multiple masks or a single mask.
            output_attentions (bool, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        if sparse_prompt_embeddings.sum().item() != 0:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.iou_token.weight.dtype)

        # Expand per-image data in batch direction to be per-point
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat(point_batch_size, 1, 1, 1)
        image_positional_embeddings = image_positional_embeddings.repeat(point_batch_size, 1, 1, 1)

        # Run the transformer, image_positional_embedding are consumed
        point_embedding, image_embeddings, attentions = self.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )

        upscaled_embedding = self.upscale_conv1(image_embeddings)
        upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)

        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
        masks = (hyper_in @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, height, width)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]
        iou_pred = iou_pred[:, :, mask_slice]

        outputs = (masks, iou_pred)

        if output_attentions:
            outputs = outputs + (attentions,)
        else:
            outputs = outputs + (None,)

        return outputs


class SamPositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = config.hidden_size // 2
        self.register_buffer("positional_embedding", self.scale * torch.randn((2, config.num_pos_feats)))

    def forward(self, input_coords, input_shape=None):
        """Positionally encode points that are normalized to [0,1]."""
        coordinates = input_coords.clone()

        if input_shape is not None:
            coordinates[:, :, :, 0] = coordinates[:, :, :, 0] / input_shape[1]
            coordinates[:, :, :, 1] = coordinates[:, :, :, 1] / input_shape[0]

        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coordinates = 2 * coordinates - 1
        coordinates = coordinates.to(self.positional_embedding.dtype)
        coordinates = coordinates @ self.positional_embedding
        coordinates = 2 * np.pi * coordinates
        # outputs d_1 x ... x d_n x channel shape
        return torch.cat([torch.sin(coordinates), torch.cos(coordinates)], dim=-1)


class SamMaskEmbedding(nn.Module):
    def __init__(self, config: SamPromptEncoderConfig):
        super().__init__()
        self.mask_input_channels = config.mask_input_channels // 4
        self.activation = ACT2FN[config.hidden_act]
        self.conv1 = nn.Conv2d(1, self.mask_input_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.mask_input_channels, config.mask_input_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(config.mask_input_channels, config.hidden_size, kernel_size=1)
        self.layer_norm1 = SamLayerNorm(
            self.mask_input_channels, eps=config.layer_norm_eps, data_format="channels_first"
        )
        self.layer_norm2 = SamLayerNorm(
            self.mask_input_channels * 4, eps=config.layer_norm_eps, data_format="channels_first"
        )

    def forward(self, masks):
        hidden_states = self.conv1(masks)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.activation(hidden_states)
        dense_embeddings = self.conv3(hidden_states)
        return dense_embeddings


class SamPromptEncoder(nn.Module):
    def __init__(self, config: SamPromptEncoderConfig, shared_patch_embedding):
        super().__init__()
        self.shared_embedding = shared_patch_embedding
        self.mask_embed = SamMaskEmbedding(config)
        self.no_mask_embed = nn.Embedding(1, config.hidden_size)

        self.image_embedding_size = (config.image_embedding_size, config.image_embedding_size)
        self.input_image_size = config.image_size

        self.point_embed = nn.ModuleList(
            [nn.Embedding(1, config.hidden_size) for i in range(config.num_point_embeddings)]
        )
        self.hidden_size = config.hidden_size
        self.not_a_point_embed = nn.Embedding(1, config.hidden_size)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            target_point_shape = (points.shape[0], points.shape[1], 1, points.shape[-1])
            target_labels_shape = (points.shape[0], points.shape[1], 1)
            padding_point = torch.zeros(target_point_shape, device=points.device)
            padding_label = -torch.ones(target_labels_shape, device=labels.device)
            points = torch.cat([points, padding_point], dim=2)
            labels = torch.cat([labels, padding_label], dim=2)
        input_shape = (self.input_image_size, self.input_image_size)
        point_embedding = self.shared_embedding(points, input_shape)

        # torch.where and expanding the labels tensor is required by the ONNX export
        point_embedding = torch.where(labels[..., None] == -1, self.not_a_point_embed.weight, point_embedding)

        # This is required for the ONNX export. The dtype, device need to be explicitly
        # specified as otherwise torch.onnx.export interprets as double
        point_embedding = torch.where(
            labels[..., None] != -10,
            point_embedding,
            torch.tensor(0.0, dtype=point_embedding.dtype, device=point_embedding.device),
        )

        point_embedding = torch.where(
            (labels == 0)[:, :, :, None],
            point_embedding + self.point_embed[0].weight[None, None, :, :],
            point_embedding,
        )

        point_embedding = torch.where(
            (labels == 1)[:, :, :, None],
            point_embedding + self.point_embed[1].weight[None, None, :, :],
            point_embedding,
        )

        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        batch_size, nb_boxes = boxes.shape[:2]
        coords = boxes.reshape(batch_size, nb_boxes, 2, 2)
        input_shape = (self.input_image_size, self.input_image_size)
        corner_embedding = self.shared_embedding(coords, input_shape)
        corner_embedding[:, :, 0, :] += self.point_embed[2].weight
        corner_embedding[:, :, 1, :] += self.point_embed[3].weight
        return corner_embedding

    def forward(
        self,
        input_points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        input_labels: Optional[torch.Tensor],
        input_boxes: Optional[torch.Tensor],
        input_masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Args:
            points (`torch.Tensor`, *optional*):
                point coordinates and labels to embed.
            boxes (`torch.Tensor`, *optional*):
                boxes to embed
            masks (`torch.Tensor`, *optional*):
                masks to embed
        """
        sparse_embeddings = None
        batch_size = 1
        target_device = self.shared_embedding.positional_embedding.device
        if input_points is not None:
            batch_size, point_batch_size = input_points.shape[:2]
            if input_labels is None:
                raise ValueError("If points are provided, labels must also be provided.")
            point_embeddings = self._embed_points(input_points, input_labels, pad=(input_boxes is None))
            sparse_embeddings = point_embeddings
        if input_boxes is not None:
            batch_size = input_boxes.shape[0]
            box_embeddings = self._embed_boxes(input_boxes)
            if sparse_embeddings is None:
                sparse_embeddings = box_embeddings
            else:
                sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=2)
        if input_masks is not None:
            dense_embeddings = self.mask_embed(input_masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch_size, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        if sparse_embeddings is None:
            sparse_embeddings = torch.zeros((batch_size, 1, 1, self.hidden_size), device=target_device)

        return sparse_embeddings, dense_embeddings


class SamVisionAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(self, config, window_size):
        super().__init__()
        input_size = (
            (config.image_size // config.patch_size, config.image_size // config.patch_size)
            if window_size == 0
            else (window_size, window_size)
        )

        self.num_attention_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.scale = head_dim**-0.5
        self.dropout = config.attention_dropout

        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.use_rel_pos = config.use_rel_pos
        if self.use_rel_pos:
            if input_size is None:
                raise ValueError("Input size must be provided if using relative positional encoding.")

            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.

        Args:
            q_size (int):
                size of the query.
            k_size (int):
                size of key k.
            rel_pos (`torch.Tensor`):
                relative position embeddings (L, channel).

        Returns:
            Extracted positional embeddings according to relative positions.
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)

        # Scale the coords with short length if shapes for q and k are different.
        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.long()]

    def add_decomposed_rel_pos(
        self,
        attn: torch.Tensor,
        query: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

        Args:
            attn (`torch.Tensor`):
                attention map.
            query (`torch.Tensor`):
                query q in the attention layer with shape (batch_size, query_height * query_width, channel).
            rel_pos_h (`torch.Tensor`):
                relative position embeddings (Lh, channel) for height axis.
            rel_pos_w (`torch.Tensor`):
                relative position embeddings (Lw, channel) for width axis.
            q_size (tuple):
                spatial sequence size of query q with (query_height, query_width).
            k_size (tuple):
                spatial sequence size of key k with (key_height, key_width).

        Returns:
            attn (`torch.Tensor`):
                attention map with added relative positional embeddings.
        """
        query_height, query_width = q_size
        key_height, key_width = k_size
        relative_position_height = self.get_rel_pos(query_height, key_height, rel_pos_h)
        relative_position_width = self.get_rel_pos(query_width, key_width, rel_pos_w)

        batch_size, _, dim = query.shape
        reshaped_query = query.reshape(batch_size, query_height, query_width, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", reshaped_query, relative_position_height)
        rel_w = torch.einsum("bhwc,wkc->bhwk", reshaped_query, relative_position_width)
        attn = attn.reshape(batch_size, query_height, query_width, key_height, key_width)
        attn = attn + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        attn = attn.reshape(batch_size, query_height * query_width, key_height * key_width)
        return attn

    def forward(self, hidden_states: torch.Tensor, output_attentions=False, output_moe_loss=False) -> torch.Tensor:
        batch_size, height, width, _ = hidden_states.shape
        # qkv with shape (3, batch_size, nHead, height * width, channel)
        ##### modify here for conv-lora
        # Paper alignment:
        # - Sec. 3.3 (Integration): Inject Conv-LoRA at attention projections (q/k/v).
        #   The adapted linear layer (possibly Conv-LoRA) may return (tensor, moe_loss, selected_experts) when output_moe_loss=True.
        qkv = self.qkv(hidden_states)
        if output_moe_loss:
            # Handle both 2-value and 3-value returns for backward compatibility
            if isinstance(qkv, tuple) and len(qkv) == 3:
                qkv, moe_loss, selected_experts = qkv
            else:
                qkv, moe_loss = qkv
        qkv = qkv.reshape(batch_size, height * width, 3, self.num_attention_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (batch_size * nHead, height * width, channel)
        query, key, value = qkv.reshape(3, batch_size * self.num_attention_heads, height * width, -1).unbind(0)

        attn_weights = (query * self.scale) @ key.transpose(-2, -1)

        if self.use_rel_pos:
            attn_weights = self.add_decomposed_rel_pos(
                attn_weights, query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = (attn_probs @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)

        attn_output = self.proj(attn_output)

        if output_attentions:
            outputs = (attn_output, attn_weights)
        else:
            outputs = (attn_output, None)
        if output_moe_loss:
            # Sec. 4: propagate MoE load-balancing loss upward for aggregation
            outputs += (moe_loss,)
        return outputs


class SamVisionLayer(nn.Module):
    def __init__(self, config, window_size):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = SamVisionAttention(config, window_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SamMLPBlock(config)
        self.window_size = window_size
        
        # Adapter Integration
        self.adapter = None
        if hasattr(config, "adapter_enabled") and config.adapter_enabled:
            adapter_dim = getattr(config, "adapter_dim", 64)
            # Optional GSPO encoder adapter policy (gated adapter). Decoder adapters are handled elsewhere.
            encoder_gate_enabled = getattr(config, "encoder_adapter_gate_enabled", False)
            encoder_gate_noise_std = getattr(config, "encoder_adapter_gate_noise_std", 0.0)
            if encoder_gate_enabled:
                self.adapter = GatedAdapterLayer(
                    in_features=config.hidden_size,
                    adapter_dim=adapter_dim,
                    gate_noise_std=encoder_gate_noise_std,
                )
            else:
                self.adapter = AdapterLayer(config.hidden_size, adapter_dim)


    def window_partition(self, hidden_states: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
        Partition into non-overlapping windows with padding if needed.
            hidden_states (tensor): input tokens with [batch_size, height, width, channel]. window_size (int): window
            size.

        Returns:
            windows: windows after partition with [batch_size * num_windows, window_size, window_size, channel].
            (pad_height, pad_width): padded height and width before partition
        """
        batch_size, height, width, channel = hidden_states.shape

        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_w, 0, pad_h))
        pad_height, pad_width = height + pad_h, width + pad_w

        hidden_states = hidden_states.reshape(
            batch_size, pad_height // window_size, window_size, pad_width // window_size, window_size, channel
        )
        windows = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, window_size, window_size, channel)
        return windows, (pad_height, pad_width)

    def window_unpartition(
        self, windows: torch.Tensor, window_size: int, padding_shape: Tuple[int, int], original_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Args:
        Window unpartition into original sequences and removing padding.
            hidden_states (tensor):
                input tokens with [batch_size * num_windows, window_size, window_size, channel].
            window_size (int):
                window size.
            padding_shape (Tuple):
                padded height and width (pad_height, pad_width).
            original_shape (Tuple): original height and width (height, width) before padding.

        Returns:
            hidden_states: unpartitioned sequences with [batch_size, height, width, channel].
        """
        pad_height, pad_width = padding_shape
        height, width = original_shape
        batch_size = windows.shape[0] // (pad_height * pad_width // window_size // window_size)
        hidden_states = windows.reshape(
            batch_size, pad_height // window_size, pad_width // window_size, window_size, window_size, -1
        )
        hidden_states = (
            hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(batch_size, pad_height, pad_width, -1)
        )

        hidden_states = hidden_states[:, :height, :width, :].contiguous()
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
        output_moe_loss: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        # Window partition
        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, padding_shape = self.window_partition(hidden_states, self.window_size)

        ### modify here
        attn_outputs = self.attn(
            hidden_states=hidden_states, output_attentions=output_attentions, output_moe_loss=output_moe_loss
        )
        if output_moe_loss:
            hidden_states, attn_weights, moe_loss = attn_outputs
        else:
            hidden_states, attn_weights = attn_outputs
        # Reverse window partition
        if self.window_size > 0:
            hidden_states = self.window_unpartition(hidden_states, self.window_size, padding_shape, (height, width))


        hidden_states = residual + hidden_states
        
        # MLP Block + Parallel Adapter
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        mlp_out = self.mlp(hidden_states)
        
        if self.adapter is not None:
            adapter_out = self.adapter(hidden_states)
            hidden_states = residual + mlp_out + adapter_out
        else:
            hidden_states = residual + mlp_out


        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        else:
            outputs += (None,)  # keep attn_weights on the same place
        if output_moe_loss:
            # Sec. 4: return per-layer MoE loss to be summed in encoder
            outputs += (moe_loss,)

        return outputs


class SamVisionNeck(nn.Module):
    def __init__(self, config: SamVisionConfig):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv2d(config.hidden_size, config.output_channels, kernel_size=1, bias=False)
        self.layer_norm1 = SamLayerNorm(config.output_channels, data_format="channels_first")
        self.conv2 = nn.Conv2d(config.output_channels, config.output_channels, kernel_size=3, padding=1, bias=False)
        self.layer_norm2 = SamLayerNorm(config.output_channels, data_format="channels_first")

    def forward(self, hidden_states):
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.layer_norm1(hidden_states)

        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        return hidden_states


class SamVisionEncoder(nn.Module):
    def __init__(self, config: SamVisionConfig):
        super().__init__()
        self.config = config
        self.image_size = config.image_size

        self.patch_embed = SamPatchEmbeddings(config)

        self.pos_embed = None
        if config.use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    config.image_size // config.patch_size,
                    config.image_size // config.patch_size,
                    config.hidden_size,
                )
            )

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = SamVisionLayer(
                config,
                window_size=config.window_size if i not in config.global_attn_indexes else 0,
            )
            self.layers.append(layer)

        self.neck = SamVisionNeck(config)

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.patch_embed

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_moe_loss: Optional[bool] = None,  # MoE loss for Conv-LoRA
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SamVisionEncoderOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_moe_loss = 0.0 if output_moe_loss else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, output_attentions=output_attentions, output_moe_loss=output_moe_loss
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if output_moe_loss:
                # Accumulate MoE load-balancing losses across layers (Sec. 4)
                all_moe_loss = all_moe_loss + layer_outputs[2]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.neck(hidden_states)

        if not return_dict:
            outputs = (
                hidden_states,
                all_hidden_states,
                all_self_attentions,
                all_moe_loss,
            )
            # if output_hidden_states:
            #     outputs = outputs + (all_hidden_states,)
            # if output_attentions:
            #     outputs = outputs + (all_self_attentions,)
            # if output_moe_loss:
            #     outputs = outputs + (all_moe_loss,)
            return outputs

        return SamVisionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            moe_loss=all_moe_loss,  # modify here
        )


class SamPreTrainedModel(PreTrainedModel):
    config_class = SamConfig
    base_model_prefix = "sam"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


SAM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SamConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


SAM_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`SamProcessor`]. See [`SamProcessor.__call__`] for
            details.
        input_points (`torch.FloatTensor` of shape `(batch_size, num_points, 2)`):
            Input 2D spatial points, this is used by the prompt encoder to encode the prompt. Generally yields to much
            better results. The points can be obtained by passing a list of list of list to the processor that will
            create corresponding `torch` tensors of dimension 4. The first dimension is the image batch size, the
            second dimension is the point batch size (i.e. how many segmentation masks do we want the model to predict
            per input point), the third dimension is the number of points per segmentation mask (it is possible to pass
            multiple points for a single mask), and the last dimension is the x (vertical) and y (horizontal)
            coordinates of the point. If a different number of points is passed either for each image, or for each
            mask, the processor will create "PAD" points that will correspond to the (0, 0) coordinate, and the
            computation of the embedding will be skipped for these points using the labels.
        input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points)`):
            Input labels for the points, this is used by the prompt encoder to encode the prompt. According to the
            official implementation, there are 3 types of labels

            - `1`: the point is a point that contains the object of interest
            - `0`: the point is a point that does not contain the object of interest
            - `-1`: the point corresponds to the background

            We added the label:

            - `-10`: the point is a padding point, thus should be ignored by the prompt encoder

            The padding labels should be automatically done by the processor.
        input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes, 4)`):
            Input boxes for the points, this is used by the prompt encoder to encode the prompt. Generally yields to
            much better generated masks. The boxes can be obtained by passing a list of list of list to the processor,
            that will generate a `torch` tensor, with each dimension corresponding respectively to the image batch
            size, the number of boxes per image and the coordinates of the top left and bottom right point of the box.
            In the order (`x1`, `y1`, `x2`, `y2`):

            - `x1`: the x coordinate of the top left point of the input box
            - `y1`: the y coordinate of the top left point of the input box
            - `x2`: the x coordinate of the bottom right point of the input box
            - `y2`: the y coordinate of the bottom right point of the input box

        input_masks (`torch.FloatTensor` of shape `(batch_size, image_size, image_size)`):
            SAM model also accepts segmentation masks as input. The mask will be embedded by the prompt encoder to
            generate a corresponding embedding, that will be fed later on to the mask decoder. These masks needs to be
            manually fed by the user, and they need to be of shape (`batch_size`, `image_size`, `image_size`).

        image_embeddings (`torch.FloatTensor` of shape `(batch_size, output_channels, window_size, window_size)`):
            Image embeddings, this is used by the mask decder to generate masks and iou scores. For more memory
            efficient computation, users can first retrieve the image embeddings using the `get_image_embeddings`
            method, and then feed them to the `forward` method instead of feeding the `pixel_values`.
        multimask_output (`bool`, *optional*):
            In the original implementation and paper, the model always outputs 3 masks per image (or per point / per
            bounding box if relevant). However, it is possible to just output a single mask, that corresponds to the
            "best" mask, by specifying `multimask_output=False`.
        attention_similarity (`torch.FloatTensor`, *optional*):
            Attention similarity tensor, to be provided to the mask decoder for target-guided attention in case the
            model is used for personalization as introduced in [PerSAM](https://arxiv.org/abs/2305.03048).
        target_embedding (`torch.FloatTensor`, *optional*):
            Embedding of the target concept, to be provided to the mask decoder for target-semantic prompting in case
            the model is used for personalization as introduced in [PerSAM](https://arxiv.org/abs/2305.03048).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "Segment Anything Model (SAM) for generating segmentation masks, given an input image and ",
    " optional 2D location and bounding boxes.",
    SAM_START_DOCSTRING,
)
class SamModel(SamPreTrainedModel):
    _tied_weights_keys = ["prompt_encoder.shared_embedding.positional_embedding"]

    def __init__(self, config):
        super().__init__(config)
        self.shared_image_embedding = SamPositionalEmbedding(config.vision_config)

        self.vision_encoder = SamVisionEncoder(config.vision_config)
        self.prompt_encoder = SamPromptEncoder(config.prompt_encoder_config, self.shared_image_embedding)
        
        # Extract decoder attention LoRA parameters from config (if available)
        decoder_attn_lora_r = getattr(config.mask_decoder_config, 'decoder_attention_lora_r', 0)
        decoder_attn_lora_alpha = getattr(config.mask_decoder_config, 'decoder_attention_lora_alpha', 1)
        decoder_attn_lora_dropout = getattr(config.mask_decoder_config, 'decoder_attention_lora_dropout', 0.0)
        
        self.mask_decoder = SamMaskDecoder(
            config.mask_decoder_config,
            lora_r=decoder_attn_lora_r,
            lora_alpha=decoder_attn_lora_alpha,
            lora_dropout=decoder_attn_lora_dropout
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.vision_encoder.get_input_embeddings()

    def get_image_wide_positional_embeddings(self):
        size = self.config.prompt_encoder_config.image_embedding_size
        target_device = self.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width

    @torch.no_grad()
    def get_image_embeddings(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns the image embeddings by passing the pixel values through the vision encoder.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Input pixel values
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        vision_output = self.vision_encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeddings = vision_output[0]
        return image_embeddings

    @torch.no_grad()
    def get_prompt_embeddings(
        self,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
    ):
        r"""
        Returns the prompt embeddings by passing the input points, labels, boxes and masks through the prompt encoder.

        Args:
            input_points (`torch.FloatTensor` of shape `(batch_size, point_batch_size, num_points_per_image, 2)`):
                Optional input points for the prompt encoder. The padding of the point is automatically done by the
                processor. `point_batch_size` refers to the number of masks that we want the model to predict per
                point. The model will output `point_batch_size` times 3 masks in total.
            input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points_per_image)`):
                Optional input labels for the prompt encoder. The padding of the labels is automatically done by the
                processor, or can be fed by the user.
            input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes_per_image, 4)`):
                Optional input boxes for the prompt encoder. The padding of the boxes is automatically done by the
                processor. users can also pass manually the input boxes.
            input_masks (`torch.LongTensor` of shape `(batch_size, image_size, image_size)`):
                Optional input masks for the prompt encoder.
        """
        prompt_output = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
        return prompt_output

    @add_start_docstrings_to_model_forward(SAM_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        multimask_output: bool = True,
        attention_similarity: Optional[torch.FloatTensor] = None,
        target_embedding: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_moe_loss: Optional[bool] = None,  # MoE loss for Conv-LoRA
        return_dict=None,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        r"""
        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoModel, AutoProcessor

        >>> model = AutoModel.from_pretrained("facebook/sam-vit-base")
        >>> processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")

        >>> img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-car.png"
        >>> raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        >>> input_points = [[[400, 650]]]  # 2D location of a window on the car
        >>> inputs = processor(images=raw_image, input_points=input_points, return_tensors="pt")

        >>> # Get segmentation mask
        >>> outputs = model(**inputs)

        >>> # Postprocess masks
        >>> masks = processor.post_process_masks(
        ...     outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        ... )
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and image_embeddings is None:
            raise ValueError("Either pixel_values or image_embeddings must be provided.")

        if pixel_values is not None and image_embeddings is not None:
            raise ValueError("Only one of pixel_values and image_embeddings can be provided.")

        if input_points is not None and len(input_points.shape) != 4:
            raise ValueError(
                "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.",
                " got {}.".format(input_points.shape),
            )
        if input_boxes is not None and len(input_boxes.shape) != 3:
            raise ValueError(
                "The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.",
                " got {}.".format(input_boxes.shape),
            )
        if input_points is not None and input_boxes is not None:
            point_batch_size = input_points.shape[1]
            box_batch_size = input_boxes.shape[1]
            if point_batch_size != box_batch_size:
                raise ValueError(
                    "You should provide as many bounding boxes as input points per box. Got {} and {}.".format(
                        point_batch_size, box_batch_size
                    )
                )

        image_positional_embeddings = self.get_image_wide_positional_embeddings()
        # repeat with batch size
        batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        vision_attentions = None
        vision_hidden_states = None
        vision_moe_loss = None

        if pixel_values is not None:
            vision_outputs = self.vision_encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_moe_loss=output_moe_loss,
                return_dict=return_dict,
            )
            image_embeddings = vision_outputs[0]

            if output_hidden_states:
                vision_hidden_states = vision_outputs[1]
            if output_attentions:
                # vision_attentions = vision_outputs[-1]
                vision_attentions = vision_outputs[2]
            if output_moe_loss:
                vision_moe_loss = vision_outputs[-1]

        if input_points is not None and input_labels is None:
            input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

        if input_points is not None and image_embeddings.shape[0] != input_points.shape[0]:
            raise ValueError(
                "The batch size of the image embeddings and the input points must be the same. ",
                "Got {} and {} respectively.".format(image_embeddings.shape[0], input_points.shape[0]),
                " if you want to pass multiple points for the same image, make sure that you passed ",
                " input_points of shape (batch_size, point_batch_size, num_points_per_image, 3) and ",
                " input_labels of shape (batch_size, point_batch_size, num_points_per_image)",
            )

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )

        low_res_masks, iou_predictions, mask_decoder_attentions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )

        if not return_dict:
            # output = (iou_predictions, low_res_masks)
            # if output_hidden_states:
            #     output = output + (vision_hidden_states,)

            # if output_attentions:
            #     output = output + (vision_attentions, mask_decoder_attentions)
            output = (
                iou_predictions,
                low_res_masks,
                vision_hidden_states,
                vision_attentions,
                mask_decoder_attentions,
                vision_moe_loss,
            )
            return output

        return SamImageSegmentationOutput(
            iou_scores=iou_predictions,
            pred_masks=low_res_masks,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
            mask_decoder_attentions=mask_decoder_attentions,
            vision_moe_loss=vision_moe_loss,
        )

```

```worktree-python
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import SamConfig

from ..constants import CLASS_LABEL, CLASS_LOGITS, COLUMN, IMAGE, IMAGE_VALID_NUM, LABEL, LOGITS, MASK_LABEL, MOE_LOSS
from .adaptation_layers import ConvLoRALinear, AdapterLayer, GatedAdapterLayer
from .custom_hf_models.modeling_sam_for_conv_lora import SamImageSegmentationOutput, SamModel
from .utils import assign_layer_ids, freeze_model_layers, image_mean_std

logger = logging.getLogger(__name__)


def multi_class_mask_decoder_forward(
    self,
    image_embeddings: torch.Tensor,
    image_positional_embeddings: torch.Tensor,
    sparse_prompt_embeddings: torch.Tensor,
    dense_prompt_embeddings: torch.Tensor,
    multimask_output: bool,
    output_attentions: Optional[bool] = None,
    attention_similarity: torch.Tensor = None,
    target_embedding: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Modify the forward method of SamMaskDecoder for multi-class semantic segmentation
    based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/modeling_sam.py#L468.

    Args:
        image_embeddings (`torch.Tensor`):
            the embeddings from the image encoder
        image_positional_embedding (`torch.Tensor`):
            positional encoding with the shape of image_embeddings
        sparse_prompt_embeddings (`torch.Tensor`):
            The embeddings of the points and boxes
        dense_prompt_embeddings (`torch.Tensor`):
            the embeddings of the mask inputs
        multimask_output (bool):
            Whether to return multiple masks or a single mask.
        output_attentions (bool, *optional*):
            Whether or not to return the attentions tensors of all attention layers.
    """
    batch_size, num_channels, height, width = image_embeddings.shape
    point_batch_size = sparse_prompt_embeddings.shape[1]
    # Concatenate output tokens
    output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
    output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

    if sparse_prompt_embeddings.sum().item() != 0:
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
    else:
        tokens = output_tokens
    point_embeddings = tokens.to(self.iou_token.weight.dtype)

    # Expand per-image data in batch direction to be per-point
    image_embeddings = image_embeddings + dense_prompt_embeddings
    image_embeddings = image_embeddings.repeat(point_batch_size, 1, 1, 1)
    image_positional_embeddings = image_positional_embeddings.repeat(point_batch_size, 1, 1, 1)

    # Run the transformer, image_positional_embedding are consumed
    point_embedding, image_embeddings, attentions = self.transformer(
        point_embeddings=point_embeddings,
        image_embeddings=image_embeddings,
        image_positional_embeddings=image_positional_embeddings,
        attention_similarity=attention_similarity,
        target_embedding=target_embedding,
        output_attentions=output_attentions,
    )
    iou_token_out = point_embedding[:, :, 0, :]
    mask_tokens_out = point_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

    # Upscale mask embeddings and predict masks using the mask tokens
    image_embeddings = image_embeddings.transpose(2, 3).reshape(
        batch_size * point_batch_size, num_channels, height, width
    )

    upscaled_embedding = self.upscale_conv1(image_embeddings)
    upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
    upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

    ################ Modify the original code. We aim at returning a single mask for each object.
    ################ Original logic is to return multiple masks.
    ################ So we use an MLP network to process all the mask proposals instead of multiple networks.
    # hyper_in_list = []
    # for i in range(self.num_mask_tokens):
    #     current_mlp = self.output_hypernetworks_mlps[i]
    #     hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
    # hyper_in = torch.stack(hyper_in_list, dim=2)
    hyper_in = self.output_hypernetworks_mlps[0](mask_tokens_out)
    ################

    _, num_channels, height, width = upscaled_embedding.shape
    upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
    masks = (hyper_in @ upscaled_embedding).reshape(
        batch_size, point_batch_size, -1, height, width
    )  # bs, 1, num_tokens, h, w

    ################ New added class prediction logic.
    class_predictions = self.output_classifier_mlps(mask_tokens_out).reshape(
        batch_size, point_batch_size, -1, self.num_classes + 1
    )
    ################

    # Generate mask quality predictions
    iou_pred = self.iou_prediction_head(iou_token_out)

    outputs = (masks, iou_pred)

    if output_attentions:
        outputs = outputs + (attentions,)
    else:
        outputs = outputs + (None,)

    return outputs + (class_predictions,)


def multi_class_sam_model_forward(
    self,
    pixel_values: Optional[torch.FloatTensor] = None,
    input_points: Optional[torch.FloatTensor] = None,
    input_labels: Optional[torch.LongTensor] = None,
    input_boxes: Optional[torch.FloatTensor] = None,
    input_masks: Optional[torch.LongTensor] = None,
    image_embeddings: Optional[torch.FloatTensor] = None,
    multimask_output: bool = True,
    attention_similarity: Optional[torch.FloatTensor] = None,
    target_embedding: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_moe_loss: Optional[bool] = None,  # MoE loss for Conv-LoRA
    return_dict=None,
    **kwargs,
) -> List[Dict[str, torch.Tensor]]:
    r"""
    Modify the forward method of SamModel for multi-class semantic segmentation
    based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/modeling_sam.py#L1279.

    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if pixel_values is None and image_embeddings is None:
        raise ValueError("Either pixel_values or image_embeddings must be provided.")

    if pixel_values is not None and image_embeddings is not None:
        raise ValueError("Only one of pixel_values and image_embeddings can be provided.")

    if input_points is not None and len(input_points.shape) != 4:
        raise ValueError(
            "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.",
            " got {}.".format(input_points.shape),
        )
    if input_boxes is not None and len(input_boxes.shape) != 3:
        raise ValueError(
            "The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.",
            " got {}.".format(input_boxes.shape),
        )
    if input_points is not None and input_boxes is not None:
        point_batch_size = input_points.shape[1]
        box_batch_size = input_boxes.shape[1]
        if point_batch_size != box_batch_size:
            raise ValueError(
                "You should provide as many bounding boxes as input points per box. Got {} and {}.".format(
                    point_batch_size, box_batch_size
                )
            )

    image_positional_embeddings = self.get_image_wide_positional_embeddings()
    # repeat with batch size
    batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]
    image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

    vision_attentions = None
    vision_hidden_states = None
    vision_moe_loss = None

    if pixel_values is not None:
        vision_outputs = self.vision_encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_moe_loss=output_moe_loss,
            return_dict=return_dict,
        )
        image_embeddings = vision_outputs[0]

        if output_hidden_states:
            vision_hidden_states = vision_outputs[1]
        if output_attentions:
            # vision_attentions = vision_outputs[-1]
            vision_attentions = vision_outputs[2]
        if output_moe_loss:
            vision_moe_loss = vision_outputs[-1]

    if input_points is not None and input_labels is None:
        input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

    if input_points is not None and image_embeddings.shape[0] != input_points.shape[0]:
        raise ValueError(
            "The batch size of the image embeddings and the input points must be the same. ",
            "Got {} and {} respectively.".format(image_embeddings.shape[0], input_points.shape[0]),
            " if you want to pass multiple points for the same image, make sure that you passed ",
            " input_points of shape (batch_size, point_batch_size, num_points_per_image, 3) and ",
            " input_labels of shape (batch_size, point_batch_size, num_points_per_image)",
        )

    sparse_embeddings, dense_embeddings = self.prompt_encoder(
        input_points=input_points,
        input_labels=input_labels,
        input_boxes=input_boxes,
        input_masks=input_masks,
    )

    low_res_masks, iou_predictions, mask_decoder_attentions, class_predictions = self.mask_decoder(
        image_embeddings=image_embeddings,
        image_positional_embeddings=image_positional_embeddings,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
        attention_similarity=attention_similarity,
        target_embedding=target_embedding,
        output_attentions=output_attentions,
    )

    if not return_dict:
        output = (
            iou_predictions,
            low_res_masks,
            vision_hidden_states,
            vision_attentions,
            mask_decoder_attentions,
            vision_moe_loss,
        )
        return output

    return (
        SamImageSegmentationOutput(
            iou_scores=iou_predictions,
            pred_masks=low_res_masks,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
            mask_decoder_attentions=mask_decoder_attentions,
            vision_moe_loss=vision_moe_loss,
        ),
        ################ New added. Return class predictions as well.
        class_predictions,
        ################
    )


class SAMForSemanticSegmentation(nn.Module):
    """
    Support SAM for semantic segmentation.
    Refer to https://huggingface.co/docs/transformers/main/model_doc/sam
    Paper alignment (Conv-LoRA, ICLR 2024):
    - Sec. 4 (Training): When Conv-LoRA is active, expose/aggregate MoE load-balancing loss produced inside
      attention projections (q/k/v). This module toggles and collects that loss when trainable Conv-LoRA exists.
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str,
        num_classes: int = 1,
        pretrained: Optional[bool] = True,
        frozen_layers: Optional[list] = None,
        num_mask_tokens: int = 1,
        image_norm: Optional[str] = None,
        adapter_enabled: bool = False,
        adapter_dim: int = 64,
        decoder_attention_lora_r: int = 0,
        decoder_attention_lora_alpha: int = 1,
        decoder_attention_lora_dropout: float = 0.0,
        decoder_attention_lora_gate_enabled: bool = False,
        decoder_attention_lora_gate_noise_std: float = 0.0,
        decoder_adapter_enabled: bool = False,
        decoder_adapter_dim: int = 64,
    ):
        """
        Load a pretrained Segment Anything Model (SAM).

        Parameters
        ----------
        prefix
            The prefix of the SAMForSemanticSegmentation model.
        checkpoint_name
            Name of the SAM checkpoint.
        num_classes
            The number of classes
        pretrained
            Whether using the pretrained SAM models. If pretrained=True, download the pretrained model.
        frozen_layers
            A list of substrings of frozen layers' names.
        num_mask_tokens
            The number of mask proposals.
        image_norm
            How to normalize an image. We now support:
            - inception
                Normalize image by IMAGENET_INCEPTION_MEAN and IMAGENET_INCEPTION_STD from timm
            - imagenet
                Normalize image by IMAGENET_DEFAULT_MEAN and IMAGENET_DEFAULT_STD from timm
            - clip
                Normalize image by mean (0.48145466, 0.4578275, 0.40821073) and
                std (0.26862954, 0.26130258, 0.27577711), used for CLIP.
        adapter_enabled
            Whether to enable standard Adapter modules in the vision encoder.
        adapter_dim
            The dimension of the adapter bottleneck.
        decoder_attention_lora_r
            LoRA rank for decoder attention layers. If 0 (default), LoRA is disabled.
            Recommended value: 8 for parameter-efficient fine-tuning.
        decoder_attention_lora_alpha
            LoRA scaling factor for decoder attention. Default: 1.
        decoder_attention_lora_dropout
            Dropout probability for LoRA in decoder attention. Default: 0.0.
        """

        super().__init__()
        self.prefix = prefix
        self.pretrained = pretrained
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes
        self.frozen_layers = frozen_layers
        self.decoder_attention_lora_r = decoder_attention_lora_r
        self.decoder_attention_lora_alpha = decoder_attention_lora_alpha
        self.decoder_attention_lora_dropout = decoder_attention_lora_dropout
        self.decoder_attention_lora_gate_enabled = decoder_attention_lora_gate_enabled
        self.decoder_attention_lora_gate_noise_std = decoder_attention_lora_gate_noise_std

        self.device = None
        self.name_to_id = {}

        # Store adapter configs
        self.adapter_enabled = adapter_enabled
        self.adapter_dim = adapter_dim
        self.decoder_adapter_enabled = decoder_adapter_enabled
        self.decoder_adapter_dim = decoder_adapter_dim

        self._load_checkpoint(checkpoint_name)

        freeze_model_layers(self.model, self.frozen_layers)

        self.image_size = self.model.vision_encoder.image_size
        self.config = self.model.config
        self.image_mean, self.image_std = image_mean_std(image_norm)

        # Inject Adapters into vision encoder if enabled
        if self.adapter_enabled:
            logger.info(f"Injecting Adapters with dim={self.adapter_dim} into Vision Encoder")
            encoder_gate_enabled = getattr(self.config, "encoder_adapter_gate_enabled", False)
            encoder_gate_noise_std = getattr(self.config, "encoder_adapter_gate_noise_std", 0.0)
            for layer in self.model.vision_encoder.layers:
                # Manually create and assign adapter since config didn't have it at init time
                if encoder_gate_enabled:
                    layer.adapter = GatedAdapterLayer(
                        in_features=self.config.hidden_size,
                        adapter_dim=self.adapter_dim,
                        gate_noise_std=encoder_gate_noise_std,
                    )
                else:
                    layer.adapter = AdapterLayer(self.config.hidden_size, self.adapter_dim)


        self.model.mask_decoder.num_mask_tokens = num_mask_tokens
        mask_token_data = self.model.mask_decoder.mask_tokens.weight.data[0]
        self.model.mask_decoder.mask_tokens = nn.Embedding(num_mask_tokens, self.model.mask_decoder.hidden_size)
        for i in range(num_mask_tokens):
            self.model.mask_decoder.mask_tokens.weight.data[i] = mask_token_data
        hyper_mlps = self.model.mask_decoder.output_hypernetworks_mlps[0]
        self.model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList([hyper_mlps])
        if num_classes > 1:
            self.model.mask_decoder.num_classes = num_classes
            self.model.mask_decoder.output_classifier_mlps = nn.Linear(
                self.model.mask_decoder.hidden_size, num_classes + 1
            )

            mask_decoder_forward = multi_class_mask_decoder_forward.__get__(
                self.model.mask_decoder, self.model.mask_decoder.__class__
            )
            setattr(self.model.mask_decoder, "forward", mask_decoder_forward)

            sam_model_forward = multi_class_sam_model_forward.__get__(self.model, self.model.__class__)
            setattr(self.model, "forward", sam_model_forward)

        # for Conv-LoRA
        self.output_moe_loss = False

    def _load_checkpoint(self, checkpoint_name):
        if self.pretrained:
            # Load configuration and inject decoder attention LoRA parameters
            try:
                config = SamConfig.from_pretrained(checkpoint_name, local_files_only=True)
            except:
                config = SamConfig.from_pretrained(checkpoint_name)
            
            # Propagate decoder adapter settings into mask decoder config (if present)
            if hasattr(config, "mask_decoder_config") and config.mask_decoder_config is not None:
                config.mask_decoder_config.decoder_adapter_enabled = self.decoder_adapter_enabled
                config.mask_decoder_config.decoder_adapter_dim = self.decoder_adapter_dim

            # Inject LoRA parameters into mask decoder config
            config.mask_decoder_config.decoder_attention_lora_r = self.decoder_attention_lora_r
            config.mask_decoder_config.decoder_attention_lora_alpha = self.decoder_attention_lora_alpha
            config.mask_decoder_config.decoder_attention_lora_dropout = self.decoder_attention_lora_dropout
            config.mask_decoder_config.decoder_attention_lora_gate_enabled = self.decoder_attention_lora_gate_enabled
            config.mask_decoder_config.decoder_attention_lora_gate_noise_std = self.decoder_attention_lora_gate_noise_std
            
            # Try to load from local cache first to avoid network issues
            try:
                self.model = SamModel.from_pretrained(checkpoint_name, config=config, local_files_only=True)
                logger.info(f"Loaded model from local cache: {checkpoint_name}")
            except Exception as e:
                logger.info(f"Local cache not found, downloading from Hugging Face: {checkpoint_name}")
                self.model = SamModel.from_pretrained(checkpoint_name, config=config)
            
            if self.decoder_attention_lora_r > 0:
                logger.info(f"Decoder Attention LoRA enabled: r={self.decoder_attention_lora_r}, "
                          f"alpha={self.decoder_attention_lora_alpha}, dropout={self.decoder_attention_lora_dropout}")
        else:
            config = SamConfig(name_or_path=checkpoint_name)
            if hasattr(config, "mask_decoder_config") and config.mask_decoder_config is not None:
                config.mask_decoder_config.decoder_adapter_enabled = self.decoder_adapter_enabled
                config.mask_decoder_config.decoder_adapter_dim = self.decoder_adapter_dim

            config.mask_decoder_config.decoder_attention_lora_r = self.decoder_attention_lora_r
            config.mask_decoder_config.decoder_attention_lora_alpha = self.decoder_attention_lora_alpha
            config.mask_decoder_config.decoder_attention_lora_dropout = self.decoder_attention_lora_dropout
            config.mask_decoder_config.decoder_attention_lora_gate_enabled = self.decoder_attention_lora_gate_enabled
            config.mask_decoder_config.decoder_attention_lora_gate_noise_std = self.decoder_attention_lora_gate_noise_std
            self.model = SamModel(config)

    def save(self, save_path: str = "./"):
        self.model.save_pretrained(save_path)
        logger.info(f"Model weights for {self.prefix} is saved to {save_path}.")

    @property
    def image_key(self):
        return f"{self.prefix}_{IMAGE}"

    @property
    def image_valid_num_key(self):
        return f"{self.prefix}_{IMAGE_VALID_NUM}"

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    @property
    def image_column_prefix(self):
        return f"{self.image_key}_{COLUMN}"

    @property
    def image_feature_dim(self):
        return self.model.num_features

    @property
    def mask_label_key(self):
        return f"{self.prefix}_{MASK_LABEL}"

    @property
    def class_label_key(self):
        return f"{self.prefix}_{CLASS_LABEL}"

    @property
    def box_key(self):
        return f"{self.prefix}_box"

    @property
    def image_path_key(self):
        return f"{self.prefix}_image_path"

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, ConvLoRALinear):
                # If Conv-LoRA is present and trainable, enable returning moe_loss from the attention blocks (Sec. 4)
                self.output_moe_loss = True
                return self

        return self

    def forward(
        self,
        batch,
    ):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with mask predictions.
        """
        # binary
        if self.num_classes == 1:
            rets = self.model(
                batch[self.image_key],
                input_boxes=batch.get(self.box_key, None),
                multimask_output=False,
                output_moe_loss=self.output_moe_loss,
            )
            pred_masks = rets.pred_masks[:, 0, :, :, :]
            pred_masks = F.interpolate(
                pred_masks, (self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
            if self.training:
                rets_dict = {self.prefix: {LOGITS: pred_masks}}
            else:
                rets_dict = {self.prefix: {LOGITS: pred_masks, LABEL: batch[self.label_key]}}
        # multi-class
        else:
            rets = self.model(
                batch[self.image_key],
                input_boxes=batch.get(self.box_key, None),
                multimask_output=False,
                output_moe_loss=self.output_moe_loss,
            )
            rets, class_predictions = rets
            pred_masks = rets.pred_masks[:, 0, :, :, :]
            pred_classes = class_predictions[:, 0, :, :]
            pred_masks = F.interpolate(
                pred_masks, (self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
            if self.training:
                rets_dict = {self.prefix: {LOGITS: pred_masks, CLASS_LOGITS: pred_classes}}
            else:
                rets_dict = {
                    self.prefix: {
                        LOGITS: pred_masks,
                        CLASS_LOGITS: pred_classes,
                        LABEL: batch[self.label_key],
                    }
                }
        if self.output_moe_loss:
            rets_dict[self.prefix].update({MOE_LOSS: rets.vision_moe_loss})
        return rets_dict

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        In the AutoModel scenario, this function may not always return the correct result.
        Thus, you can use "print(json.dumps(name_to_id, indent=2))" to manually check whether
        the layer ids are reasonable.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = "model"
        pre_encoder_patterns = (
            "vision_encoder",
            "prompt_encoder",
        )
        post_encoder_patterns = ("mask_decoder",)

        names = [n for n, _ in self.named_parameters()]

        name_to_id, names = assign_layer_ids(
            names=names,
            pre_encoder_patterns=pre_encoder_patterns,
            post_encoder_patterns=post_encoder_patterns,
            model_pre=model_prefix,
        )
        if len(names) > 0:
            logger.debug(f"outer layers are treated as head: {names}")
        for n in names:
            assert n not in name_to_id
            name_to_id[n] = 0

        return name_to_id

```

```worktree-yaml
model:
  names:
  categorical_mlp:
    hidden_size: 64
    activation: "leaky_relu"
    num_layers: 1
    dropout: 0.1
    normalization: "layer_norm"
    data_types:
      - "categorical"

  numerical_mlp:
    hidden_size: 128
    activation: "leaky_relu"
    num_layers: 1
    dropout: 0.1
    normalization: "layer_norm"
    token_dim: 8
    embedding_arch:
    data_types:
      - "numerical"
    merge: "concat"

  hf_text:
    checkpoint_name: "google/electra-base-discriminator"
    gradient_checkpointing: False
    pooling_mode: 'cls'  # The pooling mode, can be 'cls' or 'mean'
    data_types:
      - "text"
    tokenizer_name: "hf_auto"
    use_fast: True  # Use a fast Rust-based tokenizer if it is supported for a given model. If a fast tokenizer is not available for a given model, a normal Python-based tokenizer is returned instead.
    max_text_len: 512  # If None or <=0, then use the max length of pretrained models.
    insert_sep: True
    low_cpu_mem_usage: False
    text_segment_num: 2
    stochastic_chunk: False
    text_aug_detect_length: 10                # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.1            # augmentation magnitude randomly drawn from [0, text_trivial_aug_maxscale]
    text_train_augment_types:        # specify augmentation space manually, will randomly select one from the following and identity
      # - "random_swap(0.05)"          # less than 0.1 based on eda paper
      # - "random_delete(0.05)"        # less than 0.1 based on eda paper
      # - "syn_replacement(0.05)"  # less than 0.1 based on eda paper
      # - "insert_punc(0.05)"

  ner_text:
    checkpoint_name: "bert-base-cased"
    max_text_len: 512
    gradient_checkpointing: False
    low_cpu_mem_usage: False
    data_types:
      - "text_ner"
    tokenizer_name: "hf_auto"
    insert_sep: False
    text_segment_num: 2
    stochastic_chunk: False
    special_tags: 
      - X # CLS, SEP, and non-first tokens of a word will be labelled as X
      - O # Outside of a named entity

  document_transformer:
    checkpoint_name: "microsoft/layoutlmv3-base" # document foundation models
    gradient_checkpointing: False
    pooling_mode: 'cls'  # The pooling mode, can be 'cls' or 'mean'
    data_types:
      - "document"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: 224
    tokenizer_name: "hf_auto"
    max_text_len: 512  # If None or <=0, then use the max length of pretrained models.
    insert_sep: True
    low_cpu_mem_usage: False
    text_segment_num: 2
    stochastic_chunk: False
    text_aug_detect_length: 10                # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.0            # augmentation magnitude randomly drawn from [0, text_trivial_aug_maxscale]

  t_few:
    checkpoint_name: "t5-small" #"bigscience/T0_3B"
    gradient_checkpointing: False
    data_types:
      - "text"
    tokenizer_name: "hf_auto"
    length_norm: 1.0 # Normalizes length to adjust for length bias in target template
    unlikely_loss: 1.0 # Adds loss term that lowers probability of incorrect outputs
    mc_loss: 1.0 # Adds multiple choice cross entropy loss
    max_text_len: 512  # If None or <=0, then use the max length of pretrained models.
    text_segment_num: 2
    insert_sep: True
    low_cpu_mem_usage: False
    stochastic_chunk: False
    text_aug_detect_length: 10                # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.0            # augmentation magnititude randomly drawn from [0, text_trivial_aug_maxscale]
    text_train_augment_types:

  timm_image:
    checkpoint_name: "swin_base_patch4_window7_224"
    mix_choice: "all_logits"
    data_types:
      - "image"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: null
    image_chan_num: 3
    use_learnable_image: False
    max_image_num_per_column: 1

  mmdet_image:
    checkpoint_name: "yolov3_mobilenetv2_8xb24-320-300e_coco"
    config_file: ""
    data_types:
      - "image"
    max_img_num_per_col: 1
    output_bbox_format: "xyxy"  # now support xyxy or xywh, for bbox format details see https://keras.io/api/keras_cv/bounding_box/formats/
    frozen_layers: null
    coco_root: null

  mmocr_text_detection:
    checkpoint_name: "TextSnake"
    data_types:
      - "image"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: 224
    max_img_num_per_col: 2

  mmocr_text_recognition:
    checkpoint_name: "ABINet"
    data_types:
      - "image"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: 224
    max_img_num_per_col: 2

  clip:
    checkpoint_name: "openai/clip-vit-base-patch32"
    data_types:
      - "image"
      - "text"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "clip"
    image_size: 224
    image_chan_num: 3
    use_learnable_image: False
    max_image_num_per_column: 1
    tokenizer_name: "clip"
    max_text_len: 77  # The maximum possible length.
    insert_sep: False
    text_segment_num: 1
    stochastic_chunk: False
    text_aug_detect_length: 10                     # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.0                      # scale randomly drawn from [0, text_trivial_aug_maxscale]
    text_train_augment_types:        # specify augmentation space manually, will randomly select one from the following and identity
      # - "random_swap(0.05)"          # less than 0.1 based on eda paper
      # - "random_delete(0.05)"        # less than 0.1 based on eda paper
      # - "syn_replacement(0.05)"  # less than 0.1 based on eda paper
      # - "insert_punc(0.05)"

  fusion_mlp:
    aux_loss_weight:
    adapt_in_features: "max"
    hidden_sizes:
      - 128
    activation: "leaky_relu"
    dropout: 0.1
    normalization: "layer_norm"
    data_types:

  fusion_ner:
    weight:
    adapt_in_features: "max"
    hidden_sizes:
      - 128
    activation: "leaky_relu"
    drop_rate: 0.1
    normalization: "layer_norm"
    data_types:

  fusion_transformer:
    aux_loss_weight:
    hidden_size: 192
    num_blocks: 3
    attention_num_heads: 8
    adapt_in_features: "max"
    attention_dropout: 0.2
    residual_dropout: 0.0
    ffn_dropout: 0.1
    ffn_hidden_size: 192
    normalization: "layer_norm"
    ffn_activation: "geglu"
    head_activation: "relu"
    data_types:
    additive_attention: False # Whether to use lightweight additive attention, can be True, False or "auto"
    share_qv_weights: False # Whether to share weight for query and value, can be True, False or "auto"

  ft_transformer:
    data_types:
      - "categorical"
      - "numerical"
    embedding_arch:
      - 'linear'
    token_dim: 192
    hidden_size: 192
    num_blocks: 3
    attention_num_heads: 8
    attention_dropout: 0.2
    residual_dropout: 0.0
    ffn_dropout: 0.1
    ffn_hidden_size: 192
    ffn_activation: "geglu"
    head_activation: "relu"
    normalization: "layer_norm"
    merge: "concat"
    requires_all_dtypes: False
    additive_attention: False # Whether to use lightweight additive attention, can be True, False or "auto"
    share_qv_weights: False # Whether to share weight for query and value, can be True, False or "auto"
    pooling_mode: "cls"
    checkpoint_name: null

  sam:
    checkpoint_name: "facebook/sam-vit-huge"
    image_norm: "imagenet"
    data_types:
      - "semantic_segmentation_img"
    train_transforms:
      - "random_horizontal_flip"
    val_transforms: []
    img_transforms:
      - "resize_to_square"
    gt_transforms:
      - "resize_gt_to_square"
    max_img_num_per_col: 1
    frozen_layers: ["mask_decoder.iou_prediction_head", "prompt_encoder"]
    num_mask_tokens: 1
    ignore_label: 255
    adapter_enabled: False
    adapter_dim: 64
    # GSPO-Adapter extension (encoder only; kept here to build the correct adapter module type)
    encoder_adapter_gate_enabled: False
    encoder_adapter_gate_noise_std: 0.0
    decoder_attention_lora_r: 0
    decoder_attention_lora_alpha: 1
    decoder_attention_lora_dropout: 0.0
    # GSPO-AttnLoRA extension (decoder attention LoRA policy gate; only changes structure when LoRA is enabled)
    decoder_attention_lora_gate_enabled: False
    decoder_attention_lora_gate_noise_std: 0.0
    decoder_adapter_enabled: False
    decoder_adapter_dim: 64

  meta_transformer:
    data_types:
      - "image"
      - "text"
      - "categorical"
      - "numerical"
    checkpoint_path: null
    model_version: "base"
    requires_all_dtypes: False
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: 224
    image_chan_num: 3
    use_learnable_image: False
    max_image_num_per_column: 1
    tokenizer_name: "hf_auto"
    max_text_len: 512  # If None or <=0, then use the max length of pretrained models.
    insert_sep: True
    text_segment_num: 2
    stochastic_chunk: False
    text_aug_detect_length: 10     # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.1  # augmentation magnitude randomly drawn from [0, text_trivial_aug_maxscale]
    text_train_augment_types:
    merge: "concat"

```

```worktree-yaml
optim:
  optim_type: "adamw"
  lr: 1.0e-4
  weight_decay: 0.001
  lr_choice: "layerwise_decay"
  lr_decay: 0.9
  lr_schedule: "cosine_decay"
  max_epochs: 20
  max_steps: -1
  warmup_steps: 0.1
  end_lr: 0
  lr_mult: 1 # multiply lr for downstream heads
  patience: 10
  val_check_interval: 0.5 # Check validation score a fraction of epoch if float and every n steps if integer with n < total step in epoch.
  check_val_every_n_epoch: 1 # Check validation score every n epochs.
  skip_final_val: False # Flag to skip the last validation
  gradient_clip_val: 1
  gradient_clip_algorithm: "norm"
  track_grad_norm: -1 # Whether to check gradient norm. We can set it to 2 to check for gradient norm.
  log_every_n_steps: 10
  label_smoothing: 0
  top_k: 3
  top_k_average_method:
    "greedy_soup" # We support averaging method described in https://arxiv.org/pdf/2203.05482.pdf.
    # Currently support "uniform_soup", "greedy_soup", and "best".
  peft: null # Can be 'bit_fit' (only finetune bias), 'norm_fit' (finetune the normalization terms + bias terms), lora (LoRA Adaptations only), lora_bias (LoRA Adaptation + bit_fit), lora_norm (LoRA Adaptation + norm_fit), or null
  lora:
    module_filter: null # Specify which module (if any) to adapt(e.g. ".*EncDecAttention|.*DenseReluDense"). Default consider all modules in a model (i.e. empty filter).
    filter: # Specify which layer in a module to adapt. Default fine-tune only query and value attention weights, recommended in https://arxiv.org/abs/2106.09685
      - "query"
      - "value"
      - "^q$" # The filter can also be a regex. We will use re.match(filter, name) to determine if LoRA should be applied.
      - "^v$"
      - "^k$"
      - "^o$"
    r: 8
    alpha: 8
    conv_lora_expert_num: 8 # default setting for Conv-LoRA
    gspo_enabled: False # Enable GSPO (Group Sequence Policy Optimization) enhancements
    gspo_group_size: 3 # Number of experts to select in GSPO mode (TopK)
    gspo_quality_momentum: 0.9 # Momentum coefficient for expert quality history updates
    gspo_warmup_epochs: 5 # Number of epochs before enabling full GSPO training
    gspo_contrastive_weight: 0.1 # Weight for contrastive loss in GSPO training
  loss_func:
    "auto" # The replaced loss for regression. Can only support loss function in torch.nn.
    # example
    # "BCEWithLogitsLoss" or "nn.BCEWithLogitsloss"
  focal_loss:
    alpha: null
    gamma: 2.0
    reduction: "mean"
  mask2former_loss:
    loss_cross_entropy_weight: 10.0
    loss_mask_weight: 5.0
    loss_dice_weight: 5.0
  extra_trainable_params: []
  cross_modal_align: null
  cross_modal_align_weight: 0
  automatic_optimization: True
  lemda:
    turn_on: False
    arch_type: "mlp_vae"
    z_dim: 8
    num_layers: 6
    kld_weight: 0.1
    mse_weight: 0.1
    adv_weight: 1.0e-4
    consist_weight: 0.01
    consist_threshold: 0.5
    lr: 1.0e-4
    optim_type: "adamw"
    weight_decay: 1.0e-5
  gspo:
    lambda_smooth: 0.1
    lambda_boundary: 0.3
    w_boundary: 0.3
    w_smooth: 0.1
    w_thin: 0.05
    # GSPO-Adapter extension (encoder adapters only; decoder adapters excluded by design)
    # When False (default), encoder adapters will be frozen during GSPO-active epochs/steps.
    train_encoder_adapter: False
    # Stage 1: add small exploration noise on adapter outputs during GSPO rollouts
    encoder_adapter_noise_std: 0.0
    # Stage 2: gated adapter policy exploration noise (applies only if model.sam.encoder_adapter_gate_enabled=True)
    encoder_adapter_gate_noise_std: 0.0
    # GSPO-AttnLoRA extension (decoder attention LoRA only)
    # When False (default), decoder attention LoRA parameters (if enabled) are frozen during GSPO-active steps.
    train_decoder_attention_lora: False
    # Stage 1: exploration noise std added to decoder attention LoRA delta during GSPO rollouts
    decoder_attention_lora_noise_std: 0.0
    # Stage 2: exploration noise std added to gated decoder attention LoRA gate logits during GSPO rollouts
    decoder_attention_lora_gate_noise_std: 0.0

```

```worktree-python
import importlib.util
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.special import softmax

from autogluon.core.metrics import Scorer

from ..constants import LABEL, LOGITS, SEMANTIC_MASK, SEMANTIC_SEGMENTATION_IMG
from ..optim import SemanticSegmentationLitModule, get_loss_func, get_norm_layer_param_names, get_peft_param_names
from ..optim.metrics.semantic_seg_metrics import Balanced_Error_Rate_Pred as Balanced_Error_Rate
from ..optim.metrics.semantic_seg_metrics import Binary_IoU_Pred as Binary_IoU
from ..optim.metrics.semantic_seg_metrics import Binary_DICE_Pred as Binary_DICE
from ..optim.metrics.semantic_seg_metrics import COD_METRICS_NAMES_Pred as COD_METRICS_NAMES
from ..optim.metrics.semantic_seg_metrics import Multiclass_IoU_Pred as Multiclass_IoU
from ..optim.metrics.semantic_seg_metrics import Multiclass_DICE_Pred as Multiclass_DICE
from ..utils import extract_from_output, setup_save_path
from ..utils.tta_utils import TTAPredictor, dice_coefficient, iou_score
from .base import BaseLearner

logger = logging.getLogger(__name__)

from ..constants import BER, DICE, EM, FM, IOU, MAE, SEMANTIC_SEGMENTATION, SM


class SemanticSegmentationLearner(BaseLearner):
    def __init__(
        self,
        label: Optional[str] = None,
        problem_type: Optional[str] = SEMANTIC_SEGMENTATION,
        presets: Optional[str] = None,
        eval_metric: Optional[Union[str, Scorer]] = "iou",
        hyperparameters: Optional[dict] = None,
        path: Optional[str] = None,
        verbosity: Optional[int] = 2,
        num_classes: Optional[int] = None,  # TODO: can we infer this from data?
        warn_if_exist: Optional[bool] = True,
        enable_progress_bar: Optional[bool] = None,
        pretrained: Optional[bool] = True,
        validation_metric: Optional[str] = "iou",
        sample_data_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            label=label,
            problem_type=problem_type,
            presets=presets,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            path=path,
            verbosity=verbosity,
            warn_if_exist=warn_if_exist,
            enable_progress_bar=enable_progress_bar,
            pretrained=pretrained,
            validation_metric=validation_metric,
        )
        self._output_shape = num_classes
        self._sample_data_path = sample_data_path
        self._tta_predictor = None  # Will be initialized when TTA is enabled

        if self._sample_data_path is not None:
            infer_output_shape = self.get_semantic_segmentation_class_num(self._sample_data_path)
            if num_classes is not None:
                assert (
                    num_classes == infer_output_shape
                ), f"The provided number of classes '{num_classes}' and the inferred class number {infer_output_shape}' from the sample data should be consistent."
            else:
                self._output_shape = infer_output_shape

        # Training-time box prompt configuration (used when constructing LitModule)
        self._train_box_prompt_cfg = {
            "mode": "off",       # off / gt / noisy / mix / predict
            "p_no": 0.0,
            "p_gt": 0.0,
            "p_noisy": 0.0,
            "noise_frac": 0.12,
        }
        # BBoxPromptPredictor instance for "predict" mode (set via external config)
        self._train_bbox_predictor = None

    def get_semantic_segmentation_class_num(self, sample_data_path):
        """
        Get the number of classes for given data.

        Parameters
        ----------
            sample_data_path
                This is used for automatically inference num_classes of semantic segmentation dataset.
                Could be an image directory, image file or pd.DataFrame.
        Returns
        -------
            The number of classes.
        """
        if isinstance(sample_data_path, str):
            if os.path.isdir(sample_data_path):
                mask_files = os.listdir(sample_data_path)
                num_classes = []
                for mask_file in mask_files:
                    per_num_classes = self.get_semantic_segmentation_class_num(
                        os.path.join(sample_data_path, mask_file)
                    )
                    num_classes.append(per_num_classes)
                return max(num_classes)
            else:
                mask = Image.open(sample_data_path)
                mode = mask.mode

                if mode == "1":
                    return 1
                classes = np.unique(mask)
                if mode == "L" and np.array_equal(classes, np.array([0, 255])):
                    return 1

                return max(classes).item() + 1  # include background

        elif isinstance(sample_data_path, pd.DataFrame):
            num_classes = []
            for idx in range(sample_data_path.shape[0]):
                row = sample_data_path.iloc[idx]
                mask_file = row[self._label_column]
                per_num_classes = self.get_semantic_segmentation_class_num(mask_file)
                num_classes.append(per_num_classes)
            return max(num_classes)

    def infer_output_shape(self):
        if self._output_shape is None:
            self._output_shape = self.get_semantic_segmentation_class_num(self._train_data)

    @staticmethod
    def get_peft_param_names_per_run(model, config):
        peft_param_names = None
        peft = config.optim.peft
        if peft:
            norm_param_names = get_norm_layer_param_names(model)
            peft_param_names = get_peft_param_names(
                norm_param_names,
                peft=peft,
                extra_params=config.optim.extra_trainable_params,
            )
        return peft_param_names

    def get_loss_func_per_run(self, config, mixup_active=None):
        loss_func = get_loss_func(
            problem_type=self._problem_type,
            loss_func_name=config.optim.loss_func,
            config=config.optim,
            num_classes=self._output_shape,
        )
        return loss_func, None

    def evaluate_semantic_segmentation(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = False,
    ):
        """
        Evaluate semantic segmentation on a test dataset based on "torchmetrics".

        Parameters
        ----------
        data
            A dataframe, containing the same columns as the training data.
            Or a str, that is a path of the annotation file for detection.
        metrics
            Metrics used for evaluation.
        return_pred
            Whether to return the prediction result of each row.
        realtime
            Whether to do realtime inference, which is efficient for small data (default None).
            If provided None, we would infer it on based on the data modalities
            and sample number.
        """
        
        # If TTA is enabled, use TTA evaluation
        if self._tta_predictor is not None:
            # Use cache settings if configured
            cache_dir = getattr(self, '_tta_cache_dir', None)
            resume_from_cache = getattr(self, '_tta_resume_from_cache', True)
            return self._evaluate_with_tta(
                data, metrics, return_pred, 
                cache_dir=cache_dir,
                resume_from_cache=resume_from_cache
            )

        def get_metric_predict(
            metric_name: str,
            num_classes: Optional[int] = None,
        ):
            """
            Obtain a torchmerics.Metric from its name.
            Define a customized metric function in case that torchmetrics doesn't support some metric.

            Parameters
            ----------
            metric_name
                Name of metric.
            num_classes
                Number of classes.
            is_matching
                Whether is matching.
            problem_type
                Type of problem, e.g., binary and multiclass.

            Returns
            -------
            torchmetrics.Metric
                A torchmetrics.Metric object.
            custom_metric_func
                A customized metric function.
            """
            if metric_name == BER:
                return Balanced_Error_Rate()
            elif metric_name in [SM, EM, FM, MAE]:
                return COD_METRICS_NAMES[metric_name]
            elif metric_name == IOU:
                if num_classes == 1:
                    return Binary_IoU()
                else:
                    return Multiclass_IoU(num_classes=num_classes)
            elif metric_name == DICE:
                if num_classes == 1:
                    return Binary_DICE()
                else:
                    return Multiclass_DICE(num_classes=num_classes)
            else:
                raise ValueError(f"Unknown metric {metric_name}")

        outputs = self.predict_per_run(
            data=data,
            realtime=realtime,
            requires_label=False,
        )

        if self._output_shape == 1:
            logits = extract_from_output(ret_type=LOGITS, outputs=outputs, as_ndarray=False)
        else:
            logits = extract_from_output(ret_type=SEMANTIC_MASK, outputs=outputs, as_ndarray=False)
        y_pred = logits.float()
        y_true = [ele[LABEL] for ele in outputs]
        y_true = torch.cat(y_true)

        assert len(y_true) == len(y_pred)

        results = {}
        if isinstance(metrics, str):
            metrics = [metrics]
        for per_metric_name in metrics:
            per_metric = get_metric_predict(metric_name=per_metric_name.lower(), num_classes=self._output_shape)
            for y_p, y_t in zip(y_pred, y_true):
                per_metric.update(y_p.unsqueeze(0), y_t.unsqueeze(0))
            score = per_metric.compute()
            results[per_metric_name] = score.item()

        if return_pred:
            return results, outputs
        else:
            return results
    
    def _evaluate_with_tta(
        self,
        data: Union[pd.DataFrame, dict, list],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        cache_dir: Optional[str] = None,
        resume_from_cache: bool = True,
    ):
        """
        Evaluate using Test-Time Augmentation with checkpoint support.
        
        This method performs TTA for each image and evaluates the results.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, dict, list]
            Input data containing image and label paths
        metrics : Optional[Union[str, List[str]]]
            Metrics to compute
        return_pred : Optional[bool]
            Whether to return predictions
        cache_dir : Optional[str]
            Directory to cache TTA predictions for resume (default: None, no caching)
        resume_from_cache : bool
            Whether to resume from cache if exists (default: True)
        """
        logger.info(f"Evaluating with TTA ({len(self._tta_predictor.transforms)} augmentations per image)...")
        
        def get_metric_predict(metric_name: str, num_classes: Optional[int] = None):
            """Get metric function."""
            if metric_name == BER:
                return Balanced_Error_Rate()
            elif metric_name in [SM, EM, FM, MAE]:
                return COD_METRICS_NAMES[metric_name]
            elif metric_name == IOU:
                if num_classes == 1:
                    return Binary_IoU()
                else:
                    return Multiclass_IoU(num_classes=num_classes)
            elif metric_name == DICE:
                if num_classes == 1:
                    return Binary_DICE()
                else:
                    return Multiclass_DICE(num_classes=num_classes)
            else:
                raise ValueError(f"Unknown metric {metric_name}")
        
        # Get model and preprocessor
        model = self._model
        model.eval()
        
        # Ensure model is on the correct device (GPU if available)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            if next(model.parameters()).device.type != 'cuda':
                logger.info("Moving model to GPU for TTA inference...")
                model = model.cuda()
        else:
            device = torch.device('cpu')
            logger.warning("GPU not available, using CPU for TTA (will be slow)")
        
        logger.info(f"TTA inference device: {device}")
        
        # ============================================================
        # 方案 4: Sanity Check - 提前发现问题
        # ============================================================
        logger.info("🔍 Running sanity checks before full evaluation...")
        
        # Sanity Check 1: 验证数据
        if len(data) == 0:
            raise ValueError("❌ No data to evaluate!")
        logger.info(f"  ✓ Total images to process: {len(data)}")
        logger.info(f"  ✓ DataFrame columns: {data.columns.tolist()}")
        
        # Sanity Check 2: 检查设备和模型
        logger.info(f"  ✓ Model device: {device}")
        logger.info(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            logger.info(f"  ✓ GPU memory allocated: {allocated_gb:.2f} GB")
        
        # Sanity Check 3: 测试单个样本的 TTA pipeline
        logger.info(f"  Testing TTA pipeline on first image...")
        test_row = data.iloc[0]
        test_img_path = test_row['image']
        
        # Get data processors for image preprocessing
        from torchvision import transforms as T
        
        # Create preprocessing pipeline similar to model's preprocessing
        # Based on SAM's image preprocessing
        def preprocess_image_for_sam(image_np):
            """Preprocess image for SAM model."""
            from PIL import Image as PILImage
            
            # Ensure correct format
            if len(image_np.shape) == 2:
                image_np = np.stack([image_np] * 3, axis=2)
            
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            
            # Convert to PIL
            img_pil = PILImage.fromarray(image_np.astype(np.uint8))
            
            # Resize to model's expected input size (typically 1024x1024 for SAM)
            target_size = model.image_size
            img_resized = img_pil.resize((target_size, target_size), PILImage.BILINEAR)
            
            # Convert to tensor and normalize
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            
            # Close PIL objects
            img_pil.close()
            img_resized.close()
            
            # Apply model's normalization (SAM uses specific mean/std)
            mean = np.array(model.image_mean).reshape(1, 1, 3)
            std = np.array(model.image_std).reshape(1, 1, 3)
            img_array = (img_array - mean) / std
            
            # Convert to tensor (C, H, W)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().contiguous()
            
            # Free intermediate array
            del img_array
            
            return img_tensor
        
        # Create a prediction function for TTA
        def predict_fn(
            image: np.ndarray,
            box_prompt: Optional[np.ndarray] = None,
            box_ref_shape: Optional[Tuple[int, int]] = None,
        ) -> np.ndarray:
            """
            Prediction function for a single image used by TTA.
            
            Parameters
            ----------
            image : np.ndarray
                Input image (H, W, C) in range [0, 255] or [0, 1]
            
            Returns
            -------
            prob : np.ndarray
                Probability map (H, W) for binary or (C, H, W) for multi-class
            """
            # Preprocess image
            img_tensor = preprocess_image_for_sam(image)
            
            # Add batch dimension and move to device
            img_tensor = img_tensor.unsqueeze(0).to(device, non_blocking=True)

            box_tensor = None
            if box_prompt is not None:
                # box_prompt stays in ORIGINAL image coords (no TTA transform)
                ref_h, ref_w = box_ref_shape if box_ref_shape is not None else image.shape[:2]
                scale_x = model.image_size / float(ref_w)
                scale_y = model.image_size / float(ref_h)
                box_scaled = box_prompt.astype(np.float32).copy()
                box_scaled[0::2] *= scale_x
                box_scaled[1::2] *= scale_y
                box_scaled = np.clip(box_scaled, 0, model.image_size - 1)
                box_tensor = torch.from_numpy(box_scaled).to(device).unsqueeze(0).unsqueeze(0)  # (1,1,4)
            
            def forward_prob(optional_box_tensor: Optional[torch.Tensor]) -> np.ndarray:
                # Create batch dict for model
                # Need to provide a dummy label for model's forward pass
                batch = {
                    model.prefix + "_image": img_tensor,
                    model.prefix + "_label": torch.zeros(
                        (1, model.image_size, model.image_size), dtype=torch.long, device=device
                    ),
                }
                if optional_box_tensor is not None:
                    batch[model.box_key] = optional_box_tensor
                outputs = model(batch)

                logits = outputs[model.prefix][LOGITS]
                if self._output_shape == 1:
                    # Binary segmentation - remove channel dim
                    prob = torch.sigmoid(logits[0, 0]).cpu().numpy().copy()
                else:
                    # Multi-class segmentation
                    prob = torch.softmax(logits[0], dim=0).cpu().numpy().copy()

                # Free GPU memory immediately and aggressively
                del batch, outputs, logits
                return prob

            # Forward pass
            with torch.no_grad():
                mode = getattr(self, "_tta_box_prompt_mode", "off")
                if mode == "off" or box_tensor is None:
                    # No box prompt
                    prob = forward_prob(None)
                elif mode == "replace":
                    # Use only box-prompted prediction
                    prob = forward_prob(box_tensor)
                elif mode == "add":
                    # Fuse base prediction + box-prompted prediction (treat box as an extra inference view)
                    prob_base = forward_prob(None)
                    prob_box = forward_prob(box_tensor)
                    prob = ((prob_base + prob_box) / 2.0).astype(np.float32, copy=False)
                    del prob_base, prob_box
                else:
                    raise ValueError(f"Unknown tta_box_prompt_mode: {mode}")

                # Free GPU memory immediately and aggressively
                del img_tensor
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for GPU to finish

            return prob
        
        # Complete Sanity Check 3: Test TTA on first image
        try:
            test_img_pil = Image.open(test_img_path)
            test_img = np.array(test_img_pil)
            test_img_pil.close()
            
            if len(test_img.shape) == 2:
                test_img = np.stack([test_img] * 3, axis=2)
            
            logger.info(f"    Image shape: {test_img.shape}")
            
            # Test TTA prediction
            test_pred_mask, test_pred_prob = self._tta_predictor.predict_with_tta(
                test_img, predict_fn, return_probs=True
            )
            
            logger.info(f"    Output shape: {test_pred_prob.shape}")
            
            # Verify output shape matches input
            assert test_pred_prob.shape[:2] == test_img.shape[:2], \
                f"❌ Output shape mismatch! Expected {test_img.shape[:2]}, got {test_pred_prob.shape[:2]}"
            
            logger.info(f"  ✓ TTA pipeline test PASSED!")
            
            # Clean up test data
            del test_img, test_pred_mask, test_pred_prob
            
        except Exception as e:
            logger.error(f"❌ Sanity check FAILED! Error: {e}")
            logger.error("Please fix the issue before running full evaluation.")
            raise RuntimeError(f"TTA Sanity Check Failed: {e}") from e
        
        logger.info("✅ All sanity checks passed! Starting full evaluation...\n")
        
        # ============================================================
        # 方案 2: 断点续传机制
        # ============================================================
        import os
        import pickle
        
        # Setup cache directory
        processed_indices = set()
        all_preds = []
        all_labels = []
        cache_file = None
        processed_indices_file = None
        
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "tta_predictions.pkl")
            processed_indices_file = os.path.join(cache_dir, "processed_indices.txt")
            
            # Try to load cached data
            if resume_from_cache and os.path.exists(cache_file):
                try:
                    logger.info(f"📂 Loading cached predictions from {cache_file}")
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                        all_preds = cache_data['preds']
                        all_labels = cache_data['labels']
                        processed_indices = set(cache_data['indices'])
                    logger.info(f"✅ Loaded {len(all_preds)} cached predictions (resuming from checkpoint)")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load cache: {e}. Starting fresh.")
                    processed_indices = set()
                    all_preds = []
                    all_labels = []
            else:
                if cache_dir:
                    logger.info(f"💾 Caching enabled. Predictions will be saved to {cache_dir}")
        
        # Initialize metrics early so we can stream updates and avoid storing all predictions
        metric_objects = {}
        if isinstance(metrics, str):
            metrics = [metrics]
        for per_metric_name in metrics:
            metric_objects[per_metric_name] = get_metric_predict(
                metric_name=per_metric_name.lower(),
                num_classes=self._output_shape
            )

        # Determine whether to stream metrics (no caching => no large tensors kept)
        streaming_metrics = cache_dir is None

        # Process each image with TTA
        import time
        import gc
        start_time = time.time()
        total_images = len(data)

        for idx, row in data.iterrows():
            # ⭐ 跳过已处理的样本（断点续传）
            if idx in processed_indices:
                logger.info(f"⏭️  Skipping image {idx+1}/{total_images} (already processed)")
                continue

            img_start = time.time()

            # Load image and label
            image_path = row['image']
            label_path = row['label'] if 'label' in row else None

            # Load image with proper resource management
            img_pil = Image.open(image_path)
            image = np.array(img_pil)
            img_pil.close()  # Release file handle

            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=2)  # Convert grayscale to RGB

            # Load label if exists
            label = None
            gt_box = None
            if label_path:
                label_pil = Image.open(label_path)
                label = np.array(label_pil)
                label_pil.close()  # Release file handle

                if len(label.shape) == 3:
                    label = label[:, :, 0]  # Take first channel if RGB

                # Binarize label: convert [0, 255] to [0, 1]
                # This matches the preprocessing in semantic_seg_metrics.py
                if self._output_shape == 1:  # Binary segmentation
                    label = (label > 128).astype(np.int64)

                if not streaming_metrics:
                    all_labels.append(torch.from_numpy(label))

                # Compute GT bounding box (single box per sample, original coords)
                ys, xs = np.where(label > 0)
                if xs.size > 0 and ys.size > 0:
                    gt_box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

            # If predicted boxes are provided in DataFrame, override gt_box
            if {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}.issubset(row.index):
                vals = [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]]
                if not any([v is None or (isinstance(v, float) and np.isnan(v)) for v in vals]):
                    gt_box = np.array(vals, dtype=np.float32)

            # Predict with TTA
            if self._output_shape == 1:
                # Binary segmentation
                pred_mask, pred_prob = self._tta_predictor.predict_with_tta(
                    image, predict_fn, return_probs=True, box_prompt=gt_box, box_ref_shape=image.shape[:2]
                )
                pred_tensor = torch.from_numpy(pred_prob)
                if not streaming_metrics:
                    # Store only when caching/resume is enabled
                    all_preds.append(pred_tensor)
                # Free memory
                del pred_mask, pred_prob
            else:
                # Multi-class segmentation
                _, pred_prob = self._tta_predictor.predict_with_tta(
                    image, predict_fn, return_probs=True, box_prompt=gt_box, box_ref_shape=image.shape[:2]
                )
                pred_tensor = torch.from_numpy(pred_prob)
                if not streaming_metrics:
                    all_preds.append(pred_tensor)
                # Free memory
                del pred_prob

            # Stream metric updates to avoid holding all predictions in memory
            if streaming_metrics:
                y_p = pred_tensor.float()
                if label is None:
                    raise ValueError("Labels are required for evaluation")
                y_t = torch.from_numpy(label)
                y_t = y_t.long() if y_t.dtype != torch.long else y_t

                # Resize prediction to match label size if needed
                if y_p.shape[-2:] != y_t.shape[-2:]:
                    target_size = y_t.shape[-2:]
                    y_p = F.interpolate(
                        y_p.unsqueeze(0),
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)

                for metric_name, metric_obj in metric_objects.items():
                    metric_obj.update(y_p.unsqueeze(0), y_t.unsqueeze(0))

            # Free image and label memory
            del image
            if label is not None:
                del label

            # ⭐ 标记为已处理（断点续传）
            processed_indices.add(idx)

            img_time = time.time() - img_start

            if idx == 0:
                logger.info(f"First image processed in {img_time:.2f}s (includes warmup)")

            # ⭐ 定期保存缓存（每 10 张图或最后一张）
            num_processed = len(processed_indices)
            if (not streaming_metrics) and cache_file is not None and (num_processed % 10 == 0 or num_processed == total_images):
                try:
                    logger.info(f"💾 Saving checkpoint ({num_processed}/{total_images} images)")
                    with open(cache_file, 'wb') as f:
                        pickle.dump({
                            'preds': all_preds,
                            'labels': all_labels,
                            'indices': list(processed_indices)
                        }, f)

                    # Save processed indices as text file (for easy inspection)
                    with open(processed_indices_file, 'w') as f:
                        f.write('\n'.join(map(str, sorted(processed_indices))))

                    logger.info(f"✅ Checkpoint saved successfully")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to save checkpoint: {e}")

            # More aggressive garbage collection to prevent memory leaks
            if (idx + 1) % 5 == 0:  # Every 5 images instead of 10
                gc.collect()  # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear GPU cache

            # Progress logging
            if (idx + 1) % 10 == 0 or num_processed == total_images:
                elapsed = time.time() - start_time
                avg_time = elapsed / num_processed if num_processed > 0 else 0
                remaining = total_images - num_processed
                eta = avg_time * remaining if remaining > 0 else 0
                logger.info(
                    f"Processed {num_processed}/{total_images} images with TTA "
                    f"(avg: {avg_time:.2f}s/img, ETA: {eta/60:.1f}min)"
                )

                # ⭐ 中间指标：在评估过程中周期性输出当前已处理样本的指标
                # 说明：
                # - 仅在 streaming_metrics=True（即未开启 cache_dir，走低内存流式评估）时启用
                # - 使用 torchmetrics.compute() 不会清空内部状态，因此可以安全多次调用
                if streaming_metrics:
                    try:
                        intermediate_results = {}
                        for metric_name, metric_obj in metric_objects.items():
                            score = metric_obj.compute()
                            # score 可能是 tensor 或数值，这里统一为 float 便于日志展示
                            if hasattr(score, "item"):
                                score = score.item()
                            intermediate_results[metric_name] = float(score)
                        if intermediate_results:
                            logger.info(
                                f"Intermediate TTA metrics after {num_processed} images: "
                                f"{intermediate_results}"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to compute intermediate metrics: {e}")
        
        # Final garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Finished processing all {len(data)} images, computing metrics...")

        results = {}
        if streaming_metrics:
            # Metrics were updated on the fly; just compute
            for metric_name, metric_obj in metric_objects.items():
                score = metric_obj.compute()
                results[metric_name] = score.item()
        else:
            # Check that we have predictions and labels
            if len(all_labels) == 0:
                raise ValueError("Labels are required for evaluation")
            
            if len(all_preds) != len(all_labels):
                raise ValueError(f"Number of predictions ({len(all_preds)}) != number of labels ({len(all_labels)})")

            # Process each image individually (can't stack due to different sizes)
            for y_p, y_t in zip(all_preds, all_labels):
                # Ensure predictions and labels are on CPU and have matching sizes
                y_p = y_p.float()
                y_t = y_t.long() if y_t.dtype != torch.long else y_t
                
                # Resize prediction to match label size if needed
                if y_p.shape[-2:] != y_t.shape[-2:]:
                    target_size = y_t.shape[-2:]
                    y_p = F.interpolate(
                        y_p.unsqueeze(0),  # Add batch dim
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)  # Remove batch dim
                
                # Update all metrics with this image
                for metric_name, metric_obj in metric_objects.items():
                    metric_obj.update(y_p.unsqueeze(0), y_t.unsqueeze(0))

            # Compute final scores
            for metric_name, metric_obj in metric_objects.items():
                score = metric_obj.compute()
                results[metric_name] = score.item()

        logger.info(f"TTA evaluation completed: {results}")
        
        # ⭐ 清理缓存文件（评估成功完成后）
        if cache_file is not None and os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                if processed_indices_file and os.path.exists(processed_indices_file):
                    os.remove(processed_indices_file)
                logger.info(f"🗑️  Cleaned up cache files (evaluation completed successfully)")
            except Exception as e:
                logger.warning(f"⚠️  Failed to clean up cache: {e}")
        
        # Free memory
        del all_preds, all_labels, metric_objects
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if return_pred:
            logger.warning("return_pred=True not fully supported with TTA (memory optimization)")
            return results, []
        else:
            return results

    def get_litmodule_per_run(
        self,
        model=None,
        model_postprocess_fn=None,
        peft_param_names=None,
        optim_kwargs=None,
        distillation_kwargs=None,
        is_train=True,
    ):
        gspo_trainer = None
        if is_train:
            gspo_trainer = self._maybe_create_gspo_trainer()
        if is_train:
            return SemanticSegmentationLitModule(
                model=model,
                model_postprocess_fn=model_postprocess_fn,
                trainable_param_names=peft_param_names,
                gspo_trainer=gspo_trainer,
                gspo_adapter_cfg=self._get_gspo_adapter_cfg() if is_train else None,
                gspo_attn_lora_cfg=self._get_gspo_attn_lora_cfg() if is_train else None,
                train_box_prompt_cfg=self._train_box_prompt_cfg,
                train_bbox_predictor=self._train_bbox_predictor,
                **optim_kwargs,
            )
        else:
            return SemanticSegmentationLitModule(
                model=self._model,
                model_postprocess_fn=self._model_postprocess_fn,
                train_box_prompt_cfg=self._train_box_prompt_cfg,
                train_bbox_predictor=self._train_bbox_predictor,
                **optim_kwargs,
            )

    def _maybe_create_gspo_trainer(self):
        """
        Auto-create GSPO trainer if the config enables GSPO.
        The trainer implementation lives in examples/automm/Conv-LoRA/gspo_trainer.py.
        """
        if self._config is None:
            return None

    def _get_gspo_adapter_cfg(self) -> Dict:
        """
        GSPO-Adapter extension config (encoder adapters only).

        Defaults are chosen to be fully backward-compatible:
        - train_encoder_adapter: False => encoder adapters are frozen during GSPO-active steps
        - noise stds default to 0.0 => no extra exploration injected
        """
        cfg = {
            "train_encoder_adapter": False,
            "encoder_adapter_noise_std": 0.0,
            "encoder_adapter_gate_noise_std": 0.0,
        }
        if self._config is None:
            return cfg
        gspo_cfg = getattr(self._config.optim, "gspo", None)
        if gspo_cfg is None:
            return cfg
        cfg["train_encoder_adapter"] = bool(getattr(gspo_cfg, "train_encoder_adapter", False))
        cfg["encoder_adapter_noise_std"] = float(getattr(gspo_cfg, "encoder_adapter_noise_std", 0.0))
        cfg["encoder_adapter_gate_noise_std"] = float(getattr(gspo_cfg, "encoder_adapter_gate_noise_std", 0.0))
        return cfg

    def _get_gspo_attn_lora_cfg(self) -> Dict:
        """
        GSPO-AttnLoRA extension config (decoder attention LoRA only).

        Backward-compatible defaults:
        - train_decoder_attention_lora: False => decoder attention LoRA (if enabled) frozen during GSPO steps
        - noise stds default to 0.0 => no extra exploration injected
        """
        cfg = {
            "train_decoder_attention_lora": False,
            "decoder_attention_lora_noise_std": 0.0,
            "decoder_attention_lora_gate_noise_std": 0.0,
        }
        if self._config is None:
            return cfg
        gspo_cfg = getattr(self._config.optim, "gspo", None)
        if gspo_cfg is None:
            return cfg
        cfg["train_decoder_attention_lora"] = bool(getattr(gspo_cfg, "train_decoder_attention_lora", False))
        cfg["decoder_attention_lora_noise_std"] = float(getattr(gspo_cfg, "decoder_attention_lora_noise_std", 0.0))
        cfg["decoder_attention_lora_gate_noise_std"] = float(getattr(gspo_cfg, "decoder_attention_lora_gate_noise_std", 0.0))
        return cfg

        lora_cfg = getattr(self._config.optim, "lora", None)
        if not lora_cfg or not getattr(lora_cfg, "gspo_enabled", False):
            return None

        gspo_cfg = getattr(self._config.optim, "gspo", None)
        lambda_smooth = getattr(gspo_cfg, "lambda_smooth", 0.1) if gspo_cfg else 0.1
        lambda_boundary = getattr(gspo_cfg, "lambda_boundary", 0.3) if gspo_cfg else 0.3
        w_boundary = getattr(gspo_cfg, "w_boundary", 0.3) if gspo_cfg else 0.3
        w_smooth = getattr(gspo_cfg, "w_smooth", 0.1) if gspo_cfg else 0.1
        w_thin = getattr(gspo_cfg, "w_thin", 0.05) if gspo_cfg else 0.05

        # Dynamically load GSPOConvLoRATrainer from examples path.
        examples_root = Path(__file__).resolve().parents[3] / "examples" / "automm" / "Conv-LoRA" / "gspo_trainer.py"
        if not examples_root.exists():
            logger.warning("GSPO is enabled but gspo_trainer.py not found at %s; GSPO will be skipped.", examples_root)
            return None

        spec = importlib.util.spec_from_file_location("autogluon_multimodal.gspo_trainer", examples_root)
        if spec is None or spec.loader is None:
            logger.warning("Failed to load GSPO trainer spec from %s; GSPO will be skipped.", examples_root)
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        GSPOConvLoRATrainer = getattr(module, "GSPOConvLoRATrainer", None)
        if GSPOConvLoRATrainer is None:
            logger.warning("GSPOConvLoRATrainer not found in %s; GSPO will be skipped.", examples_root)
            return None

        try:
            return GSPOConvLoRATrainer(
                predictor=None,  # Not required by current implementation
                group_size=getattr(lora_cfg, "gspo_group_size", 3),
                warmup_epochs=getattr(lora_cfg, "gspo_warmup_epochs", 5),
                contrastive_weight=getattr(lora_cfg, "gspo_contrastive_weight", 0.1),
                quality_metric="iou",
                advantage_temperature=5.0,
                lambda_smooth=lambda_smooth,
                lambda_boundary=lambda_boundary,
                w_boundary=w_boundary,
                w_smooth=w_smooth,
                w_thin=w_thin,
            )
        except Exception as e:
            logger.warning("Failed to instantiate GSPOConvLoRATrainer: %s; GSPO will be skipped.", e)
            return None

    def on_predict_start(self, data: pd.DataFrame):
        data = self.data_to_df(data=data)
        if self._output_shape is None:  # for zero-shot evaluation/prediction
            self._output_shape = self.get_semantic_segmentation_class_num(data)
        self.ensure_predict_ready()
        return data
    
    def enable_tta(
        self,
        scales: List[float] = [0.75, 1.0, 1.25],
        flips: List[str] = ["none", "horizontal"],
        rotations: List[float] = [0],
        fusion_method: str = "mean",
        scale_weights: Optional[Dict[float, float]] = None,
        threshold: float = 0.5,
        min_area_ratio: float = 0.001,
        use_morphology: bool = False,
        resize_method: str = "bilinear",
        cache_dir: Optional[str] = None,
        resume_from_cache: bool = True,
        box_prompt_mode: str = "off",
    ):
        """
        Enable Test-Time Augmentation (TTA) for inference.
        
        Parameters
        ----------
        scales : List[float]
            List of scale factors for multi-scale testing.
            Recommended: [0.75, 1.0, 1.25] for fast (6 inferences).
            Or [0.5, 0.75, 1.0, 1.25, 1.5] for better results.
        flips : List[str]
            List of flip types: "none", "horizontal", "vertical".
            Recommended: ["none", "horizontal"] for most cases.
        rotations : List[float]
            List of rotation angles in degrees.
            Recommended: [0] for fast, or [-10, 0, 10] for better (18 inferences).
        fusion_method : str
            Method to fuse predictions: "mean" or "weighted_mean".
        scale_weights : Optional[Dict[float, float]]
            Weights for each scale when using weighted_mean.
        threshold : float
            Threshold for binary segmentation (can be tuned on validation set).
        min_area_ratio : float
            Remove connected components with area < min_area_ratio * image_area.
        use_morphology : bool
            Whether to apply morphological closing for smoothing.
        resize_method : str
            Resize interpolation method: "bilinear" or "bicubic".
            Recommended: "bilinear" for speed, "bicubic" for quality.
        cache_dir : Optional[str]
            Directory to cache TTA predictions for resume (default: None, no caching).
            Useful for long-running evaluations that might be interrupted.
        resume_from_cache : bool
            Whether to resume from cache if exists (default: True).
        
        Examples
        --------
        >>> # Basic TTA (6 inferences)
        >>> predictor.enable_tta()
        
        >>> # Advanced TTA (18 inferences)
        >>> predictor.enable_tta(
        ...     scales=[0.75, 1.0, 1.25],
        ...     flips=["none", "horizontal"],
        ...     rotations=[-10, 0, 10],
        ...     fusion_method="weighted_mean",
        ...     resize_method="bicubic"
        ... )
        """
        self._tta_predictor = TTAPredictor(
            scales=scales,
            flips=flips,
            rotations=rotations,
            fusion_method=fusion_method,
            scale_weights=scale_weights,
            threshold=threshold,
            min_area_ratio=min_area_ratio,
            use_morphology=use_morphology,
            resize_method=resize_method,
        )
        
        # Store cache configuration
        self._tta_cache_dir = cache_dir
        self._tta_resume_from_cache = resume_from_cache
        self._tta_box_prompt_mode = box_prompt_mode
        
        logger.info(f"TTA enabled with {len(self._tta_predictor.transforms)} augmentations")
        if cache_dir:
            logger.info(f"TTA caching enabled: {cache_dir} (resume={resume_from_cache})")
    
    def disable_tta(self):
        """Disable Test-Time Augmentation."""
        self._tta_predictor = None
        logger.info("TTA disabled")

    def evaluate(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = False,
        **kwargs,
    ):
        """
        Evaluate model on a test dataset.

        Parameters
        ----------
        data
            A dataframe, containing the same columns as the training data.
            Or a str, that is a path of the annotation file for detection.
        metrics
            A list of metric names to report.
            If None, we only return the score for the stored `_eval_metric_name`.
        return_pred
            Whether to return the prediction result of each row.
        realtime
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        A dictionary with the metric names and their corresponding scores.
        Optionally return a dataframe of prediction results.
        """
        data = self.on_predict_start(data)
        return self.evaluate_semantic_segmentation(data, metrics, realtime)

    def predict(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        realtime: Optional[bool] = False,
        save_results: Optional[bool] = None,
        **kwargs,
    ):
        """
        Predict values for the label column of new data.

        Parameters
        ----------
        data
            The data to make predictions for. Should contain same column names as training data and
            follow same format (except for the `label` column).
        realtime
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.
        save_results
            Whether to save the prediction results.

        Returns
        -------
        Array of predictions, one corresponding to each row in given dataset.
        When save_results is True, the output is a pandas dataframe containing the path of the predicted mask file for each input image.
        Otherwise, the output will have shape (#samples, height, width).
        """
        data = self.on_predict_start(data)
        if self._output_shape == 1:
            ret_type = LOGITS
        else:
            ret_type = SEMANTIC_MASK

        outputs = self.predict_per_run(
            data=data,
            realtime=realtime,
            requires_label=False,
        )

        logits = self.post_process_prediction(data, outputs, ret_type)

        pred = []
        for logit in logits:
            logit = logit.numpy()
            if ret_type == SEMANTIC_MASK:
                pred.append(logit.argmax(axis=1))
            else:
                pred.append((logit > 0.5).squeeze(axis=1))

        if save_results:
            self._save_path = setup_save_path(
                old_save_path=self._save_path,
                warn_if_exist=False,
            )
            pred = self.save_segmentation_result(
                pred=pred,
                data=data,
                result_path=self._save_path,
            )

        return pred

        # if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
        #     # TODO
        #     pred = self._as_pandas(data=data, to_be_converted=pred)

    def predict_proba(
        self,
        data: Union[pd.DataFrame, dict, list],
        as_pandas: Optional[bool] = None,
        as_multiclass: Optional[bool] = True,
        realtime: Optional[bool] = False,
        **kwargs,
    ):
        """
        Predict probabilities class probabilities rather than class labels.
        This is only for the classification. Calling it for regression will throw an exception.

        Parameters
        ----------
        data
            The data to make predictions for. Should contain same column names as training data and
              follow same format (except for the `label` column).
        as_pandas
            Whether to return the output as a pandas DataFrame(Series) (True) or numpy array (False).
        as_multiclass
            Whether to return the probability of all labels or
            just return the probability of the positive class for binary classification problems.
        realtime
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        Array of predicted class-probabilities, corresponding to each row in the given data.
        The output will always have shape (#samples, #classes, height, width).
        """
        assert (self._output_shape == 1 and as_multiclass == False) or (
            self._output_shape > 1 and as_multiclass == True
        )
        data = self.on_predict_start(data)

        outputs = self.predict_per_run(
            data=data,
            realtime=realtime,
            requires_label=False,
        )

        if as_multiclass:
            ret_type = SEMANTIC_MASK
        else:
            ret_type = LOGITS

        logits = self.post_process_prediction(data, outputs, ret_type)

        prob = []
        for logit in logits:
            logit = logit.numpy()
            if ret_type == SEMANTIC_MASK:
                prob.append(softmax(logit, axis=1))
            else:
                prob.append(logit)

        return prob

    def extract_embedding(
        self,
        data: Union[pd.DataFrame, dict, list],
        as_tensor: Optional[bool] = False,
        as_pandas: Optional[bool] = False,
        realtime: Optional[bool] = False,
        **kwargs,
    ):
        raise NotImplementedError("Semantic segmentation doesn't support calling `extract_embedding` yet.")

    def save_segmentation_result(self, pred: Iterable, data: Union[pd.DataFrame, Dict], result_path: str):
        """
        Saving segmentation results in pd.DataFrame format (per image)

        Parameters
        ----------
        pred
            List containing segmentation results for one image
        data
            Pandas data frame or dict containing the image information to be tested
        result_path
            Path to save result
        Returns
        -------
        The paths of the segmentation results as pandas DataFrame
        """

        def show_mask(mask, ax):
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)

        image_column_name = self.get_image_column_name(data)
        if isinstance(data, dict):
            image_names = data[image_column_name]
        else:
            image_names = data[image_column_name].to_list()
        results = []

        mask_path = os.path.join(result_path, "masks")
        txt_path = os.path.join(result_path, "result.txt")
        os.makedirs(mask_path, exist_ok=True)
        for image_pred, image_name in zip(pred, image_names):
            if self._output_shape == 1:
                mask = Image.fromarray(image_pred.squeeze(axis=0))
                per_mask_path = os.path.join(mask_path, os.path.basename(image_name))
                mask.save(per_mask_path)
            else:
                masks = []
                classes = np.unique(image_pred)
                for class_id in classes:
                    if class_id == 0:  # bg
                        continue
                    masks.append(image_pred == class_id)

                for mask in masks:
                    show_mask(mask, plt.gca())
                mask_name = ""
                for i in os.path.basename(image_name).split(".")[:-1]:
                    mask_name += i
                per_mask_path = os.path.join(mask_path, os.path.basename(image_name))
                plt.axis("off")
                plt.savefig(per_mask_path, bbox_inches="tight", dpi=300, pad_inches=0.0)

            results.append([image_name, per_mask_path])

        result_df = pd.DataFrame(results, columns=["image", "mask"])
        result_df.to_csv(txt_path, index=False)
        return result_df

    def post_process_prediction(self, data, outputs, ret_type):
        """
        Post-process segmentation results to match the size of original input images.

        Parameters
        ----------
        data
            Pandas data frame or dict containing the image information.
        outputs
            A list of segmentation output results.
        ret_type
            What kind of information to extract from model outputs.

        Returns
        -------
        A list of the post-processed segmentation results.
        """
        logits = [ele[ret_type] for ele in outputs]
        image_column_name = self.get_image_column_name(data)
        for idx in range(data.shape[0]):
            ori_image_size = Image.open(data[image_column_name][idx]).size  # width, height
            logits[idx] = F.interpolate(
                logits[idx].float(), (ori_image_size[1], ori_image_size[0]), mode="bilinear", align_corners=False
            )
        return logits

    def get_image_column_name(self, data: pd.DataFrame):
        if self.column_types is None:
            column_names = list(data.columns)
            if self._label_column in column_names:
                column_names.remove(self._label_column)
            assert (
                len(column_names) == 1
            ), f"More than one image columns {column_names} exist in the data. Make sure to provide data with one image column."
            return column_names[0]
        else:
            for k, v in self.column_types.items():
                if v == SEMANTIC_SEGMENTATION_IMG:
                    return k
        return None

```

```worktree-python
import logging
from typing import Callable, Dict, Optional

import torch
import torchmetrics
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss

from ..constants import CLASS_LOGITS, LOGITS, MOE_LOSS, SEMANTIC_MASK, WEIGHT
from ..models.utils import run_model
from .lit_module import LitModule
from .metrics.semantic_seg_metrics import Multiclass_IoU

logger = logging.getLogger(__name__)


class SemanticSegmentationLitModule(LitModule):
    """
    Control the loops for training, evaluation, and prediction. This module is independent of
    the model definition. This class inherits from the Pytorch Lightning's LightningModule:
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    
    GSPO enhancement:
    - Supports group-level training with quality-aware feedback
    - Integrates contrastive loss for better expert selection
    """
    
    def __init__(
        self,
        *args,
        gspo_trainer=None,
        gspo_adapter_cfg=None,
        gspo_attn_lora_cfg=None,
        train_box_prompt_cfg=None,
        train_bbox_predictor=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gspo_trainer = gspo_trainer
        # GSPO-Adapter extension config (encoder adapters only).
        # Defaults keep backward compatibility: encoder adapters are NOT trained during GSPO-active steps.
        self.gspo_adapter_cfg = gspo_adapter_cfg or {
            "train_encoder_adapter": False,
            "encoder_adapter_noise_std": 0.0,
            "encoder_adapter_gate_noise_std": 0.0,
        }
        # GSPO-AttnLoRA extension config (decoder attention LoRA only).
        self.gspo_attn_lora_cfg = gspo_attn_lora_cfg or {
            "train_decoder_attention_lora": False,
            "decoder_attention_lora_noise_std": 0.0,
            "decoder_attention_lora_gate_noise_std": 0.0,
        }
        # Training-time box prompt config (dict with keys: mode, p_no, p_gt, p_noisy, noise_frac, bbox_predictor)
        self.train_box_prompt_cfg = train_box_prompt_cfg or {
            "mode": "off",
            "p_no": 0.0,
            "p_gt": 0.0,
            "p_noisy": 0.0,
            "noise_frac": 0.12,
        }
        # BBoxPromptPredictor instance for "predict" mode
        self.train_bbox_predictor = train_bbox_predictor

    def _compute_loss(self, output: Dict, label: torch.Tensor, **kwargs):
        loss = 0
        for _, per_output in output.items():
            weight = per_output[WEIGHT] if WEIGHT in per_output else 1
            if isinstance(self.loss_func, Mask2FormerLoss):
                mask_labels = [mask_labels.to(per_output[LOGITS]) for mask_labels in kwargs["mask_labels"]]
                dict_loss = self.loss_func(
                    masks_queries_logits=per_output[LOGITS],  # bs, num_mask_tokens, height, width
                    class_queries_logits=per_output[CLASS_LOGITS],  # bs, num_mask_tokens, num_classes
                    mask_labels=mask_labels,
                    class_labels=kwargs["class_labels"],
                )
                for v in dict_loss.values():
                    loss += v
            else:
                loss += (
                    self.loss_func(
                        input=per_output[LOGITS],
                        target=label,
                    )
                    * weight
                )
            # MoE loss
            if MOE_LOSS in per_output:
                loss += per_output[MOE_LOSS]

        return loss

    def _compute_metric_score(
        self,
        metric: torchmetrics.Metric,
        custom_metric_func: Callable,
        logits: torch.Tensor,
        label: torch.Tensor,
        **kwargs,
    ):
        if isinstance(metric, Multiclass_IoU):
            metric.update(kwargs["semantic_masks"], label)
        else:
            metric.update(logits.float(), label)

    def _shared_step(
        self,
        batch: Dict,
    ):
        label = batch[self.model.label_key]

        # -------- Train-time box prompt injection (configurable) --------
        # 控制项：self.train_box_prompt_cfg = {
        #   mode: off / gt / noisy / mix / predict,
        #   p_no, p_gt, p_noisy: 仅在 mode=mix 时生效并会归一化,
        #   noise_frac: 噪声扰动比例（相对 bbox 宽高）
        # }
        # mode=predict: 使用 self.train_bbox_predictor 预测框作为 box prompt
        if self.training:
            cfg = self.train_box_prompt_cfg or {}
            mode = cfg.get("mode", "off")
            box_mode = None
            if mode != "off":
                p_no = max(float(cfg.get("p_no", 0.0)), 0.0)
                p_gt = max(float(cfg.get("p_gt", 0.0)), 0.0)
                p_noisy = max(float(cfg.get("p_noisy", 0.0)), 0.0)
                noise_frac = float(cfg.get("noise_frac", 0.12))

                if mode == "mix":
                    total = p_no + p_gt + p_noisy
                    if total > 0:
                        p_no, p_gt, p_noisy = [x / total for x in (p_no, p_gt, p_noisy)]
                    r = torch.rand(1).item()
                    if r < p_no:
                        box_mode = "off"
                    elif r < p_no + p_gt:
                        box_mode = "gt"
                    else:
                        box_mode = "noisy"
                elif mode == "predict":
                    box_mode = "predict"
                else:
                    box_mode = mode

            def _compute_gt_boxes(mask: torch.Tensor) -> torch.Tensor:
                """mask: (B,H,W) or (B,1,H,W) int/long -> boxes (B,4) in pixel coords."""
                if mask.dim() == 4 and mask.shape[1] == 1:
                    mask = mask[:, 0, :, :]
                b, h, w = mask.shape
                boxes = torch.zeros((b, 4), device=mask.device, dtype=torch.float32)
                for i in range(b):
                    ys, xs = torch.nonzero(mask[i] > 0, as_tuple=True)
                    if xs.numel() == 0:
                        continue
                    x1, x2 = xs.min(), xs.max()
                    y1, y2 = ys.min(), ys.max()
                    boxes[i] = torch.tensor([x1, y1, x2, y2], device=mask.device, dtype=torch.float32)
                return boxes

            def _jitter_boxes(boxes: torch.Tensor, h: int, w: int, frac: float = 0.12) -> torch.Tensor:
                """Uniform jitter relative to box size."""
                jittered = boxes.clone()
                for i in range(jittered.shape[0]):
                    x1, y1, x2, y2 = jittered[i]
                    if x2 <= x1 or y2 <= y1:
                        continue
                    bw = (x2 - x1).clamp(min=1.0)
                    bh = (y2 - y1).clamp(min=1.0)
                    dx = bw * frac
                    dy = bh * frac
                    rx1 = (torch.rand(1, device=boxes.device) * 2 - 1) * dx
                    ry1 = (torch.rand(1, device=boxes.device) * 2 - 1) * dy
                    rx2 = (torch.rand(1, device=boxes.device) * 2 - 1) * dx
                    ry2 = (torch.rand(1, device=boxes.device) * 2 - 1) * dy
                    x1n = torch.clamp(x1 + rx1, 0, w - 1)
                    y1n = torch.clamp(y1 + ry1, 0, h - 1)
                    x2n = torch.clamp(x2 + rx2, 0, w - 1)
                    y2n = torch.clamp(y2 + ry2, 0, h - 1)
                    jittered[i] = torch.stack(
                        [torch.min(x1n, x2n), torch.min(y1n, y2n), torch.max(x1n, x2n), torch.max(y1n, y2n)]
                    ).squeeze()
                return jittered

            def _predict_boxes_from_images(image_paths: list, target_h: int, target_w: int) -> torch.Tensor:
                """Use BBoxPromptPredictor to predict boxes from image paths.
                
                Args:
                    image_paths: List of image file paths
                    target_h: Target image height (model input size)
                    target_w: Target image width (model input size)
                
                Returns:
                    boxes: (B, 4) tensor in pixel coords, scaled to target_h x target_w
                """
                import numpy as np
                from PIL import Image
                
                if self.train_bbox_predictor is None:
                    raise ValueError("train_bbox_predictor is required for mode='predict'")
                
                b = len(image_paths)
                boxes = torch.zeros((b, 4), device=label.device, dtype=torch.float32)
                
                for i, img_path in enumerate(image_paths):
                    if not img_path:
                        continue
                    try:
                        # predict() returns bbox in original image pixels: (x1, y1, x2, y2)
                        bbox_px = self.train_bbox_predictor.predict(img_path)
                        if bbox_px is None:
                            continue
                        
                        # Get original image size to scale bbox to target size
                        with Image.open(img_path) as img:
                            orig_w, orig_h = img.size
                        
                        # Scale bbox from original coords to target coords
                        scale_x = target_w / orig_w
                        scale_y = target_h / orig_h
                        x1 = bbox_px[0] * scale_x
                        y1 = bbox_px[1] * scale_y
                        x2 = bbox_px[2] * scale_x
                        y2 = bbox_px[3] * scale_y
                        
                        # Clamp to valid range
                        x1 = max(0, min(x1, target_w - 1))
                        y1 = max(0, min(y1, target_h - 1))
                        x2 = max(0, min(x2, target_w - 1))
                        y2 = max(0, min(y2, target_h - 1))
                        
                        # Ensure x1 < x2 and y1 < y2
                        x1, x2 = min(x1, x2), max(x1, x2)
                        y1, y2 = min(y1, y2), max(y1, y2)
                        
                        boxes[i] = torch.tensor([x1, y1, x2, y2], device=label.device, dtype=torch.float32)
                    except Exception as e:
                        logger.warning(f"Failed to predict bbox for {img_path}: {e}")
                        continue
                
                return boxes

            if box_mode in ["gt", "noisy"]:
                gt_boxes = _compute_gt_boxes(label)
                h, w = label.shape[-2], label.shape[-1]
                if box_mode == "noisy":
                    gt_boxes = _jitter_boxes(gt_boxes, h=h, w=w, frac=noise_frac)
                if not (gt_boxes.sum(dim=1) == 0).all():
                    batch[self.model.box_key] = gt_boxes.unsqueeze(1)  # (B,1,4)
            elif box_mode == "predict":
                # Get image paths from batch
                image_path_key = self.model.image_path_key
                image_paths = batch.get(image_path_key, [])
                if image_paths:
                    h, w = label.shape[-2], label.shape[-1]
                    pred_boxes = _predict_boxes_from_images(image_paths, target_h=h, target_w=w)
                    if not (pred_boxes.sum(dim=1) == 0).all():
                        batch[self.model.box_key] = pred_boxes.unsqueeze(1)  # (B,1,4)
        # prepare_targets
        output = run_model(self.model, batch)
        if isinstance(self.loss_func, Mask2FormerLoss):
            loss = self._compute_loss(
                output=output,
                label=label,
                mask_labels=batch[self.model.mask_label_key],
                class_labels=batch[self.model.class_label_key],
            )
        else:
            loss = self._compute_loss(
                output=output,
                label=label,
            )

        return output, loss

    def validation_step(self, batch, batch_idx, **kwargs):
        """
        Per validation step. This function is registered by LightningModule.
        Refer to https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-loop

        Parameters
        ----------
        batch
            A dictionary containing the mini-batch data, including both input data and
            ground-truth labels. The mini-batch data are passed to each individual model,
            which indexes its required input data by keys with its model prefix. The
            ground-truth labels are used here to compute the validation loss and metric.
            The validation metric is used for top k model selection and early stopping.
        batch_idx
            Index of mini-batch.
        """
        output, loss = self._shared_step(batch)
        if self.model_postprocess_fn:
            output = self.model_postprocess_fn(output)
        # By default, on_step=False and on_epoch=True
        self.log("val_loss", loss)
        if isinstance(self.loss_func, Mask2FormerLoss):
            self._compute_metric_score(
                metric=self.validation_metric,
                custom_metric_func=self.custom_metric_func,
                logits=output[self.model.prefix][LOGITS],
                label=batch[self.model.label_key],
                semantic_masks=output[self.model.prefix][SEMANTIC_MASK],
            )
        else:
            self._compute_metric_score(
                metric=self.validation_metric,
                custom_metric_func=self.custom_metric_func,
                logits=output[self.model.prefix][LOGITS],
                label=batch[self.model.label_key],
            )

        self.log(
            self.validation_metric_name,
            self.validation_metric,
            on_step=False,
            on_epoch=True,
        )
    
    def training_step(self, batch, batch_idx):
        """
        Per training step with GSPO enhancement.
        
        If GSPO is enabled and warmed up, use group-level optimization.
        Otherwise, fall back to standard training.
        
        Parameters
        ----------
        batch
            A dictionary containing the mini-batch data
        batch_idx
            Index of mini-batch
            
        Returns
        -------
        Average loss of the mini-batch data
        """
        # Check if GSPO should be used
        use_gspo = (
            self.gspo_trainer is not None and 
            self.gspo_trainer.is_gspo_active(self.current_epoch)
        )
        
        if use_gspo:
            # GSPO-enhanced training
            loss, metrics, selected_experts = self._gspo_training_step(batch)
            
            # Log GSPO-specific metrics
            for key, value in metrics.items():
                self.log(f"train_{key}", value, on_step=True, on_epoch=True)
        else:
            # Standard training (same as parent class)
            output, loss = self._shared_step(batch)
            selected_experts = None
        
        # Handle manual optimization if needed
        if not self.automatic_optimization:
            if self.hparams.use_aug_optim:
                optimizer, aug_optimizer = self.optimizers()
            else:
                optimizer = self.optimizers()
                aug_optimizer = None

            lr_scheduler = self.lr_schedulers()
            loss = loss / self.hparams.accumulate_grad_batches
            self.manual_backward(loss)

            if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0 or self.trainer.is_last_batch:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                if aug_optimizer is not None:
                    aug_optimizer.step()
                    aug_optimizer.zero_grad()
        
        self.log("train_loss", loss)
        return loss
    
    def _gspo_training_step(self, batch):
        """
        GSPO group-level training step.
        
        This method generates multiple predictions per image and uses
        group-level advantage functions to weight the losses.
        """
        images = batch[self.model.image_key] if hasattr(self.model, "image_key") else batch["image"]
        labels = batch[self.model.label_key]

        # GSPO-Adapter extension:
        # - Enable GSPO mode on encoder adapters (adds exploration noise / gate sampling if configured)
        # - Optionally freeze encoder adapters during GSPO-active steps (default behavior)
        prev_requires_grad = None
        self._set_encoder_adapters_gspo_mode(
            active=True,
            adapter_noise_std=float(self.gspo_adapter_cfg.get("encoder_adapter_noise_std", 0.0)),
            gate_noise_std=float(self.gspo_adapter_cfg.get("encoder_adapter_gate_noise_std", 0.0)),
        )
        if not bool(self.gspo_adapter_cfg.get("train_encoder_adapter", False)):
            prev_requires_grad = self._set_encoder_adapter_requires_grad(trainable=False)

        # GSPO-AttnLoRA: activate LoRA exploration and optionally freeze decoder attention LoRA params
        prev_attn_lora_requires_grad = None
        self._set_decoder_attn_lora_gspo_mode(
            active=True,
            delta_noise_std=float(self.gspo_attn_lora_cfg.get("decoder_attention_lora_noise_std", 0.0)),
            gate_noise_std=float(self.gspo_attn_lora_cfg.get("decoder_attention_lora_gate_noise_std", 0.0)),
        )
        if not bool(self.gspo_attn_lora_cfg.get("train_decoder_attention_lora", False)):
            prev_attn_lora_requires_grad = self._set_decoder_attn_lora_requires_grad(trainable=False)
        
        # Define forward function for GSPO
        def forward_fn(images):
            batch_copy = batch.copy()
            if hasattr(self.model, 'image_key'):
                batch_copy[self.model.image_key] = images
            else:
                batch_copy['image'] = images
            output = run_model(self.model, batch_copy)
            
            # Extract predictions and MOE info
            logits = output[self.model.prefix][LOGITS]
            moe_loss = output[self.model.prefix].get(MOE_LOSS, 0)
            
            # Extract selected experts if available
            selected_experts = None
            if isinstance(output.get(self.model.prefix, None), dict) and "selected_experts" in output[self.model.prefix]:
                selected_experts = output[self.model.prefix]["selected_experts"]
            
            return logits, moe_loss, selected_experts
        
        # Define loss function for GSPO
        def loss_fn(predictions, targets):
            if isinstance(self.loss_func, Mask2FormerLoss):
                # Handle Mask2Former case
                return self.loss_func(predictions, targets)
            else:
                return self.loss_func(input=predictions, target=targets)
        
        try:
            # Run GSPO group training
            loss, metrics, selected_experts_groups = self.gspo_trainer.gspo_group_training_step(
                images=images,
                masks_gt=labels,
                forward_fn=forward_fn,
                loss_fn=loss_fn,
            )
        finally:
            # Restore adapter grad states and disable GSPO mode
            if prev_requires_grad is not None:
                self._restore_requires_grad(prev_requires_grad)
            self._set_encoder_adapters_gspo_mode(active=False)
            if prev_attn_lora_requires_grad is not None:
                self._restore_decoder_attn_lora_requires_grad(prev_attn_lora_requires_grad)
            self._set_decoder_attn_lora_gspo_mode(active=False)
        
        # Update expert quality feedback
        if selected_experts_groups:
            quality_scores = [
                self.gspo_trainer.compute_segmentation_quality(
                    pred, labels, self.gspo_trainer.quality_metric
                )
                for pred in [forward_fn(images)[0] for _ in range(self.gspo_trainer.group_size)]
            ]
            self.gspo_trainer.update_expert_feedback(
                selected_experts_groups, quality_scores, self.model
            )
        
        return loss, metrics, selected_experts_groups

    def _iter_encoder_adapter_named_modules(self):
        """
        Yield (name, module) for encoder adapters only (exclude decoder adapters).
        We match by module name path to avoid touching decoder adapter modules.
        """
        for name, module in self.model.named_modules():
            if "vision_encoder.layers" in name and name.endswith(".adapter"):
                yield name, module

    def _set_encoder_adapters_gspo_mode(self, active: bool, adapter_noise_std: float = 0.0, gate_noise_std: float = 0.0):
        for _, m in self._iter_encoder_adapter_named_modules():
            if hasattr(m, "set_gspo_mode"):
                # GatedAdapterLayer.set_gspo_mode(active, gate_noise_std, adapter_noise_std)
                try:
                    m.set_gspo_mode(active=active, gate_noise_std=gate_noise_std, adapter_noise_std=adapter_noise_std)
                except TypeError:
                    # AdapterLayer.set_gspo_mode(active, noise_std)
                    m.set_gspo_mode(active=active, noise_std=adapter_noise_std)

    def _set_encoder_adapter_requires_grad(self, trainable: bool) -> Dict[str, bool]:
        """
        Set requires_grad for encoder adapter parameters. Returns a dict for restoration.
        """
        prev: Dict[str, bool] = {}
        for name, m in self._iter_encoder_adapter_named_modules():
            for p_name, p in m.named_parameters(recurse=True):
                key = f"{name}.{p_name}"
                prev[key] = p.requires_grad
                p.requires_grad = bool(trainable)
        return prev

    def _restore_requires_grad(self, prev: Dict[str, bool]):
        for name, m in self._iter_encoder_adapter_named_modules():
            for p_name, p in m.named_parameters(recurse=True):
                key = f"{name}.{p_name}"
                if key in prev:
                    p.requires_grad = prev[key]

    def _iter_decoder_attn_lora_named_modules(self):
        """
        Yield (name, module) for decoder attention LoRA projection layers only.
        This targets LoRALinear / GatedLoRALinear inside mask decoder SamAttention modules.
        """
        # Local import to avoid heavy coupling at module import time
        from ..models.adaptation_layers import GatedLoRALinear, LoRALinear

        for name, module in self.model.named_modules():
            if "mask_decoder" not in name:
                continue
            if isinstance(module, (LoRALinear, GatedLoRALinear)):
                yield name, module

    def _set_decoder_attn_lora_gspo_mode(self, active: bool, delta_noise_std: float = 0.0, gate_noise_std: float = 0.0):
        for _, m in self._iter_decoder_attn_lora_named_modules():
            if hasattr(m, "set_gspo_mode"):
                try:
                    # GatedLoRALinear.set_gspo_mode(active, gate_noise_std, delta_noise_std)
                    m.set_gspo_mode(active=active, gate_noise_std=gate_noise_std, delta_noise_std=delta_noise_std)
                except TypeError:
                    # LoRALinear.set_gspo_mode(active, noise_std)
                    m.set_gspo_mode(active=active, noise_std=delta_noise_std)

    def _set_decoder_attn_lora_requires_grad(self, trainable: bool) -> Dict[str, bool]:
        prev: Dict[str, bool] = {}
        for name, m in self._iter_decoder_attn_lora_named_modules():
            for p_name, p in m.named_parameters(recurse=True):
                key = f"{name}.{p_name}"
                prev[key] = p.requires_grad
                p.requires_grad = bool(trainable)
        return prev

    def _restore_decoder_attn_lora_requires_grad(self, prev: Dict[str, bool]):
        for name, m in self._iter_decoder_attn_lora_named_modules():
            for p_name, p in m.named_parameters(recurse=True):
                key = f"{name}.{p_name}"
                if key in prev:
                    p.requires_grad = prev[key]

```

```worktree-python
import argparse
import os

import numpy as np
import pandas as pd
import torch

from autogluon.multimodal import MultiModalPredictor
from bbox_prompt_model import BBoxPromptPredictor


def get_default_training_setting(dataset_name):
    validation_metric = "iou"
    loss = "structure_loss"
    max_epoch = 30
    lr = 1e-4

    if dataset_name == "SBU-shadow":
        validation_metric = "ber"
        loss = "balanced_bce"
        max_epoch = 10

    elif dataset_name == "polyp":
        validation_metric = "sm"

    elif dataset_name == "camo_sem_seg":
        validation_metric = "sm"
        max_epoch = 20

    elif dataset_name == "road_segmentation":
        validation_metric = "iou"
        max_epoch = 20
        lr = 3e-4

    elif dataset_name == "leaf_disease_segmentation":
        validation_metric = "iou"
        lr = 3e-4

    return validation_metric, loss, max_epoch, lr


def expand_path(df, dataset_dir):
    for col in ["image", "label"]:
        df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script support converting voc format xmls to coco format json")
    parser.add_argument(
        "--task",
        type=str,
        default="leaf_disease_segmentation",
        choices=["polyp", "leaf_disease_segmentation", "camo_sem_seg", "isic2017", "road_segmentation", "SBU-shadow"],
    )
    parser.add_argument("--seed", type=int, default=42686693)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--expert_num", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--ckpt_path", type=str, default="outputs", help="Checkpoint path.")
    parser.add_argument("--per_gpu_batch_size", type=int, default=1, help="The batch size for each GPU.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="The effective batch size. If batch_size > per_gpu_batch_size * num_gpus, gradient accumulation would be used.",
    )
    parser.add_argument("--eval", action="store_true")
    
    # GSPO parameters
    parser.add_argument("--gspo_enable", action="store_true", help="Enable GSPO (Group Sequence Policy Optimization) training")
    parser.add_argument("--gspo_group_size", type=int, default=4, help="Number of predictions per group in GSPO")
    parser.add_argument("--gspo_warmup_epochs", type=int, default=5, help="Number of epochs before enabling GSPO")
    parser.add_argument("--gspo_contrastive_weight", type=float, default=0.1, help="Weight for contrastive loss in GSPO")
    parser.add_argument("--gspo_quality_momentum", type=float, default=0.9, help="Momentum for expert quality history")
    # GSPO reward / loss shaping hyper-parameters (可选，带默认值)
    parser.add_argument("--lambda_smooth", type=float, default=0.1, help="Smoothness loss weight (λ_smooth)")
    parser.add_argument("--lambda_boundary", type=float, default=0.3, help="Boundary loss weight (λ_boundary)")
    parser.add_argument("--w_boundary", type=float, default=0.3, help="Boundary reward weight (w_boundary)")
    parser.add_argument("--w_smooth", type=float, default=0.1, help="Smoothness reward weight (w_smooth)")
    parser.add_argument("--w_thin", type=float, default=0.05, help="Thin-structure reward weight (w_thin)")
    
    # Adapter parameters
    parser.add_argument("--adapter_enable", action="store_true", help="Enable standard Adapter modules")
    parser.add_argument("--adapter_dim", type=int, default=64, help="Dimension of the adapter bottleneck")

    # GSPO-Adapter extension (encoder adapters only)
    parser.add_argument(
        "--gspo_train_encoder_adapter",
        action="store_true",
        help="When GSPO is active, also train Vision Encoder adapters (default: False => adapters frozen during GSPO steps).",
    )
    parser.add_argument(
        "--gspo_encoder_adapter_noise_std",
        type=float,
        default=0.0,
        help="Stage 1: exploration noise std added to encoder adapter outputs during GSPO rollouts (default: 0.0).",
    )
    parser.add_argument(
        "--gspo_encoder_adapter_gate_enable",
        action="store_true",
        help="Stage 2: use gated encoder adapters (policy-like per-sample scalar gate).",
    )
    parser.add_argument(
        "--gspo_encoder_adapter_gate_noise_std",
        type=float,
        default=0.0,
        help="Stage 2: exploration noise std added to gated-adapter gate logits during GSPO rollouts (default: 0.0).",
    )

    # GSPO-AttnLoRA extension (decoder attention LoRA only)
    parser.add_argument(
        "--gspo_train_decoder_attn_lora",
        action="store_true",
        help="When GSPO is active, also train Decoder Attention LoRA parameters (default: False => frozen during GSPO steps).",
    )
    parser.add_argument(
        "--gspo_decoder_attn_lora_noise_std",
        type=float,
        default=0.0,
        help="Stage 1: exploration noise std added to decoder attention LoRA delta during GSPO rollouts (default: 0.0).",
    )
    parser.add_argument(
        "--gspo_decoder_attn_lora_gate_enable",
        action="store_true",
        help="Stage 2: use gated decoder attention LoRA (policy-like per-sample gate on LoRA delta).",
    )
    parser.add_argument(
        "--gspo_decoder_attn_lora_gate_noise_std",
        type=float,
        default=0.0,
        help="Stage 2: exploration noise std added to gated decoder-attn-LoRA gate logits during GSPO rollouts (default: 0.0).",
    )
    
    # Decoder Attention LoRA parameters (LoRA on Attention)
    parser.add_argument("--decoder_attn_lora_enable", action="store_true", 
                        help="Enable LoRA on decoder attention Q/K/V projections")
    parser.add_argument("--decoder_attn_lora_r", type=int, default=8,
                        help="LoRA rank for decoder attention (default: 8)")
    parser.add_argument("--decoder_attn_lora_alpha", type=int, default=8,
                        help="LoRA alpha for decoder attention (default: 8, same as rank)")
    parser.add_argument("--decoder_attn_lora_dropout", type=float, default=0.0,
                        help="LoRA dropout for decoder attention (default: 0.0)")
    
    # TTA (Test-Time Augmentation) parameters
    parser.add_argument("--tta_enable", action="store_true", help="Enable Test-Time Augmentation for evaluation")
    parser.add_argument("--tta_scales", type=float, nargs="+", default=[0.75, 1.0, 1.25], 
                        help="Scale factors for multi-scale TTA (default: [0.75, 1.0, 1.25]). "
                             "Recommended: [0.75, 1.0, 1.25] or [0.8, 1.0, 1.2]")
    parser.add_argument("--tta_flips", type=str, nargs="+", default=["none", "horizontal"],
                        choices=["none", "horizontal", "vertical"],
                        help="Flip types for TTA (default: ['none', 'horizontal'])")
    parser.add_argument("--tta_rotations", type=float, nargs="+", default=[0],
                        help="Rotation angles in degrees for TTA (default: [0]). "
                             "Recommended: [0] for fast, or [-10, 0, 10] for better")
    parser.add_argument("--tta_fusion", type=str, default="mean", choices=["mean", "weighted_mean"],
                        help="Fusion method for TTA predictions (default: 'mean')")
    parser.add_argument("--tta_resize_method", type=str, default="bilinear", choices=["bilinear", "bicubic"],
                        help="Resize interpolation method for multi-scale TTA (default: 'bilinear'). "
                             "bilinear: faster, bicubic: higher quality")
    parser.add_argument("--tta_threshold", type=float, default=0.5,
                        help="Threshold for binary segmentation in TTA (default: 0.5)")
    parser.add_argument("--tta_min_area", type=float, default=0.001,
                        help="Remove components with area < tta_min_area * image_area (default: 0.001)")
    parser.add_argument("--tta_morphology", action="store_true",
                        help="Apply morphological closing in TTA post-processing")
    parser.add_argument("--tta_cache_dir", type=str, default=None,
                        help="Directory to cache TTA predictions for resume (default: None, no caching)")
    parser.add_argument("--tta_no_resume", action="store_true",
                        help="Disable resume from cache (start fresh even if cache exists)")
    parser.add_argument("--tta_box_prompt_mode", type=str, default="off",
                        choices=["off", "add", "replace"],
                        help="Use GT box as SAM box prompt (off/add/replace); box not transformed with TTA")
    # BBox prompt configuration
    parser.add_argument("--bbox_prompt_source", type=str, default="none",
                        choices=["none", "gt", "predict"],
                        help="Box prompt source: none (no prompt), gt (from mask), predict (from bbox model).")
    parser.add_argument("--bbox_model_ckpt", type=str, default=None,
                        help="Checkpoint for bbox predictor when bbox_prompt_source=predict.")
    parser.add_argument("--bbox_model_image_size", type=int, default=320,
                        help="Input size for bbox predictor inference.")
    parser.add_argument("--bbox_preds_csv", type=str, default=None,
                        help="Optional CSV with columns image,bbox_x1,bbox_y1,bbox_x2,bbox_y2 to attach prompts.")
    # Training-time box prompt mixing (default off)
    parser.add_argument("--train_box_prompt_mode", type=str, default="off",
                        choices=["off", "gt", "noisy", "mix", "predict"],
                        help="Training-time box prompt injection: off/gt/noisy/mix/predict (default off). "
                             "predict: use bbox predictor model to generate box prompts from images.")
    parser.add_argument("--train_box_prob_no_prompt", type=float, default=0.2,
                        help="When mode=mix, probability of no prompt.")
    parser.add_argument("--train_box_prob_gt", type=float, default=0.3,
                        help="When mode=mix, probability of GT box prompt.")
    parser.add_argument("--train_box_prob_noisy", type=float, default=0.5,
                        help="When mode=mix, probability of noisy GT box prompt.")
    parser.add_argument("--train_box_noise_frac", type=float, default=0.12,
                        help="Noisy box jitter fraction relative to box size (e.g., 0.12 => ±12%).")
    parser.add_argument("--train_bbox_model_ckpt", type=str, default=None,
                        help="Checkpoint for bbox predictor when train_box_prompt_mode=predict.")
    parser.add_argument("--train_bbox_model_image_size", type=int, default=320,
                        help="Input size for bbox predictor during training (when train_box_prompt_mode=predict).")
    
    # Quick test / Debug parameters
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: only process first 5 images for quick validation")
    parser.add_argument("--quick_test", type=int, default=None,
                        help="Quick test mode: only process first N images")
    
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: auto based on task, typically 1e-4 or 3e-4)")
    
    # Decoder Adapter parameters
    parser.add_argument("--decoder_adapter_enable", action="store_true", help="Enable MLP-Adapter modules in the mask decoder")
    parser.add_argument("--decoder_adapter_dim", type=int, default=64, help="Dimension of the decoder adapter bottleneck")
    args = parser.parse_args()

    dataset_name = args.task
    dataset_dir = os.path.join(f"datasets/{dataset_name}", dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare dataframes
    train_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"train.csv")), dataset_dir)
    val_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"val.csv")), dataset_dir)

    # get the validation metric
    validation_metric, loss, max_epoch, lr = get_default_training_setting(dataset_name)
    
    # Override learning rate if user specified it via CLI
    if args.lr is not None:
        lr = args.lr
        print(f"Using custom learning rate: {lr}")

    hyperparameters = {}
    hyperparameters.update(
        {
            "optim.lora.r": args.rank,
            # Paper alignment (Conv-LoRA):
            # - optim.peft: choose Conv-LoRA adaptation (Sec. 3)
            # - optim.lora.r: LoRA rank r (Sec. 3)
            # - optim.lora.conv_lora_expert_num: number of MoE-Conv experts M (Sec. 3.2)
            "optim.peft": "conv_lora",
            "optim.lora.conv_lora_expert_num": args.expert_num,
            "env.num_gpus": args.num_gpus,
            "optim.loss_func": loss,
            "optim.max_epochs": max_epoch,
            "optim.lr": lr,
            "env.per_gpu_batch_size": args.per_gpu_batch_size,
            "env.batch_size": args.batch_size,
        }
    )
    
    # GSPO configuration
    if args.gspo_enable:
        print(f"Enabling GSPO with group_size={args.gspo_group_size}, warmup_epochs={args.gspo_warmup_epochs}")
        hyperparameters.update({
            "optim.lora.gspo_enabled": True,
            "optim.lora.gspo_group_size": args.gspo_group_size,
            "optim.lora.gspo_quality_momentum": args.gspo_quality_momentum,
            "optim.lora.gspo_warmup_epochs": args.gspo_warmup_epochs,
            "optim.lora.gspo_contrastive_weight": args.gspo_contrastive_weight,
            # GSPO reward / loss shaping
            "optim.gspo.lambda_smooth": args.lambda_smooth,
            "optim.gspo.lambda_boundary": args.lambda_boundary,
            "optim.gspo.w_boundary": args.w_boundary,
            "optim.gspo.w_smooth": args.w_smooth,
            "optim.gspo.w_thin": args.w_thin,
            # GSPO-Adapter extension (encoder adapters only)
            "optim.gspo.train_encoder_adapter": args.gspo_train_encoder_adapter,
            "optim.gspo.encoder_adapter_noise_std": args.gspo_encoder_adapter_noise_std,
            "optim.gspo.encoder_adapter_gate_noise_std": args.gspo_encoder_adapter_gate_noise_std,
            # GSPO-AttnLoRA extension (decoder attention LoRA only)
            "optim.gspo.train_decoder_attention_lora": args.gspo_train_decoder_attn_lora,
            "optim.gspo.decoder_attention_lora_noise_std": args.gspo_decoder_attn_lora_noise_std,
            "optim.gspo.decoder_attention_lora_gate_noise_std": args.gspo_decoder_attn_lora_gate_noise_std,
        })

    # Adapter configuration
    if args.adapter_enable:
        print(f"Enabling Standard Encoder Adapters with dim={args.adapter_dim}")
        hyperparameters.update({
            "model.sam.adapter_enabled": True,
            "model.sam.adapter_dim": args.adapter_dim,
        })

    # Stage 2: gated encoder adapter (policy-like). This only changes model structure if adapters are enabled.
    if args.gspo_encoder_adapter_gate_enable:
        print(
            f"Enabling Gated Encoder Adapters (policy) with gate_noise_std={args.gspo_encoder_adapter_gate_noise_std}"
        )
        hyperparameters.update(
            {
                "model.sam.encoder_adapter_gate_enabled": True,
                "model.sam.encoder_adapter_gate_noise_std": args.gspo_encoder_adapter_gate_noise_std,
            }
        )
    
    # Decoder Attention LoRA configuration
    if args.decoder_attn_lora_enable:
        print(f"Enabling Decoder Attention LoRA: r={args.decoder_attn_lora_r}, "
              f"alpha={args.decoder_attn_lora_alpha}, dropout={args.decoder_attn_lora_dropout}")
        hyperparameters.update({
            "model.sam.decoder_attention_lora_r": args.decoder_attn_lora_r,
            "model.sam.decoder_attention_lora_alpha": args.decoder_attn_lora_alpha,
            "model.sam.decoder_attention_lora_dropout": args.decoder_attn_lora_dropout,
        })

    # Stage 2: gated decoder attention LoRA (policy-like). Only meaningful when decoder attention LoRA is enabled.
    if args.gspo_decoder_attn_lora_gate_enable:
        print(
            f"Enabling Gated Decoder Attention LoRA (policy) with gate_noise_std={args.gspo_decoder_attn_lora_gate_noise_std}"
        )
        hyperparameters.update(
            {
                "model.sam.decoder_attention_lora_gate_enabled": True,
                "model.sam.decoder_attention_lora_gate_noise_std": args.gspo_decoder_attn_lora_gate_noise_std,
            }
        )
        
    # Decoder Adapter configuration
    if args.decoder_adapter_enable:
        print(f"Enabling Decoder MLP-Adapters with dim={args.decoder_adapter_dim}")
        hyperparameters.update({
            "model.sam.decoder_adapter_enabled": True,
            "model.sam.decoder_adapter_dim": args.decoder_adapter_dim,
        })

    if args.eval:  # load a checkpoint for evaluation
        predictor = MultiModalPredictor.load(args.ckpt_path)
    else:  # training
        predictor = MultiModalPredictor(
            problem_type="semantic_segmentation",
            validation_metric=validation_metric,
            eval_metric=validation_metric,
            hyperparameters=hyperparameters,
            label="label",
        )

        # Configure training-time box prompt mixing
        predictor._learner._train_box_prompt_cfg = {
            "mode": args.train_box_prompt_mode,
            "p_no": args.train_box_prob_no_prompt,
            "p_gt": args.train_box_prob_gt,
            "p_noisy": args.train_box_prob_noisy,
            "noise_frac": args.train_box_noise_frac,
        }

        # Setup bbox predictor for training if mode=predict
        if args.train_box_prompt_mode == "predict":
            if not args.train_bbox_model_ckpt:
                raise ValueError("train_box_prompt_mode=predict requires --train_bbox_model_ckpt.")
            train_bbox_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_bbox_predictor = BBoxPromptPredictor(
                ckpt_path=args.train_bbox_model_ckpt,
                device=train_bbox_device,
                image_size=args.train_bbox_model_image_size,
            )
            predictor._learner._train_bbox_predictor = train_bbox_predictor
            print(f"\n[Info] Training-time bbox predictor loaded from: {args.train_bbox_model_ckpt}")
            print(f"       Input size: {args.train_bbox_model_image_size}\n")

        # 打印当前实验的模型保存目录，便于后续通过 AutogluonModels 路径追溯到具体实验日志
        try:
            # MultiModalPredictor 会在内部根据时间戳生成类似 AutogluonModels/ag-YYYYMMDD_HHMMSS 的目录
            model_path = predictor.path
        except AttributeError:
            # 兼容极端情况：如果未来接口变化，没有 path 属性，则显式提示
            model_path = None

        if model_path is not None:
            print("\n========================================")
            print("Training MultiModalPredictor")
            print(f"  Task        : {dataset_name}")
            print(f"  Save path   : {model_path}")
            print("  (You can use this directory name to link back to the training log.)")
            print("========================================\n")
        else:
            print("\n[Warning] MultiModalPredictor has no 'path' attribute. "
                  "Model save directory cannot be printed.\n")

        predictor.fit(train_data=train_df, tuning_data=val_df, seed=args.seed)

    # Enable TTA if requested
    if args.tta_enable:
        print(f"\n{'='*60}")
        print(f"Enabling Test-Time Augmentation (TTA)")
        print(f"  Scales: {args.tta_scales}")
        print(f"  Flips: {args.tta_flips}")
        print(f"  Rotations: {args.tta_rotations}")
        print(f"  Fusion: {args.tta_fusion}")
        print(f"  Resize method: {args.tta_resize_method}")
        print(f"  Total augmentations: {len(args.tta_scales) * len(args.tta_flips) * len(args.tta_rotations)}")
        print(f"{'='*60}\n")
        
        predictor.enable_tta(
            scales=args.tta_scales,
            flips=args.tta_flips,
            rotations=args.tta_rotations,
            fusion_method=args.tta_fusion,
            resize_method=args.tta_resize_method,
            threshold=args.tta_threshold,
            min_area_ratio=args.tta_min_area,
            use_morphology=args.tta_morphology,
            cache_dir=args.tta_cache_dir,
            resume_from_cache=not args.tta_no_resume,
            box_prompt_mode=args.tta_box_prompt_mode,
        )

    # Optional bbox predictor for predicted box prompts
    bbox_predictor = None
    if args.bbox_prompt_source == "predict":
        if not args.bbox_model_ckpt and not args.bbox_preds_csv:
            raise ValueError("bbox_prompt_source=predict requires --bbox_model_ckpt or --bbox_preds_csv.")
        if args.bbox_model_ckpt:
            bbox_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            bbox_predictor = BBoxPromptPredictor(
                ckpt_path=args.bbox_model_ckpt,
                device=bbox_device,
                image_size=args.bbox_model_image_size,
            )
        if args.tta_box_prompt_mode == "off":
            print("[Warning] bbox_prompt_source=predict set but tta_box_prompt_mode=off; box prompts will be ignored.")

    def attach_pred_boxes(df: pd.DataFrame) -> pd.DataFrame:
        """Attach predicted or precomputed boxes to dataframe."""
        df = df.copy()
        if args.bbox_preds_csv:
            preds = pd.read_csv(args.bbox_preds_csv)
            merged = df.merge(preds, on="image", how="left")
            if merged[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].isna().all().all():
                preds["image_basename"] = preds["image"].apply(lambda p: os.path.basename(p))
                merged["image_basename"] = merged["image"].apply(lambda p: os.path.basename(p))
                merged = merged.merge(
                    preds.drop(columns=["image"]), on="image_basename", how="left", suffixes=("", "_pred")
                )
                merged.drop(columns=["image_basename"], inplace=True)
                for col in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
                    pred_col = f"{col}_pred"
                    if pred_col in merged.columns:
                        merged[col] = merged[col].fillna(merged[pred_col])
                        merged.drop(columns=[pred_col], inplace=True)
            df = merged
        elif bbox_predictor is not None:
            boxes = []
            for _, r in df.iterrows():
                box = bbox_predictor.predict(r["image"])
                boxes.append(box)
            boxes = np.stack(boxes, axis=0)
            df["bbox_x1"] = boxes[:, 0]
            df["bbox_y1"] = boxes[:, 1]
            df["bbox_x2"] = boxes[:, 2]
            df["bbox_y2"] = boxes[:, 3]
        return df

    # evaluation
    metric_file = os.path.join(args.output_dir, "metrics.txt")
    f = open(metric_file, "a")
    if dataset_name in ["isic2017", "SBU-shadow", "road_segmentation", "leaf_disease_segmentation"]:
        test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test.csv")), dataset_dir)
        
        # Apply quick test / debug mode (方案 1: 快速验证模式)
        original_size = len(test_df)
        if args.debug:
            test_df = test_df.head(5)
            print(f"\n🔍 DEBUG MODE: Processing only first 5 images (out of {original_size})\n")
        elif args.quick_test is not None:
            test_df = test_df.head(args.quick_test)
            print(f"\n🔍 QUICK TEST MODE: Processing only first {args.quick_test} images (out of {original_size})\n")
        
        if dataset_name == "SBU-shadow":
            eval_metrics = ["ber"]
        else:
            eval_metrics = ["iou", "dice"]  # Evaluate both IoU and DICE

        # Attach predicted bbox prompts if requested
        if args.bbox_prompt_source == "predict" and (args.bbox_preds_csv or bbox_predictor):
            test_df = attach_pred_boxes(test_df)

        res = predictor.evaluate(test_df, metrics=eval_metrics)
        print(f"Evaluation results for test dataset {dataset_name}: ", res)
        f.write(f"Evaluation results for test dataset {dataset_name}: {res} \n")
    elif dataset_name in ["polyp", "camo_sem_seg"]:
        if dataset_name == "polyp":
            test_datasets = ["CVC-ClinicDB", "Kvasir"]
        elif dataset_name == "camo_sem_seg":
            test_datasets = ["CAMO"]
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}.")
        for per_dataset in test_datasets:
            test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test_{per_dataset}.csv")), dataset_dir)
            
            # Apply quick test / debug mode (方案 1: 快速验证模式)
            original_size = len(test_df)
            if args.debug:
                test_df = test_df.head(5)
                print(f"\n🔍 DEBUG MODE: Processing only first 5 images (out of {original_size}) for {per_dataset}\n")
            elif args.quick_test is not None:
                test_df = test_df.head(args.quick_test)
                print(f"\n🔍 QUICK TEST MODE: Processing only first {args.quick_test} images (out of {original_size}) for {per_dataset}\n")
            
            if args.bbox_prompt_source == "predict" and (args.bbox_preds_csv or bbox_predictor):
                test_df = attach_pred_boxes(test_df)

            res = predictor.evaluate(test_df, metrics=["sm", "fm", "em", "mae"])
            print(f"Evaluation results for test dataset {per_dataset}: ", res)
            f.write(f"Evaluation results for test dataset {per_dataset}: {res} \n")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}.")

    f.close()

```

```markdown
## 目录说明（GSPO + Conv-LoRA + Encoder Adapter + LoRA on Attention 扩展）

本子目录用于集中存放本次“将 **Decoder Attention LoRA（LoRA on Attention）** 纳入 GSPO 强化学习式微调”的设计与实现说明文档。

- `design.md`：总体设计（阶段 0/1/2）、模型结构框图、公式与实现映射、损失/奖励定义、与现有GSPO对应关系
- `implementation.md`：代码落点、配置/命令行参数说明、向后兼容策略、常见排错

> 注意：本次改动不会触碰任何 `.pdf` 文件。


```

```markdown
## 目标

在现有 **Conv-LoRA + GSPO** 以及（可选）**Encoder Adapter GSPO** 的基础上，将 **LoRA on Attention**（本项目中指 *SAM Mask Decoder 的 SamAttention Q/K/V 投影上的 LoRA*）也纳入 GSPO 的强化学习式微调。

约束与原则：

- **默认向后兼容**：不开启新开关时，GSPO active step 仍只优化 Conv‑LoRA（并且会冻结 decoder-attn-LoRA，避免“被动纳入”）。
- **仅纳入 decoder attention LoRA**：不改变 Conv‑LoRA（encoder attention）与 decoder adapter 的设计边界。
- **不触碰任何 `.pdf` 文件**。

---

## LoRA on Attention 在哪里？

Decoder 侧的 `SamAttention` 对 query/key/value 的投影：

```text
Mask Decoder (Two-Way Transformer)
  SamTwoWayAttentionBlock
    - self_attn:      q_proj/k_proj/v_proj (+ LoRA)
    - cross_token2img q_proj/k_proj/v_proj (+ LoRA)
    - cross_img2token q_proj/k_proj/v_proj (+ LoRA)
  final_attn_token_to_image: q_proj/k_proj/v_proj (+ LoRA)
```

LoRA 形式：

\[
y = W_0 x + \Delta W x,\quad \Delta W = B A,\quad \text{scaling}=\alpha/r
\]

实现等价于：

\[
\Delta y = (xA^\top)B^\top \cdot \text{scaling}
\]

---

## GSPO（现有实现）的核心公式与对应

对每张图像做 \(G\) 次 rollouts（通过 dropout/噪声产生差异）：

- 预测：\(\hat{y}^{(g)} = f_\theta(x; \xi_g)\)
- 质量：\(R^{(g)} = \text{Quality}(\hat{y}^{(g)}, y)\)
- baseline：\(\bar{R} = \frac{1}{G}\sum_g R^{(g)}\)
- advantage：\(A^{(g)} = R^{(g)} - \bar{R}\)
- 权重：\(w^{(g)} = \sigma(\tau A^{(g)})\)

优化目标（主项）：

\[
\mathcal{L}
= \frac{1}{G}\sum_{g=1}^G w^{(g)} \cdot \mathcal{L}_{seg}(\hat{y}^{(g)}, y)
 + \lambda_{moe}\mathcal{L}_{moe}
 + \lambda_{ctr}\mathcal{L}_{ctr}
\]

> 这是一种 GSPO-inspired surrogate：用组内相对质量作为“策略优势”，通过加权分割损失来做 credit assignment。

---

## 阶段 0（基线，向后兼容）

当不启用本扩展时（默认）：

- 即使你开启了 decoder attention LoRA（`decoder_attention_lora_r > 0`），在 **GSPO active step** 中也会临时冻结这些 LoRA 参数：
  - `optim.gspo.train_decoder_attention_lora = False`（默认）

从语义上保证：**GSPO 仍然只用于 Conv‑LoRA**（以及你是否额外开启的 encoder adapter GSPO）。

---

## 阶段 1：对 decoder-attn-LoRA 注入探索噪声（最小改动）

### 设计

在 GSPO active 的 rollouts 中，对 LoRA 的 delta 分支加入零均值噪声：

\[
\Delta y = (xA^\top)B^\top \cdot \text{scaling} + \epsilon,\quad \epsilon\sim\mathcal{N}(0,\sigma_{\text{lora}}^2)
\]

仅在 GSPO active 且训练模式下生效，用于提高 rollouts 多样性。

### 好处

- 同一输入产生多样预测，使 advantage-weighting 更有效地对齐“更好”的 LoRA 扰动方向。
- 保持参数量与结构几乎不变，调参成本低。

---

## 阶段 2：Gated decoder-attn-LoRA（显式策略）

### 设计

只对 LoRA 的 delta 分支加 gate（不动 base path）：

\[
y = W_0x + g(x)\cdot \Delta y
\]

其中 gate 是 per-sample 的标量：

\[
g(x) = \sigma(w\cdot \text{pool}(x) + \eta),\quad \eta\sim\mathcal{N}(0,\sigma_{\text{gate}}^2)
\]

- `pool(x)`：对 `[B,P,T,C]` 或 `[B,T,C]` 的 token/point 维度做平均，得到 `[B,C]`
- \(\eta\) 仅在 GSPO active rollouts 时注入，用于探索

### 好处

- 将“LoRA 应该介入多大强度”显式变成一个可学习策略。
- GSPO 的组内比较可以更清晰地对 gate 做 credit assignment（哪些 gate 值带来更好分割质量）。

---

## 与现有组件的协同（直观框图）

```text
Encoder (Vision)                       Decoder (Mask)
┌───────────────────────────┐         ┌──────────────────────────────┐
│ Attention: Conv-LoRA MoE  │         │ SamAttention Q/K/V (+LoRA)    │
│   - GSPO feedback on gate │         │   - Stage1: delta noise       │
│ MLP: Encoder Adapter      │         │   - Stage2: gated delta (policy)│
│   - (可选) GSPO gating     │         └──────────────────────────────┘
└───────────────────────────┘

GSPO group rollouts:
  同一图像做 G 次 forward (dropout/噪声/门控采样不同) → 质量对比 → advantage-weighted loss → 反传更新
```


```

```markdown
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


```

```worktree-python
import torch
from transformers.models.sam.configuration_sam import SamMaskDecoderConfig

from autogluon.multimodal.models.adaptation_layers import GatedLoRALinear, LoRALinear
from autogluon.multimodal.models.custom_hf_models.modeling_sam_for_conv_lora import SamAttention


def test_decoder_attention_lora_default_linear_type():
    cfg = SamMaskDecoderConfig()
    cfg.hidden_size = 256
    cfg.num_attention_heads = 8
    cfg.attention_downsample_rate = 2
    cfg.decoder_attention_lora_gate_enabled = False
    cfg.decoder_attention_lora_gate_noise_std = 0.0

    attn = SamAttention(cfg, downsample_rate=1, lora_r=8, lora_alpha=8, lora_dropout=0.0)
    assert isinstance(attn.q_proj, LoRALinear)
    assert not isinstance(attn.q_proj, GatedLoRALinear)


def test_decoder_attention_lora_gated_linear_type_and_gspo_mode():
    cfg = SamMaskDecoderConfig()
    cfg.hidden_size = 256
    cfg.num_attention_heads = 8
    cfg.attention_downsample_rate = 2
    cfg.decoder_attention_lora_gate_enabled = True
    cfg.decoder_attention_lora_gate_noise_std = 0.5

    attn = SamAttention(cfg, downsample_rate=1, lora_r=8, lora_alpha=8, lora_dropout=0.0)
    assert isinstance(attn.q_proj, GatedLoRALinear)

    # Toggle GSPO mode (should not error)
    attn.q_proj.set_gspo_mode(active=True, gate_noise_std=0.1, delta_noise_std=0.02)
    attn.q_proj.set_gspo_mode(active=False)

    # Forward shape sanity check
    q = torch.randn(2, 1, 16, 256)  # [B, point_batch, tokens, C]
    out = attn.q_proj(q)
    assert out.shape == (2, 1, 16, cfg.hidden_size)


```

### ✅ 已完成：把 **LoRA on Attention（Decoder Attention LoRA）** 纳入 GSPO 强化学习式微调（并保持默认兼容）
我先确认了本项目里的 “LoRA on Attention” 指的是 **Mask Decoder 的 `SamAttention` 的 Q/K/V 投影用 `LoRALinear`**（见 `docs/LoRA_on_Attention_Design.md` 与 `modeling_sam_for_conv_lora.py` 的 `SamAttention`），然后在现有 GSPO 基础上实现了三阶段扩展，并通过开关保证默认行为不变。

---

### 1) 向后兼容（阶段 0，默认不改变旧实验语义）
新增默认策略：**GSPO active step 中，decoder attention LoRA 默认会被临时冻结**，从而保持：
- 不开新开关时，GSPO 仍只优化 Conv‑LoRA（以及你之前是否显式开启的 Encoder Adapter GSPO 扩展）。

对应配置默认值（已写入 `optim/default.yaml`）：
- `optim.gspo.train_decoder_attention_lora: False`（默认）

---

### 2) 阶段 1：让 GSPO “更有效”地作用于 attention‑LoRA（探索噪声）
在 GSPO rollouts 期间对 decoder-attn-LoRA 的 **delta 分支**注入探索噪声（只在 GSPO active + training 时生效）：
- 配置：`optim.gspo.decoder_attention_lora_noise_std`
- 实现：给 `LoRALinear` 增加 `set_gspo_mode(active, noise_std)`，在 forward 里对 `delta` 加 `N(0,σ²)`。

---

### 3) 阶段 2：显式策略（Gated LoRA on Attention）
可选把 decoder attention 的 `LoRALinear` 替换为 **`GatedLoRALinear`**（policy-like gate），仅对 LoRA delta 分支做 per-sample gating：
- `y = base + gate(x) * delta`
- GSPO rollouts 中可对 gate logits 注入噪声用于探索（`decoder_attention_lora_gate_noise_std`）

该结构**不影响 base path**，更稳定；并且只有在显式开关打开时才改变结构。

---

### 4) 你怎么用（CLI）
入口：`examples/automm/Conv-LoRA/run_semantic_segmentation.py`

#### 4.1 让 GSPO 同时训练 decoder attention LoRA（阶段 0 → 开启训练）
```bash
--gspo_enable
--decoder_attn_lora_enable
--gspo_train_decoder_attn_lora
```

#### 4.2 阶段 1（噪声）
```bash
--gspo_decoder_attn_lora_noise_std 0.02
```

#### 4.3 阶段 2（gated LoRA policy）
```bash
--gspo_decoder_attn_lora_gate_enable
--gspo_decoder_attn_lora_gate_noise_std 0.5
```

---

### 5) 文档（英文目录名，中文内容；未触碰任何 `.pdf`）
新增目录：`docs/gspo_attn_lora_rl/`
- `design.md`：框图 + 公式 + 与 GSPO surrogate 的对应关系（阶段 0/1/2）
- `implementation.md`：代码落点、配置/CLI、兼容策略
- `README.md`：索引

---

### 6) 关键实现落点（便于你 review）
- `multimodal/.../models/adaptation_layers.py`
  - `LoRALinear.set_gspo_mode(...)` + delta 噪声
  - 新增 `GatedLoRALinear`
- `multimodal/.../models/custom_hf_models/modeling_sam_for_conv_lora.py`
  - `SamAttention`：按 `decoder_attention_lora_gate_enabled` 选择 `LoRALinear` vs `GatedLoRALinear`
- `multimodal/.../models/sam.py`
  - 把 gate 配置注入 `config.mask_decoder_config.*`
- `multimodal/.../optim/lit_semantic_seg.py`
  - GSPO step 内对 decoder-attn-LoRA：`set_gspo_mode` +（默认）冻结/恢复
- `multimodal/.../learners/semantic_segmentation.py`
  - 新增 `gspo_attn_lora_cfg` 下发到 LitModule
- `examples/automm/Conv-LoRA/run_semantic_segmentation.py`
  - 新增 CLI 开关并写入 hyperparameters
- `tests/unittests/models/test_sam_decoder_attn_lora_gspo.py`
  - 覆盖 gated/非 gated 类型选择与 `set_gspo_mode`、shape

如果你希望我下一步把“GSPO 的 reward shaping”进一步针对 decoder attention（比如专门引入对 token-level 边界质量的额外 shaping 或对 gate 加 entropy 正则），我也可以在不破坏兼容性的前提下继续加到阶段 2/3。

```shellscript
--gspo_enable
--decoder_attn_lora_enable
--gspo_train_decoder_attn_lora
```

```shellscript
--gspo_decoder_attn_lora_noise_std 0.02
```

```shellscript
--gspo_decoder_attn_lora_gate_enable
--gspo_decoder_attn_lora_gate_noise_std 0.5
```

