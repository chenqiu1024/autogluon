# Conv-LoRA + RL: Architecture Diagrams

## System Architecture (Scheme B: RL Expert Routing)

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                                │
└────────────────────────────────────────────────────────────────────────┘

┌──────────────┐
│  Train Data  │
│  (images +   │
│   GT masks)  │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SAM with Conv-LoRA                            │
│  ┌────────────┐    ┌──────────────┐    ┌─────────────┐        │
│  │   Vision   │───▶│  Conv-LoRA   │───▶│    Mask     │        │
│  │  Encoder   │    │   Layers     │    │   Decoder   │        │
│  │ (frozen)   │    │              │    │  (frozen)   │        │
│  └────────────┘    └──────┬───────┘    └─────────────┘        │
│                           │                                      │
│                           ▼                                      │
│              ┌────────────────────────┐                         │
│              │  RLGate (per layer)    │                         │
│              ├────────────────────────┤                         │
│              │ lora_res → RLGate      │                         │
│              │   ├─ RoutingPolicy     │◀────── Trainable       │
│              │   │  (GAP + MLP)       │                         │
│              │   ├─ Sample action     │                         │
│              │   ├─ Compute logprob   │                         │
│              │   └─ KL to NoisyTopK   │                         │
│              │                         │                         │
│              │ Output:                 │                         │
│              │  - gates (B, M)         │                         │
│              │  - logprobs             │                         │
│              │  - KL loss              │                         │
│              └────────────────────────┘                         │
│                           │                                      │
│                           ▼                                      │
│              ┌────────────────────────┐                         │
│              │  Expert Routing        │                         │
│              │  (SparseDispatcher)    │                         │
│              └────────────────────────┘                         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
                ┌──────────────────┐
                │  Predicted Mask  │
                └─────────┬────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │       Reward Computation             │
        │                                      │
        │  IoU = iou(pred, GT)                │
        │  FLOPs = Σ expert_flops             │
        │  Imbalance = CV²(expert_usage)      │
        │                                      │
        │  Reward = IoU - α·FLOPs - β·Imbal   │
        └─────────────────┬───────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │         GRPO Update                  │
        │  ┌──────────────────────┐           │
        │  │ 1. Group advantages  │           │
        │  │    A = (R - μ) / σ   │           │
        │  └──────────────────────┘           │
        │  ┌──────────────────────┐           │
        │  │ 2. Compute losses    │           │
        │  │  L_pg = -E[A·logπ]   │           │
        │  │  L_H = -β·H(π)       │           │
        │  │  L_kl = λ·KL(π||πref)│           │
        │  │  L_l2 = λ·||θ||²     │           │
        │  └──────────────────────┘           │
        │  ┌──────────────────────┐           │
        │  │ 3. Backward + update │           │
        │  │    optimizer.step()  │           │
        │  └──────────────────────┘           │
        └─────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │    Lagrangian Update (optional)      │
        │  If FLOPs > budget:                  │
        │    dual_α += lr·(FLOPs - budget)    │
        │  Else:                               │
        │    dual_α -= lr·(budget - FLOPs)    │
        └─────────────────────────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │         Logging & Checkpointing      │
        │  - TensorBoard: metrics, histograms  │
        │  - Artifacts: heatmaps, JSON         │
        │  - Checkpoints: policy, dual_α, RNG  │
        └─────────────────────────────────────┘
```

## Data Flow Diagram

```
Input Batch
    │
    ├─ Images (B, 3, 1024, 1024)
    └─ GT Masks (B, 1024, 1024)
    │
    ▼
┌───────────────────────────────────────┐
│  SAM Vision Encoder (frozen)          │
│  Input: images                         │
│  Output: image_embeds (B, 256, 64, 64)│
└───────────────┬───────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│  Conv-LoRA Layer i (with RL gating)                   │
│  ┌─────────────────────────────────────────────┐     │
│  │ 1. Base path: x → W_frozen → result         │     │
│  └─────────────────────────────────────────────┘     │
│  ┌─────────────────────────────────────────────┐     │
│  │ 2. LoRA path:                                │     │
│  │    x → A → lora_res(B, r, H, W)             │     │
│  │                │                             │     │
│  │                ▼                             │     │
│  │    ┌───────────────────────────┐            │     │
│  │    │ RLGate(lora_res, layer_i) │            │     │
│  │    │  ├─ GAP → (B, r)          │            │     │
│  │    │  ├─ MLP → logits(B, M)    │            │     │
│  │    │  ├─ Sample → action       │            │     │
│  │    │  ├─ logprob(action)       │            │     │
│  │    │  └─ KL to NoisyTopK       │            │     │
│  │    └───────────┬───────────────┘            │     │
│  │                │                             │     │
│  │                ▼                             │     │
│  │    gates(B, M) → SparseDispatcher           │     │
│  │                │                             │     │
│  │                ├─ Expert 0 (up 1x) ─┐       │     │
│  │                ├─ Expert 1 (up 2x) ─┤       │     │
│  │                ├─ ...              ─┤       │     │
│  │                └─ Expert M-1 (up Mx)┘       │     │
│  │                │                             │     │
│  │                ▼                             │     │
│  │    Combine → temp_res + lora_res → B →      │     │
│  │                                     │        │     │
│  │                                     ▼        │     │
│  │                            result + LoRA     │     │
│  └─────────────────────────────────────────────┘     │
│                                                       │
│  Collect: logprobs_i, gates_i, kl_loss_i            │
└───────────────┬───────────────────────────────────────┘
                │
                ▼ (repeat for all N Conv-LoRA layers)
                │
                ▼
┌───────────────────────────────────────────┐
│  Mask Decoder (frozen)                     │
│  Output: pred_masks (B, 1024, 1024)       │
└───────────────┬───────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│  Reward Computation                                    │
│  ┌─────────────────────────────────────────────┐     │
│  │ IoU = iou(pred_masks, GT_masks)             │     │
│  └─────────────────────────────────────────────┘     │
│  ┌─────────────────────────────────────────────┐     │
│  │ FLOPs = Σ_i compute_expert_flops(gates_i)   │     │
│  │       = Σ active_experts · conv_flops        │     │
│  └─────────────────────────────────────────────┘     │
│  ┌─────────────────────────────────────────────┐     │
│  │ Imbalance = CV²(concat(all_gates))          │     │
│  └─────────────────────────────────────────────┘     │
│  ┌─────────────────────────────────────────────┐     │
│  │ Reward = IoU - α·(FLOPs/budget) - β·Imbal   │     │
│  └─────────────────────────────────────────────┘     │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────┐
│  GRPO Update                                           │
│  ┌─────────────────────────────────────────────┐     │
│  │ Advantages = (Reward - mean) / std           │     │
│  └─────────────────────────────────────────────┘     │
│  ┌─────────────────────────────────────────────┐     │
│  │ L_total = -Σ A·Σ_i logprobs_i               │     │
│  │         + β·(-H)                             │     │
│  │         + λ_kl·Σ_i kl_loss_i                │     │
│  │         + λ_l2·||θ_routing||²               │     │
│  └─────────────────────────────────────────────┘     │
│  ┌─────────────────────────────────────────────┐     │
│  │ optimizer.zero_grad()                        │     │
│  │ L_total.backward()                           │     │
│  │ clip_grad_norm_(θ_routing)                   │     │
│  │ optimizer.step()                             │     │
│  └─────────────────────────────────────────────┘     │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
        Update dual_α (if Lagrangian enabled)
                │
                ▼
        Log to TensorBoard & Save artifacts
```

## Module Dependency Graph

```
examples/rl_train_routing_policy.py
    │
    ├─> rl/algos/grpo.py
    │       └─> rl/policies/routing_policy.py
    │
    ├─> rl/utils/rollout.py
    ├─> rl/utils/flops.py
    ├─> rl/utils/visualization.py
    ├─> rl/utils/checkpoint.py
    │
    ├─> models/gating.py
    │       ├─> rl/policies/routing_policy.py
    │       └─> models/adaptation_layers.py (MoEGate)
    │
    └─> MultiModalPredictor
            └─> models/utils.py
                    ├─> models/gating.py (creates RLGate factory)
                    └─> models/adaptation_layers.py
                            └─> ConvLoRALinear (uses gate_factory)
```

## Config-to-Code Flow

```
User Config (YAML or dict)
    │
    optim.lora.conv_lora_gating: "rl"
    optim.lora.rl_routing_ckpt: "path/to/ckpt.pt"
    │
    ▼
models/utils.py::apply_peft_adaptation()
    │
    ├─ Check if conv_lora_gating == "rl"
    │
    ├─ Load RoutingPolicy from rl_routing_ckpt
    │       │
    │       └─> rl/utils/checkpoint.py::load_checkpoint()
    │
    ├─ Create RLGate factory
    │       │
    │       └─> models/gating.py::RLGate(routing_policy, ...)
    │
    └─ Pass gate_factory to inject_adaptation_to_linear_layer()
            │
            └─> For each attention layer:
                    │
                    └─> create_adaptation(..., gate_factory=gate_factory)
                            │
                            └─> ConvLoRALinear(..., gate_factory=gate_factory)
                                    │
                                    └─> self.lora_moe_gating = gate_factory()
                                            │
                                            └─> RLGate instance created
```

## Forward Pass Flow (Inference/Eval with RL Gate)

```
Input: image (B, 3, 1024, 1024)
    │
    ▼
SAM Vision Encoder
    │
    ▼
Conv-LoRA Layer 0
    │
    ├─ x @ A.T → lora_res(B, r, H, W)
    │
    ├─ RLGate.forward(lora_res, layer_idx=0)
    │       │
    │       ├─ GAP(lora_res) → (B, r)
    │       ├─ Concat layer_embed(0) → (B, r+32)
    │       ├─ MLP → logits(B, M)
    │       ├─ argmax → action (greedy in eval)
    │       └─ Scatter → gates(B, M)  [one-hot]
    │
    ├─ SparseDispatcher(gates)
    │       │
    │       ├─ Dispatch to experts
    │       ├─ Expert_i: upsample → Conv3x3 → GELU → downsample
    │       └─ Combine outputs
    │
    ├─ lora_res + expert_res → lora_res'
    │
    └─ result + lora_res' @ B.T → output
    │
    ▼
(repeat for layers 1 to N-1)
    │
    ▼
Mask Decoder → pred_masks
```

## Training Loop Detailed Steps

```
Step t:
┌─────────────────────────────────────────────────────┐
│ 1. Sample batch                                      │
│    images, labels = next(dataloader)                 │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ 2. Forward with RL gates                             │
│    all_logprobs = []                                 │
│    all_gates = []                                    │
│    all_kl_losses = []                                │
│                                                      │
│    for layer_idx, conv_lora_layer in enumerate(...):│
│        RLGate.forward(lora_res, layer_idx)          │
│          ├─ Sample action                            │
│          ├─ Store logprob                            │
│          ├─ Compute KL to NoisyTopK                  │
│          └─ Return gates                             │
│                                                      │
│    pred_masks = SAM.mask_decoder(...)               │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ 3. Compute reward components                         │
│    IoU = iou(pred_masks, labels)                    │
│    FLOPs = compute_layer_flops(all_gates)           │
│    Imbalance = compute_imbalance(all_gates)         │
│    Reward = IoU - dual_α·FLOPs - β·Imbalance        │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ 4. GRPO advantage computation                        │
│    Buffer.add(logprobs, reward, gates, ...)         │
│    Advantages = (Rewards - mean) / std               │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ 5. GRPO update                                       │
│    GRPOTrainer.update(rollout_data, advantages)     │
│      ├─ L_pg = -E[A·Σ logprobs]                     │
│      ├─ L_entropy from all logits                   │
│      ├─ L_kl from all kl_losses                     │
│      ├─ L_adapter = ||routing_policy.params||²      │
│      ├─ L_total = L_pg + β·L_H + λ·L_kl + λ·L_l2   │
│      ├─ Backward()                                   │
│      └─ optimizer.step()                             │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ 6. Lagrangian update                                 │
│    If FLOPs > budget:                               │
│        dual_α += lr · (FLOPs - budget)              │
│    Clip dual_α >= 0                                 │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ 7. Logging                                           │
│    TensorBoard: scalars, histograms                  │
│    Local: heatmaps, JSON stats                       │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│ 8. Checkpoint (every N steps)                        │
│    Save: policy, optimizer, dual_α, RNG, config     │
└─────────────────────────────────────────────────────┘
```

## Class Hierarchy

```
nn.Module
    │
    ├─ RoutingPolicy
    │    └─ forward(feats, layer_idx) -> logits
    │
    ├─ NoisyTopKGate
    │    ├─ forward(feats, layer_idx) -> (gates, loss, info)
    │    └─ get_ref_logits(feats) -> logits
    │
    └─ RLGate
         ├─ __init__(routing_policy, reference_gate, k)
         └─ forward(feats, layer_idx) -> (gates, aux_loss, info)

GRPOTrainer (not nn.Module)
    ├─ compute_policy_loss(logprobs, advantages)
    ├─ compute_entropy_loss(logits_list)
    ├─ compute_kl_loss(kl_losses)
    ├─ compute_adapter_l2_loss()
    └─ update(rollout_data, advantages) -> metrics

RolloutBuffer
    ├─ add(state, action, logprob, reward, ...)
    ├─ get() -> dict
    └─ reset()
```

## Visualization Outputs

### During Training

```
rl_routing/
├── logs/                           # TensorBoard logs
│   └── events.out.tfevents.*
├── artifacts/
│   ├── gates_step_100.png          # Expert usage at step 100
│   ├── gates_step_200.png
│   └── ...
└── checkpoints/
    ├── step_500.pt
    ├── step_1000.pt
    └── final.pt
```

### After Evaluation

```
eval_results/
├── summary.json                    # Overall metrics
├── stats/
│   ├── image_001.json              # Per-image: IoU, FLOPs, gates
│   ├── image_002.json
│   └── ...
├── heatmaps/
│   ├── expert_usage.png            # Aggregated expert usage
│   └── per_layer_usage.png         # (optional) Usage per layer
└── features/                       # (if --save_features)
    ├── encoder_layer_10.npy
    ├── expert_0_output.npy
    └── ...
```

## Hyperparameter Tuning Guide

### Key Hyperparameters

| Param | Default | Range | Effect |
|-------|---------|-------|--------|
| `kl_coef` | 0.05 | 0.01-0.2 | Higher = stay closer to Noisy-TopK |
| `entropy_coef` | 0.01 | 0.001-0.1 | Higher = more exploration |
| `adapter_l2_coef` | 0.001 | 1e-4-1e-2 | Higher = more regularization |
| `compute_budget` | 1e10 | - | Target FLOPs per batch |
| `lagrangian_lr` | 0.001 | 1e-4-1e-2 | Speed of dual_α adaptation |
| `imbalance_weight` | 0.01 | 0.001-0.1 | Penalty for uneven expert usage |

### Recommended Tuning Strategy

1. **Start with defaults**: Run baseline to get initial metrics
2. **Ablate one at a time**:
   - Disable KL (`kl_coef=0`): Check if policy diverges
   - Disable Lagrangian (`use_lagrangian=False`): Check FLOPs behavior
   - Vary compute_budget: Plot IoU vs FLOPs curve
3. **Monitor training**:
   - KL should stabilize around 0.05-0.1
   - dual_α should converge (oscillate around a value)
   - Reward should increase over time
4. **Compare to baseline**:
   - Same FLOPs: expect +1-3% IoU
   - Lower FLOPs: expect <1% IoU drop at 70% FLOPs

## Code-to-Paper Mapping (Extended)

### Original Conv-LoRA Paper
| Paper Section | Code Location |
|---------------|---------------|
| Sec. 3 (Conv-LoRA) | `adaptation_layers.py::ConvLoRALinear` |
| Sec. 3.2 (MoE-Conv, Noisy-TopK) | `adaptation_layers.py::MoEGate` |
| Sec. 3.3 (Integration) | `models/utils.py::inject_adaptation_to_linear_layer` |
| Sec. 4 (Training, moe_loss) | `models/sam.py::SAMForSemanticSegmentation` |

### PERL Paper (Enhancements)
| Technique | Code Location |
|-----------|---------------|
| Parameter-efficient update | `routing_policy.py` (small adapter) |
| KL to reference | `grpo.py::compute_kl_loss` |
| Adapter L2 | `grpo.py::compute_adapter_l2_loss` |
| BC warmstart | `rl_bc_routing_policy.py` |

### GRPO Algorithm
| Component | Code Location |
|-----------|---------------|
| Group-wise advantages | `rollout.py::compute_group_advantages` |
| Policy gradient | `grpo.py::compute_policy_loss` |
| Entropy regularization | `grpo.py::compute_entropy_loss` |

## FAQ

### Q: Why use GRPO instead of PPO?
A: GRPO is simpler (no clipping, no value network required) and works well for problems where groups of rollouts can be compared (e.g., same batch). We can add PPO later if needed.

### Q: Why KL to Noisy-TopK instead of uniform?
A: Noisy-TopK is a strong baseline that already works. KL to it ensures RL explores locally rather than randomly, improving sample efficiency.

### Q: Can I disable Lagrangian and just use fixed α?
A: Yes, set `use_lagrangian=False` and tune α manually via `compute_budget`.

### Q: How to choose compute_budget?
A: Run baseline Noisy-TopK first, measure average FLOPs, then set budget to 0.7-1.0x that value.

### Q: What if training is unstable?
A: Increase `kl_coef` and `adapter_l2_coef`, decrease `lr`, check gradient norms.

### Q: How to visualize expert specialization?
A: Use `--save_features` in eval, then analyze which experts activate for which image regions.

---

End of Architecture Diagram

