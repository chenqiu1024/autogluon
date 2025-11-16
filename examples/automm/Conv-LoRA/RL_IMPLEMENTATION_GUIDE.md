# RL-based Expert Routing Implementation Guide (Scheme B)

## Overview

This document provides a comprehensive guide to the RL-based expert routing implementation for Conv-LoRA, detailing the architecture, training flow, and integration points.

## Architecture Components

### 1. Core RL Modules (`multimodal/src/autogluon/multimodal/rl/`)

#### 1.1 Routing Policy (`policies/routing_policy.py`)
```
RoutingPolicy: Lightweight MLP that learns expert selection

Input: (feats, layer_idx)
  - feats: (B, C, H, W) spatial features from Conv-LoRA
  - layer_idx: scalar indicating which layer

Architecture:
  feats -> GAP -> (B, C) -> concat(GAP, embed(layer_idx)) -> MLP -> logits(B, M)

Output: logits over M experts

Parameters: ~1-2M (negligible vs SAM's 600M+)
```

#### 1.2 GRPO Trainer (`algos/grpo.py`)
```
GRPOTrainer: Implements GRPO with PERL regularizers

Loss = L_pg + β·L_entropy + λ_kl·L_kl + λ_l2·L_adapter

Components:
- L_pg: Policy gradient with group-wise normalized advantages
- L_entropy: Entropy regularization for exploration
- L_kl: KL(π || π_NoisyTopK) to reference policy
- L_adapter: L2 penalty on routing policy parameters

Advantages: A_i = (R_i - mean(R_group)) / std(R_group)
```

#### 1.3 Gating Abstraction (`models/gating.py`)
```
BaseGate (Protocol):
  forward(feats, layer_idx) -> (gates, aux_loss, info_dict)

NoisyTopKGate:
  Wraps existing MoEGate
  Provides get_ref_logits() for KL computation

RLGate:
  Uses RoutingPolicy for expert selection
  Computes KL to NoisyTopKGate during training
  Supports Top-1 sampling (Categorical) or Top-K (Gumbel-Softmax)
```

### 2. Utilities

#### 2.1 FLOPs Estimation (`utils/flops.py`)
- `compute_conv_flops()`: Theoretical FLOPs for Conv2d
- `compute_expert_flops()`: Batch FLOPs given gates
- Used in reward: IoU - α·(FLOPs/budget)

#### 2.2 Rollout Management (`utils/rollout.py`)
- `RolloutBuffer`: Stores trajectories
- `compute_group_advantages()`: GRPO core
- `compute_imbalance()`: CV^2 load balance metric

#### 2.3 Visualization (`utils/visualization.py`)
- TensorBoard: `log_scalars()`, `log_histograms()`
- Local artifacts: `save_expert_heatmap()`, `save_json_stats()`

#### 2.4 Checkpointing (`utils/checkpoint.py`)
- `save_checkpoint()`: Policy + optimizer + dual_α + RNG
- `restore_training_state()`: Full state recovery for resume

#### 2.5 Feature Hooks (`utils/hooks.py`)
- `FeatureHook`: Context manager for capturing intermediate features
- `downsample_feature_map()`, `feature_map_to_heatmap()`: Preprocessing for viz

## Training Flow

### Phase 1: BC Warmstart (Optional)

```
Script: rl_bc_routing_policy.py

1. Load Conv-LoRA model with Noisy-TopK gating
2. For each batch:
   - Forward through SAM
   - Hook Conv-LoRA layers to capture (feats, layer_idx, expert_action)
3. Train RoutingPolicy via cross-entropy:
   L_BC = -Σ log π(a_expert | feats, layer_idx)
4. Save checkpoint as π_ref

Output: bc_warmstart/routing_bc.pt
```

### Phase 2: GRPO Training

```
Script: rl_train_routing_policy.py

Initialization:
1. Create RoutingPolicy (load BC ckpt if provided)
2. Create NoisyTopKGate as reference (for KL)
3. Create RLGate wrapping RoutingPolicy
4. Inject RLGate into ConvLoRALinear via gate_factory
5. Create GRPOTrainer with loss coefficients

Training Loop (each step):
1. Sample batch from train_df
2. Forward SAM with RLGate:
   - Each Conv-LoRA layer calls RLGate(feats, layer_idx)
   - RLGate samples actions, computes logprobs, KL
   - Collect all_gates, all_logprobs, all_info_dicts
3. Compute reward:
   - IoU = iou(pred_masks, gt_masks)
   - FLOPs = compute_layer_flops_from_gates(all_gates)
   - Imbalance = compute_imbalance(all_gates)
   - Reward = IoU - α·(FLOPs/budget) - β·imbalance
4. Compute advantages (group-wise, within batch)
5. GRPO update:
   - Aggregate logprobs across layers
   - Compute total loss with PERL regularizers
   - Backward + gradient clip + optimizer step
6. Update dual_α (Lagrangian):
   - If FLOPs > budget: dual_α += lr·(FLOPs - budget)
   - Else: dual_α -= lr·(budget - FLOPs)
   - dual_α = max(0, dual_α)
7. Log to TensorBoard:
   - Scalars: reward, IoU, FLOPs, dual_α, losses
   - Histograms: gate distributions
8. Save artifacts (every N steps):
   - Expert usage heatmaps
9. Save checkpoint (every M steps):
   - Policy, optimizer, dual_α, RNG, config

Output: rl_routing/checkpoints/{step_*.pt, final.pt}
```

### Phase 3: Evaluation

```
Script: rl_eval_routing_policy.py

1. Load RoutingPolicy from checkpoint
2. Create RLGate (eval mode, greedy selection)
3. Inject into SAM model
4. For each test image:
   - Forward with RL gates
   - Compute IoU, FLOPs
   - Collect gate decisions
   - Save per-image JSON stats
5. Aggregate metrics:
   - Avg IoU, FLOPs
   - Expert usage distribution
6. Generate visualizations:
   - Expert usage heatmap
   - Optional: encoder features, expert outputs

Output: eval_results/{summary.json, stats/*.json, heatmaps/*.png}
```

## Integration with Existing Code

### Modified Files

#### `models/adaptation_layers.py` (ConvLoRALinear)
```python
# Added parameters:
- gate_factory: callable (optional)
- layer_idx: int (optional)

# Forward changes:
- Support both legacy MoEGate (2-tuple) and BaseGate (3-tuple) returns
- Pass layer_idx to gate if available
- Extract gate_info for RL updates
```

#### `models/utils.py`
```python
# apply_peft_adaptation():
- Check config.optim.lora.conv_lora_gating
- If 'rl': load RoutingPolicy, create RLGate factory
- Pass gate_factory to inject_adaptation_to_linear_layer

# inject_adaptation_to_linear_layer():
- Track layer_idx counter
- Pass layer_idx and gate_factory to create_adaptation

# create_adaptation():
- Accept gate_factory and layer_idx in kwargs
- Pass to ConvLoRALinear constructor
```

#### `configs/optim/default.yaml`
```yaml
# Added config options:
optim:
  lora:
    conv_lora_gating: noisy_topk  # or 'rl'
    rl_routing_ckpt: null  # path to trained policy
```

### A/B Testing Usage

#### Baseline (Noisy-TopK)
```python
predictor = MultiModalPredictor(
    problem_type="semantic_segmentation",
    hyperparameters={
        "optim.peft": "conv_lora",
        "optim.lora.conv_lora_gating": "noisy_topk",  # default
        ...
    }
)
```

#### RL Routing
```python
predictor = MultiModalPredictor(
    problem_type="semantic_segmentation",
    hyperparameters={
        "optim.peft": "conv_lora",
        "optim.lora.conv_lora_gating": "rl",
        "optim.lora.rl_routing_ckpt": "rl_routing/checkpoints/final.pt",
        ...
    }
)
```

## Reward Function Details

```
Reward = IoU - α·(FLOPs / budget) - β·imbalance

Components:
1. IoU: Intersection over Union with ground truth
2. FLOPs term:
   - Compute per-expert FLOPs: 2·in_c·out_c·k²·H·W + 8·out_c·H·W (GELU)
   - Sum over active experts (gates > 0) across all layers
   - Normalize by budget, multiply by dual variable α
3. Imbalance term:
   - CV² = var(expert_loads) / (mean(expert_loads)² + ε)
   - Encourages balanced expert usage

Lagrangian update (optional):
- If avg_FLOPs > budget: increase α (penalize FLOPs more)
- Else: decrease α (allow more FLOPs)
- Converges to satisfying budget constraint while maximizing IoU
```

## PERL Enhancements Summary

### 1. Parameter Efficiency
- Freeze SAM base model + Conv-LoRA A/B matrices
- Train only RoutingPolicy (~1-2M params)
- Reduces training cost and overfitting risk

### 2. KL to Reference Policy
```
L_kl = λ_kl · KL(π || π_NoisyTopK)
     = λ_kl · Σ [π(a|s) · (log π(a|s) - log π_ref(a|s))]

Purpose:
- Prevents policy from diverging too far from working baseline
- Stabilizes training
- Enables safe exploration
```

### 3. Adapter L2 Regularization
```
L_adapter = λ_l2 · Σ [||θ_routing||²]

Purpose:
- Prevents routing policy from fitting noise
- Improves generalization
- Complements KL regularization
```

### 4. BC Warmstart
- Initialize policy by imitating Noisy-TopK
- Provides strong starting point
- Reduces random exploration phase

## Monitoring & Debugging

### TensorBoard Metrics

#### Training
- `train/reward`: Total reward per batch
- `train/iou`: IoU metric
- `train/flops`: FLOPs consumed
- `train/imbalance`: Load imbalance (CV²)
- `train/dual_alpha`: Lagrangian multiplier
- `loss/total`: Total loss
- `loss/policy_gradient`: PG component
- `loss/kl`: KL to reference
- `loss/entropy`: Entropy term
- `loss/adapter_l2`: L2 penalty
- `loss/grad_norm`: Gradient norm
- `train/gates_dist`: Histogram of expert selections

### Local Artifacts

#### Heatmaps
- `artifacts/gates_step_*.png`: Bar chart of expert usage
- Shows which experts are preferred

#### JSON Stats
- Per-checkpoint or per-image
- Contains: IoU, FLOPs, gate decisions, layer-wise stats

### Optional Internal Visualizations

#### Encoder Features
```python
from autogluon.multimodal.utils.hooks import FeatureHook

with FeatureHook(sam.vision_encoder, target_layers=['neck']) as hook:
    output = sam(image)
    features = hook.get_features()['neck']
    # Save heatmap
```

#### Expert Outputs
```python
# Hook into specific expert
with FeatureHook(conv_lora_layer.lora_moe_experts[i]) as hook:
    output = layer(input)
    expert_output = hook.get_features()
    # Visualize
```

## Checkpointing & Resume

### Checkpoint Contents
```python
{
    'policy': routing_policy.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scaler': scaler.state_dict(),  # AMP
    'dual_alpha': float,
    'global_step': int,
    'rng_state': torch.get_rng_state(),
    'config': dict,
}
```

### Resume Training
```bash
python rl_train_routing_policy.py \
  --resume_from rl_routing/checkpoints/step_5000.pt \
  --task polyp \
  ...
```

Restores:
- Policy weights
- Optimizer state (momentum, etc.)
- Dual variable
- Global step counter
- RNG state (for reproducibility)

## Limitations & Future Work

### Current Limitations

1. **Placeholder Integration**
   - Current scripts show structure but need actual hooks into SAM forward
   - Need to modify SAM's forward to collect and use RL gates
   - Requires access to intermediate Conv-LoRA features

2. **Single-GPU Training**
   - No distributed RL support yet
   - Batch size limited by memory

3. **Fixed Architecture**
   - Routing policy architecture is fixed
   - No AutoML for policy design

### Full Implementation Checklist

To make this production-ready:

- [ ] Implement actual forward hooks into Conv-LoRA layers
- [ ] Modify SAM forward to use RLGate and collect info_dicts
- [ ] Implement BC data collection from Noisy-TopK
- [ ] Add distributed training support
- [ ] Add more reward options (Dice, boundary F-score, etc.)
- [ ] Implement value network (optional baseline for advantage)
- [ ] Add PPO clipping as alternative to pure GRPO
- [ ] Support variable K (Top-K routing)
- [ ] Add curriculum learning (anneal α, β over training)

### Scheme A (Future Work)

Interactive Prompt Policy - placeholder files created:
- `rl/envs/sam_prompt_env.py`
- `rl/policies/prompt_policy.py`
- `examples/.../rl_train_prompt_policy.py`
- `examples/.../rl_infer_prompt_policy.py`

To implement:
1. Define state representation (image features, current mask, uncertainty)
2. Implement multi-step episode environment
3. Design candidate proposal mechanism
4. Train with GRPO (reward = ΔIoU - λ·clicks)

## Quick Start Examples

### 1. BC Warmstart
```bash
python rl_bc_routing_policy.py \
  --task polyp \
  --output_dir bc_warmstart/ \
  --num_experts 8 \
  --rank 3
```

### 2. RL Training
```bash
python rl_train_routing_policy.py \
  --task polyp \
  --output_dir rl_routing/ \
  --warmstart bc_warmstart/routing_bc.pt \
  --max_steps 10000 \
  --kl_coef 0.05 \
  --use_lagrangian
```

### 3. Evaluation
```bash
python rl_eval_routing_policy.py \
  --task polyp \
  --ckpt_path rl_routing/checkpoints/final.pt \
  --output_dir eval_results/ \
  --save_heatmaps
```

### 4. Inference with RL Routing
```python
from autogluon.multimodal import MultiModalPredictor

predictor = MultiModalPredictor(
    problem_type="semantic_segmentation",
    hyperparameters={
        "optim.peft": "conv_lora",
        "optim.lora.r": 3,
        "optim.lora.conv_lora_expert_num": 8,
        "optim.lora.conv_lora_gating": "rl",
        "optim.lora.rl_routing_ckpt": "rl_routing/checkpoints/final.pt",
    }
)

# Training uses RL policy for routing
predictor.fit(train_data, tuning_data)

# Evaluation
metrics = predictor.evaluate(test_data)
```

## Code Navigation

### Key Files Map

| Component | File Path |
|-----------|-----------|
| Routing Policy | `multimodal/src/.../rl/policies/routing_policy.py` |
| GRPO Trainer | `multimodal/src/.../rl/algos/grpo.py` |
| Gating Interface | `multimodal/src/.../models/gating.py` |
| Conv-LoRA Integration | `multimodal/src/.../models/adaptation_layers.py` |
| PEFT Application | `multimodal/src/.../models/utils.py` |
| Config Schema | `multimodal/src/.../configs/optim/default.yaml` |
| BC Warmstart | `examples/.../rl_bc_routing_policy.py` |
| RL Training | `examples/.../rl_train_routing_policy.py` |
| Evaluation | `examples/.../rl_eval_routing_policy.py` |
| Example Config | `examples/.../configs/routing_rl_default.yaml` |

### Paper Alignment

| Code Component | Paper Section |
|----------------|---------------|
| `RoutingPolicy` | Extension beyond paper (RL routing) |
| `GRPOTrainer` | GRPO algorithm (external) + PERL (external) |
| `RLGate` | Replaces Noisy-TopK from Sec. 3.2 |
| `ConvLoRALinear.forward` | Sec. 3 (method), Sec. 3.2 (MoE-Conv) |
| Expert routing loop | Sec. 3.2 (dispatcher, upsample, combine) |
| Load balance loss | Sec. 4 (auxiliary loss) |

## Debugging Tips

### Check if RL gating is active
```python
# Print gate type
for name, module in model.named_modules():
    if hasattr(module, 'lora_moe_gating'):
        print(f"{name}: {type(module.lora_moe_gating)}")
        # Should show RLGate if config.optim.lora.conv_lora_gating == 'rl'
```

### Verify gate outputs
```python
# In RLGate.forward, add:
print(f"Layer {layer_idx}: actions = {actions}, logprobs = {logprobs}")
```

### Check FLOPs computation
```python
from autogluon.multimodal.rl.utils.flops import compute_expert_flops

flops = compute_expert_flops(gates, expert_configs, (H, W))
print(f"Batch FLOPs: {flops:.2e}")
```

### Monitor KL divergence
```python
# Should decrease during training if policy is learning
# But not go to zero (indicates policy is copying reference)
# Target: 0.01 - 0.1 range
```

## References

1. **Conv-LoRA**: Zhong et al., "Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model", ICLR 2024
2. **GRPO**: "Group Relative Policy Optimization" (algorithm details from RL literature)
3. **PERL**: "Parameter Efficient Reinforcement Learning from Human Feedback" (regularization techniques)
4. **MoE**: Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", 2017

## Contact & Support

For issues or questions about the RL routing implementation, please:
1. Check TensorBoard logs for anomalies
2. Verify checkpoint loading with `--resume_from`
3. Compare RL vs baseline metrics
4. Review this guide and code comments

Happy researching!

