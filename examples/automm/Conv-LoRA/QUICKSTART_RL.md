# Quick Start: RL Expert Routing (Scheme B)

## Prerequisites

**IMPORTANT**: You need a trained baseline Conv-LoRA model first.

### Step 0: Train Baseline Conv-LoRA (if not done already)

#### Option A: Use Existing Checkpoint

If you already have a trained Conv-LoRA model:

```bash
# Example: Use the checkpoint from the hierarchical RL experiments
BASELINE_CKPT="AutogluonModels/ag-20251113_165105/model.ckpt"

# Verify it exists
ls -lh $BASELINE_CKPT
```

#### Option B: Train from Scratch

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA

# Train baseline Conv-LoRA on ISIC 2017
python run_semantic_segmentation.py \
  --task isic2017 \
  --seed 42686693 \
  --rank 3 \
  --expert_num 8 \
  --num_gpus 1 \
  --per_gpu_batch_size 1 \
  --batch_size 4 \
  --output_dir baseline_conv_lora

# This takes ~6-8 hours on single GPU (30 epochs)
```

**Verify baseline exists**:
```bash
ls baseline_conv_lora/
# Should see: model.ckpt, config.yaml, hparams.yaml, metrics.txt
```

**Expected Baseline Performance**:
- Validation IoU: ~0.76-0.77
- Validation DICE: ~0.84-0.85

---

## Three-Step RL Training

### Approach A: Full Pipeline (Recommended)

#### Step 1: BC Warmstart (~15 min on L20)

**IMPORTANT NOTICE**: The current BC implementation uses a **simplified synthetic data approach** because true data collection requires deeper hooks into Conv-LoRA internals (which would need modifying the forward pass).

**For quick testing**, use synthetic BC:

```bash
python rl_bc_routing_policy.py \
  --task isic2017 \
  --model_path outputs_baseline/ \
  --output_dir bc_warmstart/ \
  --num_epochs 5 \
  --max_samples 1000 \
  --device cuda
```

**What this does**:
- Loads your trained Conv-LoRA model
- Detects rank, #experts, #layers automatically
- Generates synthetic BC data (simulates Noisy-TopK behavior)
- Trains routing policy to match
- Saves `bc_warmstart/routing_bc.pt`

**Output**:
```
✓ BC warmstart checkpoint saved to bc_warmstart/routing_bc.pt
  - Trained on 1000+ samples
  - X layers, 8 experts, rank 3
```

---

#### Step 2: RL Training (~35 min on L20)

**NOTICE**: Current RL training also uses simplified approach. Full integration requires model forward hooks.

```bash
python rl_train_routing_policy.py \
  --task isic2017 \
  --output_dir rl_routing/ \
  --warmstart bc_warmstart/routing_bc.pt \
  --num_experts 8 \
  --rank 3 \
  --lr 1e-4 \
  --max_steps 6000 \
  --batch_size 12 \
  --kl_coef 0.05 \
  --entropy_coef 0.01 \
  --adapter_l2_coef 0.001 \
  --compute_budget 1e10 \
  --use_lagrangian \
  --ckpt_interval 500 \
  --vis_interval 100 \
  --device cuda
```

**Monitor training**:
```bash
# In another terminal
tensorboard --logdir rl_routing/logs --port 6006 --bind_all
```

**Output**:
- `rl_routing/checkpoints/step_*.pt`
- `rl_routing/checkpoints/final.pt`
- `rl_routing/logs/` (TensorBoard)
- `rl_routing/artifacts/gates_step_*.png`

---

#### Step 3: Evaluation (~2 min)

```bash
python rl_eval_routing_policy.py \
  --task isic2017 \
  --ckpt_path rl_routing/checkpoints/final.pt \
  --output_dir eval_results/ \
  --num_experts 8 \
  --rank 3 \
  --save_heatmaps \
  --device cuda
```

**Output**:
- `eval_results/summary.json` - Overall IoU, FLOPs
- `eval_results/heatmaps/expert_usage.png`
- `eval_results/stats/*.json` - Per-image details

---

### Approach B: Skip BC (Faster, Less Stable)

If you want to test quickly:

```bash
python rl_train_routing_policy.py \
  --task isic2017 \
  --output_dir rl_routing_no_bc/ \
  --num_experts 8 \
  --rank 3 \
  --max_steps 10000 \
  --batch_size 12 \
  --device cuda
```

**Tradeoff**: Saves 15 min BC time, but may need more RL steps (10K vs 6K).

---

## 🚨 Current Limitations

### What Works Now
✅ Full module structure and interfaces
✅ GRPO trainer with PERL regularizers  
✅ Routing policy network
✅ BC training on synthetic data
✅ Config flags and A/B switching
✅ TensorBoard logging infrastructure
✅ Checkpointing and resume

### What Needs Integration
❌ Real BC data collection (hooks into Conv-LoRA forward)
❌ RL training forward pass (inject RLGate into SAM)
❌ Actual reward computation from real model outputs

### Why Simplified?

The core challenge: Conv-LoRA's forward pass needs to expose intermediate `lora_res` and `gates` for collection. Options:

1. **Modify `adaptation_layers.py`**: Add `self._last_lora_res = lora_res.clone()` in forward
2. **Deep hooks**: Intercept mid-forward (complex, fragile)
3. **Synthetic data** (current): Fast to test, but not real Noisy-TopK behavior

**For production experiments**, I recommend Option 1 (small modification to save intermediates).

---

## 🔧 Quick Integration Fix (Do This First)

To make BC collect real data, add 2 lines to `ConvLoRALinear.forward`:

```python
# In multimodal/src/autogluon/multimodal/models/adaptation_layers.py
# Around line 727-737, after computing gates:

gates, moe_loss, gate_info = gate_output  # (or legacy 2-tuple)

# ADD THESE TWO LINES:
self._last_lora_res = lora_res.clone().detach()  # Save for BC collection
self._last_gates = gates.clone().detach()         # Save gate decisions
```

Then re-run BC collection - it will capture real data!

---

## Minimal Working Example (Test Now)

Even with current placeholders, you can test the workflow:

```bash
# 1. Synthetic BC (works now, ~2 min)
python rl_bc_routing_policy.py \
  --task isic2017 \
  --model_path outputs_baseline/ \
  --output_dir bc_test/ \
  --num_epochs 2 \
  --max_samples 100

# 2. Check output
ls bc_test/routing_bc.pt
# Should exist

# 3. Test RL training structure (placeholder, ~1 min)
python rl_train_routing_policy.py \
  --task isic2017 \
  --warmstart bc_test/routing_bc.pt \
  --output_dir rl_test/ \
  --max_steps 10 \
  --device cuda

# 4. Check TensorBoard
tensorboard --logdir rl_test/logs
```

This verifies infrastructure works, even though using placeholders.

---

## Recommended Next Action

**For real experiments**: Switch to agent mode and say:

> "Please add the 2-line modification to ConvLoRALinear.forward to enable real BC data collection, then create an end-to-end integration test"

I'll implement the missing hooks and test on a small batch.

---

## Alternative: Start with Current Synthetic Version

If you want to see the full pipeline structure work immediately (even with synthetic data):

```bash
# This will work RIGHT NOW (uses synthetic BC data)
python rl_bc_routing_policy.py \
  --task isic2017 \
  --model_path outputs_baseline/ \
  --output_dir bc_warmstart/ \
  --num_epochs 5
```

Then examine the TensorBoard logs and checkpoint to understand the system.

**When ready for real experiments**, request the integration fixes.

