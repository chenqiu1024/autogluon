# RL-based Conv-LoRA Layer Selection for Medical Image Segmentation

This directory contains the implementation of reinforcement learning-based dynamic layer selection for Conv-LoRA in medical image segmentation tasks.

## Overview

The method uses RL to learn a policy that dynamically selects which Conv-LoRA layers to activate for each input image, aiming to improve segmentation performance (IoU/DICE) on the ISIC 2017 dataset.

### Key Components

1. **Layer Selection Policy** (`multimodal/src/autogluon/multimodal/rl/policies/layer_selection_policy.py`)
   - Neural network that decides which layers to activate
   - Input: Patch embeddings from SAM encoder
   - Output: Binary mask for 32 transformer layers

2. **RL Environment** (`multimodal/src/autogluon/multimodal/rl/envs/conv_lora_env.py`)
   - Wraps trained Conv-LoRA model
   - Provides reward signals based on segmentation performance

3. **REINFORCE Algorithm** (`multimodal/src/autogluon/multimodal/rl/algos/reinforce.py`)
   - Policy gradient optimization with variance reduction
   - Moving average baseline and entropy regularization

4. **Training Script** (`train_rl_layer_selection.py`)
   - End-to-end RL training pipeline

5. **Evaluation Script** (`evaluate_rl_policy.py`)
   - Test set evaluation with visualizations

## Usage

### Step 1: Train Baseline Conv-LoRA Model

First, train a baseline Conv-LoRA model with all 32 layers activated:

```bash
python run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --output_dir baseline_conv_lora \
    --num_gpus 1 \
    --per_gpu_batch_size 1 \
    --batch_size 4
```

This will create a checkpoint in `AutogluonModels/ag-YYYYMMDD_HHMMSS/`.

Expected baseline performance on ISIC 2017:
- **IoU**: ~0.77
- **DICE**: ~0.85

### Step 2: Train RL Layer Selection Policy

Train the RL policy using the pre-trained Conv-LoRA model:

```bash
python train_rl_layer_selection.py \
    --task isic2017 \
    --ckpt_path AutogluonModels/ag-YYYYMMDD_HHMMSS \
    --output_dir rl_layer_selection \
    --num_episodes 1000 \
    --batch_size 16 \
    --eval_subset_size 200 \
    --learning_rate 1e-4 \
    --entropy_coef 0.01 \
    --reward_metric iou
```

**Key hyperparameters:**
- `num_episodes`: Number of training episodes (default: 1000)
- `batch_size`: Images per episode (default: 16)
- `eval_subset_size`: Validation subset size for reward computation (default: 200)
- `learning_rate`: Policy learning rate (default: 1e-4)
- `entropy_coef`: Entropy regularization coefficient for exploration (default: 0.01)
- `baseline_decay`: Moving average baseline decay rate (default: 0.99)

**Output:**
- Checkpoints: `rl_layer_selection/checkpoints/`
- Logs: `rl_layer_selection/logs/` (view with TensorBoard)
- Training curves: `rl_layer_selection/training_curves.png`
- Summary: `rl_layer_selection/training_summary.json`

**Monitoring training:**
```bash
tensorboard --logdir rl_layer_selection/logs
```

### Step 3: Evaluate Trained Policy

Evaluate the trained policy on the test set:

```bash
python evaluate_rl_policy.py \
    --task isic2017 \
    --ckpt_path AutogluonModels/ag-YYYYMMDD_HHMMSS \
    --policy_path rl_layer_selection/checkpoints/best.pt \
    --output_dir rl_evaluation \
    --compare_random
```

**Output:**
- Metrics: `rl_evaluation/evaluation_results.json`
- Layer activation analysis: `rl_evaluation/layer_activation_analysis.png`
- Performance comparison: `rl_evaluation/performance_comparison.png`

## Expected Results

Based on the plan's success criteria:

| Metric | Baseline (32 layers) | RL Policy (dynamic) | Improvement |
|--------|---------------------|---------------------|-------------|
| IoU    | ~0.77               | ~0.78-0.79          | +1-2%       |
| DICE   | ~0.85               | ~0.86-0.87          | +1-2%       |
| Active Layers | 32            | ~15-25              | Variable    |

**Key insights:**
- Different images activate different layers (dynamic adaptation)
- Layer activation frequency varies by layer depth
- Policy learns meaningful patterns (not random selection)

## Implementation Details

### Modified Files

1. **`multimodal/src/autogluon/multimodal/models/adaptation_layers.py`**
   - Modified `ConvLoRALinear.forward()` to accept `layer_mask` parameter
   - Returns zero moe_loss when layer is masked

2. **`multimodal/src/autogluon/multimodal/models/custom_hf_models/modeling_sam_for_conv_lora.py`**
   - Modified `SamVisionEncoder.forward()` to accept `layer_masks` parameter
   - Modified `SamVisionLayer.forward()` to pass `layer_mask` to attention
   - Modified `SamVisionAttention.forward()` to pass `layer_mask` to Conv-LoRA layers

### Design Choices

1. **Dynamic per-image layer selection**: Each image gets its own layer configuration
2. **Fixed parameter budget**: All 32 layers' parameters exist; policy selects which to use
3. **Search mode training**: First train Conv-LoRA, then train RL policy (frozen Conv-LoRA)

### Reward Function

```python
reward = IoU (or DICE) score on validation subset
```

Simple and direct: optimize segmentation performance.

### Variance Reduction Techniques

1. **Moving average baseline**: Reduces gradient variance
2. **Entropy regularization**: Encourages exploration
3. **Batch averaging**: Average rewards across batch
4. **Gradient clipping**: Stabilizes training

## Troubleshooting

### Issue: Policy always activates all layers

**Solution**: Increase entropy coefficient or adjust init_bias:
```bash
--entropy_coef 0.05 --init_bias 1.0
```

### Issue: Policy degenerates to activating no layers

**Solution**: Use positive init_bias and check baseline reward:
```bash
--init_bias 3.0
```

### Issue: High variance in training

**Solution**:
- Increase eval_subset_size for more stable rewards
- Increase baseline_decay for smoother baseline
- Reduce learning_rate

### Issue: No improvement over baseline

This is expected if:
- The baseline model is already near-optimal
- ISIC 2017 task benefits from all layer adaptations
- Evaluation is on a different distribution

**Next steps**:
- Try on more diverse datasets
- Analyze layer selection patterns for insights
- Consider joint training (future work)

## Future Extensions (Not Currently Implemented)

### Alternative Training Modes

1. **Fixed configuration mode**: Learn single layer config for all images
   - Simpler, faster training
   - Good for identifying universally important/unimportant layers

2. **Hybrid mode**: Learn 2-3 layer config templates
   - Balance between flexibility and simplicity

3. **Joint training**: Train Conv-LoRA and policy together
   - Potentially higher ceiling
   - More unstable

4. **Alternating training**: Iteratively update Conv-LoRA and policy
   - Progressive optimization

### Alternative Parameter Budgets

1. **Constant total parameters**: Reallocate parameters from inactive to active layers
   - Requires dynamic model architecture

2. **Reduced parameters**: Simply use fewer parameters
   - Focuses on efficiency

## Citation

If you use this code, please cite:

```bibtex
@article{zhong2024convolution,
  title={Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model},
  author={Zhong, Zihan and Tang, Zhiqiang and He, Tong and Fang, Haoyang and Yuan, Chun},
  journal={arXiv preprint arXiv:2401.17868},
  year={2024}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

