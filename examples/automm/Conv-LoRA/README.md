# Conv-LoRA: Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model (ICLR 2024)

Examples showing how to use `Conv-LoRA` for parameter efficient fine-tuning SAM.

## 1. Installation
The installation may take a while since AutoGluon Multimodal has multiple dependencies.
```shell
  conda create -n conv-lora python=3.10
  conda activate conv-lora
  pip install -U pip
  pip install -U setuptools wheel
  git clone https://github.com/autogluon/autogluon
  cd autogluon && pip install -e multimodal/[tests]
  ```

## 2. Dataset

Enter the `autogluon/examples/automm/Conv-LoRA` directory and run the following script to download the datasets.

`python prepare_semantic_segmentation_datasets.py`

## 3. Training

`python run_semantic_segmentation.py --<flag> <value>`

- `task` refers to the dataset name, i.e., one of the datasets we have downloaded. Options are `polyp, leaf_disease_segmentation, camo_sem_seg, isic2017, road_segmentation, or SBU-shadow`.
- `seed` determines the random seed.
- `rank` determines the rank of Conv-LoRA. Default is 3.
- `expert_num` determines the used expert number of Conv-LoRA. Default is 8.
- `num_gpus` determines the number of gpu used for training. Default is 1.
- `output_dir` determines the path of output directory. Default is "outputs" folder.
- `ckpt_path` determines the path of model for evaluation. Default is "outputs" folder.
- `per_gpu_batch_size` is the batch size for each GPU. Default is 1.
- `batch_size` effective batch size. If batch_size > per_gpu_batch_size * num_gpus, gradient accumulation would be used. Default is 4.

## 4. Evaluation

After running the benchmark, the evaluation results of test set are stored in "{output_dir}/metrics.txt".

You can also run the following command to evaluate a checkpoint:

`python3 run_semantic_segmentation.py --task {dataset_name} --output_dir {output_dir} --ckpt_path {ckpt_path} --eval`

---

## 5. RL-based Expert Routing (Scheme B - Experimental)

### Overview
This extends Conv-LoRA with RL-based expert routing, replacing the Noisy Top-K gating with a learned policy trained via GRPO (Group Relative Policy Optimization) with PERL-style regularizers.

**Goal**: Learn to route tokens to experts for better IoU vs FLOPs trade-off.

**Key Features**:
- GRPO training with group-wise normalized advantages
- PERL regularizers: KL to Noisy-TopK reference, adapter L2
- Optional Lagrangian constraint for compute budget
- TensorBoard logging + local artifacts (heatmaps, JSON stats)
- Resumable training with full state checkpointing

### 5.1 Behavior Cloning (BC) Warmstart (Optional)

Pretrain routing policy to imitate Noisy-TopK decisions:

```bash
python rl_bc_routing_policy.py \
  --task polyp \
  --output_dir bc_warmstart/ \
  --num_experts 8 \
  --rank 3 \
  --lr 1e-3 \
  --num_epochs 10
```

Output: `bc_warmstart/routing_bc.pt`

### 5.2 RL Training

Train routing policy with GRPO:

```bash
python rl_train_routing_policy.py \
  --task polyp \
  --output_dir rl_routing/ \
  --warmstart bc_warmstart/routing_bc.pt \
  --num_experts 8 \
  --rank 3 \
  --lr 1e-4 \
  --max_steps 10000 \
  --kl_coef 0.05 \
  --entropy_coef 0.01 \
  --adapter_l2_coef 0.001 \
  --compute_budget 1e10 \
  --use_lagrangian \
  --ckpt_interval 500 \
  --vis_interval 100
```

**Key flags**:
- `--warmstart`: BC checkpoint for initialization
- `--kl_coef`: Weight for KL(π || π_NoisyTopK) regularization
- `--adapter_l2_coef`: Weight for adapter L2 penalty
- `--compute_budget`: Target FLOPs budget
- `--use_lagrangian`: Enable Lagrangian constraint on FLOPs
- `--resume_from`: Resume from checkpoint

**Outputs**:
- Checkpoints: `rl_routing/checkpoints/step_*.pt`, `final.pt`
- TensorBoard logs: `rl_routing/logs/`
- Artifacts: `rl_routing/artifacts/gates_step_*.png`

### 5.3 Evaluation

Evaluate trained routing policy:

```bash
python rl_eval_routing_policy.py \
  --task polyp \
  --ckpt_path rl_routing/checkpoints/final.pt \
  --output_dir eval_results/ \
  --save_heatmaps \
  --save_features
```

**Outputs**:
- Summary statistics: `eval_results/summary.json`
- Per-image stats: `eval_results/stats/*.json`
- Expert usage heatmaps: `eval_results/heatmaps/*.png`
- Optional features: `eval_results/features/*.npy` (if `--save_features`)

### 5.4 A/B Testing

Compare RL routing vs baseline Noisy-TopK by toggling config:

```python
# Baseline (Noisy-TopK)
hyperparameters = {
    "optim.lora.conv_lora_gating": "noisy_topk",  # default
    ...
}

# RL routing
hyperparameters = {
    "optim.lora.conv_lora_gating": "rl",
    "optim.lora.rl_routing_ckpt": "rl_routing/checkpoints/final.pt",
    ...
}
```

### 5.5 Monitoring & Debugging

**TensorBoard metrics**:
- `train/reward`, `train/iou`, `train/flops`, `train/imbalance`
- `train/dual_alpha` (Lagrangian multiplier)
- `loss/total`, `loss/kl`, `loss/entropy`, `loss/adapter_l2`
- `train/gates_dist` (histogram of expert selections)

**Artifacts**:
- Expert usage heatmaps: visualize which experts are selected
- JSON stats: per-image IoU, FLOPs, gate decisions

**Resume training**:
```bash
python rl_train_routing_policy.py \
  --resume_from rl_routing/checkpoints/step_5000.pt \
  ...
```

### 5.6 Limitations & Future Work

**Current limitations**:
- Placeholder integration (requires hooking into actual Conv-LoRA forward)
- Single-GPU training only
- No distributed RL

**Future enhancements (Scheme A)**:
- Interactive prompt policy for point/box selection
- Multi-step episodes with ΔIoU rewards
- Preference-based warmstart (DPO)

---

### Citation

```
@article{zhong2024convolution,
  title={Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model},
  author={Zhong, Zihan and Tang, Zhiqiang and He, Tong and Fang, Haoyang and Yuan, Chun},
  journal={arXiv preprint arXiv:2401.17868},
  year={2024}
}
```