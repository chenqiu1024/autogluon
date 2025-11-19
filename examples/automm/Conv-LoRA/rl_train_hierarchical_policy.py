"""
Hierarchical RL training for Conv-LoRA:

- High-level LayerPolicy decides which Conv-LoRA layers are active.
- Low-level RoutingPolicy routes tokens to experts (Scheme B).

Phases:
    - routing: only train RoutingPolicy (similar to rl_train_routing_policy.py).
    - layer  : freeze RoutingPolicy, train LayerPolicy (which layers use LoRA).
    - joint  : jointly finetune both LayerPolicy and RoutingPolicy.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.cuda.amp import autocast
import torchvision.transforms as transforms

# Add multimodal src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "multimodal" / "src"))

from autogluon.multimodal.models.adaptation_layers import ConvLoRALinear
from autogluon.multimodal.models.gating import NoisyTopKGate, RLGate
# from autogluon.multimodal.multimodal_predictor import MultiModalPredictor  # noqa: F401
from autogluon.multimodal.rl.algos.grpo import GRPOTrainer
from autogluon.multimodal.rl.policies.layer_policy import LayerPolicy
from autogluon.multimodal.rl.policies.routing_policy import RoutingPolicy
from autogluon.multimodal.rl.utils import (
    build_hierarchical_rollout,
    compute_shared_advantages,
    compute_expert_flops,
    compute_conv_flops,
    load_checkpoint,
    log_scalars,
    save_checkpoint,
)

from rl_utils import load_trained_conv_lora_model, prepare_dataset, compute_iou


def _collect_conv_lora_layers(sam_model) -> List[Tuple[str, ConvLoRALinear]]:
    """Collect all ConvLoRALinear layers from SAM model."""
    conv_lora_layers: List[Tuple[str, ConvLoRALinear]] = []
    for name, module in sam_model.model.named_modules():
        if isinstance(module, ConvLoRALinear):
            conv_lora_layers.append((name, module))
    return conv_lora_layers


def _extract_global_features(sam_model, batch_images: torch.Tensor) -> torch.Tensor:
    """
    Extract sample-level global features for LayerPolicy.

    For SAM-based models, we run a forward pass and use image embeddings.
    """
    batch_dict = {"sam_image": batch_images}
    
    # Force train mode to avoid requiring sam_label in forward
    was_training = sam_model.training
    sam_model.train()
    with torch.no_grad():
        output = sam_model(batch_dict)
    if not was_training:
        sam_model.eval()

    sam_outputs = output["sam"]
    if "image_embeds" in sam_outputs:
        feats = sam_outputs["image_embeds"]  # (B, C, H, W)
        feats = feats.mean(dim=[2, 3])  # (B, C)
    else:
        logits = sam_outputs["logits"]  # (B, 1, H, W)
        feats = logits.mean(dim=[1, 2, 3], keepdim=False)  # (B,)
        feats = feats.unsqueeze(-1)  # (B, 1)
    return feats


class HierarchicalRLTrainer:
    """
    Trainer coordinating LayerPolicy and RoutingPolicy for Conv-LoRA.
    """

    def __init__(
        self,
        sam_model,
        layer_policy: LayerPolicy,
        routing_policy: RoutingPolicy,
        grpo_layer: GRPOTrainer,
        grpo_routing: GRPOTrainer,
        device: str = "cuda",
        compute_budget: float = 1e10,
        imbalance_weight: float = 0.01,
        layer_penalty_coef: float = 0.01,
        use_lagrangian: bool = True,
        lagrangian_lr: float = 1e-3,
    ):
        self.sam_model = sam_model
        self.layer_policy = layer_policy
        self.routing_policy = routing_policy
        self.grpo_layer = grpo_layer
        self.grpo_routing = grpo_routing
        self.device = device
        self.compute_budget = compute_budget
        self.imbalance_weight = imbalance_weight
        self.layer_penalty_coef = layer_penalty_coef
        self.use_lagrangian = use_lagrangian
        self.lagrangian_lr = lagrangian_lr

        # Dual variable for compute budget (Lagrangian)
        self.dual_alpha = 0.0

    def train_step(
        self,
        batch_images: torch.Tensor,
        batch_labels: torch.Tensor | None,
        phase: str = "layer",
    ) -> dict:
        """
        One training step for hierarchical RL.

        Parameters
        ----------
        batch_images
            Image tensor of shape (B, 3, H, W).
        batch_labels
            Mask tensor of shape (B, 1, H, W) or (B, H, W), optional.
        phase
            - 'routing': only update routing policy (Scheme B).
            - 'layer'  : freeze routing, update layer policy.
            - 'joint'  : update both.
        """
        conv_lora_layers = _collect_conv_lora_layers(self.sam_model)
        num_layers = len(conv_lora_layers)

        # ---- 1) LayerPolicy: decide active layers ----
        global_feats = _extract_global_features(self.sam_model, batch_images).to(self.device)
        layer_logits = self.layer_policy(global_feats)  # (B, L)
        layer_probs = torch.sigmoid(layer_logits)       # (B, L)

        # Simple strategy: use mean prob over batch to form mask
        layer_mean_probs = layer_probs.mean(dim=0)      # (L,)
        layer_mask = (layer_mean_probs > 0.5).float()   # (L,)

        # Apply mask to ConvLoRALinear modules
        for layer_idx, (_, module) in enumerate(conv_lora_layers):
            active = layer_mask[layer_idx].item()
            module.set_lora_active(active)

        # ---- 2) Inject RLGate for active layers (reuse Scheme B logic) ----
        original_gates = {}
        rl_gate_outputs = {i: None for i in range(num_layers)}

        from autogluon.multimodal.models.adaptation_layers import MoEGate

        for layer_idx, (name, module) in enumerate(conv_lora_layers):
            original_gates[name] = module.lora_moe_gating

            if layer_mask[layer_idx] < 0.5:
                # Layer effectively deactivated for LoRA; keep existing gate.
                continue

            if isinstance(module.lora_moe_gating, MoEGate):
                reference_gate = NoisyTopKGate(module.lora_moe_gating)
            else:
                reference_gate = None

            rl_gate = RLGate(
                routing_policy=self.routing_policy,
                reference_gate=reference_gate,
                k=1,
                compute_kl=True,
            )
            rl_gate._layer_idx = layer_idx
            module.lora_moe_gating = rl_gate

            def make_capture_hook(l_idx):
                def hook_fn(gate_module, inputs, outputs):
                    if len(outputs) == 3:
                        rl_gate_outputs[l_idx] = {
                            "gates": outputs[0].detach().clone(),
                            "info_dict": {
                                k: v.detach().clone()
                                if isinstance(v, torch.Tensor)
                                else v
                                for k, v in outputs[2].items()
                            },
                        }

                return hook_fn

            rl_gate.register_forward_hook(make_capture_hook(layer_idx))

        # ---- 3) Forward SAM with RL gates (no grad on SAM) ----
        batch_dict = {"sam_image": batch_images}
        was_training = self.sam_model.training
        self.sam_model.train()
        with torch.no_grad():
            output = self.sam_model(batch_dict)
            pred_masks = output["sam"]["logits"]
        if not was_training:
            self.sam_model.eval()

        # Restore original gates
        for name, module in conv_lora_layers:
            module.lora_moe_gating = original_gates[name]

        # ---- 4) Recompute routing logprobs with grad ----
        collected_logprobs = []
        collected_gates = []
        collected_info_dicts = []

        for layer_idx in range(num_layers):
            if rl_gate_outputs[layer_idx] is not None:
                gate_output = rl_gate_outputs[layer_idx]
                info = gate_output["info_dict"]
                gates = gate_output["gates"]  # (B_tokens, M)
                actions = info.get("actions", gates.argmax(dim=1))

                if "_feats" in info:
                    feats = info["_feats"]  # features with grad
                    logits = self.routing_policy(feats, layer_idx)
                    log_probs = torch.log_softmax(logits, dim=1)
                    logprobs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
                else:
                    logprobs = info.get(
                        "logprobs",
                        torch.zeros(gates.size(0), device=self.device),
                    )
                    if not logprobs.requires_grad:
                        logprobs = logprobs.detach().requires_grad_(True)

                collected_logprobs.append(logprobs)
                collected_gates.append(gates)
                collected_info_dicts.append(info)

        # ---- 5) Compute metrics & reward ----
        if batch_labels is not None:
            iou = compute_iou(pred_masks, batch_labels).mean().item()
        else:
            iou = 0.0

        # FLOPs: reuse conv/expert FLOPs utilities for approximation
        if len(collected_gates) > 0:
            feature_shapes = [(14, 14)] * len(collected_gates)  # approximate
            # Example conv config; adapt to actual r & channels if needed
            expert_configs_per_layer = [
                [{"in_c": 3, "out_c": 3, "kernel_size": 3}] * collected_gates[0].size(1)
            ] * len(collected_gates)
            flops = 0.0
            for gates, feats_shape, expert_configs in zip(
                collected_gates, feature_shapes, expert_configs_per_layer
            ):
                flops += compute_expert_flops(gates, expert_configs, feats_shape)
        else:
            flops = self.compute_budget * 0.8

        # Imbalance across experts
        from autogluon.multimodal.rl.utils.rollout import compute_imbalance

        imbalance = compute_imbalance(collected_gates) if len(collected_gates) > 0 else 0.0

        num_active_layers = layer_mask.sum().item()

        # reward = IoU - α·(FLOPs / budget) - β·imbalance - γ·(#active_layers / L)
        compute_term = self.dual_alpha * (flops / self.compute_budget)
        layer_penalty = self.layer_penalty_coef * (
            num_active_layers / max(1, num_layers)
        )
        reward = iou - compute_term - self.imbalance_weight * imbalance - layer_penalty

        routing_rollout = {
            "logprobs": collected_logprobs,
            "info_dicts": collected_info_dicts,
            "gates": collected_gates,
        }

        # ---- 6) Build hierarchical rollout & advantages ----
        # Here we approximate Bernoulli logprob via log_sigmoid for active layers.
        layer_logprobs = torch.nn.functional.logsigmoid(layer_logits)  # (B, L)
        layer_logprobs_flat = layer_logprobs.view(-1)

        hier_rollout = build_hierarchical_rollout(
            layer_logprobs_flat,
            layer_logits,
            routing_rollout,
        )

        adv_layer, adv_routing = compute_shared_advantages(
            reward,
            routing_rollout,
            layer_rollout_size=layer_logprobs_flat.numel(),
            device=self.device,
            normalize=False,
        )

        # ---- 7) GRPO updates according to phase ----
        metrics: dict = {
            "iou": iou,
            "flops": flops,
            "imbalance": imbalance,
            "reward": reward,
            "active_layers": num_active_layers,
        }

        if phase in ("layer", "joint"):
            layer_metrics = self.grpo_layer.update(hier_rollout["layer"], adv_layer)
            metrics.update({f"layer_{k}": v for k, v in layer_metrics.items()})

        if phase in ("routing", "joint") and len(collected_logprobs) > 0:
            routing_metrics = self.grpo_routing.update(
                hier_rollout["routing"],
                adv_routing,
            )
            metrics.update({f"routing_{k}": v for k, v in routing_metrics.items()})

        # Lagrangian update
        if self.use_lagrangian:
            flops_violation = flops - self.compute_budget
            self.dual_alpha += self.lagrangian_lr * flops_violation
            self.dual_alpha = max(0.0, self.dual_alpha)
        metrics["dual_alpha"] = self.dual_alpha

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Hierarchical RL for Conv-LoRA")
    parser.add_argument("--task", type=str, default="isic2017")
    parser.add_argument("--phase", type=str, default="layer", choices=["routing", "layer", "joint"])
    parser.add_argument("--model_path", type=str, required=True, help="Trained Conv-LoRA checkpoint")
    parser.add_argument("--routing_ckpt", type=str, default=None, help="Routing policy checkpoint from Scheme B")
    parser.add_argument("--layer_ckpt", type=str, default=None, help="Layer policy checkpoint from Phase 2")
    parser.add_argument("--output_dir", type=str, default="rl_hierarchical")
    parser.add_argument("--device", type=str, default="cuda")

    # RL / compute config
    parser.add_argument("--compute_budget", type=float, default=1e10)
    parser.add_argument("--imbalance_weight", type=float, default=0.01)
    parser.add_argument("--layer_penalty_coef", type=float, default=0.01)
    parser.add_argument("--lagrangian_lr", type=float, default=1e-3)

    # Training config
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr_layer", type=float, default=1e-4)
    parser.add_argument("--lr_routing", type=float, default=1e-4)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--adapter_l2_coef", type=float, default=0.001)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # TensorBoard writer (optional)
    from autogluon.multimodal.rl.utils.visualization import create_tensorboard_writer

    writer = create_tensorboard_writer(os.path.join(args.output_dir, "logs"))

    # Load Conv-LoRA model
    predictor, sam_model, conv_lora_layers = load_trained_conv_lora_model(
        args.task,
        args.model_path,
        device=args.device,
    )
    num_layers = len(conv_lora_layers)
    print(f"Loaded Conv-LoRA model with {num_layers} ConvLoRALinear layers")

    # RoutingPolicy: reuse Scheme B config (rank = Conv-LoRA r)
    # NOTE: adjust feature_dim / num_experts to match your Conv-LoRA config.
    num_experts = conv_lora_layers[0][1].num_experts if num_layers > 0 else 8
    rank = conv_lora_layers[0][1].r if num_layers > 0 else 3

    routing_policy = RoutingPolicy(
        num_experts=num_experts,
        feature_dim=rank,
        hidden_dim=256,
        layer_embed_dim=32,
        num_layers=num_layers,
    ).to(args.device)

    if args.routing_ckpt:
        print(f"Loading routing policy from {args.routing_ckpt}")
        ckpt = load_checkpoint(args.routing_ckpt, device=args.device)
        routing_policy.load_state_dict(ckpt["policy"])

    # LayerPolicy: feature_dim from global features. We infer from example run.
    # If SAM exposes image_embeds of shape (B, C, H, W), C is used.
    # Here we do a dummy forward to infer C.
    dummy_img = torch.zeros(1, 3, 1024, 1024, device=args.device)
    dummy_feats = _extract_global_features(sam_model, dummy_img).to(args.device)
    feature_dim = dummy_feats.size(1)

    layer_policy = LayerPolicy(
        num_layers=num_layers,
        feature_dim=feature_dim,
        hidden_dim=256,
        dropout=0.1,
        use_patterns=False,
    ).to(args.device)

    # Load layer policy checkpoint if provided (for Phase 3 joint training)
    if args.layer_ckpt:
        print(f"Loading layer policy from {args.layer_ckpt}")
        ckpt = load_checkpoint(args.layer_ckpt, device=args.device)
        layer_policy.load_state_dict(ckpt["layer_policy"])

    # Optimizers and GRPO trainers
    opt_layer = torch.optim.AdamW(layer_policy.parameters(), lr=args.lr_layer)
    opt_routing = torch.optim.AdamW(routing_policy.parameters(), lr=args.lr_routing)

    grpo_layer = GRPOTrainer(
        policy=layer_policy,
        optimizer=opt_layer,
        entropy_coef=args.entropy_coef,
        kl_coef=0.0,
        adapter_l2_coef=args.adapter_l2_coef,
        device=args.device,
    )
    grpo_routing = GRPOTrainer(
        policy=routing_policy,
        optimizer=opt_routing,
        entropy_coef=args.entropy_coef,
        kl_coef=args.kl_coef,
        adapter_l2_coef=args.adapter_l2_coef,
        device=args.device,
    )

    # Freeze as needed
    if args.phase == "routing":
        for p in layer_policy.parameters():
            p.requires_grad = False
    elif args.phase == "layer":
        for p in routing_policy.parameters():
            p.requires_grad = False

    trainer = HierarchicalRLTrainer(
        sam_model=sam_model,
        layer_policy=layer_policy,
        routing_policy=routing_policy,
        grpo_layer=grpo_layer,
        grpo_routing=grpo_routing,
        device=args.device,
        compute_budget=args.compute_budget,
        imbalance_weight=args.imbalance_weight,
        layer_penalty_coef=args.layer_penalty_coef,
        use_lagrangian=True,
        lagrangian_lr=args.lagrangian_lr,
    )

    # Dataset & transforms (reuse Conv-LoRA settings)
    train_df, dataset_dir = prepare_dataset(args.task, split="train")
    print(f"Training samples: {len(train_df)}")

    transform = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    def load_batch(indices):
        images = []
        masks = []
        for idx in indices:
            row = train_df.iloc[idx]
            # prepare_dataset already expanded paths, use them directly
            img = Image.open(row["image"]).convert("RGB")
            msk = Image.open(row["label"]).convert("L")
            img_tensor = transform(img)
            mask_tensor = transforms.Resize((1024, 1024))(transforms.ToTensor()(msk))
            images.append(img_tensor)
            masks.append(mask_tensor)
        return torch.stack(images, dim=0), torch.stack(masks, dim=0)

    all_indices = np.arange(len(train_df))

    from tqdm import tqdm

    global_step = 0
    for step in tqdm(range(args.max_steps)):
        batch_indices = np.random.choice(
            all_indices, size=args.batch_size, replace=True
        )
        batch_images, batch_labels = load_batch(batch_indices)
        batch_images = batch_images.to(args.device)
        batch_labels = batch_labels.to(args.device)

        with autocast():
            metrics = trainer.train_step(batch_images, batch_labels, phase=args.phase)

        if writer and step % 10 == 0:
            log_scalars(
                writer,
                {
                    "train/reward": metrics["reward"],
                    "train/iou": metrics["iou"],
                    "train/flops": metrics["flops"],
                    "train/imbalance": metrics["imbalance"],
                    "train/dual_alpha": metrics["dual_alpha"],
                    "train/active_layers": metrics["active_layers"],
                },
                global_step,
            )

        if (step + 1) % 50 == 0:
            print(
                f"[{step+1}/{args.max_steps}] "
                f"phase={args.phase} "
                f"reward={metrics['reward']:.4f} "
                f"iou={metrics['iou']:.4f} "
                f"flops={metrics['flops']:.2e} "
                f"active_layers={metrics['active_layers']:.1f}"
            )

        if (step + 1) % 500 == 0:
            ckpt_path = os.path.join(ckpt_dir, f"step_{step+1}.pt")
            save_checkpoint(
                {
                    "layer_policy": layer_policy.state_dict(),
                    "routing_policy": routing_policy.state_dict(),
                    "dual_alpha": trainer.dual_alpha,
                    "step": step + 1,
                    "config": vars(args),
                },
                ckpt_path,
            )

        global_step += 1

    final_ckpt = os.path.join(ckpt_dir, "final.pt")
    save_checkpoint(
        {
            "layer_policy": layer_policy.state_dict(),
            "routing_policy": routing_policy.state_dict(),
            "dual_alpha": trainer.dual_alpha,
            "step": global_step,
            "config": vars(args),
        },
        final_ckpt,
    )
    print(f"Training complete. Final checkpoint: {final_ckpt}")

    if writer:
        writer.close()


if __name__ == "__main__":
    main()


