# ACDC 半监督 Conv-LoRA 训练错误
_Exported on 1/7/2026 at 22:29:37 PST from Cursor (2.3.29)_

---

**User**

@autogluon/docs/cursor_semisup_gpt_abd-260105.md 从该文件恢复会话，然后继续回答我的问题：我用如下命令`cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA

nohup python3 run_acdc_conv_lora_semi.py \
  --dataset_dir datasets/acdc_conv_lora/acdc_conv_lora \
  --rank 3 \
  --expert_num 8 \
  --gspo_enable \
  --adapter_enable \
  --adapter_dim 64 \
  --gspo_adapter_enable \
  --gspo_allow_semisup \
  --loss mask2former_loss \
  --num_mask_tokens 10 \
  --semi_labeled_fraction 0.1 \
  --semi_labeled_seed 123 \
  --weak_box_jitter_mode box \
  --weak_box_jitter_amount 0.1 \
  --weak_box_outward_only \
  --weak_loss_outside_weight 1.0 \
  --weak_loss_entropy_weight 0.05 \
  --weak_loss_tv_weight 0.0 \
  --max_epochs 100 \
  --patience 30 \
  --vis_samples 8 \
  --vis_output_dir outputs/acdc_vis_mask2former-semi-260107 \
  --output_dir outputs/acdc_mask2former-semi-260107 \
> outputs/acdc_mask2former-semi-260107.log 2>&1 &`尝试用我们的半监督方法在ACDC数据集上跑训练，但是出错，控制台输出日志在 @autogluon/examples/automm/Conv-LoRA/outputs/acdc_mask2former-semi-260107.log 

---

**Cursor**

```python
import logging
import hashlib
from typing import Callable, Dict, Optional

import torch
import torchmetrics
import torch.nn.functional as F
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
    
    def __init__(self, *args, gspo_trainer=None, train_box_prompt_cfg=None, train_bbox_predictor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gspo_trainer = gspo_trainer
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
    
    @staticmethod
    def _stable_uniform_0_1(key: str, seed: int = 0) -> float:
        """
        Stable deterministic pseudo-random number in [0, 1) from a string key + seed.
        Avoids python's randomized hash across processes.
        """
        h = hashlib.md5(f"{seed}::{key}".encode("utf-8")).digest()
        # use first 8 bytes as uint64
        x = int.from_bytes(h[:8], byteorder="little", signed=False)
        return (x % (10**12)) / float(10**12)
    
    def _get_labeled_mask_from_paths(self, image_paths, labeled_fraction: float, seed: int) -> torch.Tensor:
        """
        image_paths: list[str] (len=B)
        returns: bool tensor (B,)
        """
        labeled_fraction = float(labeled_fraction)
        labeled_fraction = max(0.0, min(1.0, labeled_fraction))
        if labeled_fraction >= 1.0:
            return torch.ones(len(image_paths), dtype=torch.bool, device=self.device)
        if labeled_fraction <= 0.0:
            return torch.zeros(len(image_paths), dtype=torch.bool, device=self.device)
        flags = []
        for p in image_paths:
            p = p or ""
            u = self._stable_uniform_0_1(p, seed=seed)
            flags.append(u < labeled_fraction)
        return torch.tensor(flags, dtype=torch.bool, device=self.device)
    
    @staticmethod
    def _compute_boxes_from_mask(mask: torch.Tensor) -> torch.Tensor:
        """mask: (B,H,W) or (B,1,H,W) -> boxes (B,4) in pixel coords (x1,y1,x2,y2)."""
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
    
    @staticmethod
    def _jitter_boxes_outward(
        boxes: torch.Tensor,
        h: int,
        w: int,
        amount: float,
        mode: str = "box",  # box / image / pixel
        outward_only: bool = True,
    ) -> torch.Tensor:
        """
        Jitter boxes to simulate coarse human boxes.
        - outward_only=True: never shrink vs original (always expand or keep).
        - amount meaning:
          - mode=box: fraction of box width/height
          - mode=image: fraction of image width/height
          - mode=pixel: absolute pixels
        """
        jittered = boxes.clone()
        mode = (mode or "box").lower()
        amount = float(amount)
        amount = max(0.0, amount)
        for i in range(jittered.shape[0]):
            x1, y1, x2, y2 = jittered[i]
            if x2 <= x1 or y2 <= y1:
                continue
            bw = (x2 - x1).clamp(min=1.0)
            bh = (y2 - y1).clamp(min=1.0)
            if mode == "box":
                dx = bw * amount
                dy = bh * amount
            elif mode == "image":
                dx = float(w) * amount
                dy = float(h) * amount
                dx = torch.tensor(dx, device=boxes.device, dtype=torch.float32)
                dy = torch.tensor(dy, device=boxes.device, dtype=torch.float32)
            elif mode == "pixel":
                dx = torch.tensor(amount, device=boxes.device, dtype=torch.float32)
                dy = torch.tensor(amount, device=boxes.device, dtype=torch.float32)
            else:
                # fallback to box
                dx = bw * amount
                dy = bh * amount
            
            # sample non-negative deltas (mostly outward expansion)
            # use uniform [0, dx] / [0, dy]
            ex1 = torch.rand(1, device=boxes.device) * dx
            ey1 = torch.rand(1, device=boxes.device) * dy
            ex2 = torch.rand(1, device=boxes.device) * dx
            ey2 = torch.rand(1, device=boxes.device) * dy
            
            if outward_only:
                # outward: x1 decreases, y1 decreases, x2 increases, y2 increases
                x1n = x1 - ex1
                y1n = y1 - ey1
                x2n = x2 + ex2
                y2n = y2 + ey2
            else:
                # allow inward with small probability by random sign
                s1 = torch.where(torch.rand(1, device=boxes.device) < 0.9, -1.0, 1.0)  # x1: outward is -1
                t1 = torch.where(torch.rand(1, device=boxes.device) < 0.9, -1.0, 1.0)  # y1
                s2 = torch.where(torch.rand(1, device=boxes.device) < 0.9, 1.0, -1.0)   # x2: outward is +1
                t2 = torch.where(torch.rand(1, device=boxes.device) < 0.9, 1.0, -1.0)   # y2
                x1n = x1 + s1 * ex1
                y1n = y1 + t1 * ey1
                x2n = x2 + s2 * ex2
                y2n = y2 + t2 * ey2
            
            x1n = torch.clamp(x1n, 0, w - 1)
            y1n = torch.clamp(y1n, 0, h - 1)
            x2n = torch.clamp(x2n, 0, w - 1)
            y2n = torch.clamp(y2n, 0, h - 1)
            # ensure valid ordering
            jittered[i] = torch.stack(
                [torch.min(x1n, x2n), torch.min(y1n, y2n), torch.max(x1n, x2n), torch.max(y1n, y2n)]
            ).squeeze()
        return jittered
    
    @staticmethod
    def _structure_loss_per_sample(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Per-sample StructureLoss (same as optim/losses/structure_loss.py) without final mean.
        Returns tensor of shape (B,).
        """
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
        wbce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)).clamp(min=1e-6)
        probs = torch.sigmoid(logits)
        inter = ((probs * target) * weit).sum(dim=(2, 3))
        union = ((probs + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).view(-1)
    
    @staticmethod
    def _weak_box_losses(
        logits: torch.Tensor,  # (B,1,H,W) or (B,H,W)
        boxes: torch.Tensor,   # (B,4) x1,y1,x2,y2 in pixel coords
        outside_weight: float = 1.0,
        entropy_weight: float = 0.05,
        tv_weight: float = 0.0,
    ) -> torch.Tensor:
        """
        Box-only weak supervision loss:
        - outside-box should be background (0)
        - inside-box entropy minimization (push confident predictions without assuming full positive)
        - optional smoothness (TV) inside box
        Returns scalar loss.
        """
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        b, _, h, w = logits.shape
        device = logits.device
        outside_weight = float(outside_weight)
        entropy_weight = float(entropy_weight)
        tv_weight = float(tv_weight)
        
        total = logits.new_tensor(0.0)
        count = 0
        for i in range(b):
            x1, y1, x2, y2 = boxes[i]
            if x2 <= x1 or y2 <= y1:
                continue
            # integer pixel bounds (inclusive)
            x1i = int(torch.clamp(x1.round(), 0, w - 1).item())
            y1i = int(torch.clamp(y1.round(), 0, h - 1).item())
            x2i = int(torch.clamp(x2.round(), 0, w - 1).item())
            y2i = int(torch.clamp(y2.round(), 0, h - 1).item())
            if x2i <= x1i or y2i <= y1i:
                continue
            
            logit = logits[i : i + 1]  # (1,1,H,W)
            outside_mask = torch.ones((1, 1, h, w), device=device, dtype=logit.dtype)
            outside_mask[:, :, y1i : y2i + 1, x1i : x2i + 1] = 0.0
            inside_mask = 1.0 - outside_mask
            
            # Outside-box background constraint
            if outside_weight > 0:
                zeros = torch.zeros_like(logit)
                bce = F.binary_cross_entropy_with_logits(logit, zeros, reduction="none")
                outside_loss = (bce * outside_mask).sum() / outside_mask.sum().clamp(min=1.0)
            else:
                outside_loss = logit.new_tensor(0.0)
            
            # Inside-box entropy minimization
            if entropy_weight > 0:
                p = torch.sigmoid(logit).clamp(min=1e-6, max=1 - 1e-6)
                ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
                ent_loss = (ent * inside_mask).sum() / inside_mask.sum().clamp(min=1.0)
            else:
                ent_loss = logit.new_tensor(0.0)
            
            # Total variation smoothness inside the box
            if tv_weight > 0:
                p = torch.sigmoid(logit)
                dy = torch.abs(p[:, :, 1:, :] - p[:, :, :-1, :])
                dx = torch.abs(p[:, :, :, 1:] - p[:, :, :, :-1])
                # match shapes with masks
                inside_y = inside_mask[:, :, 1:, :]
                inside_x = inside_mask[:, :, :, 1:]
                tv = (dy * inside_y).sum() / inside_y.sum().clamp(min=1.0) + (dx * inside_x).sum() / inside_x.sum().clamp(min=1.0)
            else:
                tv = logit.new_tensor(0.0)
            
            total = total + outside_weight * outside_loss + entropy_weight * ent_loss + tv_weight * tv
            count += 1
        
        if count == 0:
            return logits.new_tensor(0.0)
        return total / float(count)

    def _predict_boxes_from_images(self, image_paths: list, target_h: int, target_w: int) -> torch.Tensor:
        """
        Wrapper around train_bbox_predictor to predict boxes in pixel coords and scale to target size.
        """
        import numpy as np
        from PIL import Image
        if self.train_bbox_predictor is None:
            raise ValueError("train_bbox_predictor is required for mode='predict'")
        b = len(image_paths)
        boxes = torch.zeros((b, 4), device=self.device, dtype=torch.float32)
        for i, img_path in enumerate(image_paths):
            if not img_path:
                continue
            try:
                bbox_px = self.train_bbox_predictor.predict(img_path)
                if bbox_px is None:
                    continue
                with Image.open(img_path) as img:
                    orig_w, orig_h = img.size
                scale_x = target_w / orig_w
                scale_y = target_h / orig_h
                x1 = bbox_px[0] * scale_x
                y1 = bbox_px[1] * scale_y
                x2 = bbox_px[2] * scale_x
                y2 = bbox_px[3] * scale_y
                x1 = max(0, min(x1, target_w - 1))
                y1 = max(0, min(y1, target_h - 1))
                x2 = max(0, min(x2, target_w - 1))
                y2 = max(0, min(y2, target_h - 1))
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                boxes[i] = torch.tensor([x1, y1, x2, y2], device=self.device, dtype=torch.float32)
            except Exception as e:
                logger.warning(f"Failed to predict bbox for {img_path}: {e}")
                continue
        return boxes

    def _apply_train_box_prompts(self, batch: Dict, label: torch.Tensor) -> None:
        """
        In-place box prompt injection for training (gt/noisy/mix/predict) and for semi/weak supervision.
        This is factored out so GSPO/standard paths can share the same augmentation.
        """
        cfg = self.train_box_prompt_cfg or {}
        mode = cfg.get("mode", "off")
        if mode == "off" and "semi_labeled_fraction" not in cfg:
            return

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
        else:
            noise_frac = float(cfg.get("noise_frac", 0.12))

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

        if box_mode in ["gt", "noisy"]:
            gt_boxes = self._compute_boxes_from_mask(label)
            h, w = label.shape[-2], label.shape[-1]
            if box_mode == "noisy":
                gt_boxes = _jitter_boxes(gt_boxes, h=h, w=w, frac=noise_frac)
            if not (gt_boxes.sum(dim=1) == 0).all():
                batch[self.model.box_key] = gt_boxes.unsqueeze(1)  # (B,1,4)
        elif box_mode == "predict":
            image_path_key = self.model.image_path_key
            image_paths = batch.get(image_path_key, [])
            if image_paths:
                h, w = label.shape[-2], label.shape[-1]
                pred_boxes = self._predict_boxes_from_images(
                    image_paths=image_paths, target_h=h, target_w=w
                )
                if not (pred_boxes.sum(dim=1) == 0).all():
                    batch[self.model.box_key] = pred_boxes.unsqueeze(1)  # (B,1,4)

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
                logits = per_output[LOGITS]
                
                # Handle SAM multi-class output format: combine mask and class predictions
                # SAM multi-class outputs: LOGITS [B, num_mask_tokens, H, W] + CLASS_LOGITS [B, num_mask_tokens, num_classes+1]
                # When num_mask_tokens == num_classes, we can use LOGITS directly as per-class masks
                # Otherwise, we need to combine mask and class predictions
                if CLASS_LOGITS in per_output:
                    mask_logits = logits  # [B, num_mask_tokens, H, W]
                    class_logits = per_output[CLASS_LOGITS]  # [B, num_mask_tokens, num_classes+1]
                    
                    # Apply semantic inference to combine mask and class predictions
                    # This follows Mask2Former's approach: semantic_mask = einsum(softmax(class), sigmoid(mask))
                    mask_cls = F.softmax(class_logits, dim=-1)[..., :-1]  # [B, num_mask_tokens, num_classes]
                    mask_pred = torch.sigmoid(mask_logits)  # [B, num_mask_tokens, H, W]
                    
                    # Combine: for each pixel, sum over mask tokens weighted by class probability
                    # semantic_mask[b, c, h, w] = sum_q(mask_cls[b, q, c] * mask_pred[b, q, h, w])
                    logits = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
                    
                    # Convert probabilities back to logits for CrossEntropyLoss (add small epsilon to avoid log(0))
                    logits = torch.log(logits.clamp(min=1e-7))
                
                loss += (
                    self.loss_func(
                        input=logits,
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
        
        # ---- Semi/weak supervision config (optional) ----
        cfg = self.train_box_prompt_cfg or {}
        semi_labeled_fraction = float(cfg.get("semi_labeled_fraction", 1.0))
        semi_labeled_seed = int(cfg.get("semi_labeled_seed", 0))
        weak_box_jitter_mode = str(cfg.get("weak_box_jitter_mode", "box"))
        weak_box_jitter_amount = float(cfg.get("weak_box_jitter_amount", cfg.get("noise_frac", 0.12)))
        weak_box_outward_only = bool(cfg.get("weak_box_outward_only", True))
        weak_outside_w = float(cfg.get("weak_loss_outside_weight", 1.0))
        weak_entropy_w = float(cfg.get("weak_loss_entropy_weight", 0.05))
        weak_tv_w = float(cfg.get("weak_loss_tv_weight", 0.0))
        
        # Determine which samples are "labeled" (for simulation) using stable hashing on image paths.
        image_paths = batch.get(self.model.image_path_key, [])
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        labeled_mask = None
        if self.training and semi_labeled_fraction < 1.0:
            labeled_mask = self._get_labeled_mask_from_paths(
                image_paths=image_paths, labeled_fraction=semi_labeled_fraction, seed=semi_labeled_seed
            )

        # -------- Train-time box prompt injection (configurable) --------
        # 控制项：self.train_box_prompt_cfg = {
        #   mode: off / gt / noisy / mix / predict,
        #   p_no, p_gt, p_noisy: 仅在 mode=mix 时生效并会归一化,
        #   noise_frac: 噪声扰动比例（相对 bbox 宽高）
        # }
        # mode=predict: 使用 self.train_bbox_predictor 预测框作为 box prompt
        if self.training:
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

            # If semi/weak enabled: for "unlabeled" samples, force weak coarse boxes generated from GT bbox with outward-biased jitter.
            if labeled_mask is not None and hasattr(self.model, "box_key"):
                gt_boxes_all = self._compute_boxes_from_mask(label)
                h, w = label.shape[-2], label.shape[-1]
                weak_boxes = self._jitter_boxes_outward(
                    gt_boxes_all,
                    h=h,
                    w=w,
                    amount=weak_box_jitter_amount,
                    mode=weak_box_jitter_mode,
                    outward_only=weak_box_outward_only,
                )
                # apply only to unlabeled samples
                if (~labeled_mask).any():
                    boxes_for_batch = gt_boxes_all.clone()
                    boxes_for_batch[~labeled_mask] = weak_boxes[~labeled_mask]
                    if not (boxes_for_batch.sum(dim=1) == 0).all():
                        batch[self.model.box_key] = boxes_for_batch.unsqueeze(1)  # (B,1,4)

            if box_mode in ["gt", "noisy"]:
                gt_boxes = self._compute_boxes_from_mask(label)
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
        
        # ---- Loss: fully-supervised OR semi/weak mixed ----
        if self.training and labeled_mask is not None:
            # Semi/weak supervision path
            if isinstance(self.loss_func, Mask2FormerLoss):
                # 1) 有标签子集：使用 Mask2FormerLoss
                sup_loss = output[self.model.prefix][LOGITS].new_tensor(0.0)
                if labeled_mask.any():
                    # 子集化输出与标签
                    output_labeled = {}
                    for key, per_output in output.items():
                        sub_out = {}
                        for sub_k, val in per_output.items():
                            if torch.is_tensor(val) and val.shape[0] == label.shape[0]:
                                sub_out[sub_k] = val[labeled_mask]
                            else:
                                sub_out[sub_k] = val
                        output_labeled[key] = sub_out
                    mask_labels = batch.get(self.model.mask_label_key, [])
                    class_labels = batch.get(self.model.class_label_key, [])
                    if isinstance(mask_labels, torch.Tensor):
                        mask_labels_l = mask_labels[labeled_mask]
                    else:
                        mask_labels_l = [m for m, keep in zip(mask_labels, labeled_mask) if keep]
                    if isinstance(class_labels, torch.Tensor):
                        class_labels_l = class_labels[labeled_mask]
                    else:
                        class_labels_l = [c for c, keep in zip(class_labels, labeled_mask) if keep]
                    label_l = label[labeled_mask]
                    sup_loss = self._compute_loss(
                        output=output_labeled,
                        label=label_l,
                        mask_labels=mask_labels_l,
                        class_labels=class_labels_l,
                    )
                
                # 2) 无标签子集：对聚合前景概率做盒弱监督
                weak_loss = output[self.model.prefix][LOGITS].new_tensor(0.0)
                if hasattr(self.model, "box_key") and self.model.box_key in batch and (~labeled_mask).any():
                    per_output = output[self.model.prefix]
                    mask_logits_u = per_output[LOGITS][~labeled_mask]
                    boxes_u = batch[self.model.box_key][:, 0, :][~labeled_mask]
                    if CLASS_LOGITS in per_output:
                        class_logits_u = per_output[CLASS_LOGITS][~labeled_mask]
                        # 语义概率：softmax(class) * sigmoid(mask) -> foreground 概率
                        mask_cls = F.softmax(class_logits_u, dim=-1)[..., :-1]
                        mask_prob = torch.sigmoid(mask_logits_u)
                        semantic_prob = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_prob)
                        fg_prob = semantic_prob.sum(dim=1, keepdim=True).clamp(min=1e-7, max=1 - 1e-7)
                        fg_logits = torch.logit(fg_prob)
                    else:
                        logits_u = mask_logits_u
                        if logits_u.dim() == 4 and logits_u.shape[1] > 1:
                            fg_prob = torch.sigmoid(logits_u).mean(dim=1, keepdim=True).clamp(min=1e-7, max=1 - 1e-7)
                            fg_logits = torch.logit(fg_prob)
                        else:
                            fg_logits = logits_u
                    weak_loss = self._weak_box_losses(
                        logits=fg_logits,
                        boxes=boxes_u,
                        outside_weight=weak_outside_w,
                        entropy_weight=weak_entropy_w,
                        tv_weight=weak_tv_w,
                    )
                
                moe_loss = output[self.model.prefix].get(MOE_LOSS, 0.0)
                loss = sup_loss + weak_loss + moe_loss
                self.log("train_sup_loss", sup_loss, on_step=True, on_epoch=True)
                self.log("train_weak_loss", weak_loss, on_step=True, on_epoch=True)
                self.log("train_labeled_frac", labeled_mask.float().mean(), on_step=True, on_epoch=True)
            else:
                # 二值结构损失路径
                logits = output[self.model.prefix][LOGITS]
                if logits.dim() == 3:
                    logits = logits.unsqueeze(1)
                if label.dim() == 3:
                    label_ = label.unsqueeze(1)
                else:
                    label_ = label
                
                per_sample_sup = self._structure_loss_per_sample(logits, label_)
                sup_count = labeled_mask.sum().clamp(min=1)
                sup_loss = (per_sample_sup * labeled_mask.float()).sum() / sup_count.float()
                
                weak_loss = logits.new_tensor(0.0)
                if hasattr(self.model, "box_key") and self.model.box_key in batch and (~labeled_mask).any():
                    boxes_b1 = batch[self.model.box_key]  # (B,1,4)
                    boxes = boxes_b1[:, 0, :]
                    logits_u = logits[~labeled_mask]
                    boxes_u = boxes[~labeled_mask]
                    weak_loss = self._weak_box_losses(
                        logits=logits_u,
                        boxes=boxes_u,
                        outside_weight=weak_outside_w,
                        entropy_weight=weak_entropy_w,
                        tv_weight=weak_tv_w,
                    )
                
                moe_loss = output[self.model.prefix].get(MOE_LOSS, 0.0)
                loss = sup_loss + weak_loss + moe_loss
                self.log("train_sup_loss", sup_loss, on_step=True, on_epoch=True)
                self.log("train_weak_loss", weak_loss, on_step=True, on_epoch=True)
                self.log("train_labeled_frac", labeled_mask.float().mean(), on_step=True, on_epoch=True)
        else:
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
        cfg = getattr(self, "train_box_prompt_cfg", {}) or {}
        semi_labeled_fraction = float(cfg.get("semi_labeled_fraction", 1.0))
        use_gspo = (
            self.gspo_trainer is not None and 
            self.gspo_trainer.is_gspo_active(self.current_epoch)
        )
        allow_semisup_gspo = bool(getattr(self.gspo_trainer, "allow_semisup", False)) if self.gspo_trainer else False
        
        if use_gspo and allow_semisup_gspo and semi_labeled_fraction < 1.0:
            # Semi-supervised with GSPO on labeled subset, weak loss on unlabeled subset.
            label = batch[self.model.label_key]
            # Apply box prompts (gt/noisy/predict + weak jitter boxes already handled inside)
            if self.training:
                self._apply_train_box_prompts(batch, label)
            image_paths = batch.get(self.model.image_path_key, [])
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            labeled_mask = self._get_labeled_mask_from_paths(
                image_paths=image_paths, labeled_fraction=semi_labeled_fraction, seed=int(cfg.get("semi_labeled_seed", 0))
            )
            has_labeled = bool(labeled_mask.any())
            has_unlabeled = bool((~labeled_mask).any())
            loss = torch.tensor(0.0, device=self.device)
            selected_experts = None

            def _subset_batch(b, mask_bool: torch.Tensor):
                new = {}
                for k, v in b.items():
                    if torch.is_tensor(v) and v.shape[0] == mask_bool.shape[0]:
                        new[k] = v[mask_bool]
                    elif isinstance(v, list) and len(v) == mask_bool.shape[0]:
                        new[k] = [v[i] for i in range(len(v)) if mask_bool[i].item()]
                    else:
                        new[k] = v
                return new

            # Weak loss on unlabeled subset
            if has_unlabeled and hasattr(self.model, "box_key") and self.model.box_key in batch:
                batch_u = _subset_batch(batch, ~labeled_mask)
                label_u = batch_u[self.model.label_key]
                output_u = run_model(self.model, batch_u)
                logits_u = output_u[self.model.prefix][LOGITS]
                if logits_u.dim() == 3:
                    logits_u = logits_u.unsqueeze(1)
                boxes_u = batch_u[self.model.box_key][:, 0, :]
                weak_loss = self._weak_box_losses(
                    logits=logits_u,
                    boxes=boxes_u,
                    outside_weight=float(cfg.get("weak_loss_outside_weight", 1.0)),
                    entropy_weight=float(cfg.get("weak_loss_entropy_weight", 0.05)),
                    tv_weight=float(cfg.get("weak_loss_tv_weight", 0.0)),
                )
                moe_loss_u = output_u[self.model.prefix].get(MOE_LOSS, 0.0)
                loss = loss + weak_loss + moe_loss_u
                self.log("train_weak_loss", weak_loss, on_step=True, on_epoch=True)

            # GSPO on labeled subset
            if has_labeled:
                batch_l = _subset_batch(batch, labeled_mask)
                gspo_loss, metrics, selected_experts = self._gspo_training_step(batch_l)
                loss = loss + gspo_loss
                for key, value in metrics.items():
                    self.log(f"train_{key}", value, on_step=True, on_epoch=True)

            self.log("train_labeled_frac", labeled_mask.float().mean(), on_step=True, on_epoch=True)

        elif use_gspo:
            # GSPO-enhanced training (full GT)
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
        
        Extended to support:
        - Encoder Adapter quality feedback when gspo_adapter_enabled=True
        - Decoder LoRA on Attention quality feedback when gspo_lora_attention_enabled=True
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
        
        # Update expert quality feedback (Conv-LoRA MoE)
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
            
            # Update Adapter quality feedback (GSPO-Adapter extension)
            if hasattr(self.gspo_trainer, 'gspo_adapter_enabled') and self.gspo_trainer.gspo_adapter_enabled:
                self.gspo_trainer.update_adapter_feedback(quality_scores, self.model)
            
            # Update LoRA on Attention quality feedback (GSPO-LoRA extension)
            if hasattr(self.gspo_trainer, 'gspo_lora_attention_enabled') and self.gspo_trainer.gspo_lora_attention_enabled:
                self.gspo_trainer.update_lora_attention_feedback(quality_scores, self.model)
        
        return loss, metrics, selected_experts_groups

```

变更说明：为 Mask2Former 半监督路径补上实现，避免之前 `semi_labeled_fraction < 1.0` 直接抛出 NotImplementedError。现在有标签子集用 Mask2FormerLoss 计算监督，无标签子集从多 token + class logits 聚合前景概率后施加盒弱监督，并保持 MoE/日志记录。

代码要点：
```617:685:autogluon/multimodal/src/autogluon/multimodal/optim/lit_semantic_seg.py
        # ---- Loss: fully-supervised OR semi/weak mixed ----
        if self.training and labeled_mask is not None:
            # Semi/weak supervision path
            if isinstance(self.loss_func, Mask2FormerLoss):
                # 1) 有标签子集：使用 Mask2FormerLoss
                sup_loss = output[self.model.prefix][LOGITS].new_tensor(0.0)
                if labeled_mask.any():
                    ...
                    sup_loss = self._compute_loss(
                        output=output_labeled,
                        label=label_l,
                        mask_labels=mask_labels_l,
                        class_labels=class_labels_l,
                    )
                
                # 2) 无标签子集：对聚合前景概率做盒弱监督
                weak_loss = output[self.model.prefix][LOGITS].new_tensor(0.0)
                if hasattr(self.model, "box_key") and self.model.box_key in batch and (~labeled_mask).any():
                    ...
                    weak_loss = self._weak_box_losses(
                        logits=fg_logits,
                        boxes=boxes_u,
                        outside_weight=weak_outside_w,
                        entropy_weight=weak_entropy_w,
                        tv_weight=weak_tv_w,
                    )
                
                moe_loss = output[self.model.prefix].get(MOE_LOSS, 0.0)
                loss = sup_loss + weak_loss + moe_loss
                self.log("train_sup_loss", sup_loss, on_step=True, on_epoch=True)
                self.log("train_weak_loss", weak_loss, on_step=True, on_epoch=True)
                self.log("train_labeled_frac", labeled_mask.float().mean(), on_step=True, on_epoch=True)
```

下一步运行建议：
- 直接用同一命令重新启动训练即可（代码已修正），日志仍写入 `outputs/acdc_mask2former-semi-260107.log` 或换一个新输出目录避免覆盖。  
- 训练日志中应能看到 `train_sup_loss/train_weak_loss/train_labeled_frac`，若仍异常请告知具体栈信息。

```python
        # ---- Loss: fully-supervised OR semi/weak mixed ----
        if self.training and labeled_mask is not None:
            # Semi/weak supervision path
            if isinstance(self.loss_func, Mask2FormerLoss):
                # 1) 有标签子集：使用 Mask2FormerLoss
                sup_loss = output[self.model.prefix][LOGITS].new_tensor(0.0)
                if labeled_mask.any():
                    ...
                    sup_loss = self._compute_loss(
                        output=output_labeled,
                        label=label_l,
                        mask_labels=mask_labels_l,
                        class_labels=class_labels_l,
                    )
                
                # 2) 无标签子集：对聚合前景概率做盒弱监督
                weak_loss = output[self.model.prefix][LOGITS].new_tensor(0.0)
                if hasattr(self.model, "box_key") and self.model.box_key in batch and (~labeled_mask).any():
                    ...
                    weak_loss = self._weak_box_losses(
                        logits=fg_logits,
                        boxes=boxes_u,
                        outside_weight=weak_outside_w,
                        entropy_weight=weak_entropy_w,
                        tv_weight=weak_tv_w,
                    )
                
                moe_loss = output[self.model.prefix].get(MOE_LOSS, 0.0)
                loss = sup_loss + weak_loss + moe_loss
                self.log("train_sup_loss", sup_loss, on_step=True, on_epoch=True)
                self.log("train_weak_loss", weak_loss, on_step=True, on_epoch=True)
                self.log("train_labeled_frac", labeled_mask.float().mean(), on_step=True, on_epoch=True)
```

---

**User**

请重新仔细审查你为ACDC做的半监督跟我们之前在ISIC2017、2018数据集上做的半监督的方法是一样的吗？

---

**Cursor**

结论：整体思路与 ISIC2017/2018 的半/弱监督保持一致（同样的 labeled/unlabeled 划分、GT bbox 抖动生成弱盒、盒外背景+盒内熵/TV 的弱损失、可选 GSPO 仅作用于有标签子集），但在 ACDC 上为适配 Mask2Former 多类输出做了必要的改动：

- 配置层面仍然通过 `_train_box_prompt_cfg` 传递 `semi_labeled_fraction/seed` 与 `weak_box_jitter_*`、`weak_loss_*`，与 ISIC 命令行开关一致。
- 有标签子集：ISIC 用二分类 `StructureLoss`；ACDC 现在对子集用 `Mask2FormerLoss`，保持全监督分割能力。  
- 无标签子集：ISIC 直接对二值 logits 用盒弱监督；ACDC 先把多 token + class logits 聚合为前景概率（softmax(class)*sigmoid(mask) 并求和），再取 logit 输入同一盒弱监督函数，实现与 ISIC 相同的约束形式。  
- GSPO 半监督路径保持原有逻辑：有标签子集走 GSPO，未标注子集走弱损失，`gspo_allow_semisup` 依旧有效。

关键实现片段（ACDC 半监督多类适配）：
```
617:685:autogluon/multimodal/src/autogluon/multimodal/optim/lit_semantic_seg.py
# 1) labeled 子集用 Mask2FormerLoss
# 2) unlabeled 子集聚合前景概率后走 _weak_box_losses(outside/entropy/TV)
sup_loss = self._compute_loss(...)
...
mask_cls = F.softmax(class_logits_u, dim=-1)[..., :-1]
mask_prob = torch.sigmoid(mask_logits_u)
semantic_prob = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_prob)
fg_prob = semantic_prob.sum(dim=1, keepdim=True).clamp(min=1e-7, max=1 - 1e-7)
fg_logits = torch.logit(fg_prob)
weak_loss = self._weak_box_losses(...)
```

配置一致性示例（ISIC 脚本）：
```
336:352:autogluon/examples/automm/Conv-LoRA/run_semantic_segmentation.py
predictor._learner._train_box_prompt_cfg = {
    "semi_labeled_fraction": args.semi_labeled_fraction,
    "semi_labeled_seed": args.semi_labeled_seed,
    "weak_box_jitter_mode": args.weak_box_jitter_mode,
    "weak_box_jitter_amount": args.weak_box_jitter_amount,
    "weak_box_outward_only": bool(args.weak_box_outward_only),
    "weak_loss_outside_weight": args.weak_loss_outside_weight,
    "weak_loss_entropy_weight": args.weak_loss_entropy_weight,
    "weak_loss_tv_weight": args.weak_loss_tv_weight,
    ...
}
```

差异与注意点：
- 唯一必要差异是多类 Mask2Former 的聚合步骤；弱监督仍然针对“前景概率”做盒外/盒内约束，因此语义一致。
- 如果需要类别敏感的弱监督（盒内只对应特定器官），目前实现是对全部前景求和后约束，仍有潜在改进空间，但行为与 ISIC 的“前景 vs 背景”假设一致。
- 运行命令可继续使用之前的 ACDC 半监督命令；如要保留旧日志，建议改 `--output_dir` / `--vis_output_dir`。

```plaintext
617:685:autogluon/multimodal/src/autogluon/multimodal/optim/lit_semantic_seg.py
# 1) labeled 子集用 Mask2FormerLoss
# 2) unlabeled 子集聚合前景概率后走 _weak_box_losses(outside/entropy/TV)
sup_loss = self._compute_loss(...)
...
mask_cls = F.softmax(class_logits_u, dim=-1)[..., :-1]
mask_prob = torch.sigmoid(mask_logits_u)
semantic_prob = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_prob)
fg_prob = semantic_prob.sum(dim=1, keepdim=True).clamp(min=1e-7, max=1 - 1e-7)
fg_logits = torch.logit(fg_prob)
weak_loss = self._weak_box_losses(...)
```

```plaintext
336:352:autogluon/examples/automm/Conv-LoRA/run_semantic_segmentation.py
predictor._learner._train_box_prompt_cfg = {
    "semi_labeled_fraction": args.semi_labeled_fraction,
    "semi_labeled_seed": args.semi_labeled_seed,
    "weak_box_jitter_mode": args.weak_box_jitter_mode,
    "weak_box_jitter_amount": args.weak_box_jitter_amount,
    "weak_box_outward_only": bool(args.weak_box_outward_only),
    "weak_loss_outside_weight": args.weak_loss_outside_weight,
    "weak_loss_entropy_weight": args.weak_loss_entropy_weight,
    "weak_loss_tv_weight": args.weak_loss_tv_weight,
    ...
}
```

---

**User**

又错了：@autogluon/examples/automm/Conv-LoRA/outputs/acdc_mask2former-semi-260107.log 

---

**Cursor**

```python
import logging
import hashlib
from typing import Callable, Dict, Optional

import torch
import torchmetrics
import torch.nn.functional as F
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
    
    def __init__(self, *args, gspo_trainer=None, train_box_prompt_cfg=None, train_bbox_predictor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gspo_trainer = gspo_trainer
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
    
    @staticmethod
    def _stable_uniform_0_1(key: str, seed: int = 0) -> float:
        """
        Stable deterministic pseudo-random number in [0, 1) from a string key + seed.
        Avoids python's randomized hash across processes.
        """
        h = hashlib.md5(f"{seed}::{key}".encode("utf-8")).digest()
        # use first 8 bytes as uint64
        x = int.from_bytes(h[:8], byteorder="little", signed=False)
        return (x % (10**12)) / float(10**12)
    
    def _get_labeled_mask_from_paths(self, image_paths, labeled_fraction: float, seed: int) -> torch.Tensor:
        """
        image_paths: list[str] (len=B)
        returns: bool tensor (B,)
        """
        labeled_fraction = float(labeled_fraction)
        labeled_fraction = max(0.0, min(1.0, labeled_fraction))
        if labeled_fraction >= 1.0:
            return torch.ones(len(image_paths), dtype=torch.bool, device=self.device)
        if labeled_fraction <= 0.0:
            return torch.zeros(len(image_paths), dtype=torch.bool, device=self.device)
        flags = []
        for p in image_paths:
            p = p or ""
            u = self._stable_uniform_0_1(p, seed=seed)
            flags.append(u < labeled_fraction)
        return torch.tensor(flags, dtype=torch.bool, device=self.device)
    
    @staticmethod
    def _compute_boxes_from_mask(mask: torch.Tensor) -> torch.Tensor:
        """mask: (B,H,W) or (B,1,H,W) -> boxes (B,4) in pixel coords (x1,y1,x2,y2)."""
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
    
    @staticmethod
    def _jitter_boxes_outward(
        boxes: torch.Tensor,
        h: int,
        w: int,
        amount: float,
        mode: str = "box",  # box / image / pixel
        outward_only: bool = True,
    ) -> torch.Tensor:
        """
        Jitter boxes to simulate coarse human boxes.
        - outward_only=True: never shrink vs original (always expand or keep).
        - amount meaning:
          - mode=box: fraction of box width/height
          - mode=image: fraction of image width/height
          - mode=pixel: absolute pixels
        """
        jittered = boxes.clone()
        mode = (mode or "box").lower()
        amount = float(amount)
        amount = max(0.0, amount)
        for i in range(jittered.shape[0]):
            x1, y1, x2, y2 = jittered[i]
            if x2 <= x1 or y2 <= y1:
                continue
            bw = (x2 - x1).clamp(min=1.0)
            bh = (y2 - y1).clamp(min=1.0)
            if mode == "box":
                dx = bw * amount
                dy = bh * amount
            elif mode == "image":
                dx = float(w) * amount
                dy = float(h) * amount
                dx = torch.tensor(dx, device=boxes.device, dtype=torch.float32)
                dy = torch.tensor(dy, device=boxes.device, dtype=torch.float32)
            elif mode == "pixel":
                dx = torch.tensor(amount, device=boxes.device, dtype=torch.float32)
                dy = torch.tensor(amount, device=boxes.device, dtype=torch.float32)
            else:
                # fallback to box
                dx = bw * amount
                dy = bh * amount
            
            # sample non-negative deltas (mostly outward expansion)
            # use uniform [0, dx] / [0, dy]
            ex1 = torch.rand(1, device=boxes.device) * dx
            ey1 = torch.rand(1, device=boxes.device) * dy
            ex2 = torch.rand(1, device=boxes.device) * dx
            ey2 = torch.rand(1, device=boxes.device) * dy
            
            if outward_only:
                # outward: x1 decreases, y1 decreases, x2 increases, y2 increases
                x1n = x1 - ex1
                y1n = y1 - ey1
                x2n = x2 + ex2
                y2n = y2 + ey2
            else:
                # allow inward with small probability by random sign
                s1 = torch.where(torch.rand(1, device=boxes.device) < 0.9, -1.0, 1.0)  # x1: outward is -1
                t1 = torch.where(torch.rand(1, device=boxes.device) < 0.9, -1.0, 1.0)  # y1
                s2 = torch.where(torch.rand(1, device=boxes.device) < 0.9, 1.0, -1.0)   # x2: outward is +1
                t2 = torch.where(torch.rand(1, device=boxes.device) < 0.9, 1.0, -1.0)   # y2
                x1n = x1 + s1 * ex1
                y1n = y1 + t1 * ey1
                x2n = x2 + s2 * ex2
                y2n = y2 + t2 * ey2
            
            x1n = torch.clamp(x1n, 0, w - 1)
            y1n = torch.clamp(y1n, 0, h - 1)
            x2n = torch.clamp(x2n, 0, w - 1)
            y2n = torch.clamp(y2n, 0, h - 1)
            # ensure valid ordering
            jittered[i] = torch.stack(
                [torch.min(x1n, x2n), torch.min(y1n, y2n), torch.max(x1n, x2n), torch.max(y1n, y2n)]
            ).squeeze()
        return jittered
    
    @staticmethod
    def _structure_loss_per_sample(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Per-sample StructureLoss (same as optim/losses/structure_loss.py) without final mean.
        Returns tensor of shape (B,).
        """
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
        wbce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)).clamp(min=1e-6)
        probs = torch.sigmoid(logits)
        inter = ((probs * target) * weit).sum(dim=(2, 3))
        union = ((probs + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).view(-1)
    
    @staticmethod
    def _weak_box_losses(
        logits: torch.Tensor,  # (B,1,H,W) or (B,H,W)
        boxes: torch.Tensor,   # (B,4) x1,y1,x2,y2 in pixel coords
        outside_weight: float = 1.0,
        entropy_weight: float = 0.05,
        tv_weight: float = 0.0,
    ) -> torch.Tensor:
        """
        Box-only weak supervision loss:
        - outside-box should be background (0)
        - inside-box entropy minimization (push confident predictions without assuming full positive)
        - optional smoothness (TV) inside box
        Returns scalar loss.
        """
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        b, _, h, w = logits.shape
        device = logits.device
        outside_weight = float(outside_weight)
        entropy_weight = float(entropy_weight)
        tv_weight = float(tv_weight)
        
        total = logits.new_tensor(0.0)
        count = 0
        for i in range(b):
            x1, y1, x2, y2 = boxes[i]
            if x2 <= x1 or y2 <= y1:
                continue
            # integer pixel bounds (inclusive)
            x1i = int(torch.clamp(x1.round(), 0, w - 1).item())
            y1i = int(torch.clamp(y1.round(), 0, h - 1).item())
            x2i = int(torch.clamp(x2.round(), 0, w - 1).item())
            y2i = int(torch.clamp(y2.round(), 0, h - 1).item())
            if x2i <= x1i or y2i <= y1i:
                continue
            
            logit = logits[i : i + 1]  # (1,1,H,W)
            outside_mask = torch.ones((1, 1, h, w), device=device, dtype=logit.dtype)
            outside_mask[:, :, y1i : y2i + 1, x1i : x2i + 1] = 0.0
            inside_mask = 1.0 - outside_mask
            
            # Outside-box background constraint
            if outside_weight > 0:
                zeros = torch.zeros_like(logit)
                bce = F.binary_cross_entropy_with_logits(logit, zeros, reduction="none")
                outside_loss = (bce * outside_mask).sum() / outside_mask.sum().clamp(min=1.0)
            else:
                outside_loss = logit.new_tensor(0.0)
            
            # Inside-box entropy minimization
            if entropy_weight > 0:
                p = torch.sigmoid(logit).clamp(min=1e-6, max=1 - 1e-6)
                ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
                ent_loss = (ent * inside_mask).sum() / inside_mask.sum().clamp(min=1.0)
            else:
                ent_loss = logit.new_tensor(0.0)
            
            # Total variation smoothness inside the box
            if tv_weight > 0:
                p = torch.sigmoid(logit)
                dy = torch.abs(p[:, :, 1:, :] - p[:, :, :-1, :])
                dx = torch.abs(p[:, :, :, 1:] - p[:, :, :, :-1])
                # match shapes with masks
                inside_y = inside_mask[:, :, 1:, :]
                inside_x = inside_mask[:, :, :, 1:]
                tv = (dy * inside_y).sum() / inside_y.sum().clamp(min=1.0) + (dx * inside_x).sum() / inside_x.sum().clamp(min=1.0)
            else:
                tv = logit.new_tensor(0.0)
            
            total = total + outside_weight * outside_loss + entropy_weight * ent_loss + tv_weight * tv
            count += 1
        
        if count == 0:
            return logits.new_tensor(0.0)
        return total / float(count)

    def _predict_boxes_from_images(self, image_paths: list, target_h: int, target_w: int) -> torch.Tensor:
        """
        Wrapper around train_bbox_predictor to predict boxes in pixel coords and scale to target size.
        """
        import numpy as np
        from PIL import Image
        if self.train_bbox_predictor is None:
            raise ValueError("train_bbox_predictor is required for mode='predict'")
        b = len(image_paths)
        boxes = torch.zeros((b, 4), device=self.device, dtype=torch.float32)
        for i, img_path in enumerate(image_paths):
            if not img_path:
                continue
            try:
                bbox_px = self.train_bbox_predictor.predict(img_path)
                if bbox_px is None:
                    continue
                with Image.open(img_path) as img:
                    orig_w, orig_h = img.size
                scale_x = target_w / orig_w
                scale_y = target_h / orig_h
                x1 = bbox_px[0] * scale_x
                y1 = bbox_px[1] * scale_y
                x2 = bbox_px[2] * scale_x
                y2 = bbox_px[3] * scale_y
                x1 = max(0, min(x1, target_w - 1))
                y1 = max(0, min(y1, target_h - 1))
                x2 = max(0, min(x2, target_w - 1))
                y2 = max(0, min(y2, target_h - 1))
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                boxes[i] = torch.tensor([x1, y1, x2, y2], device=self.device, dtype=torch.float32)
            except Exception as e:
                logger.warning(f"Failed to predict bbox for {img_path}: {e}")
                continue
        return boxes

    def _apply_train_box_prompts(self, batch: Dict, label: torch.Tensor) -> None:
        """
        In-place box prompt injection for training (gt/noisy/mix/predict) and for semi/weak supervision.
        This is factored out so GSPO/standard paths can share the same augmentation.
        """
        cfg = self.train_box_prompt_cfg or {}
        mode = cfg.get("mode", "off")
        if mode == "off" and "semi_labeled_fraction" not in cfg:
            return

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
        else:
            noise_frac = float(cfg.get("noise_frac", 0.12))

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

        if box_mode in ["gt", "noisy"]:
            gt_boxes = self._compute_boxes_from_mask(label)
            h, w = label.shape[-2], label.shape[-1]
            if box_mode == "noisy":
                gt_boxes = _jitter_boxes(gt_boxes, h=h, w=w, frac=noise_frac)
            if not (gt_boxes.sum(dim=1) == 0).all():
                batch[self.model.box_key] = gt_boxes.unsqueeze(1)  # (B,1,4)
        elif box_mode == "predict":
            image_path_key = self.model.image_path_key
            image_paths = batch.get(image_path_key, [])
            if image_paths:
                h, w = label.shape[-2], label.shape[-1]
                pred_boxes = self._predict_boxes_from_images(
                    image_paths=image_paths, target_h=h, target_w=w
                )
                if not (pred_boxes.sum(dim=1) == 0).all():
                    batch[self.model.box_key] = pred_boxes.unsqueeze(1)  # (B,1,4)

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
                logits = per_output[LOGITS]
                
                # Handle SAM multi-class output format: combine mask and class predictions
                # SAM multi-class outputs: LOGITS [B, num_mask_tokens, H, W] + CLASS_LOGITS [B, num_mask_tokens, num_classes+1]
                # When num_mask_tokens == num_classes, we can use LOGITS directly as per-class masks
                # Otherwise, we need to combine mask and class predictions
                if CLASS_LOGITS in per_output:
                    mask_logits = logits  # [B, num_mask_tokens, H, W]
                    class_logits = per_output[CLASS_LOGITS]  # [B, num_mask_tokens, num_classes+1]
                    
                    # Apply semantic inference to combine mask and class predictions
                    # This follows Mask2Former's approach: semantic_mask = einsum(softmax(class), sigmoid(mask))
                    mask_cls = F.softmax(class_logits, dim=-1)[..., :-1]  # [B, num_mask_tokens, num_classes]
                    mask_pred = torch.sigmoid(mask_logits)  # [B, num_mask_tokens, H, W]
                    
                    # Combine: for each pixel, sum over mask tokens weighted by class probability
                    # semantic_mask[b, c, h, w] = sum_q(mask_cls[b, q, c] * mask_pred[b, q, h, w])
                    logits = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
                    
                    # Convert probabilities back to logits for CrossEntropyLoss (add small epsilon to avoid log(0))
                    logits = torch.log(logits.clamp(min=1e-7))
                
                loss += (
                    self.loss_func(
                        input=logits,
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
        
        # ---- Semi/weak supervision config (optional) ----
        cfg = self.train_box_prompt_cfg or {}
        semi_labeled_fraction = float(cfg.get("semi_labeled_fraction", 1.0))
        semi_labeled_seed = int(cfg.get("semi_labeled_seed", 0))
        weak_box_jitter_mode = str(cfg.get("weak_box_jitter_mode", "box"))
        weak_box_jitter_amount = float(cfg.get("weak_box_jitter_amount", cfg.get("noise_frac", 0.12)))
        weak_box_outward_only = bool(cfg.get("weak_box_outward_only", True))
        weak_outside_w = float(cfg.get("weak_loss_outside_weight", 1.0))
        weak_entropy_w = float(cfg.get("weak_loss_entropy_weight", 0.05))
        weak_tv_w = float(cfg.get("weak_loss_tv_weight", 0.0))
        
        # Determine which samples are "labeled" (for simulation) using stable hashing on image paths.
        image_paths = batch.get(self.model.image_path_key, [])
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        labeled_mask = None
        if self.training and semi_labeled_fraction < 1.0:
            labeled_mask = self._get_labeled_mask_from_paths(
                image_paths=image_paths, labeled_fraction=semi_labeled_fraction, seed=semi_labeled_seed
            )

        # -------- Train-time box prompt injection (configurable) --------
        # 控制项：self.train_box_prompt_cfg = {
        #   mode: off / gt / noisy / mix / predict,
        #   p_no, p_gt, p_noisy: 仅在 mode=mix 时生效并会归一化,
        #   noise_frac: 噪声扰动比例（相对 bbox 宽高）
        # }
        # mode=predict: 使用 self.train_bbox_predictor 预测框作为 box prompt
        if self.training:
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

            # If semi/weak enabled: for "unlabeled" samples, force weak coarse boxes generated from GT bbox with outward-biased jitter.
            if labeled_mask is not None and hasattr(self.model, "box_key"):
                gt_boxes_all = self._compute_boxes_from_mask(label)
                h, w = label.shape[-2], label.shape[-1]
                weak_boxes = self._jitter_boxes_outward(
                    gt_boxes_all,
                    h=h,
                    w=w,
                    amount=weak_box_jitter_amount,
                    mode=weak_box_jitter_mode,
                    outward_only=weak_box_outward_only,
                )
                # apply only to unlabeled samples
                if (~labeled_mask).any():
                    boxes_for_batch = gt_boxes_all.clone()
                    boxes_for_batch[~labeled_mask] = weak_boxes[~labeled_mask]
                    if not (boxes_for_batch.sum(dim=1) == 0).all():
                        batch[self.model.box_key] = boxes_for_batch.unsqueeze(1)  # (B,1,4)

            if box_mode in ["gt", "noisy"]:
                gt_boxes = self._compute_boxes_from_mask(label)
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
        
        # ---- Loss: fully-supervised OR semi/weak mixed ----
        if self.training and labeled_mask is not None:
            # Semi/weak supervision path
            if isinstance(self.loss_func, Mask2FormerLoss):
                # 1) 有标签子集：使用 Mask2FormerLoss
                sup_loss = output[self.model.prefix][LOGITS].new_tensor(0.0)
                if labeled_mask.any():
                    # 子集化输出与标签
                    output_labeled = {}
                    for key, per_output in output.items():
                        sub_out = {}
                        for sub_k, val in per_output.items():
                            # 仅当第 0 维与 batch 对齐时才子集化，避免标量/非 batch 维度触发索引错误
                            if torch.is_tensor(val) and val.ndim > 0 and val.shape[0] == label.shape[0]:
                                sub_out[sub_k] = val[labeled_mask]
                            else:
                                sub_out[sub_k] = val
                        output_labeled[key] = sub_out
                    mask_labels = batch.get(self.model.mask_label_key, [])
                    class_labels = batch.get(self.model.class_label_key, [])
                    if isinstance(mask_labels, torch.Tensor):
                        mask_labels_l = mask_labels[labeled_mask]
                    else:
                        mask_labels_l = [m for m, keep in zip(mask_labels, labeled_mask) if keep]
                    if isinstance(class_labels, torch.Tensor):
                        class_labels_l = class_labels[labeled_mask]
                    else:
                        class_labels_l = [c for c, keep in zip(class_labels, labeled_mask) if keep]
                    label_l = label[labeled_mask]
                    sup_loss = self._compute_loss(
                        output=output_labeled,
                        label=label_l,
                        mask_labels=mask_labels_l,
                        class_labels=class_labels_l,
                    )
                
                # 2) 无标签子集：对聚合前景概率做盒弱监督
                weak_loss = output[self.model.prefix][LOGITS].new_tensor(0.0)
                if hasattr(self.model, "box_key") and self.model.box_key in batch and (~labeled_mask).any():
                    per_output = output[self.model.prefix]
                    mask_logits_u = per_output[LOGITS][~labeled_mask]
                    boxes_u = batch[self.model.box_key][:, 0, :][~labeled_mask]
                    if CLASS_LOGITS in per_output:
                        class_logits_u = per_output[CLASS_LOGITS][~labeled_mask]
                        # 语义概率：softmax(class) * sigmoid(mask) -> foreground 概率
                        mask_cls = F.softmax(class_logits_u, dim=-1)[..., :-1]
                        mask_prob = torch.sigmoid(mask_logits_u)
                        semantic_prob = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_prob)
                        fg_prob = semantic_prob.sum(dim=1, keepdim=True).clamp(min=1e-7, max=1 - 1e-7)
                        fg_logits = torch.logit(fg_prob)
                    else:
                        logits_u = mask_logits_u
                        if logits_u.dim() == 4 and logits_u.shape[1] > 1:
                            fg_prob = torch.sigmoid(logits_u).mean(dim=1, keepdim=True).clamp(min=1e-7, max=1 - 1e-7)
                            fg_logits = torch.logit(fg_prob)
                        else:
                            fg_logits = logits_u
                    weak_loss = self._weak_box_losses(
                        logits=fg_logits,
                        boxes=boxes_u,
                        outside_weight=weak_outside_w,
                        entropy_weight=weak_entropy_w,
                        tv_weight=weak_tv_w,
                    )
                
                moe_loss = output[self.model.prefix].get(MOE_LOSS, 0.0)
                loss = sup_loss + weak_loss + moe_loss
                self.log("train_sup_loss", sup_loss, on_step=True, on_epoch=True)
                self.log("train_weak_loss", weak_loss, on_step=True, on_epoch=True)
                self.log("train_labeled_frac", labeled_mask.float().mean(), on_step=True, on_epoch=True)
            else:
                # 二值结构损失路径
                logits = output[self.model.prefix][LOGITS]
                if logits.dim() == 3:
                    logits = logits.unsqueeze(1)
                if label.dim() == 3:
                    label_ = label.unsqueeze(1)
                else:
                    label_ = label
                
                per_sample_sup = self._structure_loss_per_sample(logits, label_)
                sup_count = labeled_mask.sum().clamp(min=1)
                sup_loss = (per_sample_sup * labeled_mask.float()).sum() / sup_count.float()
                
                weak_loss = logits.new_tensor(0.0)
                if hasattr(self.model, "box_key") and self.model.box_key in batch and (~labeled_mask).any():
                    boxes_b1 = batch[self.model.box_key]  # (B,1,4)
                    boxes = boxes_b1[:, 0, :]
                    logits_u = logits[~labeled_mask]
                    boxes_u = boxes[~labeled_mask]
                    weak_loss = self._weak_box_losses(
                        logits=logits_u,
                        boxes=boxes_u,
                        outside_weight=weak_outside_w,
                        entropy_weight=weak_entropy_w,
                        tv_weight=weak_tv_w,
                    )
                
                moe_loss = output[self.model.prefix].get(MOE_LOSS, 0.0)
                loss = sup_loss + weak_loss + moe_loss
                self.log("train_sup_loss", sup_loss, on_step=True, on_epoch=True)
                self.log("train_weak_loss", weak_loss, on_step=True, on_epoch=True)
                self.log("train_labeled_frac", labeled_mask.float().mean(), on_step=True, on_epoch=True)
        else:
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
        cfg = getattr(self, "train_box_prompt_cfg", {}) or {}
        semi_labeled_fraction = float(cfg.get("semi_labeled_fraction", 1.0))
        use_gspo = (
            self.gspo_trainer is not None and 
            self.gspo_trainer.is_gspo_active(self.current_epoch)
        )
        allow_semisup_gspo = bool(getattr(self.gspo_trainer, "allow_semisup", False)) if self.gspo_trainer else False
        
        if use_gspo and allow_semisup_gspo and semi_labeled_fraction < 1.0:
            # Semi-supervised with GSPO on labeled subset, weak loss on unlabeled subset.
            label = batch[self.model.label_key]
            # Apply box prompts (gt/noisy/predict + weak jitter boxes already handled inside)
            if self.training:
                self._apply_train_box_prompts(batch, label)
            image_paths = batch.get(self.model.image_path_key, [])
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            labeled_mask = self._get_labeled_mask_from_paths(
                image_paths=image_paths, labeled_fraction=semi_labeled_fraction, seed=int(cfg.get("semi_labeled_seed", 0))
            )
            has_labeled = bool(labeled_mask.any())
            has_unlabeled = bool((~labeled_mask).any())
            loss = torch.tensor(0.0, device=self.device)
            selected_experts = None

            def _subset_batch(b, mask_bool: torch.Tensor):
                new = {}
                for k, v in b.items():
                    if torch.is_tensor(v) and v.shape[0] == mask_bool.shape[0]:
                        new[k] = v[mask_bool]
                    elif isinstance(v, list) and len(v) == mask_bool.shape[0]:
                        new[k] = [v[i] for i in range(len(v)) if mask_bool[i].item()]
                    else:
                        new[k] = v
                return new

            # Weak loss on unlabeled subset
            if has_unlabeled and hasattr(self.model, "box_key") and self.model.box_key in batch:
                batch_u = _subset_batch(batch, ~labeled_mask)
                label_u = batch_u[self.model.label_key]
                output_u = run_model(self.model, batch_u)
                logits_u = output_u[self.model.prefix][LOGITS]
                if logits_u.dim() == 3:
                    logits_u = logits_u.unsqueeze(1)
                boxes_u = batch_u[self.model.box_key][:, 0, :]
                weak_loss = self._weak_box_losses(
                    logits=logits_u,
                    boxes=boxes_u,
                    outside_weight=float(cfg.get("weak_loss_outside_weight", 1.0)),
                    entropy_weight=float(cfg.get("weak_loss_entropy_weight", 0.05)),
                    tv_weight=float(cfg.get("weak_loss_tv_weight", 0.0)),
                )
                moe_loss_u = output_u[self.model.prefix].get(MOE_LOSS, 0.0)
                loss = loss + weak_loss + moe_loss_u
                self.log("train_weak_loss", weak_loss, on_step=True, on_epoch=True)

            # GSPO on labeled subset
            if has_labeled:
                batch_l = _subset_batch(batch, labeled_mask)
                gspo_loss, metrics, selected_experts = self._gspo_training_step(batch_l)
                loss = loss + gspo_loss
                for key, value in metrics.items():
                    self.log(f"train_{key}", value, on_step=True, on_epoch=True)

            self.log("train_labeled_frac", labeled_mask.float().mean(), on_step=True, on_epoch=True)

        elif use_gspo:
            # GSPO-enhanced training (full GT)
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
        
        Extended to support:
        - Encoder Adapter quality feedback when gspo_adapter_enabled=True
        - Decoder LoRA on Attention quality feedback when gspo_lora_attention_enabled=True
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
        
        # Update expert quality feedback (Conv-LoRA MoE)
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
            
            # Update Adapter quality feedback (GSPO-Adapter extension)
            if hasattr(self.gspo_trainer, 'gspo_adapter_enabled') and self.gspo_trainer.gspo_adapter_enabled:
                self.gspo_trainer.update_adapter_feedback(quality_scores, self.model)
            
            # Update LoRA on Attention quality feedback (GSPO-LoRA extension)
            if hasattr(self.gspo_trainer, 'gspo_lora_attention_enabled') and self.gspo_trainer.gspo_lora_attention_enabled:
                self.gspo_trainer.update_lora_attention_feedback(quality_scores, self.model)
        
        return loss, metrics, selected_experts_groups

```

找到问题：`_shared_step` 在提取有标签子集时，对每个 `per_output` 条目的张量直接用 `val.shape[0]` 判断是否需要子集化；遇到标量（0 维张量）时 `shape[0]` 抛出 IndexError。

修复：仅当张量维度>0 且第 0 维等于 batch 大小时才子集化，否则原样保留，避免标量/非 batch 维度被误索引。
```
autogluon/multimodal/src/autogluon/multimodal/optim/lit_semantic_seg.py
- if torch.is_tensor(val) and val.shape[0] == label.shape[0]:
+ if torch.is_tensor(val) and val.ndim > 0 and val.shape[0] == label.shape[0]:
```

请重新运行同一训练命令（或继续当前 nohup 被中断的作业），日志仍写在 `outputs/acdc_mask2former-semi-260107.log`。如再有报错，把新日志尾部贴给我。

```plaintext
autogluon/multimodal/src/autogluon/multimodal/optim/lit_semantic_seg.py
- if torch.is_tensor(val) and val.shape[0] == label.shape[0]:
+ if torch.is_tensor(val) and val.ndim > 0 and val.shape[0] == label.shape[0]:
```

---

**User**

现在能跑起来，但是val_dice又上不去了。记得你之前也曾经解决过类似的 val_dice始终在0.2+左右且不随训练epochs增加而提升 的问题，不过那时好像是针对ISIC数据集的。那么现在你再看看对于ACDC数据集，要如何改才能使训练正常？

---

**Cursor**

我看了一下当前代码路径，Mask2Former 在半监督 + GSPO 下有两处关键不匹配，容易导致 val_dice 卡在很低水平：

1) GSPO 分支并未按 Mask2Former 的格式计算损失  
在 `_gspo_training_step` 里，`forward_fn` 只取了 `output[prefix][LOGITS]`，再把它直接塞给 `Mask2FormerLoss(predictions, targets)`。Mask2FormerLoss 需要的是 `(masks_queries_logits, class_queries_logits, mask_labels, class_labels)` 的组合，而 class logits、mask_labels/class_labels 都没传进去，所以 GSPO 分支的梯度基本无效/噪声。

2) GSPO 质量分数是“二值化”的  
`compute_segmentation_quality` 会把多类预测/GT 都 argmax 后再二值化（>0.5），在多类 ACDC 上这会把类别信息丢光，质量排序失真，GSPO 反而可能拖慢/扰乱优化。

最快让 val_dice 正常爬升的做法（无需改代码）：
- 暂时关掉 GSPO 及 GSPO-Adapter：去掉 `--gspo_enable` / `--gspo_adapter_enable` / `--gspo_allow_semisup`，保留 Conv-LoRA + Adapter + Mask2FormerLoss 的标准训练。
- 适当提高有标签比例：把 `--semi_labeled_fraction` 提到 0.3 或 0.5（ACDC 样本不算大，0.1 可能太少）。
- 若显存允许，可把 `--per_gpu_batch_size` 提到 2，并同步调高 `--batch_size`（例如 8），稳定 BatchNorm/统计。

示例命令（去掉 GSPO，并把半监督比例调高）：
```bash
python3 run_acdc_conv_lora_semi.py \
  --dataset_dir datasets/acdc_conv_lora/acdc_conv_lora \
  --rank 3 \
  --expert_num 8 \
  --adapter_enable \
  --adapter_dim 64 \
  --loss mask2former_loss \
  --num_mask_tokens 10 \
  --semi_labeled_fraction 0.3 \
  --semi_labeled_seed 123 \
  --weak_box_jitter_mode box \
  --weak_box_jitter_amount 0.1 \
  --weak_box_outward_only \
  --weak_loss_outside_weight 1.0 \
  --weak_loss_entropy_weight 0.05 \
  --weak_loss_tv_weight 0.0 \
  --max_epochs 100 \
  --patience 30 \
  --vis_samples 8 \
  --vis_output_dir outputs/acdc_vis_mask2former-semi-260107 \
  --output_dir outputs/acdc_mask2former-semi-260107-noGSPO
```

如果必须保留 GSPO，需要补的代码较多（摘要）：
- 在 `_gspo_training_step` 构造 Mask2FormerLoss 的输入：取 `output[prefix][CLASS_LOGITS]`、`batch[mask_label_key]`、`batch[class_label_key]`，调用时用与 `_compute_loss` 一致的参数。
- 在 `gspo_trainer.compute_segmentation_quality` 中，对多类改为“逐类 Dice/IoU 后取均值（可忽略背景类）”，不要再二值化。
- 在半监督场景，GSPO 只在有标签子集上跑，并确保质量分数的标签与子集对齐。

建议先按“关闭 GSPO + 提高有标签比例”跑一轮看 val_dice 是否回升，再决定是否投入时间修补 GSPO 的多类支持。

```shellscript
python3 run_acdc_conv_lora_semi.py \
  --dataset_dir datasets/acdc_conv_lora/acdc_conv_lora \
  --rank 3 \
  --expert_num 8 \
  --adapter_enable \
  --adapter_dim 64 \
  --loss mask2former_loss \
  --num_mask_tokens 10 \
  --semi_labeled_fraction 0.3 \
  --semi_labeled_seed 123 \
  --weak_box_jitter_mode box \
  --weak_box_jitter_amount 0.1 \
  --weak_box_outward_only \
  --weak_loss_outside_weight 1.0 \
  --weak_loss_entropy_weight 0.05 \
  --weak_loss_tv_weight 0.0 \
  --max_epochs 100 \
  --patience 30 \
  --vis_samples 8 \
  --vis_output_dir outputs/acdc_vis_mask2former-semi-260107 \
  --output_dir outputs/acdc_mask2former-semi-260107-noGSPO
```

---

**User**

还是要补上GSPO 的多类支持。请仔细规划和实现，确保代码修改符合设计意图，不改坏原有正确的逻辑

---

**Cursor**

```python
"""
GSPO (Group Sequence Policy Optimization) Trainer for Conv-LoRA, Adapters, and Decoder LoRA

This module implements the GSPO training strategy that enhances Conv-LoRA's
MoE mechanism through group-level optimization and quality-aware feedback.

Extended to support:
1. Encoder Adapters (Phase 1-3):
   - Phase 1: Unified advantage weighting (adapters receive same quality-weighted gradients)
   - Phase 2: Adapter scale adaptation (contribution score based on quality history)
   - Phase 3: Adapter MoE architecture (optional, multiple adapter variants)

2. Decoder LoRA on Attention (NEW):
   - Phase 1: Quality history tracking for LoRA layers
   - Phase 2: Dynamic scaling adaptation based on quality feedback

Key components:
1. Group-level sampling: Generate multiple predictions per image
2. Advantage function: Compute relative quality within groups
3. Contrastive loss: Pull high-quality predictions together, push low-quality apart
4. Quality feedback: Update expert selection, adapter quality, AND LoRA quality based on actual performance
"""

import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np


class GSPOConvLoRATrainer:
    """
    GSPO-enhanced trainer for Conv-LoRA semantic segmentation.
    
    This trainer implements group-level policy optimization inspired by GSPO,
    adapted for the Conv-LoRA MoE architecture in semantic segmentation tasks.
    
    Args:
        predictor: The MultiModalPredictor instance
        group_size: Number of predictions to generate per image in a group
        warmup_epochs: Number of epochs to train without GSPO before enabling it
        contrastive_weight: Weight for the contrastive loss component
        quality_metric: Metric to use for quality evaluation ('iou', 'dice', 'both')
        advantage_temperature: Temperature for computing advantage weights
    """
    
    def __init__(
        self,
        predictor,
        group_size: int = 4,
        warmup_epochs: int = 5,
        contrastive_weight: float = 0.1,
        quality_metric: str = 'iou',
        advantage_temperature: float = 5.0,
        lambda_smooth: float = 0.1,
        lambda_boundary: float = 0.3,
        w_boundary: float = 0.3,
        w_smooth: float = 0.1,
        w_thin: float = 0.05,
        # GSPO-Adapter extension parameters
        gspo_adapter_enabled: bool = False,
        gspo_adapter_momentum: float = 0.9,
        # GSPO-LoRA on Attention extension parameters (NEW)
        gspo_lora_attention_enabled: bool = False,
        gspo_lora_attention_momentum: float = 0.9,
        allow_semisup: bool = False,
    ):
        self.predictor = predictor
        self.group_size = group_size
        self.warmup_epochs = warmup_epochs
        self.contrastive_weight = contrastive_weight
        self.quality_metric = quality_metric
        self.advantage_temperature = advantage_temperature
        # Reward / loss shaping hyper-parameters
        self.lambda_smooth = lambda_smooth
        self.lambda_boundary = lambda_boundary
        self.w_boundary = w_boundary
        self.w_smooth = w_smooth
        self.w_thin = w_thin
        
        # GSPO-Adapter extension (Phase 1-2)
        self.gspo_adapter_enabled = gspo_adapter_enabled
        self.gspo_adapter_momentum = gspo_adapter_momentum
        
        # GSPO-LoRA on Attention extension (NEW)
        self.gspo_lora_attention_enabled = gspo_lora_attention_enabled
        self.gspo_lora_attention_momentum = gspo_lora_attention_momentum
        self.allow_semisup = allow_semisup
        
        # Statistics tracking
        self.expert_performance_log = defaultdict(list)
        self.group_quality_history = []
        self.adapter_quality_log = []  # Track adapter quality over time
        self.lora_quality_log = []  # Track LoRA quality over time (NEW)
        self.current_epoch = 0
        
    def is_gspo_active(self, epoch: int) -> bool:
        """Check if GSPO should be active for the given epoch."""
        return epoch >= self.warmup_epochs
    
    def compute_segmentation_quality(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        metric: str = 'iou',
        class_logits: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute segmentation quality metrics.
        
        - If class_logits is provided (Mask2Former-style), compute per-class
          IoU/Dice excluding background, then average over foreground classes
          present in GT. This preserves multi类信息（ACDC）而非二值化。
        - Otherwise fallback to binary IoU/Dice with threshold 0.5.
        """
        eps = 1e-6
        
        if class_logits is not None:
            # Mask2Former multi-class: aggregate semantic probabilities
            # class_logits: [B, Q, C+1], pred_masks: [B, Q, H, W]
            class_prob = torch.softmax(class_logits, dim=-1)[..., :-1]  # drop background
            mask_prob = torch.sigmoid(pred_masks)
            sem_prob = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob).clamp(min=0.0, max=1.0)
            B, C, H, W = sem_prob.shape
            
            # GT one-hot (exclude background channel 0)
            gt_onehot = torch.nn.functional.one_hot(gt_masks.long(), num_classes=class_logits.shape[-1])
            gt_onehot = gt_onehot[..., 1:].permute(0, 3, 1, 2).float()  # [B, C, H, W]
            
            inter = (sem_prob * gt_onehot).sum(dim=(2, 3))
            union = sem_prob.sum(dim=(2, 3)) + gt_onehot.sum(dim=(2, 3)) - inter
            iou = (inter + eps) / (union + eps)
            dice = (2 * inter + eps) / (sem_prob.sum(dim=(2, 3)) + gt_onehot.sum(dim=(2, 3)) + eps)
            
            if metric == 'iou':
                score = iou
            elif metric == 'dice':
                score = dice
            else:
                score = (iou + dice) / 2.0
            
            # 平均仅在 GT 中出现的前景类；若某样本全背景，退化为全类平均
            valid = gt_onehot.sum(dim=(2, 3)) > 0  # [B, C]
            valid_count = valid.sum(dim=1).clamp(min=1)
            score = (score * valid.float()).sum(dim=1) / valid_count
            return score
        
        # ---- Binary fallback (原逻辑) ----
        if pred_masks.dim() == 4 and pred_masks.shape[1] > 1:
            pred_masks = torch.argmax(pred_masks, dim=1)
        if gt_masks.dim() == 4 and gt_masks.shape[1] > 1:
            gt_masks = torch.argmax(gt_masks, dim=1)
        
        pred_flat = pred_masks.view(pred_masks.shape[0], -1).float()
        gt_flat = gt_masks.view(gt_masks.shape[0], -1).float()
        pred_flat = (pred_flat > 0.5).float()
        gt_flat = (gt_flat > 0.5).float()
        
        intersection = (pred_flat * gt_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1) - intersection
        iou = (intersection + eps) / (union + eps)
        
        if metric == 'iou':
            return iou
        
        dice = (2 * intersection + eps) / (pred_flat.sum(dim=1) + gt_flat.sum(dim=1) + eps)
        
        if metric == 'dice':
            return dice
        return (iou + dice) / 2.0

    def _prepare_binary_masks(
        self, pred_masks: torch.Tensor, gt_masks: torch.Tensor, class_logits: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert logits/probabilities to single-channel foreground probability
        maps for boundary/smoothness computation.
        
        - For Mask2Former (class_logits provided): sum foreground class probs.
        - For multi-channel logits without class_logits: take max prob as foreground.
        - For binary logits: sigmoid.
        """
        # pred_masks: logits; gt_masks: integer labels or one-hot
        if class_logits is not None:
            # Mask2Former aggregation to semantic foreground prob (sum over classes)
            class_prob = torch.softmax(class_logits, dim=-1)[..., :-1]  # drop background
            mask_prob = torch.sigmoid(pred_masks)
            sem_prob = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)
            pred_probs = sem_prob.sum(dim=1, keepdim=True).clamp(min=0.0, max=1.0)  # [B,1,H,W]
            if gt_masks.dim() == 3:
                gt_fg = (gt_masks > 0).float()
            else:
                gt_fg = torch.argmax(gt_masks, dim=1).float()
            gt_probs = gt_fg.unsqueeze(1)
            return pred_probs, gt_probs
        
        # Ensure channel dimension
        if pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(1)
        if gt_masks.dim() == 3:
            gt_masks = gt_masks.unsqueeze(1)

        if pred_masks.shape[1] > 1:
            pred_probs = torch.softmax(pred_masks, dim=1).max(dim=1, keepdim=True).values
        else:
            pred_probs = torch.sigmoid(pred_masks)

        if gt_masks.shape[1] > 1:
            gt_probs = torch.argmax(gt_masks, dim=1, keepdim=True).float()
        else:
            gt_probs = gt_masks.float()

        return pred_probs, gt_probs

    def _boundary_map(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Approximate boundary map via spatial gradients.
        """
        grad_x = torch.abs(masks[..., :, 1:] - masks[..., :, :-1])
        grad_y = torch.abs(masks[..., 1:, :] - masks[..., :-1, :])
        grad_x = torch.nn.functional.pad(grad_x, (0, 1, 0, 0))
        grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1))
        return torch.clamp(grad_x + grad_y, 0.0, 1.0)

    def compute_smoothness_loss(self, pred_probs: torch.Tensor) -> torch.Tensor:
        """
        Laplacian smoothness loss (per-sample).
        """
        lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=pred_probs.device, dtype=pred_probs.dtype)
        lap_kernel = lap_kernel.view(1, 1, 3, 3)
        lap = torch.nn.functional.conv2d(pred_probs, lap_kernel, padding=1)
        return (lap ** 2).mean(dim=(1, 2, 3))

    def compute_boundary_loss(self, pred_probs: torch.Tensor, gt_probs: torch.Tensor) -> torch.Tensor:
        """
        L1 difference between predicted and GT boundary maps (per-sample).
        """
        pred_edge = self._boundary_map(pred_probs)
        gt_edge = self._boundary_map(gt_probs)
        return torch.mean(torch.abs(pred_edge - gt_edge), dim=(1, 2, 3))

    def compute_thin_reward(self, pred_probs: torch.Tensor, gt_probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Dice-style reward on boundary skeletons (per-sample).
        """
        pred_edge = self._boundary_map(pred_probs)
        gt_edge = self._boundary_map(gt_probs)
        intersection = (pred_edge * gt_edge).flatten(1).sum(dim=1)
        union = pred_edge.flatten(1).sum(dim=1) + gt_edge.flatten(1).sum(dim=1)
        return (2 * intersection + eps) / (union + eps)
    
    def gspo_group_training_step(
        self,
        images: torch.Tensor,
        masks_gt: torch.Tensor,
        forward_fn,
        loss_fn,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        GSPO core training step: generate group predictions and compute weighted loss.
        
        Args:
            images: Input images [B, C, H, W]
            masks_gt: Ground truth masks [B, C, H, W] or [B, H, W]
            forward_fn: Forward function that returns (predictions, moe_loss, selected_experts)
            loss_fn: Loss function for segmentation
            
        Returns:
            total_loss: Combined loss for backpropagation
            metrics: Dictionary of metrics for logging
        """
        B = images.shape[0]
        G = self.group_size
        
        group_predictions = []
        group_quality_scores = []
        group_base_quality_scores = []
        group_smooth_losses = []
        group_boundary_losses = []
        group_thin_rewards = []
        group_moe_losses = []
        group_selected_experts = []
        group_seg_losses = []
        
        # 1. Generate group predictions with variations
        for g in range(G):
            # Apply slight variations via dropout to create diversity
            # (dropout is automatically different across forward passes)
            with torch.set_grad_enabled(True):
                pred_masks, moe_loss, selected_experts = forward_fn(images)
            
            pred_probs, gt_probs = self._prepare_binary_masks(pred_masks, masks_gt)
            smooth_loss = self.compute_smoothness_loss(pred_probs)
            boundary_loss = self.compute_boundary_loss(pred_probs, gt_probs)
            thin_reward = self.compute_thin_reward(pred_probs, gt_probs)

            # Compute quality scores
            quality = self.compute_segmentation_quality(
                pred_masks, masks_gt, metric=self.quality_metric
            )
            shaped_quality = (
                quality
                + self.w_boundary * (1.0 - boundary_loss.detach())
                - self.w_smooth * smooth_loss.detach()
                + self.w_thin * thin_reward.detach()
            )
            
            # Compute segmentation loss
            seg_loss = loss_fn(pred_masks, masks_gt)
            seg_loss = (
                seg_loss
                + self.lambda_smooth * smooth_loss.mean()
                + self.lambda_boundary * boundary_loss.mean()
            )
            
            group_predictions.append(pred_masks)
            group_quality_scores.append(shaped_quality)
            group_base_quality_scores.append(quality)
            group_smooth_losses.append(smooth_loss.detach())
            group_boundary_losses.append(boundary_loss.detach())
            group_thin_rewards.append(thin_reward.detach())
            group_moe_losses.append(moe_loss)
            group_selected_experts.append(selected_experts)
            group_seg_losses.append(seg_loss)
        
        # 2. Compute group-level advantage function (GSPO core)
        quality_tensor = torch.stack(group_quality_scores)  # [G, B]
        baseline = quality_tensor.mean(dim=0, keepdim=True)  # [1, B]
        advantages = quality_tensor - baseline  # [G, B]
        
        # 3. GSPO weighted loss
        total_seg_loss = 0
        total_moe_loss = 0
        
        for g in range(G):
            advantage = advantages[g]  # [B]
            
            # Compute adaptive weights based on advantage
            # Higher advantage -> higher weight (stronger gradient)
            weight = torch.sigmoid(advantage * self.advantage_temperature)
            
            # Weighted segmentation loss
            seg_loss = group_seg_losses[g]
            if seg_loss.dim() == 0:  # scalar loss
                weighted_seg_loss = seg_loss * weight.mean()
            else:  # per-sample loss
                weighted_seg_loss = (seg_loss * weight).mean()
            
            total_seg_loss += weighted_seg_loss
            total_moe_loss += group_moe_losses[g]
        
        total_seg_loss /= G
        total_moe_loss /= G
        
        # 4. GSPO contrastive loss
        contrastive_loss = self.compute_group_contrastive_loss(
            group_predictions, group_quality_scores
        )
        
        # 5. Combine losses
        total_loss = total_seg_loss + 1e-2 * total_moe_loss + self.contrastive_weight * contrastive_loss
        
        # 6. Collect metrics
        metrics = {
            'loss': total_loss.item(),
            'seg_loss': total_seg_loss.item(),
            'moe_loss': total_moe_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'avg_quality': quality_tensor.mean().item(),
            'avg_quality_base': torch.stack(group_base_quality_scores).mean().item(),
            'avg_smooth_loss': torch.stack(group_smooth_losses).mean().item(),
            'avg_boundary_loss': torch.stack(group_boundary_losses).mean().item(),
            'avg_thin_reward': torch.stack(group_thin_rewards).mean().item(),
            'quality_variance': quality_tensor.var().item(),
            'max_advantage': advantages.max().item(),
            'min_advantage': advantages.min().item(),
        }
        
        # 7. Store group quality for analysis
        self.group_quality_history.append({
            'epoch': self.current_epoch,
            'qualities': quality_tensor.detach().cpu().numpy(),
            'advantages': advantages.detach().cpu().numpy(),
        })
        
        return total_loss, metrics, group_selected_experts
    
    def compute_group_contrastive_loss(
        self,
        predictions: List[torch.Tensor],
        quality_scores: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        GSPO-inspired contrastive loss.
        
        The idea is to make predictions with similar quality have similar features,
        while predictions with different quality should have different features.
        
        Args:
            predictions: List of G prediction tensors [B, C, H, W]
            quality_scores: List of G quality tensors [B]
            
        Returns:
            contrastive_loss: Scalar loss
        """
        G = len(predictions)
        
        if G < 2:
            return torch.tensor(0.0, device=predictions[0].device)
        
        quality_tensor = torch.stack(quality_scores)  # [G, B]
        
        # Extract features from predictions (global average pooling)
        features = []
        for pred in predictions:
            # Apply global average pooling
            feat = F.adaptive_avg_pool2d(pred, (1, 1)).flatten(1)  # [B, C]
            feat = F.normalize(feat, dim=1)  # Normalize for cosine similarity
            features.append(feat)
        features = torch.stack(features)  # [G, B, C]
        
        # Compute pairwise cosine similarities
        # Reshape for batch matrix multiplication
        feat_flat = features.view(G, -1)  # [G, B*C]
        similarities = torch.mm(feat_flat, feat_flat.t())  # [G, G]
        
        # Normalize by number of features
        similarities = similarities / (features.shape[1] * features.shape[2])
        
        # Compute quality differences
        quality_mean = quality_tensor.mean(dim=1)  # [G]
        quality_diff = torch.abs(
            quality_mean.unsqueeze(0) - quality_mean.unsqueeze(1)
        )  # [G, G]
        
        # Contrastive objective:
        # - If quality is similar (small diff), encourage high similarity
        # - If quality is different (large diff), encourage low similarity
        target_similarity = 1.0 - quality_diff / (quality_diff.max() + 1e-6)
        
        # MSE loss between actual and target similarities
        contrastive_loss = F.mse_loss(similarities, target_similarity)
        
        return contrastive_loss
    
    def update_expert_feedback(
        self,
        selected_experts_groups: List[Optional[torch.Tensor]],
        quality_scores: List[torch.Tensor],
        model
    ):
        """
        Update expert quality history in the model based on GSPO feedback.
        
        Args:
            selected_experts_groups: List of selected expert tensors from each group
            quality_scores: List of quality score tensors from each group
            model: The model containing MoEGate modules
        """
        if not selected_experts_groups or selected_experts_groups[0] is None:
            return
        
        # Collect all MoEGate modules from the model
        moe_gates = []
        for module in model.modules():
            if hasattr(module, 'update_quality_history'):
                moe_gates.append(module)
        
        if not moe_gates:
            return
        
        # Update each MoE gate with quality feedback
        for g_idx, (selected_experts, quality) in enumerate(zip(selected_experts_groups, quality_scores)):
            if selected_experts is not None:
                for moe_gate in moe_gates:
                    moe_gate.update_quality_history(selected_experts, quality)
    
    def update_adapter_feedback(
        self,
        quality_scores: List[torch.Tensor],
        model
    ):
        """
        GSPO-Adapter Extension: Update adapter quality history based on GSPO feedback.
        
        This method is called after each GSPO training step to provide quality feedback
        to all AdapterLayer modules in the model.
        
        Args:
            quality_scores: List of quality score tensors from each group [G] of [B]
            model: The model containing AdapterLayer modules
        """
        if not self.gspo_adapter_enabled:
            return
        
        # Compute average quality across groups
        if not quality_scores:
            return
            
        avg_quality = torch.stack([q.mean() for q in quality_scores]).mean()
        
        # Collect all AdapterLayer modules from the model
        adapters = []
        for name, module in model.named_modules():
            # Check if this is an AdapterLayer with GSPO enabled
            if hasattr(module, 'update_quality_feedback') and hasattr(module, 'gspo_enabled'):
                if module.gspo_enabled:
                    adapters.append((name, module))
        
        if not adapters:
            return
        
        # Update each adapter with quality feedback
        for name, adapter in adapters:
            adapter.update_quality_feedback(avg_quality)
        
        # Log adapter quality for analysis
        self.adapter_quality_log.append({
            'epoch': self.current_epoch,
            'avg_quality': avg_quality.item(),
            'num_adapters': len(adapters),
        })
    
    def get_adapter_stats(self, model) -> Dict:
        """
        Get GSPO statistics from all adapters in the model.
        
        Args:
            model: The model containing AdapterLayer modules
            
        Returns:
            Dictionary with adapter statistics
        """
        stats = {}
        for name, module in model.named_modules():
            if hasattr(module, 'get_gspo_stats'):
                adapter_stats = module.get_gspo_stats()
                if adapter_stats:
                    stats[name] = adapter_stats
        return stats
    
    def update_lora_attention_feedback(
        self,
        quality_scores: List[torch.Tensor],
        model
    ):
        """
        GSPO-LoRA Extension: Update LoRA layer quality history based on GSPO feedback.
        
        This method is called after each GSPO training step to provide quality feedback
        to all LoRALinear modules (in Decoder Attention) that have GSPO enabled.
        
        The feedback mechanism allows LoRA layers to dynamically adjust their contribution
        based on the quality of predictions, following the GSPO principle of
        quality-aware optimization.
        
        Args:
            quality_scores: List of quality score tensors from each group [G] of [B]
            model: The model containing LoRALinear modules (with gspo_enabled=True)
        """
        if not self.gspo_lora_attention_enabled:
            return
        
        # Compute average quality across groups
        if not quality_scores:
            return
            
        avg_quality = torch.stack([q.mean() for q in quality_scores]).mean()
        
        # Import LoRALinear to check module type
        # Note: We check for the method instead of type to avoid circular imports
        lora_layers = []
        for name, module in model.named_modules():
            # Check if this is a LoRALinear with GSPO enabled
            # LoRALinear has: update_quality_feedback, gspo_enabled, r > 0
            if (hasattr(module, 'update_quality_feedback') and 
                hasattr(module, 'gspo_enabled') and 
                hasattr(module, 'r') and
                module.gspo_enabled and
                module.r > 0):
                lora_layers.append((name, module))
        
        if not lora_layers:
            return
        
        # Update each LoRA layer with quality feedback
        for name, lora in lora_layers:
            lora.update_quality_feedback(avg_quality)
        
        # Log LoRA quality for analysis
        self.lora_quality_log.append({
            'epoch': self.current_epoch,
            'avg_quality': avg_quality.item(),
            'num_lora_layers': len(lora_layers),
        })
    
    def get_lora_stats(self, model) -> Dict:
        """
        Get GSPO statistics from all LoRA layers in the model.
        
        Args:
            model: The model containing LoRALinear modules
            
        Returns:
            Dictionary with LoRA statistics (quality_history, contribution_score, etc.)
        """
        stats = {}
        for name, module in model.named_modules():
            if (hasattr(module, 'get_gspo_stats') and 
                hasattr(module, 'r') and 
                hasattr(module, 'gspo_enabled')):
                if module.gspo_enabled and module.r > 0:
                    lora_stats = module.get_gspo_stats()
                    if lora_stats:
                        stats[name] = lora_stats
        return stats
    
    def get_statistics(self) -> Dict:
        """Get training statistics for analysis."""
        return {
            'expert_performance': dict(self.expert_performance_log),
            'group_quality_history': self.group_quality_history,
            'adapter_quality_log': self.adapter_quality_log,
            'lora_quality_log': self.lora_quality_log,  # NEW
        }
    
    def reset_statistics(self):
        """Reset accumulated statistics."""
        self.expert_performance_log.clear()
        self.group_quality_history.clear()
        self.adapter_quality_log.clear()
        self.lora_quality_log.clear()  # NEW


```

```python
*** Begin Patch
*** Update File: /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA/gspo_trainer.py
@@
-        group_predictions = []
+        group_predictions = []
+        group_class_logits = []
         group_quality_scores = []
         group_base_quality_scores = []
         group_smooth_losses = []
         group_boundary_losses = []
         group_thin_rewards = []
@@
-            with torch.set_grad_enabled(True):
-                pred_masks, moe_loss, selected_experts = forward_fn(images)
-            
-            pred_probs, gt_probs = self._prepare_binary_masks(pred_masks, masks_gt)
+            with torch.set_grad_enabled(True):
+                pred_output, moe_loss, selected_experts = forward_fn(images)
+            
+            # pred_output can be logits tensor or dict/tuple
+            if isinstance(pred_output, dict):
+                pred_masks = pred_output.get("logits", pred_output.get("masks", None))
+                class_logits = pred_output.get("class_logits", None)
+            elif isinstance(pred_output, (tuple, list)) and len(pred_output) >= 2:
+                pred_masks, class_logits = pred_output[0], pred_output[1]
+            else:
+                pred_masks, class_logits = pred_output, None
+            
+            pred_probs, gt_probs = self._prepare_binary_masks(pred_masks, masks_gt, class_logits=class_logits)
             smooth_loss = self.compute_smoothness_loss(pred_probs)
             boundary_loss = self.compute_boundary_loss(pred_probs, gt_probs)
             thin_reward = self.compute_thin_reward(pred_probs, gt_probs)
 
             # Compute quality scores
             quality = self.compute_segmentation_quality(
-                pred_masks, masks_gt, metric=self.quality_metric
+                pred_masks, masks_gt, metric=self.quality_metric, class_logits=class_logits
             )
@@
-            seg_loss = loss_fn(pred_masks, masks_gt)
+            # loss_fn can handle Mask2Former via class_logits
+            seg_loss = loss_fn((pred_masks, class_logits), masks_gt)
             seg_loss = (
                 seg_loss
                 + self.lambda_smooth * smooth_loss.mean()
                 + self.lambda_boundary * boundary_loss.mean()
             )
@@
-            group_predictions.append(pred_masks)
+            group_predictions.append(pred_masks)
+            group_class_logits.append(class_logits)
             group_quality_scores.append(shaped_quality)
             group_base_quality_scores.append(quality)
             group_smooth_losses.append(smooth_loss.detach())
             group_boundary_losses.append(boundary_loss.detach())
             group_thin_rewards.append(thin_reward.detach())
@@
-        for g in range(G):
+        for g in range(G):
             advantage = advantages[g]  # [B]
@@
-            group_moe_losses.append(moe_loss)
+            group_moe_losses.append(moe_loss)
*** End Patch
```

```python
"""
GSPO (Group Sequence Policy Optimization) Trainer for Conv-LoRA, Adapters, and Decoder LoRA

This module implements the GSPO training strategy that enhances Conv-LoRA's
MoE mechanism through group-level optimization and quality-aware feedback.

Extended to support:
1. Encoder Adapters (Phase 1-3):
   - Phase 1: Unified advantage weighting (adapters receive same quality-weighted gradients)
   - Phase 2: Adapter scale adaptation (contribution score based on quality history)
   - Phase 3: Adapter MoE architecture (optional, multiple adapter variants)

2. Decoder LoRA on Attention (NEW):
   - Phase 1: Quality history tracking for LoRA layers
   - Phase 2: Dynamic scaling adaptation based on quality feedback

Key components:
1. Group-level sampling: Generate multiple predictions per image
2. Advantage function: Compute relative quality within groups
3. Contrastive loss: Pull high-quality predictions together, push low-quality apart
4. Quality feedback: Update expert selection, adapter quality, AND LoRA quality based on actual performance
"""

import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np


class GSPOConvLoRATrainer:
    """
    GSPO-enhanced trainer for Conv-LoRA semantic segmentation.
    
    This trainer implements group-level policy optimization inspired by GSPO,
    adapted for the Conv-LoRA MoE architecture in semantic segmentation tasks.
    
    Args:
        predictor: The MultiModalPredictor instance
        group_size: Number of predictions to generate per image in a group
        warmup_epochs: Number of epochs to train without GSPO before enabling it
        contrastive_weight: Weight for the contrastive loss component
        quality_metric: Metric to use for quality evaluation ('iou', 'dice', 'both')
        advantage_temperature: Temperature for computing advantage weights
    """
    
    def __init__(
        self,
        predictor,
        group_size: int = 4,
        warmup_epochs: int = 5,
        contrastive_weight: float = 0.1,
        quality_metric: str = 'iou',
        advantage_temperature: float = 5.0,
        lambda_smooth: float = 0.1,
        lambda_boundary: float = 0.3,
        w_boundary: float = 0.3,
        w_smooth: float = 0.1,
        w_thin: float = 0.05,
        # GSPO-Adapter extension parameters
        gspo_adapter_enabled: bool = False,
        gspo_adapter_momentum: float = 0.9,
        # GSPO-LoRA on Attention extension parameters (NEW)
        gspo_lora_attention_enabled: bool = False,
        gspo_lora_attention_momentum: float = 0.9,
        allow_semisup: bool = False,
    ):
        self.predictor = predictor
        self.group_size = group_size
        self.warmup_epochs = warmup_epochs
        self.contrastive_weight = contrastive_weight
        self.quality_metric = quality_metric
        self.advantage_temperature = advantage_temperature
        # Reward / loss shaping hyper-parameters
        self.lambda_smooth = lambda_smooth
        self.lambda_boundary = lambda_boundary
        self.w_boundary = w_boundary
        self.w_smooth = w_smooth
        self.w_thin = w_thin
        
        # GSPO-Adapter extension (Phase 1-2)
        self.gspo_adapter_enabled = gspo_adapter_enabled
        self.gspo_adapter_momentum = gspo_adapter_momentum
        
        # GSPO-LoRA on Attention extension (NEW)
        self.gspo_lora_attention_enabled = gspo_lora_attention_enabled
        self.gspo_lora_attention_momentum = gspo_lora_attention_momentum
        self.allow_semisup = allow_semisup
        
        # Statistics tracking
        self.expert_performance_log = defaultdict(list)
        self.group_quality_history = []
        self.adapter_quality_log = []  # Track adapter quality over time
        self.lora_quality_log = []  # Track LoRA quality over time (NEW)
        self.current_epoch = 0
        
    def is_gspo_active(self, epoch: int) -> bool:
        """Check if GSPO should be active for the given epoch."""
        return epoch >= self.warmup_epochs
    
    def compute_segmentation_quality(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        metric: str = 'iou',
        class_logits: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute segmentation quality metrics.
        
        - If class_logits is provided (Mask2Former-style), compute per-class
          IoU/Dice excluding background, then average over foreground classes
          present in GT. This preserves multi类信息（ACDC）而非二值化。
        - Otherwise fallback to binary IoU/Dice with threshold 0.5.
        """
        eps = 1e-6
        
        if class_logits is not None:
            # Mask2Former multi-class: aggregate semantic probabilities
            # class_logits: [B, Q, C+1], pred_masks: [B, Q, H, W]
            class_prob = torch.softmax(class_logits, dim=-1)[..., :-1]  # drop background
            mask_prob = torch.sigmoid(pred_masks)
            sem_prob = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob).clamp(min=0.0, max=1.0)
            B, C, H, W = sem_prob.shape
            
            # GT one-hot (exclude background channel 0)
            gt_onehot = torch.nn.functional.one_hot(gt_masks.long(), num_classes=class_logits.shape[-1])
            gt_onehot = gt_onehot[..., 1:].permute(0, 3, 1, 2).float()  # [B, C, H, W]
            
            inter = (sem_prob * gt_onehot).sum(dim=(2, 3))
            union = sem_prob.sum(dim=(2, 3)) + gt_onehot.sum(dim=(2, 3)) - inter
            iou = (inter + eps) / (union + eps)
            dice = (2 * inter + eps) / (sem_prob.sum(dim=(2, 3)) + gt_onehot.sum(dim=(2, 3)) + eps)
            
            if metric == 'iou':
                score = iou
            elif metric == 'dice':
                score = dice
            else:
                score = (iou + dice) / 2.0
            
            # 平均仅在 GT 中出现的前景类；若某样本全背景，退化为全类平均
            valid = gt_onehot.sum(dim=(2, 3)) > 0  # [B, C]
            valid_count = valid.sum(dim=1).clamp(min=1)
            score = (score * valid.float()).sum(dim=1) / valid_count
            return score
        
        # ---- Binary fallback (原逻辑) ----
        if pred_masks.dim() == 4 and pred_masks.shape[1] > 1:
            pred_masks = torch.argmax(pred_masks, dim=1)
        if gt_masks.dim() == 4 and gt_masks.shape[1] > 1:
            gt_masks = torch.argmax(gt_masks, dim=1)
        
        pred_flat = pred_masks.view(pred_masks.shape[0], -1).float()
        gt_flat = gt_masks.view(gt_masks.shape[0], -1).float()
        pred_flat = (pred_flat > 0.5).float()
        gt_flat = (gt_flat > 0.5).float()
        
        intersection = (pred_flat * gt_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1) - intersection
        iou = (intersection + eps) / (union + eps)
        
        if metric == 'iou':
            return iou
        
        dice = (2 * intersection + eps) / (pred_flat.sum(dim=1) + gt_flat.sum(dim=1) + eps)
        
        if metric == 'dice':
            return dice
        return (iou + dice) / 2.0

    def _prepare_binary_masks(
        self, pred_masks: torch.Tensor, gt_masks: torch.Tensor, class_logits: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert logits/probabilities to single-channel foreground probability
        maps for boundary/smoothness computation.
        
        - For Mask2Former (class_logits provided): sum foreground class probs.
        - For multi-channel logits without class_logits: take max prob as foreground.
        - For binary logits: sigmoid.
        """
        # pred_masks: logits; gt_masks: integer labels or one-hot
        if class_logits is not None:
            # Mask2Former aggregation to semantic foreground prob (sum over classes)
            class_prob = torch.softmax(class_logits, dim=-1)[..., :-1]  # drop background
            mask_prob = torch.sigmoid(pred_masks)
            sem_prob = torch.einsum("bqc,bqhw->bchw", class_prob, mask_prob)
            pred_probs = sem_prob.sum(dim=1, keepdim=True).clamp(min=0.0, max=1.0)  # [B,1,H,W]
            if gt_masks.dim() == 3:
                gt_fg = (gt_masks > 0).float()
            else:
                gt_fg = torch.argmax(gt_masks, dim=1).float()
            gt_probs = gt_fg.unsqueeze(1)
            return pred_probs, gt_probs
        
        # Ensure channel dimension
        if pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(1)
        if gt_masks.dim() == 3:
            gt_masks = gt_masks.unsqueeze(1)

        if pred_masks.shape[1] > 1:
            pred_probs = torch.softmax(pred_masks, dim=1).max(dim=1, keepdim=True).values
        else:
            pred_probs = torch.sigmoid(pred_masks)

        if gt_masks.shape[1] > 1:
            gt_probs = torch.argmax(gt_masks, dim=1, keepdim=True).float()
        else:
            gt_probs = gt_masks.float()

        return pred_probs, gt_probs

    def _boundary_map(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Approximate boundary map via spatial gradients.
        """
        grad_x = torch.abs(masks[..., :, 1:] - masks[..., :, :-1])
        grad_y = torch.abs(masks[..., 1:, :] - masks[..., :-1, :])
        grad_x = torch.nn.functional.pad(grad_x, (0, 1, 0, 0))
        grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1))
        return torch.clamp(grad_x + grad_y, 0.0, 1.0)

    def compute_smoothness_loss(self, pred_probs: torch.Tensor) -> torch.Tensor:
        """
        Laplacian smoothness loss (per-sample).
        """
        lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=pred_probs.device, dtype=pred_probs.dtype)
        lap_kernel = lap_kernel.view(1, 1, 3, 3)
        lap = torch.nn.functional.conv2d(pred_probs, lap_kernel, padding=1)
        return (lap ** 2).mean(dim=(1, 2, 3))

    def compute_boundary_loss(self, pred_probs: torch.Tensor, gt_probs: torch.Tensor) -> torch.Tensor:
        """
        L1 difference between predicted and GT boundary maps (per-sample).
        """
        pred_edge = self._boundary_map(pred_probs)
        gt_edge = self._boundary_map(gt_probs)
        return torch.mean(torch.abs(pred_edge - gt_edge), dim=(1, 2, 3))

    def compute_thin_reward(self, pred_probs: torch.Tensor, gt_probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Dice-style reward on boundary skeletons (per-sample).
        """
        pred_edge = self._boundary_map(pred_probs)
        gt_edge = self._boundary_map(gt_probs)
        intersection = (pred_edge * gt_edge).flatten(1).sum(dim=1)
        union = pred_edge.flatten(1).sum(dim=1) + gt_edge.flatten(1).sum(dim=1)
        return (2 * intersection + eps) / (union + eps)
    
    def gspo_group_training_step(
        self,
        images: torch.Tensor,
        masks_gt: torch.Tensor,
        forward_fn,
        loss_fn,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        GSPO core training step: generate group predictions and compute weighted loss.
        
        Args:
            images: Input images [B, C, H, W]
            masks_gt: Ground truth masks [B, C, H, W] or [B, H, W]
            forward_fn: Forward function that returns (predictions, moe_loss, selected_experts)
            loss_fn: Loss function for segmentation
            
        Returns:
            total_loss: Combined loss for backpropagation
            metrics: Dictionary of metrics for logging
        """
        B = images.shape[0]
        G = self.group_size
        
        group_predictions = []
        group_class_logits = []
        group_quality_scores = []
        group_base_quality_scores = []
        group_smooth_losses = []
        group_boundary_losses = []
        group_thin_rewards = []
        group_moe_losses = []
        group_selected_experts = []
        group_seg_losses = []
        
        # 1. Generate group predictions with variations
        for g in range(G):
            # Apply slight variations via dropout to create diversity
            # (dropout is automatically different across forward passes)
            with torch.set_grad_enabled(True):
                pred_out, moe_loss, selected_experts = forward_fn(images)
            
            # Normalize outputs: support dict/tuple for Mask2Former
            if isinstance(pred_out, dict):
                pred_masks = pred_out.get("logits", pred_out.get("masks", None))
                class_logits = pred_out.get("class_logits", None)
            elif isinstance(pred_out, (tuple, list)) and len(pred_out) >= 2:
                pred_masks, class_logits = pred_out[0], pred_out[1]
            else:
                pred_masks, class_logits = pred_out, None
            
            pred_probs, gt_probs = self._prepare_binary_masks(pred_masks, masks_gt, class_logits=class_logits)
            smooth_loss = self.compute_smoothness_loss(pred_probs)
            boundary_loss = self.compute_boundary_loss(pred_probs, gt_probs)
            thin_reward = self.compute_thin_reward(pred_probs, gt_probs)

            # Compute quality scores
            quality = self.compute_segmentation_quality(
                pred_masks, masks_gt, metric=self.quality_metric, class_logits=class_logits
            )
            shaped_quality = (
                quality
                + self.w_boundary * (1.0 - boundary_loss.detach())
                - self.w_smooth * smooth_loss.detach()
                + self.w_thin * thin_reward.detach()
            )
            
            # Compute segmentation loss
            seg_loss = loss_fn((pred_masks, class_logits), masks_gt)
            seg_loss = (
                seg_loss
                + self.lambda_smooth * smooth_loss.mean()
                + self.lambda_boundary * boundary_loss.mean()
            )
            
            group_predictions.append(pred_masks)
            group_class_logits.append(class_logits)
            group_quality_scores.append(shaped_quality)
            group_base_quality_scores.append(quality)
            group_smooth_losses.append(smooth_loss.detach())
            group_boundary_losses.append(boundary_loss.detach())
            group_thin_rewards.append(thin_reward.detach())
            group_moe_losses.append(moe_loss)
            group_selected_experts.append(selected_experts)
            group_seg_losses.append(seg_loss)
        
        # 2. Compute group-level advantage function (GSPO core)
        quality_tensor = torch.stack(group_quality_scores)  # [G, B]
        baseline = quality_tensor.mean(dim=0, keepdim=True)  # [1, B]
        advantages = quality_tensor - baseline  # [G, B]
        
        # 3. GSPO weighted loss
        total_seg_loss = 0
        total_moe_loss = 0
        
        for g in range(G):
            advantage = advantages[g]  # [B]
            
            # Compute adaptive weights based on advantage
            # Higher advantage -> higher weight (stronger gradient)
            weight = torch.sigmoid(advantage * self.advantage_temperature)
            
            # Weighted segmentation loss
            seg_loss = group_seg_losses[g]
            if seg_loss.dim() == 0:  # scalar loss
                weighted_seg_loss = seg_loss * weight.mean()
            else:  # per-sample loss
                weighted_seg_loss = (seg_loss * weight).mean()
            
            total_seg_loss += weighted_seg_loss
            total_moe_loss += group_moe_losses[g]
        
        total_seg_loss /= G
        total_moe_loss /= G
        
        # 4. GSPO contrastive loss
        contrastive_loss = self.compute_group_contrastive_loss(
            group_predictions, group_quality_scores
        )
        
        # 5. Combine losses
        total_loss = total_seg_loss + 1e-2 * total_moe_loss + self.contrastive_weight * contrastive_loss
        
        # 6. Collect metrics
        metrics = {
            'loss': total_loss.item(),
            'seg_loss': total_seg_loss.item(),
            'moe_loss': total_moe_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'avg_quality': quality_tensor.mean().item(),
            'avg_quality_base': torch.stack(group_base_quality_scores).mean().item(),
            'avg_smooth_loss': torch.stack(group_smooth_losses).mean().item(),
            'avg_boundary_loss': torch.stack(group_boundary_losses).mean().item(),
            'avg_thin_reward': torch.stack(group_thin_rewards).mean().item(),
            'quality_variance': quality_tensor.var().item(),
            'max_advantage': advantages.max().item(),
            'min_advantage': advantages.min().item(),
        }
        
        # 7. Store group quality for analysis
        self.group_quality_history.append({
            'epoch': self.current_epoch,
            'qualities': quality_tensor.detach().cpu().numpy(),
            'advantages': advantages.detach().cpu().numpy(),
        })
        
        return total_loss, metrics, group_selected_experts
    
    def compute_group_contrastive_loss(
        self,
        predictions: List[torch.Tensor],
        quality_scores: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        GSPO-inspired contrastive loss.
        
        The idea is to make predictions with similar quality have similar features,
        while predictions with different quality should have different features.
        
        Args:
            predictions: List of G prediction tensors [B, C, H, W]
            quality_scores: List of G quality tensors [B]
            
        Returns:
            contrastive_loss: Scalar loss
        """
        G = len(predictions)
        
        if G < 2:
            return torch.tensor(0.0, device=predictions[0].device)
        
        quality_tensor = torch.stack(quality_scores)  # [G, B]
        
        # Extract features from predictions (global average pooling)
        features = []
        for pred in predictions:
            # Apply global average pooling
            feat = F.adaptive_avg_pool2d(pred, (1, 1)).flatten(1)  # [B, C]
            feat = F.normalize(feat, dim=1)  # Normalize for cosine similarity
            features.append(feat)
        features = torch.stack(features)  # [G, B, C]
        
        # Compute pairwise cosine similarities
        # Reshape for batch matrix multiplication
        feat_flat = features.view(G, -1)  # [G, B*C]
        similarities = torch.mm(feat_flat, feat_flat.t())  # [G, G]
        
        # Normalize by number of features
        similarities = similarities / (features.shape[1] * features.shape[2])
        
        # Compute quality differences
        quality_mean = quality_tensor.mean(dim=1)  # [G]
        quality_diff = torch.abs(
            quality_mean.unsqueeze(0) - quality_mean.unsqueeze(1)
        )  # [G, G]
        
        # Contrastive objective:
        # - If quality is similar (small diff), encourage high similarity
        # - If quality is different (large diff), encourage low similarity
        target_similarity = 1.0 - quality_diff / (quality_diff.max() + 1e-6)
        
        # MSE loss between actual and target similarities
        contrastive_loss = F.mse_loss(similarities, target_similarity)
        
        return contrastive_loss
    
    def update_expert_feedback(
        self,
        selected_experts_groups: List[Optional[torch.Tensor]],
        quality_scores: List[torch.Tensor],
        model
    ):
        """
        Update expert quality history in the model based on GSPO feedback.
        
        Args:
            selected_experts_groups: List of selected expert tensors from each group
            quality_scores: List of quality score tensors from each group
            model: The model containing MoEGate modules
        """
        if not selected_experts_groups or selected_experts_groups[0] is None:
            return
        
        # Collect all MoEGate modules from the model
        moe_gates = []
        for module in model.modules():
            if hasattr(module, 'update_quality_history'):
                moe_gates.append(module)
        
        if not moe_gates:
            return
        
        # Update each MoE gate with quality feedback
        for g_idx, (selected_experts, quality) in enumerate(zip(selected_experts_groups, quality_scores)):
            if selected_experts is not None:
                for moe_gate in moe_gates:
                    moe_gate.update_quality_history(selected_experts, quality)
    
    def update_adapter_feedback(
        self,
        quality_scores: List[torch.Tensor],
        model
    ):
        """
        GSPO-Adapter Extension: Update adapter quality history based on GSPO feedback.
        
        This method is called after each GSPO training step to provide quality feedback
        to all AdapterLayer modules in the model.
        
        Args:
            quality_scores: List of quality score tensors from each group [G] of [B]
            model: The model containing AdapterLayer modules
        """
        if not self.gspo_adapter_enabled:
            return
        
        # Compute average quality across groups
        if not quality_scores:
            return
            
        avg_quality = torch.stack([q.mean() for q in quality_scores]).mean()
        
        # Collect all AdapterLayer modules from the model
        adapters = []
        for name, module in model.named_modules():
            # Check if this is an AdapterLayer with GSPO enabled
            if hasattr(module, 'update_quality_feedback') and hasattr(module, 'gspo_enabled'):
                if module.gspo_enabled:
                    adapters.append((name, module))
        
        if not adapters:
            return
        
        # Update each adapter with quality feedback
        for name, adapter in adapters:
            adapter.update_quality_feedback(avg_quality)
        
        # Log adapter quality for analysis
        self.adapter_quality_log.append({
            'epoch': self.current_epoch,
            'avg_quality': avg_quality.item(),
            'num_adapters': len(adapters),
        })
    
    def get_adapter_stats(self, model) -> Dict:
        """
        Get GSPO statistics from all adapters in the model.
        
        Args:
            model: The model containing AdapterLayer modules
            
        Returns:
            Dictionary with adapter statistics
        """
        stats = {}
        for name, module in model.named_modules():
            if hasattr(module, 'get_gspo_stats'):
                adapter_stats = module.get_gspo_stats()
                if adapter_stats:
                    stats[name] = adapter_stats
        return stats
    
    def update_lora_attention_feedback(
        self,
        quality_scores: List[torch.Tensor],
        model
    ):
        """
        GSPO-LoRA Extension: Update LoRA layer quality history based on GSPO feedback.
        
        This method is called after each GSPO training step to provide quality feedback
        to all LoRALinear modules (in Decoder Attention) that have GSPO enabled.
        
        The feedback mechanism allows LoRA layers to dynamically adjust their contribution
        based on the quality of predictions, following the GSPO principle of
        quality-aware optimization.
        
        Args:
            quality_scores: List of quality score tensors from each group [G] of [B]
            model: The model containing LoRALinear modules (with gspo_enabled=True)
        """
        if not self.gspo_lora_attention_enabled:
            return
        
        # Compute average quality across groups
        if not quality_scores:
            return
            
        avg_quality = torch.stack([q.mean() for q in quality_scores]).mean()
        
        # Import LoRALinear to check module type
        # Note: We check for the method instead of type to avoid circular imports
        lora_layers = []
        for name, module in model.named_modules():
            # Check if this is a LoRALinear with GSPO enabled
            # LoRALinear has: update_quality_feedback, gspo_enabled, r > 0
            if (hasattr(module, 'update_quality_feedback') and 
                hasattr(module, 'gspo_enabled') and 
                hasattr(module, 'r') and
                module.gspo_enabled and
                module.r > 0):
                lora_layers.append((name, module))
        
        if not lora_layers:
            return
        
        # Update each LoRA layer with quality feedback
        for name, lora in lora_layers:
            lora.update_quality_feedback(avg_quality)
        
        # Log LoRA quality for analysis
        self.lora_quality_log.append({
            'epoch': self.current_epoch,
            'avg_quality': avg_quality.item(),
            'num_lora_layers': len(lora_layers),
        })
    
    def get_lora_stats(self, model) -> Dict:
        """
        Get GSPO statistics from all LoRA layers in the model.
        
        Args:
            model: The model containing LoRALinear modules
            
        Returns:
            Dictionary with LoRA statistics (quality_history, contribution_score, etc.)
        """
        stats = {}
        for name, module in model.named_modules():
            if (hasattr(module, 'get_gspo_stats') and 
                hasattr(module, 'r') and 
                hasattr(module, 'gspo_enabled')):
                if module.gspo_enabled and module.r > 0:
                    lora_stats = module.get_gspo_stats()
                    if lora_stats:
                        stats[name] = lora_stats
        return stats
    
    def get_statistics(self) -> Dict:
        """Get training statistics for analysis."""
        return {
            'expert_performance': dict(self.expert_performance_log),
            'group_quality_history': self.group_quality_history,
            'adapter_quality_log': self.adapter_quality_log,
            'lora_quality_log': self.lora_quality_log,  # NEW
        }
    
    def reset_statistics(self):
        """Reset accumulated statistics."""
        self.expert_performance_log.clear()
        self.group_quality_history.clear()
        self.adapter_quality_log.clear()
        self.lora_quality_log.clear()  # NEW


```

```python
import logging
import hashlib
from typing import Callable, Dict, Optional

import torch
import torchmetrics
import torch.nn.functional as F
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
    
    def __init__(self, *args, gspo_trainer=None, train_box_prompt_cfg=None, train_bbox_predictor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gspo_trainer = gspo_trainer
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
    
    @staticmethod
    def _stable_uniform_0_1(key: str, seed: int = 0) -> float:
        """
        Stable deterministic pseudo-random number in [0, 1) from a string key + seed.
        Avoids python's randomized hash across processes.
        """
        h = hashlib.md5(f"{seed}::{key}".encode("utf-8")).digest()
        # use first 8 bytes as uint64
        x = int.from_bytes(h[:8], byteorder="little", signed=False)
        return (x % (10**12)) / float(10**12)
    
    def _get_labeled_mask_from_paths(self, image_paths, labeled_fraction: float, seed: int) -> torch.Tensor:
        """
        image_paths: list[str] (len=B)
        returns: bool tensor (B,)
        """
        labeled_fraction = float(labeled_fraction)
        labeled_fraction = max(0.0, min(1.0, labeled_fraction))
        if labeled_fraction >= 1.0:
            return torch.ones(len(image_paths), dtype=torch.bool, device=self.device)
        if labeled_fraction <= 0.0:
            return torch.zeros(len(image_paths), dtype=torch.bool, device=self.device)
        flags = []
        for p in image_paths:
            p = p or ""
            u = self._stable_uniform_0_1(p, seed=seed)
            flags.append(u < labeled_fraction)
        return torch.tensor(flags, dtype=torch.bool, device=self.device)
    
    @staticmethod
    def _compute_boxes_from_mask(mask: torch.Tensor) -> torch.Tensor:
        """mask: (B,H,W) or (B,1,H,W) -> boxes (B,4) in pixel coords (x1,y1,x2,y2)."""
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
    
    @staticmethod
    def _jitter_boxes_outward(
        boxes: torch.Tensor,
        h: int,
        w: int,
        amount: float,
        mode: str = "box",  # box / image / pixel
        outward_only: bool = True,
    ) -> torch.Tensor:
        """
        Jitter boxes to simulate coarse human boxes.
        - outward_only=True: never shrink vs original (always expand or keep).
        - amount meaning:
          - mode=box: fraction of box width/height
          - mode=image: fraction of image width/height
          - mode=pixel: absolute pixels
        """
        jittered = boxes.clone()
        mode = (mode or "box").lower()
        amount = float(amount)
        amount = max(0.0, amount)
        for i in range(jittered.shape[0]):
            x1, y1, x2, y2 = jittered[i]
            if x2 <= x1 or y2 <= y1:
                continue
            bw = (x2 - x1).clamp(min=1.0)
            bh = (y2 - y1).clamp(min=1.0)
            if mode == "box":
                dx = bw * amount
                dy = bh * amount
            elif mode == "image":
                dx = float(w) * amount
                dy = float(h) * amount
                dx = torch.tensor(dx, device=boxes.device, dtype=torch.float32)
                dy = torch.tensor(dy, device=boxes.device, dtype=torch.float32)
            elif mode == "pixel":
                dx = torch.tensor(amount, device=boxes.device, dtype=torch.float32)
                dy = torch.tensor(amount, device=boxes.device, dtype=torch.float32)
            else:
                # fallback to box
                dx = bw * amount
                dy = bh * amount
            
            # sample non-negative deltas (mostly outward expansion)
            # use uniform [0, dx] / [0, dy]
            ex1 = torch.rand(1, device=boxes.device) * dx
            ey1 = torch.rand(1, device=boxes.device) * dy
            ex2 = torch.rand(1, device=boxes.device) * dx
            ey2 = torch.rand(1, device=boxes.device) * dy
            
            if outward_only:
                # outward: x1 decreases, y1 decreases, x2 increases, y2 increases
                x1n = x1 - ex1
                y1n = y1 - ey1
                x2n = x2 + ex2
                y2n = y2 + ey2
            else:
                # allow inward with small probability by random sign
                s1 = torch.where(torch.rand(1, device=boxes.device) < 0.9, -1.0, 1.0)  # x1: outward is -1
                t1 = torch.where(torch.rand(1, device=boxes.device) < 0.9, -1.0, 1.0)  # y1
                s2 = torch.where(torch.rand(1, device=boxes.device) < 0.9, 1.0, -1.0)   # x2: outward is +1
                t2 = torch.where(torch.rand(1, device=boxes.device) < 0.9, 1.0, -1.0)   # y2
                x1n = x1 + s1 * ex1
                y1n = y1 + t1 * ey1
                x2n = x2 + s2 * ex2
                y2n = y2 + t2 * ey2
            
            x1n = torch.clamp(x1n, 0, w - 1)
            y1n = torch.clamp(y1n, 0, h - 1)
            x2n = torch.clamp(x2n, 0, w - 1)
            y2n = torch.clamp(y2n, 0, h - 1)
            # ensure valid ordering
            jittered[i] = torch.stack(
                [torch.min(x1n, x2n), torch.min(y1n, y2n), torch.max(x1n, x2n), torch.max(y1n, y2n)]
            ).squeeze()
        return jittered
    
    @staticmethod
    def _structure_loss_per_sample(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Per-sample StructureLoss (same as optim/losses/structure_loss.py) without final mean.
        Returns tensor of shape (B,).
        """
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
        wbce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)).clamp(min=1e-6)
        probs = torch.sigmoid(logits)
        inter = ((probs * target) * weit).sum(dim=(2, 3))
        union = ((probs + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).view(-1)
    
    @staticmethod
    def _weak_box_losses(
        logits: torch.Tensor,  # (B,1,H,W) or (B,H,W)
        boxes: torch.Tensor,   # (B,4) x1,y1,x2,y2 in pixel coords
        outside_weight: float = 1.0,
        entropy_weight: float = 0.05,
        tv_weight: float = 0.0,
    ) -> torch.Tensor:
        """
        Box-only weak supervision loss:
        - outside-box should be background (0)
        - inside-box entropy minimization (push confident predictions without assuming full positive)
        - optional smoothness (TV) inside box
        Returns scalar loss.
        """
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        b, _, h, w = logits.shape
        device = logits.device
        outside_weight = float(outside_weight)
        entropy_weight = float(entropy_weight)
        tv_weight = float(tv_weight)
        
        total = logits.new_tensor(0.0)
        count = 0
        for i in range(b):
            x1, y1, x2, y2 = boxes[i]
            if x2 <= x1 or y2 <= y1:
                continue
            # integer pixel bounds (inclusive)
            x1i = int(torch.clamp(x1.round(), 0, w - 1).item())
            y1i = int(torch.clamp(y1.round(), 0, h - 1).item())
            x2i = int(torch.clamp(x2.round(), 0, w - 1).item())
            y2i = int(torch.clamp(y2.round(), 0, h - 1).item())
            if x2i <= x1i or y2i <= y1i:
                continue
            
            logit = logits[i : i + 1]  # (1,1,H,W)
            outside_mask = torch.ones((1, 1, h, w), device=device, dtype=logit.dtype)
            outside_mask[:, :, y1i : y2i + 1, x1i : x2i + 1] = 0.0
            inside_mask = 1.0 - outside_mask
            
            # Outside-box background constraint
            if outside_weight > 0:
                zeros = torch.zeros_like(logit)
                bce = F.binary_cross_entropy_with_logits(logit, zeros, reduction="none")
                outside_loss = (bce * outside_mask).sum() / outside_mask.sum().clamp(min=1.0)
            else:
                outside_loss = logit.new_tensor(0.0)
            
            # Inside-box entropy minimization
            if entropy_weight > 0:
                p = torch.sigmoid(logit).clamp(min=1e-6, max=1 - 1e-6)
                ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
                ent_loss = (ent * inside_mask).sum() / inside_mask.sum().clamp(min=1.0)
            else:
                ent_loss = logit.new_tensor(0.0)
            
            # Total variation smoothness inside the box
            if tv_weight > 0:
                p = torch.sigmoid(logit)
                dy = torch.abs(p[:, :, 1:, :] - p[:, :, :-1, :])
                dx = torch.abs(p[:, :, :, 1:] - p[:, :, :, :-1])
                # match shapes with masks
                inside_y = inside_mask[:, :, 1:, :]
                inside_x = inside_mask[:, :, :, 1:]
                tv = (dy * inside_y).sum() / inside_y.sum().clamp(min=1.0) + (dx * inside_x).sum() / inside_x.sum().clamp(min=1.0)
            else:
                tv = logit.new_tensor(0.0)
            
            total = total + outside_weight * outside_loss + entropy_weight * ent_loss + tv_weight * tv
            count += 1
        
        if count == 0:
            return logits.new_tensor(0.0)
        return total / float(count)

    def _predict_boxes_from_images(self, image_paths: list, target_h: int, target_w: int) -> torch.Tensor:
        """
        Wrapper around train_bbox_predictor to predict boxes in pixel coords and scale to target size.
        """
        import numpy as np
        from PIL import Image
        if self.train_bbox_predictor is None:
            raise ValueError("train_bbox_predictor is required for mode='predict'")
        b = len(image_paths)
        boxes = torch.zeros((b, 4), device=self.device, dtype=torch.float32)
        for i, img_path in enumerate(image_paths):
            if not img_path:
                continue
            try:
                bbox_px = self.train_bbox_predictor.predict(img_path)
                if bbox_px is None:
                    continue
                with Image.open(img_path) as img:
                    orig_w, orig_h = img.size
                scale_x = target_w / orig_w
                scale_y = target_h / orig_h
                x1 = bbox_px[0] * scale_x
                y1 = bbox_px[1] * scale_y
                x2 = bbox_px[2] * scale_x
                y2 = bbox_px[3] * scale_y
                x1 = max(0, min(x1, target_w - 1))
                y1 = max(0, min(y1, target_h - 1))
                x2 = max(0, min(x2, target_w - 1))
                y2 = max(0, min(y2, target_h - 1))
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                boxes[i] = torch.tensor([x1, y1, x2, y2], device=self.device, dtype=torch.float32)
            except Exception as e:
                logger.warning(f"Failed to predict bbox for {img_path}: {e}")
                continue
        return boxes

    def _apply_train_box_prompts(self, batch: Dict, label: torch.Tensor) -> None:
        """
        In-place box prompt injection for training (gt/noisy/mix/predict) and for semi/weak supervision.
        This is factored out so GSPO/standard paths can share the same augmentation.
        """
        cfg = self.train_box_prompt_cfg or {}
        mode = cfg.get("mode", "off")
        if mode == "off" and "semi_labeled_fraction" not in cfg:
            return

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
        else:
            noise_frac = float(cfg.get("noise_frac", 0.12))

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

        if box_mode in ["gt", "noisy"]:
            gt_boxes = self._compute_boxes_from_mask(label)
            h, w = label.shape[-2], label.shape[-1]
            if box_mode == "noisy":
                gt_boxes = _jitter_boxes(gt_boxes, h=h, w=w, frac=noise_frac)
            if not (gt_boxes.sum(dim=1) == 0).all():
                batch[self.model.box_key] = gt_boxes.unsqueeze(1)  # (B,1,4)
        elif box_mode == "predict":
            image_path_key = self.model.image_path_key
            image_paths = batch.get(image_path_key, [])
            if image_paths:
                h, w = label.shape[-2], label.shape[-1]
                pred_boxes = self._predict_boxes_from_images(
                    image_paths=image_paths, target_h=h, target_w=w
                )
                if not (pred_boxes.sum(dim=1) == 0).all():
                    batch[self.model.box_key] = pred_boxes.unsqueeze(1)  # (B,1,4)

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
                logits = per_output[LOGITS]
                
                # Handle SAM multi-class output format: combine mask and class predictions
                # SAM multi-class outputs: LOGITS [B, num_mask_tokens, H, W] + CLASS_LOGITS [B, num_mask_tokens, num_classes+1]
                # When num_mask_tokens == num_classes, we can use LOGITS directly as per-class masks
                # Otherwise, we need to combine mask and class predictions
                if CLASS_LOGITS in per_output:
                    mask_logits = logits  # [B, num_mask_tokens, H, W]
                    class_logits = per_output[CLASS_LOGITS]  # [B, num_mask_tokens, num_classes+1]
                    
                    # Apply semantic inference to combine mask and class predictions
                    # This follows Mask2Former's approach: semantic_mask = einsum(softmax(class), sigmoid(mask))
                    mask_cls = F.softmax(class_logits, dim=-1)[..., :-1]  # [B, num_mask_tokens, num_classes]
                    mask_pred = torch.sigmoid(mask_logits)  # [B, num_mask_tokens, H, W]
                    
                    # Combine: for each pixel, sum over mask tokens weighted by class probability
                    # semantic_mask[b, c, h, w] = sum_q(mask_cls[b, q, c] * mask_pred[b, q, h, w])
                    logits = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
                    
                    # Convert probabilities back to logits for CrossEntropyLoss (add small epsilon to avoid log(0))
                    logits = torch.log(logits.clamp(min=1e-7))
                
                loss += (
                    self.loss_func(
                        input=logits,
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
        
        # ---- Semi/weak supervision config (optional) ----
        cfg = self.train_box_prompt_cfg or {}
        semi_labeled_fraction = float(cfg.get("semi_labeled_fraction", 1.0))
        semi_labeled_seed = int(cfg.get("semi_labeled_seed", 0))
        weak_box_jitter_mode = str(cfg.get("weak_box_jitter_mode", "box"))
        weak_box_jitter_amount = float(cfg.get("weak_box_jitter_amount", cfg.get("noise_frac", 0.12)))
        weak_box_outward_only = bool(cfg.get("weak_box_outward_only", True))
        weak_outside_w = float(cfg.get("weak_loss_outside_weight", 1.0))
        weak_entropy_w = float(cfg.get("weak_loss_entropy_weight", 0.05))
        weak_tv_w = float(cfg.get("weak_loss_tv_weight", 0.0))
        
        # Determine which samples are "labeled" (for simulation) using stable hashing on image paths.
        image_paths = batch.get(self.model.image_path_key, [])
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        labeled_mask = None
        if self.training and semi_labeled_fraction < 1.0:
            labeled_mask = self._get_labeled_mask_from_paths(
                image_paths=image_paths, labeled_fraction=semi_labeled_fraction, seed=semi_labeled_seed
            )

        # -------- Train-time box prompt injection (configurable) --------
        # 控制项：self.train_box_prompt_cfg = {
        #   mode: off / gt / noisy / mix / predict,
        #   p_no, p_gt, p_noisy: 仅在 mode=mix 时生效并会归一化,
        #   noise_frac: 噪声扰动比例（相对 bbox 宽高）
        # }
        # mode=predict: 使用 self.train_bbox_predictor 预测框作为 box prompt
        if self.training:
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

            # If semi/weak enabled: for "unlabeled" samples, force weak coarse boxes generated from GT bbox with outward-biased jitter.
            if labeled_mask is not None and hasattr(self.model, "box_key"):
                gt_boxes_all = self._compute_boxes_from_mask(label)
                h, w = label.shape[-2], label.shape[-1]
                weak_boxes = self._jitter_boxes_outward(
                    gt_boxes_all,
                    h=h,
                    w=w,
                    amount=weak_box_jitter_amount,
                    mode=weak_box_jitter_mode,
                    outward_only=weak_box_outward_only,
                )
                # apply only to unlabeled samples
                if (~labeled_mask).any():
                    boxes_for_batch = gt_boxes_all.clone()
                    boxes_for_batch[~labeled_mask] = weak_boxes[~labeled_mask]
                    if not (boxes_for_batch.sum(dim=1) == 0).all():
                        batch[self.model.box_key] = boxes_for_batch.unsqueeze(1)  # (B,1,4)

            if box_mode in ["gt", "noisy"]:
                gt_boxes = self._compute_boxes_from_mask(label)
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
        
        # ---- Loss: fully-supervised OR semi/weak mixed ----
        if self.training and labeled_mask is not None:
            # Semi/weak supervision path
            if isinstance(self.loss_func, Mask2FormerLoss):
                # 1) 有标签子集：使用 Mask2FormerLoss
                sup_loss = output[self.model.prefix][LOGITS].new_tensor(0.0)
                if labeled_mask.any():
                    # 子集化输出与标签
                    output_labeled = {}
                    for key, per_output in output.items():
                        sub_out = {}
                        for sub_k, val in per_output.items():
                            # 仅当第 0 维与 batch 对齐时才子集化，避免标量/非 batch 维度触发索引错误
                            if torch.is_tensor(val) and val.ndim > 0 and val.shape[0] == label.shape[0]:
                                sub_out[sub_k] = val[labeled_mask]
                            else:
                                sub_out[sub_k] = val
                        output_labeled[key] = sub_out
                    mask_labels = batch.get(self.model.mask_label_key, [])
                    class_labels = batch.get(self.model.class_label_key, [])
                    if isinstance(mask_labels, torch.Tensor):
                        mask_labels_l = mask_labels[labeled_mask]
                    else:
                        mask_labels_l = [m for m, keep in zip(mask_labels, labeled_mask) if keep]
                    if isinstance(class_labels, torch.Tensor):
                        class_labels_l = class_labels[labeled_mask]
                    else:
                        class_labels_l = [c for c, keep in zip(class_labels, labeled_mask) if keep]
                    label_l = label[labeled_mask]
                    sup_loss = self._compute_loss(
                        output=output_labeled,
                        label=label_l,
                        mask_labels=mask_labels_l,
                        class_labels=class_labels_l,
                    )
                
                # 2) 无标签子集：对聚合前景概率做盒弱监督
                weak_loss = output[self.model.prefix][LOGITS].new_tensor(0.0)
                if hasattr(self.model, "box_key") and self.model.box_key in batch and (~labeled_mask).any():
                    per_output = output[self.model.prefix]
                    mask_logits_u = per_output[LOGITS][~labeled_mask]
                    boxes_u = batch[self.model.box_key][:, 0, :][~labeled_mask]
                    if CLASS_LOGITS in per_output:
                        class_logits_u = per_output[CLASS_LOGITS][~labeled_mask]
                        # 语义概率：softmax(class) * sigmoid(mask) -> foreground 概率
                        mask_cls = F.softmax(class_logits_u, dim=-1)[..., :-1]
                        mask_prob = torch.sigmoid(mask_logits_u)
                        semantic_prob = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_prob)
                        fg_prob = semantic_prob.sum(dim=1, keepdim=True).clamp(min=1e-7, max=1 - 1e-7)
                        fg_logits = torch.logit(fg_prob)
                    else:
                        logits_u = mask_logits_u
                        if logits_u.dim() == 4 and logits_u.shape[1] > 1:
                            fg_prob = torch.sigmoid(logits_u).mean(dim=1, keepdim=True).clamp(min=1e-7, max=1 - 1e-7)
                            fg_logits = torch.logit(fg_prob)
                        else:
                            fg_logits = logits_u
                    weak_loss = self._weak_box_losses(
                        logits=fg_logits,
                        boxes=boxes_u,
                        outside_weight=weak_outside_w,
                        entropy_weight=weak_entropy_w,
                        tv_weight=weak_tv_w,
                    )
                
                moe_loss = output[self.model.prefix].get(MOE_LOSS, 0.0)
                loss = sup_loss + weak_loss + moe_loss
                self.log("train_sup_loss", sup_loss, on_step=True, on_epoch=True)
                self.log("train_weak_loss", weak_loss, on_step=True, on_epoch=True)
                self.log("train_labeled_frac", labeled_mask.float().mean(), on_step=True, on_epoch=True)
            else:
                # 二值结构损失路径
                logits = output[self.model.prefix][LOGITS]
                if logits.dim() == 3:
                    logits = logits.unsqueeze(1)
                if label.dim() == 3:
                    label_ = label.unsqueeze(1)
                else:
                    label_ = label
                
                per_sample_sup = self._structure_loss_per_sample(logits, label_)
                sup_count = labeled_mask.sum().clamp(min=1)
                sup_loss = (per_sample_sup * labeled_mask.float()).sum() / sup_count.float()
                
                weak_loss = logits.new_tensor(0.0)
                if hasattr(self.model, "box_key") and self.model.box_key in batch and (~labeled_mask).any():
                    boxes_b1 = batch[self.model.box_key]  # (B,1,4)
                    boxes = boxes_b1[:, 0, :]
                    logits_u = logits[~labeled_mask]
                    boxes_u = boxes[~labeled_mask]
                    weak_loss = self._weak_box_losses(
                        logits=logits_u,
                        boxes=boxes_u,
                        outside_weight=weak_outside_w,
                        entropy_weight=weak_entropy_w,
                        tv_weight=weak_tv_w,
                    )
                
                moe_loss = output[self.model.prefix].get(MOE_LOSS, 0.0)
                loss = sup_loss + weak_loss + moe_loss
                self.log("train_sup_loss", sup_loss, on_step=True, on_epoch=True)
                self.log("train_weak_loss", weak_loss, on_step=True, on_epoch=True)
                self.log("train_labeled_frac", labeled_mask.float().mean(), on_step=True, on_epoch=True)
        else:
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
        cfg = getattr(self, "train_box_prompt_cfg", {}) or {}
        semi_labeled_fraction = float(cfg.get("semi_labeled_fraction", 1.0))
        use_gspo = (
            self.gspo_trainer is not None and 
            self.gspo_trainer.is_gspo_active(self.current_epoch)
        )
        allow_semisup_gspo = bool(getattr(self.gspo_trainer, "allow_semisup", False)) if self.gspo_trainer else False
        
        if use_gspo and allow_semisup_gspo and semi_labeled_fraction < 1.0:
            # Semi-supervised with GSPO on labeled subset, weak loss on unlabeled subset.
            label = batch[self.model.label_key]
            # Apply box prompts (gt/noisy/predict + weak jitter boxes already handled inside)
            if self.training:
                self._apply_train_box_prompts(batch, label)
            image_paths = batch.get(self.model.image_path_key, [])
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            labeled_mask = self._get_labeled_mask_from_paths(
                image_paths=image_paths, labeled_fraction=semi_labeled_fraction, seed=int(cfg.get("semi_labeled_seed", 0))
            )
            has_labeled = bool(labeled_mask.any())
            has_unlabeled = bool((~labeled_mask).any())
            loss = torch.tensor(0.0, device=self.device)
            selected_experts = None

            def _subset_batch(b, mask_bool: torch.Tensor):
                new = {}
                for k, v in b.items():
                    if torch.is_tensor(v) and v.shape[0] == mask_bool.shape[0]:
                        new[k] = v[mask_bool]
                    elif isinstance(v, list) and len(v) == mask_bool.shape[0]:
                        new[k] = [v[i] for i in range(len(v)) if mask_bool[i].item()]
                    else:
                        new[k] = v
                return new

            # Weak loss on unlabeled subset
            if has_unlabeled and hasattr(self.model, "box_key") and self.model.box_key in batch:
                batch_u = _subset_batch(batch, ~labeled_mask)
                label_u = batch_u[self.model.label_key]
                output_u = run_model(self.model, batch_u)
                logits_u = output_u[self.model.prefix][LOGITS]
                if logits_u.dim() == 3:
                    logits_u = logits_u.unsqueeze(1)
                boxes_u = batch_u[self.model.box_key][:, 0, :]
                weak_loss = self._weak_box_losses(
                    logits=logits_u,
                    boxes=boxes_u,
                    outside_weight=float(cfg.get("weak_loss_outside_weight", 1.0)),
                    entropy_weight=float(cfg.get("weak_loss_entropy_weight", 0.05)),
                    tv_weight=float(cfg.get("weak_loss_tv_weight", 0.0)),
                )
                moe_loss_u = output_u[self.model.prefix].get(MOE_LOSS, 0.0)
                loss = loss + weak_loss + moe_loss_u
                self.log("train_weak_loss", weak_loss, on_step=True, on_epoch=True)

            # GSPO on labeled subset
            if has_labeled:
                batch_l = _subset_batch(batch, labeled_mask)
                gspo_loss, metrics, selected_experts = self._gspo_training_step(batch_l)
                loss = loss + gspo_loss
                for key, value in metrics.items():
                    self.log(f"train_{key}", value, on_step=True, on_epoch=True)

            self.log("train_labeled_frac", labeled_mask.float().mean(), on_step=True, on_epoch=True)

        elif use_gspo:
            # GSPO-enhanced training (full GT)
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
        
        Extended to support:
        - Encoder Adapter quality feedback when gspo_adapter_enabled=True
        - Decoder LoRA on Attention quality feedback when gspo_lora_attention_enabled=True
        """
        images = batch[self.model.image_key] if hasattr(self.model, 'image_key') else batch['image']
        labels = batch[self.model.label_key]
        
        mask_labels = batch.get(self.model.mask_label_key, None)
        class_labels = batch.get(self.model.class_label_key, None)
        
        # Define forward function for GSPO
        def forward_fn(images):
            batch_copy = batch.copy()
            if hasattr(self.model, 'image_key'):
                batch_copy[self.model.image_key] = images
            else:
                batch_copy['image'] = images
            output = run_model(self.model, batch_copy)
            
            # Extract predictions and MOE info
            pred_dict = output[self.model.prefix]
            logits = pred_dict[LOGITS]
            class_logits = pred_dict.get(CLASS_LOGITS, None)
            moe_loss = pred_dict.get(MOE_LOSS, 0)
            
            # Extract selected experts if available
            selected_experts = None
            if hasattr(pred_dict, 'selected_experts'):
                selected_experts = pred_dict['selected_experts']
            
            return {"logits": logits, "class_logits": class_logits}, moe_loss, selected_experts
        
        # Define loss function for GSPO
        def loss_fn(predictions, targets):
            if isinstance(self.loss_func, Mask2FormerLoss):
                mask_logits, class_logits = None, None
                if isinstance(predictions, dict):
                    mask_logits = predictions.get("logits", predictions.get("masks", None))
                    class_logits = predictions.get("class_logits", None)
                elif isinstance(predictions, (tuple, list)) and len(predictions) >= 2:
                    mask_logits, class_logits = predictions[0], predictions[1]
                else:
                    mask_logits = predictions
                    class_logits = None
                if class_logits is None:
                    raise ValueError("Mask2FormerLoss requires class_logits when GSPO is enabled.")
                return self.loss_func(
                    masks_queries_logits=mask_logits,
                    class_queries_logits=class_logits,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                )
            # non Mask2Former path
            if isinstance(predictions, (tuple, list)) and len(predictions) >= 1:
                predictions = predictions[0]
            return self.loss_func(input=predictions, target=targets)
        
        # Run GSPO group training
        loss, metrics, selected_experts_groups = self.gspo_trainer.gspo_group_training_step(
            images=images,
            masks_gt=labels,
            forward_fn=forward_fn,
            loss_fn=loss_fn,
        )
        
        # Update expert quality feedback (Conv-LoRA MoE)
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
            
            # Update Adapter quality feedback (GSPO-Adapter extension)
            if hasattr(self.gspo_trainer, 'gspo_adapter_enabled') and self.gspo_trainer.gspo_adapter_enabled:
                self.gspo_trainer.update_adapter_feedback(quality_scores, self.model)
            
            # Update LoRA on Attention quality feedback (GSPO-LoRA extension)
            if hasattr(self.gspo_trainer, 'gspo_lora_attention_enabled') and self.gspo_trainer.gspo_lora_attention_enabled:
                self.gspo_trainer.update_lora_attention_feedback(quality_scores, self.model)
        
        return loss, metrics, selected_experts_groups

```

```python
import logging
import hashlib
from typing import Callable, Dict, Optional

import torch
import torchmetrics
import torch.nn.functional as F
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
    
    def __init__(self, *args, gspo_trainer=None, train_box_prompt_cfg=None, train_bbox_predictor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gspo_trainer = gspo_trainer
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
    
    @staticmethod
    def _stable_uniform_0_1(key: str, seed: int = 0) -> float:
        """
        Stable deterministic pseudo-random number in [0, 1) from a string key + seed.
        Avoids python's randomized hash across processes.
        """
        h = hashlib.md5(f"{seed}::{key}".encode("utf-8")).digest()
        # use first 8 bytes as uint64
        x = int.from_bytes(h[:8], byteorder="little", signed=False)
        return (x % (10**12)) / float(10**12)
    
    def _get_labeled_mask_from_paths(self, image_paths, labeled_fraction: float, seed: int) -> torch.Tensor:
        """
        image_paths: list[str] (len=B)
        returns: bool tensor (B,)
        """
        labeled_fraction = float(labeled_fraction)
        labeled_fraction = max(0.0, min(1.0, labeled_fraction))
        if labeled_fraction >= 1.0:
            return torch.ones(len(image_paths), dtype=torch.bool, device=self.device)
        if labeled_fraction <= 0.0:
            return torch.zeros(len(image_paths), dtype=torch.bool, device=self.device)
        flags = []
        for p in image_paths:
            p = p or ""
            u = self._stable_uniform_0_1(p, seed=seed)
            flags.append(u < labeled_fraction)
        return torch.tensor(flags, dtype=torch.bool, device=self.device)
    
    @staticmethod
    def _compute_boxes_from_mask(mask: torch.Tensor) -> torch.Tensor:
        """mask: (B,H,W) or (B,1,H,W) -> boxes (B,4) in pixel coords (x1,y1,x2,y2)."""
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
    
    @staticmethod
    def _jitter_boxes_outward(
        boxes: torch.Tensor,
        h: int,
        w: int,
        amount: float,
        mode: str = "box",  # box / image / pixel
        outward_only: bool = True,
    ) -> torch.Tensor:
        """
        Jitter boxes to simulate coarse human boxes.
        - outward_only=True: never shrink vs original (always expand or keep).
        - amount meaning:
          - mode=box: fraction of box width/height
          - mode=image: fraction of image width/height
          - mode=pixel: absolute pixels
        """
        jittered = boxes.clone()
        mode = (mode or "box").lower()
        amount = float(amount)
        amount = max(0.0, amount)
        for i in range(jittered.shape[0]):
            x1, y1, x2, y2 = jittered[i]
            if x2 <= x1 or y2 <= y1:
                continue
            bw = (x2 - x1).clamp(min=1.0)
            bh = (y2 - y1).clamp(min=1.0)
            if mode == "box":
                dx = bw * amount
                dy = bh * amount
            elif mode == "image":
                dx = float(w) * amount
                dy = float(h) * amount
                dx = torch.tensor(dx, device=boxes.device, dtype=torch.float32)
                dy = torch.tensor(dy, device=boxes.device, dtype=torch.float32)
            elif mode == "pixel":
                dx = torch.tensor(amount, device=boxes.device, dtype=torch.float32)
                dy = torch.tensor(amount, device=boxes.device, dtype=torch.float32)
            else:
                # fallback to box
                dx = bw * amount
                dy = bh * amount
            
            # sample non-negative deltas (mostly outward expansion)
            # use uniform [0, dx] / [0, dy]
            ex1 = torch.rand(1, device=boxes.device) * dx
            ey1 = torch.rand(1, device=boxes.device) * dy
            ex2 = torch.rand(1, device=boxes.device) * dx
            ey2 = torch.rand(1, device=boxes.device) * dy
            
            if outward_only:
                # outward: x1 decreases, y1 decreases, x2 increases, y2 increases
                x1n = x1 - ex1
                y1n = y1 - ey1
                x2n = x2 + ex2
                y2n = y2 + ey2
            else:
                # allow inward with small probability by random sign
                s1 = torch.where(torch.rand(1, device=boxes.device) < 0.9, -1.0, 1.0)  # x1: outward is -1
                t1 = torch.where(torch.rand(1, device=boxes.device) < 0.9, -1.0, 1.0)  # y1
                s2 = torch.where(torch.rand(1, device=boxes.device) < 0.9, 1.0, -1.0)   # x2: outward is +1
                t2 = torch.where(torch.rand(1, device=boxes.device) < 0.9, 1.0, -1.0)   # y2
                x1n = x1 + s1 * ex1
                y1n = y1 + t1 * ey1
                x2n = x2 + s2 * ex2
                y2n = y2 + t2 * ey2
            
            x1n = torch.clamp(x1n, 0, w - 1)
            y1n = torch.clamp(y1n, 0, h - 1)
            x2n = torch.clamp(x2n, 0, w - 1)
            y2n = torch.clamp(y2n, 0, h - 1)
            # ensure valid ordering
            jittered[i] = torch.stack(
                [torch.min(x1n, x2n), torch.min(y1n, y2n), torch.max(x1n, x2n), torch.max(y1n, y2n)]
            ).squeeze()
        return jittered
    
    @staticmethod
    def _structure_loss_per_sample(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Per-sample StructureLoss (same as optim/losses/structure_loss.py) without final mean.
        Returns tensor of shape (B,).
        """
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
        wbce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3)).clamp(min=1e-6)
        probs = torch.sigmoid(logits)
        inter = ((probs * target) * weit).sum(dim=(2, 3))
        union = ((probs + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).view(-1)
    
    @staticmethod
    def _weak_box_losses(
        logits: torch.Tensor,  # (B,1,H,W) or (B,H,W)
        boxes: torch.Tensor,   # (B,4) x1,y1,x2,y2 in pixel coords
        outside_weight: float = 1.0,
        entropy_weight: float = 0.05,
        tv_weight: float = 0.0,
    ) -> torch.Tensor:
        """
        Box-only weak supervision loss:
        - outside-box should be background (0)
        - inside-box entropy minimization (push confident predictions without assuming full positive)
        - optional smoothness (TV) inside box
        Returns scalar loss.
        """
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        b, _, h, w = logits.shape
        device = logits.device
        outside_weight = float(outside_weight)
        entropy_weight = float(entropy_weight)
        tv_weight = float(tv_weight)
        
        total = logits.new_tensor(0.0)
        count = 0
        for i in range(b):
            x1, y1, x2, y2 = boxes[i]
            if x2 <= x1 or y2 <= y1:
                continue
            # integer pixel bounds (inclusive)
            x1i = int(torch.clamp(x1.round(), 0, w - 1).item())
            y1i = int(torch.clamp(y1.round(), 0, h - 1).item())
            x2i = int(torch.clamp(x2.round(), 0, w - 1).item())
            y2i = int(torch.clamp(y2.round(), 0, h - 1).item())
            if x2i <= x1i or y2i <= y1i:
                continue
            
            logit = logits[i : i + 1]  # (1,1,H,W)
            outside_mask = torch.ones((1, 1, h, w), device=device, dtype=logit.dtype)
            outside_mask[:, :, y1i : y2i + 1, x1i : x2i + 1] = 0.0
            inside_mask = 1.0 - outside_mask
            
            # Outside-box background constraint
            if outside_weight > 0:
                zeros = torch.zeros_like(logit)
                bce = F.binary_cross_entropy_with_logits(logit, zeros, reduction="none")
                outside_loss = (bce * outside_mask).sum() / outside_mask.sum().clamp(min=1.0)
            else:
                outside_loss = logit.new_tensor(0.0)
            
            # Inside-box entropy minimization
            if entropy_weight > 0:
                p = torch.sigmoid(logit).clamp(min=1e-6, max=1 - 1e-6)
                ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
                ent_loss = (ent * inside_mask).sum() / inside_mask.sum().clamp(min=1.0)
            else:
                ent_loss = logit.new_tensor(0.0)
            
            # Total variation smoothness inside the box
            if tv_weight > 0:
                p = torch.sigmoid(logit)
                dy = torch.abs(p[:, :, 1:, :] - p[:, :, :-1, :])
                dx = torch.abs(p[:, :, :, 1:] - p[:, :, :, :-1])
                # match shapes with masks
                inside_y = inside_mask[:, :, 1:, :]
                inside_x = inside_mask[:, :, :, 1:]
                tv = (dy * inside_y).sum() / inside_y.sum().clamp(min=1.0) + (dx * inside_x).sum() / inside_x.sum().clamp(min=1.0)
            else:
                tv = logit.new_tensor(0.0)
            
            total = total + outside_weight * outside_loss + entropy_weight * ent_loss + tv_weight * tv
            count += 1
        
        if count == 0:
            return logits.new_tensor(0.0)
        return total / float(count)

    def _predict_boxes_from_images(self, image_paths: list, target_h: int, target_w: int) -> torch.Tensor:
        """
        Wrapper around train_bbox_predictor to predict boxes in pixel coords and scale to target size.
        """
        import numpy as np
        from PIL import Image
        if self.train_bbox_predictor is None:
            raise ValueError("train_bbox_predictor is required for mode='predict'")
        b = len(image_paths)
        boxes = torch.zeros((b, 4), device=self.device, dtype=torch.float32)
        for i, img_path in enumerate(image_paths):
            if not img_path:
                continue
            try:
                bbox_px = self.train_bbox_predictor.predict(img_path)
                if bbox_px is None:
                    continue
                with Image.open(img_path) as img:
                    orig_w, orig_h = img.size
                scale_x = target_w / orig_w
                scale_y = target_h / orig_h
                x1 = bbox_px[0] * scale_x
                y1 = bbox_px[1] * scale_y
                x2 = bbox_px[2] * scale_x
                y2 = bbox_px[3] * scale_y
                x1 = max(0, min(x1, target_w - 1))
                y1 = max(0, min(y1, target_h - 1))
                x2 = max(0, min(x2, target_w - 1))
                y2 = max(0, min(y2, target_h - 1))
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                boxes[i] = torch.tensor([x1, y1, x2, y2], device=self.device, dtype=torch.float32)
            except Exception as e:
                logger.warning(f"Failed to predict bbox for {img_path}: {e}")
                continue
        return boxes

    def _apply_train_box_prompts(self, batch: Dict, label: torch.Tensor) -> None:
        """
        In-place box prompt injection for training (gt/noisy/mix/predict) and for semi/weak supervision.
        This is factored out so GSPO/standard paths can share the same augmentation.
        """
        cfg = self.train_box_prompt_cfg or {}
        mode = cfg.get("mode", "off")
        if mode == "off" and "semi_labeled_fraction" not in cfg:
            return

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
        else:
            noise_frac = float(cfg.get("noise_frac", 0.12))

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

        if box_mode in ["gt", "noisy"]:
            gt_boxes = self._compute_boxes_from_mask(label)
            h, w = label.shape[-2], label.shape[-1]
            if box_mode == "noisy":
                gt_boxes = _jitter_boxes(gt_boxes, h=h, w=w, frac=noise_frac)
            if not (gt_boxes.sum(dim=1) == 0).all():
                batch[self.model.box_key] = gt_boxes.unsqueeze(1)  # (B,1,4)
        elif box_mode == "predict":
            image_path_key = self.model.image_path_key
            image_paths = batch.get(image_path_key, [])
            if image_paths:
                h, w = label.shape[-2], label.shape[-1]
                pred_boxes = self._predict_boxes_from_images(
                    image_paths=image_paths, target_h=h, target_w=w
                )
                if not (pred_boxes.sum(dim=1) == 0).all():
                    batch[self.model.box_key] = pred_boxes.unsqueeze(1)  # (B,1,4)

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
                logits = per_output[LOGITS]
                
                # Handle SAM multi-class output format: combine mask and class predictions
                # SAM multi-class outputs: LOGITS [B, num_mask_tokens, H, W] + CLASS_LOGITS [B, num_mask_tokens, num_classes+1]
                # When num_mask_tokens == num_classes, we can use LOGITS directly as per-class masks
                # Otherwise, we need to combine mask and class predictions
                if CLASS_LOGITS in per_output:
                    mask_logits = logits  # [B, num_mask_tokens, H, W]
                    class_logits = per_output[CLASS_LOGITS]  # [B, num_mask_tokens, num_classes+1]
                    
                    # Apply semantic inference to combine mask and class predictions
                    # This follows Mask2Former's approach: semantic_mask = einsum(softmax(class), sigmoid(mask))
                    mask_cls = F.softmax(class_logits, dim=-1)[..., :-1]  # [B, num_mask_tokens, num_classes]
                    mask_pred = torch.sigmoid(mask_logits)  # [B, num_mask_tokens, H, W]
                    
                    # Combine: for each pixel, sum over mask tokens weighted by class probability
                    # semantic_mask[b, c, h, w] = sum_q(mask_cls[b, q, c] * mask_pred[b, q, h, w])
                    logits = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
                    
                    # Convert probabilities back to logits for CrossEntropyLoss (add small epsilon to avoid log(0))
                    logits = torch.log(logits.clamp(min=1e-7))
                
                loss += (
                    self.loss_func(
                        input=logits,
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
        
        # ---- Semi/weak supervision config (optional) ----
        cfg = self.train_box_prompt_cfg or {}
        semi_labeled_fraction = float(cfg.get("semi_labeled_fraction", 1.0))
        semi_labeled_seed = int(cfg.get("semi_labeled_seed", 0))
        weak_box_jitter_mode = str(cfg.get("weak_box_jitter_mode", "box"))
        weak_box_jitter_amount = float(cfg.get("weak_box_jitter_amount", cfg.get("noise_frac", 0.12)))
        weak_box_outward_only = bool(cfg.get("weak_box_outward_only", True))
        weak_outside_w = float(cfg.get("weak_loss_outside_weight", 1.0))
        weak_entropy_w = float(cfg.get("weak_loss_entropy_weight", 0.05))
        weak_tv_w = float(cfg.get("weak_loss_tv_weight", 0.0))
        
        # Determine which samples are "labeled" (for simulation) using stable hashing on image paths.
        image_paths = batch.get(self.model.image_path_key, [])
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        labeled_mask = None
        if self.training and semi_labeled_fraction < 1.0:
            labeled_mask = self._get_labeled_mask_from_paths(
                image_paths=image_paths, labeled_fraction=semi_labeled_fraction, seed=semi_labeled_seed
            )

        # -------- Train-time box prompt injection (configurable) --------
        # 控制项：self.train_box_prompt_cfg = {
        #   mode: off / gt / noisy / mix / predict,
        #   p_no, p_gt, p_noisy: 仅在 mode=mix 时生效并会归一化,
        #   noise_frac: 噪声扰动比例（相对 bbox 宽高）
        # }
        # mode=predict: 使用 self.train_bbox_predictor 预测框作为 box prompt
        if self.training:
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

            # If semi/weak enabled: for "unlabeled" samples, force weak coarse boxes generated from GT bbox with outward-biased jitter.
            if labeled_mask is not None and hasattr(self.model, "box_key"):
                gt_boxes_all = self._compute_boxes_from_mask(label)
                h, w = label.shape[-2], label.shape[-1]
                weak_boxes = self._jitter_boxes_outward(
                    gt_boxes_all,
                    h=h,
                    w=w,
                    amount=weak_box_jitter_amount,
                    mode=weak_box_jitter_mode,
                    outward_only=weak_box_outward_only,
                )
                # apply only to unlabeled samples
                if (~labeled_mask).any():
                    boxes_for_batch = gt_boxes_all.clone()
                    boxes_for_batch[~labeled_mask] = weak_boxes[~labeled_mask]
                    if not (boxes_for_batch.sum(dim=1) == 0).all():
                        batch[self.model.box_key] = boxes_for_batch.unsqueeze(1)  # (B,1,4)

            if box_mode in ["gt", "noisy"]:
                gt_boxes = self._compute_boxes_from_mask(label)
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
        
        # ---- Loss: fully-supervised OR semi/weak mixed ----
        if self.training and labeled_mask is not None:
            # Semi/weak supervision path
            if isinstance(self.loss_func, Mask2FormerLoss):
                # 1) 有标签子集：使用 Mask2FormerLoss
                sup_loss = output[self.model.prefix][LOGITS].new_tensor(0.0)
                if labeled_mask.any():
                    # 子集化输出与标签
                    output_labeled = {}
                    for key, per_output in output.items():
                        sub_out = {}
                        for sub_k, val in per_output.items():
                            # 仅当第 0 维与 batch 对齐时才子集化，避免标量/非 batch 维度触发索引错误
                            if torch.is_tensor(val) and val.ndim > 0 and val.shape[0] == label.shape[0]:
                                sub_out[sub_k] = val[labeled_mask]
                            else:
                                sub_out[sub_k] = val
                        output_labeled[key] = sub_out
                    mask_labels = batch.get(self.model.mask_label_key, [])
                    class_labels = batch.get(self.model.class_label_key, [])
                    if isinstance(mask_labels, torch.Tensor):
                        mask_labels_l = mask_labels[labeled_mask]
                    else:
                        mask_labels_l = [m for m, keep in zip(mask_labels, labeled_mask) if keep]
                    if isinstance(class_labels, torch.Tensor):
                        class_labels_l = class_labels[labeled_mask]
                    else:
                        class_labels_l = [c for c, keep in zip(class_labels, labeled_mask) if keep]
                    label_l = label[labeled_mask]
                    sup_loss = self._compute_loss(
                        output=output_labeled,
                        label=label_l,
                        mask_labels=mask_labels_l,
                        class_labels=class_labels_l,
                    )
                
                # 2) 无标签子集：对聚合前景概率做盒弱监督
                weak_loss = output[self.model.prefix][LOGITS].new_tensor(0.0)
                if hasattr(self.model, "box_key") and self.model.box_key in batch and (~labeled_mask).any():
                    per_output = output[self.model.prefix]
                    mask_logits_u = per_output[LOGITS][~labeled_mask]
                    boxes_u = batch[self.model.box_key][:, 0, :][~labeled_mask]
                    if CLASS_LOGITS in per_output:
                        class_logits_u = per_output[CLASS_LOGITS][~labeled_mask]
                        # 语义概率：softmax(class) * sigmoid(mask) -> foreground 概率
                        mask_cls = F.softmax(class_logits_u, dim=-1)[..., :-1]
                        mask_prob = torch.sigmoid(mask_logits_u)
                        semantic_prob = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_prob)
                        fg_prob = semantic_prob.sum(dim=1, keepdim=True).clamp(min=1e-7, max=1 - 1e-7)
                        fg_logits = torch.logit(fg_prob)
                    else:
                        logits_u = mask_logits_u
                        if logits_u.dim() == 4 and logits_u.shape[1] > 1:
                            fg_prob = torch.sigmoid(logits_u).mean(dim=1, keepdim=True).clamp(min=1e-7, max=1 - 1e-7)
                            fg_logits = torch.logit(fg_prob)
                        else:
                            fg_logits = logits_u
                    weak_loss = self._weak_box_losses(
                        logits=fg_logits,
                        boxes=boxes_u,
                        outside_weight=weak_outside_w,
                        entropy_weight=weak_entropy_w,
                        tv_weight=weak_tv_w,
                    )
                
                moe_loss = output[self.model.prefix].get(MOE_LOSS, 0.0)
                loss = sup_loss + weak_loss + moe_loss
                self.log("train_sup_loss", sup_loss, on_step=True, on_epoch=True)
                self.log("train_weak_loss", weak_loss, on_step=True, on_epoch=True)
                self.log("train_labeled_frac", labeled_mask.float().mean(), on_step=True, on_epoch=True)
            else:
                # 二值结构损失路径
                logits = output[self.model.prefix][LOGITS]
                if logits.dim() == 3:
                    logits = logits.unsqueeze(1)
                if label.dim() == 3:
                    label_ = label.unsqueeze(1)
                else:
                    label_ = label
                
                per_sample_sup = self._structure_loss_per_sample(logits, label_)
                sup_count = labeled_mask.sum().clamp(min=1)
                sup_loss = (per_sample_sup * labeled_mask.float()).sum() / sup_count.float()
                
                weak_loss = logits.new_tensor(0.0)
                if hasattr(self.model, "box_key") and self.model.box_key in batch and (~labeled_mask).any():
                    boxes_b1 = batch[self.model.box_key]  # (B,1,4)
                    boxes = boxes_b1[:, 0, :]
                    logits_u = logits[~labeled_mask]
                    boxes_u = boxes[~labeled_mask]
                    weak_loss = self._weak_box_losses(
                        logits=logits_u,
                        boxes=boxes_u,
                        outside_weight=weak_outside_w,
                        entropy_weight=weak_entropy_w,
                        tv_weight=weak_tv_w,
                    )
                
                moe_loss = output[self.model.prefix].get(MOE_LOSS, 0.0)
                loss = sup_loss + weak_loss + moe_loss
                self.log("train_sup_loss", sup_loss, on_step=True, on_epoch=True)
                self.log("train_weak_loss", weak_loss, on_step=True, on_epoch=True)
                self.log("train_labeled_frac", labeled_mask.float().mean(), on_step=True, on_epoch=True)
        else:
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
        cfg = getattr(self, "train_box_prompt_cfg", {}) or {}
        semi_labeled_fraction = float(cfg.get("semi_labeled_fraction", 1.0))
        use_gspo = (
            self.gspo_trainer is not None and 
            self.gspo_trainer.is_gspo_active(self.current_epoch)
        )
        allow_semisup_gspo = bool(getattr(self.gspo_trainer, "allow_semisup", False)) if self.gspo_trainer else False
        
        if use_gspo and allow_semisup_gspo and semi_labeled_fraction < 1.0:
            # Semi-supervised with GSPO on labeled subset, weak loss on unlabeled subset.
            label = batch[self.model.label_key]
            # Apply box prompts (gt/noisy/predict + weak jitter boxes already handled inside)
            if self.training:
                self._apply_train_box_prompts(batch, label)
            image_paths = batch.get(self.model.image_path_key, [])
            if isinstance(image_paths, str):
                image_paths = [image_paths]
            labeled_mask = self._get_labeled_mask_from_paths(
                image_paths=image_paths, labeled_fraction=semi_labeled_fraction, seed=int(cfg.get("semi_labeled_seed", 0))
            )
            has_labeled = bool(labeled_mask.any())
            has_unlabeled = bool((~labeled_mask).any())
            loss = torch.tensor(0.0, device=self.device)
            selected_experts = None

            def _subset_batch(b, mask_bool: torch.Tensor):
                new = {}
                for k, v in b.items():
                    if torch.is_tensor(v) and v.shape[0] == mask_bool.shape[0]:
                        new[k] = v[mask_bool]
                    elif isinstance(v, list) and len(v) == mask_bool.shape[0]:
                        new[k] = [v[i] for i in range(len(v)) if mask_bool[i].item()]
                    else:
                        new[k] = v
                return new

            # Weak loss on unlabeled subset
            if has_unlabeled and hasattr(self.model, "box_key") and self.model.box_key in batch:
                batch_u = _subset_batch(batch, ~labeled_mask)
                label_u = batch_u[self.model.label_key]
                output_u = run_model(self.model, batch_u)
                logits_u = output_u[self.model.prefix][LOGITS]
                if logits_u.dim() == 3:
                    logits_u = logits_u.unsqueeze(1)
                boxes_u = batch_u[self.model.box_key][:, 0, :]
                weak_loss = self._weak_box_losses(
                    logits=logits_u,
                    boxes=boxes_u,
                    outside_weight=float(cfg.get("weak_loss_outside_weight", 1.0)),
                    entropy_weight=float(cfg.get("weak_loss_entropy_weight", 0.05)),
                    tv_weight=float(cfg.get("weak_loss_tv_weight", 0.0)),
                )
                moe_loss_u = output_u[self.model.prefix].get(MOE_LOSS, 0.0)
                loss = loss + weak_loss + moe_loss_u
                self.log("train_weak_loss", weak_loss, on_step=True, on_epoch=True)

            # GSPO on labeled subset
            if has_labeled:
                batch_l = _subset_batch(batch, labeled_mask)
                gspo_loss, metrics, selected_experts = self._gspo_training_step(batch_l)
                loss = loss + gspo_loss
                for key, value in metrics.items():
                    self.log(f"train_{key}", value, on_step=True, on_epoch=True)

            self.log("train_labeled_frac", labeled_mask.float().mean(), on_step=True, on_epoch=True)

        elif use_gspo:
            # GSPO-enhanced training (full GT)
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
        
        Extended to support:
        - Encoder Adapter quality feedback when gspo_adapter_enabled=True
        - Decoder LoRA on Attention quality feedback when gspo_lora_attention_enabled=True
        """
        images = batch[self.model.image_key] if hasattr(self.model, 'image_key') else batch['image']
        labels = batch[self.model.label_key]
        
        mask_labels = batch.get(self.model.mask_label_key, None)
        class_labels = batch.get(self.model.class_label_key, None)
        
        # Define forward function for GSPO
        def forward_fn(images):
            batch_copy = batch.copy()
            if hasattr(self.model, 'image_key'):
                batch_copy[self.model.image_key] = images
            else:
                batch_copy['image'] = images
            output = run_model(self.model, batch_copy)
            
            # Extract predictions and MOE info
            pred_dict = output[self.model.prefix]
            logits = pred_dict[LOGITS]
            class_logits = pred_dict.get(CLASS_LOGITS, None)
            moe_loss = pred_dict.get(MOE_LOSS, 0)
            
            # Extract selected experts if available
            selected_experts = None
            if hasattr(pred_dict, 'selected_experts'):
                selected_experts = pred_dict['selected_experts']
            
            return {"logits": logits, "class_logits": class_logits}, moe_loss, selected_experts
        
        # Define loss function for GSPO
        def loss_fn(predictions, targets):
            if isinstance(self.loss_func, Mask2FormerLoss):
                mask_logits, class_logits = None, None
                if isinstance(predictions, dict):
                    mask_logits = predictions.get("logits", predictions.get("masks", None))
                    class_logits = predictions.get("class_logits", None)
                elif isinstance(predictions, (tuple, list)) and len(predictions) >= 2:
                    mask_logits, class_logits = predictions[0], predictions[1]
                else:
                    mask_logits = predictions
                    class_logits = None
                if class_logits is None:
                    raise ValueError("Mask2FormerLoss requires class_logits when GSPO is enabled.")
                return self.loss_func(
                    masks_queries_logits=mask_logits,
                    class_queries_logits=class_logits,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                )
            # non Mask2Former path
            if isinstance(predictions, (tuple, list)) and len(predictions) >= 1:
                predictions = predictions[0]
            return self.loss_func(input=predictions, target=targets)
        
        # Run GSPO group training
        loss, metrics, selected_experts_groups = self.gspo_trainer.gspo_group_training_step(
            images=images,
            masks_gt=labels,
            forward_fn=forward_fn,
            loss_fn=loss_fn,
        )
        
        # Update expert quality feedback (Conv-LoRA MoE)
        if selected_experts_groups:
            quality_scores = []
            for _ in range(self.gspo_trainer.group_size):
                pred_out, _, _ = forward_fn(images)
                if isinstance(pred_out, dict):
                    mask_logits = pred_out.get("logits", pred_out.get("masks", None))
                    class_logits = pred_out.get("class_logits", None)
                elif isinstance(pred_out, (tuple, list)) and len(pred_out) >= 2:
                    mask_logits, class_logits = pred_out[0], pred_out[1]
                else:
                    mask_logits, class_logits = pred_out, None
                quality_scores.append(
                    self.gspo_trainer.compute_segmentation_quality(
                        mask_logits,
                        labels,
                        self.gspo_trainer.quality_metric,
                        class_logits=class_logits,
                    )
                )
            self.gspo_trainer.update_expert_feedback(
                selected_experts_groups, quality_scores, self.model
            )
            
            # Update Adapter quality feedback (GSPO-Adapter extension)
            if hasattr(self.gspo_trainer, 'gspo_adapter_enabled') and self.gspo_trainer.gspo_adapter_enabled:
                self.gspo_trainer.update_adapter_feedback(quality_scores, self.model)
            
            # Update LoRA on Attention quality feedback (GSPO-LoRA extension)
            if hasattr(self.gspo_trainer, 'gspo_lora_attention_enabled') and self.gspo_trainer.gspo_lora_attention_enabled:
                self.gspo_trainer.update_lora_attention_feedback(quality_scores, self.model)
        
        return loss, metrics, selected_experts_groups

```

已补全 GSPO 的多类 Mask2Former 支持，并修复半监督 + GSPO 分支的损失与质量计算路径，避免之前多类信息被二值化或损失被错误调用。

变更要点：
- `gspo_trainer.py`  
  - `compute_segmentation_quality` 支持 Mask2Former：使用 class logits 聚合语义概率，按前景类逐类 IoU/Dice 再对 GT 中出现的类取均值，保留多类信息。  
  - `gspo_group_training_step` 兼容 `{"logits", "class_logits"}` / (mask_logits, class_logits) 形式；弱约束/平滑/对比损失使用聚合后前景概率；seg loss 调用时传入 (mask_logits, class_logits)。
  - `_prepare_binary_masks` 对 Mask2Former 聚合前景概率，用于边界/平滑/细线奖励。
- `lit_semantic_seg.py`  
  - GSPO 前向返回 `{"logits", "class_logits"}`，loss_fn 对 Mask2FormerLoss 传入 mask/class logits 以及 mask_labels/class_labels。  
  - GSPO 质量反馈时带上 class_logits，质量计算与主干一致。

建议重新运行（保留原超参）：
```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA
nohup python3 run_acdc_conv_lora_semi.py \
  --dataset_dir datasets/acdc_conv_lora/acdc_conv_lora \
  --rank 3 \
  --expert_num 8 \
  --gspo_enable \
  --adapter_enable \
  --adapter_dim 64 \
  --gspo_adapter_enable \
  --gspo_allow_semisup \
  --loss mask2former_loss \
  --num_mask_tokens 10 \
  --semi_labeled_fraction 0.1 \
  --semi_labeled_seed 123 \
  --weak_box_jitter_mode box \
  --weak_box_jitter_amount 0.1 \
  --weak_box_outward_only \
  --weak_loss_outside_weight 1.0 \
  --weak_loss_entropy_weight 0.05 \
  --weak_loss_tv_weight 0.0 \
  --max_epochs 100 \
  --patience 30 \
  --vis_samples 8 \
  --vis_output_dir outputs/acdc_vis_mask2former-semi-260107 \
  --output_dir outputs/acdc_mask2former-semi-260107-gspo-fix \
  > outputs/acdc_mask2former-semi-260107-gspo-fix.log 2>&1 &
```

观察点：
- 日志中 `train_sup_loss/train_weak_loss/train_labeled_frac` 继续正常；
- 若 val_dice 仍停滞，可先提升 `--semi_labeled_fraction` 至 0.3~0.5 验证上限；或在确认 GSPO 路径无误后再调参（如 group_size/contrastive_weight）。

```shellscript
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA
nohup python3 run_acdc_conv_lora_semi.py \
  --dataset_dir datasets/acdc_conv_lora/acdc_conv_lora \
  --rank 3 \
  --expert_num 8 \
  --gspo_enable \
  --adapter_enable \
  --adapter_dim 64 \
  --gspo_adapter_enable \
  --gspo_allow_semisup \
  --loss mask2former_loss \
  --num_mask_tokens 10 \
  --semi_labeled_fraction 0.1 \
  --semi_labeled_seed 123 \
  --weak_box_jitter_mode box \
  --weak_box_jitter_amount 0.1 \
  --weak_box_outward_only \
  --weak_loss_outside_weight 1.0 \
  --weak_loss_entropy_weight 0.05 \
  --weak_loss_tv_weight 0.0 \
  --max_epochs 100 \
  --patience 30 \
  --vis_samples 8 \
  --vis_output_dir outputs/acdc_vis_mask2former-semi-260107 \
  --output_dir outputs/acdc_mask2former-semi-260107-gspo-fix \
  > outputs/acdc_mask2former-semi-260107-gspo-fix.log 2>&1 &
```

