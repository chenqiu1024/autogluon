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
