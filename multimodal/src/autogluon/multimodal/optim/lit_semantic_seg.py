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
    
    def __init__(self, *args, gspo_trainer=None, train_box_prompt_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gspo_trainer = gspo_trainer
        # Training-time box prompt config (dict with keys: mode, p_no, p_gt, p_noisy, noise_frac)
        self.train_box_prompt_cfg = train_box_prompt_cfg or {
            "mode": "off",
            "p_no": 0.0,
            "p_gt": 0.0,
            "p_noisy": 0.0,
            "noise_frac": 0.12,
        }

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
        #   mode: off / gt / noisy / mix,
        #   p_no, p_gt, p_noisy: 仅在 mode=mix 时生效并会归一化,
        #   noise_frac: 噪声扰动比例（相对 bbox 宽高）
        # }
        if self.training:
            cfg = self.train_box_prompt_cfg or {}
            mode = cfg.get("mode", "off")
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
                else:
                    box_mode = mode

            def _compute_gt_boxes(mask: torch.Tensor) -> torch.Tensor:
                """mask: (B,H,W) int/long -> boxes (B,4) in pixel coords."""
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

            if box_mode in ["gt", "noisy"]:
                gt_boxes = _compute_gt_boxes(label)
                h, w = label.shape[-2], label.shape[-1]
                if box_mode == "noisy":
                    gt_boxes = _jitter_boxes(gt_boxes, h=h, w=w, frac=noise_frac)
                if not (gt_boxes.sum(dim=1) == 0).all():
                    batch[self.model.box_key] = gt_boxes.unsqueeze(1)  # (B,1,4)
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
