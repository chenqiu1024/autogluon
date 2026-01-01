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

    def _compute_loss(self, output: Dict, label: torch.Tensor, **kwargs):
        # Guard against missing loss_fn (would raise "NoneType is not callable")
        if self.loss_func is None:
            raise ValueError(
                "loss_func is None. Please set loss_func or specify optim.loss_func in config."
            )
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
        Per training step with GSPO and Semi-Supervised enhancement.
        
        支持三种训练模式：
        1. 标准监督学习
        2. GSPO 策略优化（如果启用）
        3. 半监督学习（如果启用）- 新增
        
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
        # Check if semi-supervised learning is enabled
        use_semi_supervised = (
            hasattr(self, 'ema_teacher') and 
            hasattr(self, 'quality_estimator') and
            hasattr(self, 'pseudo_label_gen')
        )

        if use_semi_supervised:
            # 半监督训练（整合 Phase 1-4）
            loss = self._semi_supervised_training_step(batch, batch_idx)
        else:
            # Check if GSPO should be used
            use_gspo = (
                self.gspo_trainer is not None
                and self.gspo_trainer.is_gspo_active(self.current_epoch)
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
    
    def _semi_supervised_training_step(self, batch, batch_idx):
        """
        半监督训练步骤（完整实现 Phase 1-4）
        
        对应设计文档中的训练流程：
        1. 区分 labeled 和 weak 样本
        2. Teacher K 次采样 → 质量评估 → 伪标签生成
        3. Student 前向 → 计算半监督损失
        4. 更新 EMA Teacher
        
        对应公式 2.1-2.5 的完整实现
        """
        device = self.model.device
        
        # 1. 区分 labeled 和 weak 样本
        # AutoGluon 会过滤掉自定义列，batch 里通常没有 is_labeled。
        # 若缺失，则根据 mask 是否为空来推断：有正像素视为 labeled。
        if 'is_labeled' in batch:
            is_labeled = batch['is_labeled'].to(device)
        else:
            label_tensor = batch[self.model.label_key].to(device)
            if label_tensor.dim() == 4 and label_tensor.shape[1] == 1:
                label_tensor = label_tensor[:, 0, ...]
            # mask 像素和为 0 视为 weak，占位或空 mask 不会被误判为 labeled
            is_labeled = (label_tensor.flatten(1).sum(dim=1) > 0)

        has_weak = (~is_labeled).any()
        has_labeled = is_labeled.any()
        
        # 2. 对 weak 样本：Teacher 生成伪标签（Phase 1-2）
        pseudo_labels = None
        quality_scores = None
        valid_mask = None
        
        if has_weak:
            with torch.no_grad():
                # 2.1 Teacher K 次采样（对应公式 2.1）
                K = self.semi_supervised_config.get('quality_k_samples', 5)
                teacher_predictions = self.ema_teacher.forward_k_times(batch, K=K)
                
                # 2.2 质量评估（对应公式 2.2）
                quality_scores_full, q_cons, q_conf = self.quality_estimator.estimate_quality(
                    teacher_predictions
                )
                
                # 2.3 生成伪标签（对应公式 2.3）
                pseudo_labels_full, valid_mask_full = self.pseudo_label_gen.generate_from_predictions(
                    teacher_predictions, quality_scores_full
                )
            
            # 日志记录
            stats = self.pseudo_label_gen.get_statistics(quality_scores_full, valid_mask_full)
            for key, value in stats.items():
                self.log(f"semi/{key}", value, on_step=False, on_epoch=True, prog_bar=(key=='pseudo_label_ratio'))
            
            self.log("semi/q_cons", q_cons.mean(), on_step=False, on_epoch=True)
            self.log("semi/q_conf", q_conf.mean(), on_step=False, on_epoch=True)
            
            # 只保留 weak 样本的部分
            quality_scores = quality_scores_full
            pseudo_labels = pseudo_labels_full
            valid_mask = valid_mask_full
        
        # 3. Student 前向传播
        output, _ = self._shared_step(batch)
        student_pred = output[self.model.prefix][LOGITS]
        
        # 转换为概率 [B, H, W]
        if student_pred.dim() == 4 and student_pred.shape[1] > 1:
            student_pred_prob = torch.softmax(student_pred, dim=1)[:, 1]
        elif student_pred.dim() == 4 and student_pred.shape[1] == 1:
            student_pred_prob = torch.sigmoid(student_pred).squeeze(1)
        else:
            student_pred_prob = torch.sigmoid(student_pred) if student_pred.max() > 1 else student_pred
        
        # 4. 计算半监督损失（Phase 3）
        # 检查是否有 GSPO 半监督训练器
        use_gspo_semi = (
            hasattr(self, 'gspo_trainer') and 
            self.gspo_trainer is not None and
            hasattr(self.gspo_trainer, 'compute_semi_supervised_loss')
        )
        
        if use_gspo_semi:
            # 使用 GSPO 半监督训练器（含质量联动）
            
            # 准备 GT mask
            gt_mask = batch[self.model.label_key] if has_labeled else None
            
            # 计算质量权重
            if has_weak and quality_scores is not None:
                quality_weight = self.quality_estimator.compute_quality_weight(
                    quality_scores,
                    beta=self.semi_supervised_config.get('quality_weighting_beta', 10.0),
                    q0=0.5
                )
                # 应用质量过滤
                quality_weight = quality_weight * valid_mask.float()
            else:
                quality_weight = None
            
            # 计算损失（对应公式 2.4）
            loss, loss_dict = self.gspo_trainer.compute_semi_supervised_loss(
                student_pred=student_pred_prob,
                gt_mask=gt_mask,
                pseudo_label=pseudo_labels,
                quality_weight=quality_weight,
                is_labeled=is_labeled,
                epoch=self.current_epoch,
                loss_fn=self.loss_func
            )
            
            # 记录各项损失
            for key, value in loss_dict.items():
                self.log(f"train/{key}", value, on_step=True, on_epoch=True)
        
        else:
            # 简化版（无 GSPO）：直接计算监督 + 伪监督损失
            loss = 0
            
            # 监督损失
            if has_labeled:
                labeled_pred = student_pred[is_labeled]
                labeled_gt = batch[self.model.label_key][is_labeled]
                loss_supervised = self.loss_func(input=labeled_pred, target=labeled_gt)
                loss += loss_supervised
                self.log("train/loss_supervised", loss_supervised, on_step=True, on_epoch=True)
            
            # 伪监督损失（计算质量权重）
            if has_weak and pseudo_labels is not None and quality_scores is not None:
                # 计算质量权重
                quality_weight = self.quality_estimator.compute_quality_weight(
                    quality_scores,
                    beta=self.semi_supervised_config.get('quality_weighting_beta', 10.0),
                    q0=0.5
                )
                # 应用质量过滤
                quality_weight = quality_weight * valid_mask.float()
                weak_pred = student_pred_prob[~is_labeled]
                weak_pseudo = pseudo_labels[~is_labeled]
                weak_weight = quality_weight[~is_labeled] * valid_mask[~is_labeled].float()
                
                # 计算加权伪监督损失
                loss_pseudo_per_sample = self.loss_func(input=weak_pred, target=weak_pseudo)
                loss_pseudo = (loss_pseudo_per_sample * weak_weight).sum() / (weak_weight.sum() + 1e-6)
                
                # 应用 warmup
                config = getattr(self, 'semi_supervised_config', {})
                warmup_epochs = config.get('pseudo_lambda_warmup_epochs', 5)
                lambda_u = min(1.0, self.current_epoch / max(warmup_epochs, 1))
                
                loss_pseudo_weighted = lambda_u * loss_pseudo
                loss += loss_pseudo_weighted
                
                self.log("train/loss_pseudo", loss_pseudo, on_step=True, on_epoch=True)
                self.log("train/lambda_u", lambda_u, on_step=True, on_epoch=True)
        
        # 5. 更新 EMA Teacher（每步更新，对应公式 2.1）
        config = getattr(self, 'semi_supervised_config', {})
        ema_update_freq = config.get('ema_update_freq', 'step')
        if ema_update_freq == 'step':
            self.ema_teacher.update(self.model._model if hasattr(self.model, '_model') else self.model)
        
        return loss
    
    def on_train_epoch_end(self):
        """
        Epoch 结束时的处理
        
        如果 EMA 是 epoch 级别更新，在这里执行
        """
        # EMA Teacher 更新（如果是 epoch 级别）
        if hasattr(self, 'ema_teacher'):
            config = getattr(self, 'semi_supervised_config', {})
            ema_update_freq = config.get('ema_update_freq', 'step')
            if ema_update_freq == 'epoch':
                self.ema_teacher.update(self.model._model if hasattr(self.model, '_model') else self.model)
        
        # 调用父类方法
        try:
            super().on_train_epoch_end()
        except AttributeError:
            pass
    
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
