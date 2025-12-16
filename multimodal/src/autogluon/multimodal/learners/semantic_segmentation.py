import importlib.util
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.special import softmax

from autogluon.core.metrics import Scorer

from ..constants import LABEL, LOGITS, SEMANTIC_MASK, SEMANTIC_SEGMENTATION_IMG
from ..optim import SemanticSegmentationLitModule, get_loss_func, get_norm_layer_param_names, get_peft_param_names
from ..optim.metrics.semantic_seg_metrics import Balanced_Error_Rate_Pred as Balanced_Error_Rate
from ..optim.metrics.semantic_seg_metrics import Binary_IoU_Pred as Binary_IoU
from ..optim.metrics.semantic_seg_metrics import Binary_DICE_Pred as Binary_DICE
from ..optim.metrics.semantic_seg_metrics import COD_METRICS_NAMES_Pred as COD_METRICS_NAMES
from ..optim.metrics.semantic_seg_metrics import Multiclass_IoU_Pred as Multiclass_IoU
from ..optim.metrics.semantic_seg_metrics import Multiclass_DICE_Pred as Multiclass_DICE
from ..utils import extract_from_output, setup_save_path
from ..utils.tta_utils import TTAPredictor, dice_coefficient, iou_score
from .base import BaseLearner

logger = logging.getLogger(__name__)

from ..constants import BER, DICE, EM, FM, IOU, MAE, SEMANTIC_SEGMENTATION, SM


class SemanticSegmentationLearner(BaseLearner):
    def __init__(
        self,
        label: Optional[str] = None,
        problem_type: Optional[str] = SEMANTIC_SEGMENTATION,
        presets: Optional[str] = None,
        eval_metric: Optional[Union[str, Scorer]] = "iou",
        hyperparameters: Optional[dict] = None,
        path: Optional[str] = None,
        verbosity: Optional[int] = 2,
        num_classes: Optional[int] = None,  # TODO: can we infer this from data?
        warn_if_exist: Optional[bool] = True,
        enable_progress_bar: Optional[bool] = None,
        pretrained: Optional[bool] = True,
        validation_metric: Optional[str] = "iou",
        sample_data_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            label=label,
            problem_type=problem_type,
            presets=presets,
            eval_metric=eval_metric,
            hyperparameters=hyperparameters,
            path=path,
            verbosity=verbosity,
            warn_if_exist=warn_if_exist,
            enable_progress_bar=enable_progress_bar,
            pretrained=pretrained,
            validation_metric=validation_metric,
        )
        self._output_shape = num_classes
        self._sample_data_path = sample_data_path
        self._tta_predictor = None  # Will be initialized when TTA is enabled

        if self._sample_data_path is not None:
            infer_output_shape = self.get_semantic_segmentation_class_num(self._sample_data_path)
            if num_classes is not None:
                assert (
                    num_classes == infer_output_shape
                ), f"The provided number of classes '{num_classes}' and the inferred class number {infer_output_shape}' from the sample data should be consistent."
            else:
                self._output_shape = infer_output_shape

        # Training-time box prompt configuration (used when constructing LitModule)
        self._train_box_prompt_cfg = {
            "mode": "off",       # off / gt / noisy / mix / predict
            "p_no": 0.0,
            "p_gt": 0.0,
            "p_noisy": 0.0,
            "noise_frac": 0.12,
        }
        # BBoxPromptPredictor instance for "predict" mode (set via external config)
        self._train_bbox_predictor = None

    def get_semantic_segmentation_class_num(self, sample_data_path):
        """
        Get the number of classes for given data.

        Parameters
        ----------
            sample_data_path
                This is used for automatically inference num_classes of semantic segmentation dataset.
                Could be an image directory, image file or pd.DataFrame.
        Returns
        -------
            The number of classes.
        """
        if isinstance(sample_data_path, str):
            if os.path.isdir(sample_data_path):
                mask_files = os.listdir(sample_data_path)
                num_classes = []
                for mask_file in mask_files:
                    per_num_classes = self.get_semantic_segmentation_class_num(
                        os.path.join(sample_data_path, mask_file)
                    )
                    num_classes.append(per_num_classes)
                return max(num_classes)
            else:
                mask = Image.open(sample_data_path)
                mode = mask.mode

                if mode == "1":
                    return 1
                classes = np.unique(mask)
                if mode == "L" and np.array_equal(classes, np.array([0, 255])):
                    return 1

                return max(classes).item() + 1  # include background

        elif isinstance(sample_data_path, pd.DataFrame):
            num_classes = []
            for idx in range(sample_data_path.shape[0]):
                row = sample_data_path.iloc[idx]
                mask_file = row[self._label_column]
                per_num_classes = self.get_semantic_segmentation_class_num(mask_file)
                num_classes.append(per_num_classes)
            return max(num_classes)

    def infer_output_shape(self):
        if self._output_shape is None:
            self._output_shape = self.get_semantic_segmentation_class_num(self._train_data)

    @staticmethod
    def get_peft_param_names_per_run(model, config):
        peft_param_names = None
        peft = config.optim.peft
        if peft:
            norm_param_names = get_norm_layer_param_names(model)
            peft_param_names = get_peft_param_names(
                norm_param_names,
                peft=peft,
                extra_params=config.optim.extra_trainable_params,
            )
        return peft_param_names

    def get_loss_func_per_run(self, config, mixup_active=None):
        loss_func = get_loss_func(
            problem_type=self._problem_type,
            loss_func_name=config.optim.loss_func,
            config=config.optim,
            num_classes=self._output_shape,
        )
        return loss_func, None

    def evaluate_semantic_segmentation(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = False,
    ):
        """
        Evaluate semantic segmentation on a test dataset based on "torchmetrics".

        Parameters
        ----------
        data
            A dataframe, containing the same columns as the training data.
            Or a str, that is a path of the annotation file for detection.
        metrics
            Metrics used for evaluation.
        return_pred
            Whether to return the prediction result of each row.
        realtime
            Whether to do realtime inference, which is efficient for small data (default None).
            If provided None, we would infer it on based on the data modalities
            and sample number.
        """
        
        # If TTA is enabled, use TTA evaluation
        if self._tta_predictor is not None:
            # Use cache settings if configured
            cache_dir = getattr(self, '_tta_cache_dir', None)
            resume_from_cache = getattr(self, '_tta_resume_from_cache', True)
            return self._evaluate_with_tta(
                data, metrics, return_pred, 
                cache_dir=cache_dir,
                resume_from_cache=resume_from_cache
            )

        def get_metric_predict(
            metric_name: str,
            num_classes: Optional[int] = None,
        ):
            """
            Obtain a torchmerics.Metric from its name.
            Define a customized metric function in case that torchmetrics doesn't support some metric.

            Parameters
            ----------
            metric_name
                Name of metric.
            num_classes
                Number of classes.
            is_matching
                Whether is matching.
            problem_type
                Type of problem, e.g., binary and multiclass.

            Returns
            -------
            torchmetrics.Metric
                A torchmetrics.Metric object.
            custom_metric_func
                A customized metric function.
            """
            if metric_name == BER:
                return Balanced_Error_Rate()
            elif metric_name in [SM, EM, FM, MAE]:
                return COD_METRICS_NAMES[metric_name]
            elif metric_name == IOU:
                if num_classes == 1:
                    return Binary_IoU()
                else:
                    return Multiclass_IoU(num_classes=num_classes)
            elif metric_name == DICE:
                if num_classes == 1:
                    return Binary_DICE()
                else:
                    return Multiclass_DICE(num_classes=num_classes)
            else:
                raise ValueError(f"Unknown metric {metric_name}")

        outputs = self.predict_per_run(
            data=data,
            realtime=realtime,
            requires_label=False,
        )

        if self._output_shape == 1:
            logits = extract_from_output(ret_type=LOGITS, outputs=outputs, as_ndarray=False)
        else:
            logits = extract_from_output(ret_type=SEMANTIC_MASK, outputs=outputs, as_ndarray=False)
        y_pred = logits.float()
        y_true = [ele[LABEL] for ele in outputs]
        y_true = torch.cat(y_true)

        assert len(y_true) == len(y_pred)

        results = {}
        if isinstance(metrics, str):
            metrics = [metrics]
        for per_metric_name in metrics:
            per_metric = get_metric_predict(metric_name=per_metric_name.lower(), num_classes=self._output_shape)
            for y_p, y_t in zip(y_pred, y_true):
                per_metric.update(y_p.unsqueeze(0), y_t.unsqueeze(0))
            score = per_metric.compute()
            results[per_metric_name] = score.item()

        if return_pred:
            return results, outputs
        else:
            return results
    
    def _evaluate_with_tta(
        self,
        data: Union[pd.DataFrame, dict, list],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        cache_dir: Optional[str] = None,
        resume_from_cache: bool = True,
    ):
        """
        Evaluate using Test-Time Augmentation with checkpoint support.
        
        This method performs TTA for each image and evaluates the results.
        
        Parameters
        ----------
        data : Union[pd.DataFrame, dict, list]
            Input data containing image and label paths
        metrics : Optional[Union[str, List[str]]]
            Metrics to compute
        return_pred : Optional[bool]
            Whether to return predictions
        cache_dir : Optional[str]
            Directory to cache TTA predictions for resume (default: None, no caching)
        resume_from_cache : bool
            Whether to resume from cache if exists (default: True)
        """
        logger.info(f"Evaluating with TTA ({len(self._tta_predictor.transforms)} augmentations per image)...")
        
        def get_metric_predict(metric_name: str, num_classes: Optional[int] = None):
            """Get metric function."""
            if metric_name == BER:
                return Balanced_Error_Rate()
            elif metric_name in [SM, EM, FM, MAE]:
                return COD_METRICS_NAMES[metric_name]
            elif metric_name == IOU:
                if num_classes == 1:
                    return Binary_IoU()
                else:
                    return Multiclass_IoU(num_classes=num_classes)
            elif metric_name == DICE:
                if num_classes == 1:
                    return Binary_DICE()
                else:
                    return Multiclass_DICE(num_classes=num_classes)
            else:
                raise ValueError(f"Unknown metric {metric_name}")
        
        # Get model and preprocessor
        model = self._model
        model.eval()
        
        # Ensure model is on the correct device (GPU if available)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            if next(model.parameters()).device.type != 'cuda':
                logger.info("Moving model to GPU for TTA inference...")
                model = model.cuda()
        else:
            device = torch.device('cpu')
            logger.warning("GPU not available, using CPU for TTA (will be slow)")
        
        logger.info(f"TTA inference device: {device}")
        
        # ============================================================
        # 方案 4: Sanity Check - 提前发现问题
        # ============================================================
        logger.info("🔍 Running sanity checks before full evaluation...")
        
        # Sanity Check 1: 验证数据
        if len(data) == 0:
            raise ValueError("❌ No data to evaluate!")
        logger.info(f"  ✓ Total images to process: {len(data)}")
        logger.info(f"  ✓ DataFrame columns: {data.columns.tolist()}")
        
        # Sanity Check 2: 检查设备和模型
        logger.info(f"  ✓ Model device: {device}")
        logger.info(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            logger.info(f"  ✓ GPU memory allocated: {allocated_gb:.2f} GB")
        
        # Sanity Check 3: 测试单个样本的 TTA pipeline
        logger.info(f"  Testing TTA pipeline on first image...")
        test_row = data.iloc[0]
        test_img_path = test_row['image']
        
        # Get data processors for image preprocessing
        from torchvision import transforms as T
        
        # Create preprocessing pipeline similar to model's preprocessing
        # Based on SAM's image preprocessing
        def preprocess_image_for_sam(image_np):
            """Preprocess image for SAM model."""
            from PIL import Image as PILImage
            
            # Ensure correct format
            if len(image_np.shape) == 2:
                image_np = np.stack([image_np] * 3, axis=2)
            
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            
            # Convert to PIL
            img_pil = PILImage.fromarray(image_np.astype(np.uint8))
            
            # Resize to model's expected input size (typically 1024x1024 for SAM)
            target_size = model.image_size
            img_resized = img_pil.resize((target_size, target_size), PILImage.BILINEAR)
            
            # Convert to tensor and normalize
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            
            # Close PIL objects
            img_pil.close()
            img_resized.close()
            
            # Apply model's normalization (SAM uses specific mean/std)
            mean = np.array(model.image_mean).reshape(1, 1, 3)
            std = np.array(model.image_std).reshape(1, 1, 3)
            img_array = (img_array - mean) / std
            
            # Convert to tensor (C, H, W)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float().contiguous()
            
            # Free intermediate array
            del img_array
            
            return img_tensor
        
        # Create a prediction function for TTA
        def predict_fn(
            image: np.ndarray,
            box_prompt: Optional[np.ndarray] = None,
            box_ref_shape: Optional[Tuple[int, int]] = None,
        ) -> np.ndarray:
            """
            Prediction function for a single image used by TTA.
            
            Parameters
            ----------
            image : np.ndarray
                Input image (H, W, C) in range [0, 255] or [0, 1]
            
            Returns
            -------
            prob : np.ndarray
                Probability map (H, W) for binary or (C, H, W) for multi-class
            """
            # Preprocess image
            img_tensor = preprocess_image_for_sam(image)
            
            # Add batch dimension and move to device
            img_tensor = img_tensor.unsqueeze(0).to(device, non_blocking=True)

            box_tensor = None
            if box_prompt is not None:
                # box_prompt stays in ORIGINAL image coords (no TTA transform)
                ref_h, ref_w = box_ref_shape if box_ref_shape is not None else image.shape[:2]
                scale_x = model.image_size / float(ref_w)
                scale_y = model.image_size / float(ref_h)
                box_scaled = box_prompt.astype(np.float32).copy()
                box_scaled[0::2] *= scale_x
                box_scaled[1::2] *= scale_y
                box_scaled = np.clip(box_scaled, 0, model.image_size - 1)
                box_tensor = torch.from_numpy(box_scaled).to(device).unsqueeze(0).unsqueeze(0)  # (1,1,4)
            
            def forward_prob(optional_box_tensor: Optional[torch.Tensor]) -> np.ndarray:
                # Create batch dict for model
                # Need to provide a dummy label for model's forward pass
                batch = {
                    model.prefix + "_image": img_tensor,
                    model.prefix + "_label": torch.zeros(
                        (1, model.image_size, model.image_size), dtype=torch.long, device=device
                    ),
                }
                if optional_box_tensor is not None:
                    batch[model.box_key] = optional_box_tensor
                outputs = model(batch)

                logits = outputs[model.prefix][LOGITS]
                if self._output_shape == 1:
                    # Binary segmentation - remove channel dim
                    prob = torch.sigmoid(logits[0, 0]).cpu().numpy().copy()
                else:
                    # Multi-class segmentation
                    prob = torch.softmax(logits[0], dim=0).cpu().numpy().copy()

                # Free GPU memory immediately and aggressively
                del batch, outputs, logits
                return prob

            # Forward pass
            with torch.no_grad():
                mode = getattr(self, "_tta_box_prompt_mode", "off")
                if mode == "off" or box_tensor is None:
                    # No box prompt
                    prob = forward_prob(None)
                elif mode == "replace":
                    # Use only box-prompted prediction
                    prob = forward_prob(box_tensor)
                elif mode == "add":
                    # Fuse base prediction + box-prompted prediction (treat box as an extra inference view)
                    prob_base = forward_prob(None)
                    prob_box = forward_prob(box_tensor)
                    prob = ((prob_base + prob_box) / 2.0).astype(np.float32, copy=False)
                    del prob_base, prob_box
                else:
                    raise ValueError(f"Unknown tta_box_prompt_mode: {mode}")

                # Free GPU memory immediately and aggressively
                del img_tensor
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for GPU to finish

            return prob
        
        # Complete Sanity Check 3: Test TTA on first image
        try:
            test_img_pil = Image.open(test_img_path)
            test_img = np.array(test_img_pil)
            test_img_pil.close()
            
            if len(test_img.shape) == 2:
                test_img = np.stack([test_img] * 3, axis=2)
            
            logger.info(f"    Image shape: {test_img.shape}")
            
            # Test TTA prediction
            test_pred_mask, test_pred_prob = self._tta_predictor.predict_with_tta(
                test_img, predict_fn, return_probs=True
            )
            
            logger.info(f"    Output shape: {test_pred_prob.shape}")
            
            # Verify output shape matches input
            assert test_pred_prob.shape[:2] == test_img.shape[:2], \
                f"❌ Output shape mismatch! Expected {test_img.shape[:2]}, got {test_pred_prob.shape[:2]}"
            
            logger.info(f"  ✓ TTA pipeline test PASSED!")
            
            # Clean up test data
            del test_img, test_pred_mask, test_pred_prob
            
        except Exception as e:
            logger.error(f"❌ Sanity check FAILED! Error: {e}")
            logger.error("Please fix the issue before running full evaluation.")
            raise RuntimeError(f"TTA Sanity Check Failed: {e}") from e
        
        logger.info("✅ All sanity checks passed! Starting full evaluation...\n")
        
        # ============================================================
        # 方案 2: 断点续传机制
        # ============================================================
        import os
        import pickle
        
        # Setup cache directory
        processed_indices = set()
        all_preds = []
        all_labels = []
        cache_file = None
        processed_indices_file = None
        
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, "tta_predictions.pkl")
            processed_indices_file = os.path.join(cache_dir, "processed_indices.txt")
            
            # Try to load cached data
            if resume_from_cache and os.path.exists(cache_file):
                try:
                    logger.info(f"📂 Loading cached predictions from {cache_file}")
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                        all_preds = cache_data['preds']
                        all_labels = cache_data['labels']
                        processed_indices = set(cache_data['indices'])
                    logger.info(f"✅ Loaded {len(all_preds)} cached predictions (resuming from checkpoint)")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load cache: {e}. Starting fresh.")
                    processed_indices = set()
                    all_preds = []
                    all_labels = []
            else:
                if cache_dir:
                    logger.info(f"💾 Caching enabled. Predictions will be saved to {cache_dir}")
        
        # Initialize metrics early so we can stream updates and avoid storing all predictions
        metric_objects = {}
        if isinstance(metrics, str):
            metrics = [metrics]
        for per_metric_name in metrics:
            metric_objects[per_metric_name] = get_metric_predict(
                metric_name=per_metric_name.lower(),
                num_classes=self._output_shape
            )

        # Determine whether to stream metrics (no caching => no large tensors kept)
        streaming_metrics = cache_dir is None

        # Process each image with TTA
        import time
        import gc
        start_time = time.time()
        total_images = len(data)

        for idx, row in data.iterrows():
            # ⭐ 跳过已处理的样本（断点续传）
            if idx in processed_indices:
                logger.info(f"⏭️  Skipping image {idx+1}/{total_images} (already processed)")
                continue

            img_start = time.time()

            # Load image and label
            image_path = row['image']
            label_path = row['label'] if 'label' in row else None

            # Load image with proper resource management
            img_pil = Image.open(image_path)
            image = np.array(img_pil)
            img_pil.close()  # Release file handle

            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=2)  # Convert grayscale to RGB

            # Load label if exists
            label = None
            gt_box = None
            if label_path:
                label_pil = Image.open(label_path)
                label = np.array(label_pil)
                label_pil.close()  # Release file handle

                if len(label.shape) == 3:
                    label = label[:, :, 0]  # Take first channel if RGB

                # Binarize label: convert [0, 255] to [0, 1]
                # This matches the preprocessing in semantic_seg_metrics.py
                if self._output_shape == 1:  # Binary segmentation
                    label = (label > 128).astype(np.int64)

                if not streaming_metrics:
                    all_labels.append(torch.from_numpy(label))

                # Compute GT bounding box (single box per sample, original coords)
                ys, xs = np.where(label > 0)
                if xs.size > 0 and ys.size > 0:
                    gt_box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

            # If predicted boxes are provided in DataFrame, override gt_box
            if {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"}.issubset(row.index):
                vals = [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]]
                if not any([v is None or (isinstance(v, float) and np.isnan(v)) for v in vals]):
                    gt_box = np.array(vals, dtype=np.float32)

            # Predict with TTA
            if self._output_shape == 1:
                # Binary segmentation
                pred_mask, pred_prob = self._tta_predictor.predict_with_tta(
                    image, predict_fn, return_probs=True, box_prompt=gt_box, box_ref_shape=image.shape[:2]
                )
                pred_tensor = torch.from_numpy(pred_prob)
                if not streaming_metrics:
                    # Store only when caching/resume is enabled
                    all_preds.append(pred_tensor)
                # Free memory
                del pred_mask, pred_prob
            else:
                # Multi-class segmentation
                _, pred_prob = self._tta_predictor.predict_with_tta(
                    image, predict_fn, return_probs=True, box_prompt=gt_box, box_ref_shape=image.shape[:2]
                )
                pred_tensor = torch.from_numpy(pred_prob)
                if not streaming_metrics:
                    all_preds.append(pred_tensor)
                # Free memory
                del pred_prob

            # Stream metric updates to avoid holding all predictions in memory
            if streaming_metrics:
                y_p = pred_tensor.float()
                if label is None:
                    raise ValueError("Labels are required for evaluation")
                y_t = torch.from_numpy(label)
                y_t = y_t.long() if y_t.dtype != torch.long else y_t

                # Resize prediction to match label size if needed
                if y_p.shape[-2:] != y_t.shape[-2:]:
                    target_size = y_t.shape[-2:]
                    y_p = F.interpolate(
                        y_p.unsqueeze(0),
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)

                for metric_name, metric_obj in metric_objects.items():
                    metric_obj.update(y_p.unsqueeze(0), y_t.unsqueeze(0))

            # Free image and label memory
            del image
            if label is not None:
                del label

            # ⭐ 标记为已处理（断点续传）
            processed_indices.add(idx)

            img_time = time.time() - img_start

            if idx == 0:
                logger.info(f"First image processed in {img_time:.2f}s (includes warmup)")

            # ⭐ 定期保存缓存（每 10 张图或最后一张）
            num_processed = len(processed_indices)
            if (not streaming_metrics) and cache_file is not None and (num_processed % 10 == 0 or num_processed == total_images):
                try:
                    logger.info(f"💾 Saving checkpoint ({num_processed}/{total_images} images)")
                    with open(cache_file, 'wb') as f:
                        pickle.dump({
                            'preds': all_preds,
                            'labels': all_labels,
                            'indices': list(processed_indices)
                        }, f)

                    # Save processed indices as text file (for easy inspection)
                    with open(processed_indices_file, 'w') as f:
                        f.write('\n'.join(map(str, sorted(processed_indices))))

                    logger.info(f"✅ Checkpoint saved successfully")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to save checkpoint: {e}")

            # More aggressive garbage collection to prevent memory leaks
            if (idx + 1) % 5 == 0:  # Every 5 images instead of 10
                gc.collect()  # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Clear GPU cache

            # Progress logging
            if (idx + 1) % 10 == 0 or num_processed == total_images:
                elapsed = time.time() - start_time
                avg_time = elapsed / num_processed if num_processed > 0 else 0
                remaining = total_images - num_processed
                eta = avg_time * remaining if remaining > 0 else 0
                logger.info(
                    f"Processed {num_processed}/{total_images} images with TTA "
                    f"(avg: {avg_time:.2f}s/img, ETA: {eta/60:.1f}min)"
                )

                # ⭐ 中间指标：在评估过程中周期性输出当前已处理样本的指标
                # 说明：
                # - 仅在 streaming_metrics=True（即未开启 cache_dir，走低内存流式评估）时启用
                # - 使用 torchmetrics.compute() 不会清空内部状态，因此可以安全多次调用
                if streaming_metrics:
                    try:
                        intermediate_results = {}
                        for metric_name, metric_obj in metric_objects.items():
                            score = metric_obj.compute()
                            # score 可能是 tensor 或数值，这里统一为 float 便于日志展示
                            if hasattr(score, "item"):
                                score = score.item()
                            intermediate_results[metric_name] = float(score)
                        if intermediate_results:
                            logger.info(
                                f"Intermediate TTA metrics after {num_processed} images: "
                                f"{intermediate_results}"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to compute intermediate metrics: {e}")
        
        # Final garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"Finished processing all {len(data)} images, computing metrics...")

        results = {}
        if streaming_metrics:
            # Metrics were updated on the fly; just compute
            for metric_name, metric_obj in metric_objects.items():
                score = metric_obj.compute()
                results[metric_name] = score.item()
        else:
            # Check that we have predictions and labels
            if len(all_labels) == 0:
                raise ValueError("Labels are required for evaluation")
            
            if len(all_preds) != len(all_labels):
                raise ValueError(f"Number of predictions ({len(all_preds)}) != number of labels ({len(all_labels)})")

            # Process each image individually (can't stack due to different sizes)
            for y_p, y_t in zip(all_preds, all_labels):
                # Ensure predictions and labels are on CPU and have matching sizes
                y_p = y_p.float()
                y_t = y_t.long() if y_t.dtype != torch.long else y_t
                
                # Resize prediction to match label size if needed
                if y_p.shape[-2:] != y_t.shape[-2:]:
                    target_size = y_t.shape[-2:]
                    y_p = F.interpolate(
                        y_p.unsqueeze(0),  # Add batch dim
                        size=target_size,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)  # Remove batch dim
                
                # Update all metrics with this image
                for metric_name, metric_obj in metric_objects.items():
                    metric_obj.update(y_p.unsqueeze(0), y_t.unsqueeze(0))

            # Compute final scores
            for metric_name, metric_obj in metric_objects.items():
                score = metric_obj.compute()
                results[metric_name] = score.item()

        logger.info(f"TTA evaluation completed: {results}")
        
        # ⭐ 清理缓存文件（评估成功完成后）
        if cache_file is not None and os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                if processed_indices_file and os.path.exists(processed_indices_file):
                    os.remove(processed_indices_file)
                logger.info(f"🗑️  Cleaned up cache files (evaluation completed successfully)")
            except Exception as e:
                logger.warning(f"⚠️  Failed to clean up cache: {e}")
        
        # Free memory
        del all_preds, all_labels, metric_objects
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if return_pred:
            logger.warning("return_pred=True not fully supported with TTA (memory optimization)")
            return results, []
        else:
            return results

    def get_litmodule_per_run(
        self,
        model=None,
        model_postprocess_fn=None,
        peft_param_names=None,
        optim_kwargs=None,
        distillation_kwargs=None,
        is_train=True,
    ):
        gspo_trainer = None
        if is_train:
            gspo_trainer = self._maybe_create_gspo_trainer()
        if is_train:
            return SemanticSegmentationLitModule(
                model=model,
                model_postprocess_fn=model_postprocess_fn,
                trainable_param_names=peft_param_names,
                gspo_trainer=gspo_trainer,
                train_box_prompt_cfg=self._train_box_prompt_cfg,
                train_bbox_predictor=self._train_bbox_predictor,
                **optim_kwargs,
            )
        else:
            return SemanticSegmentationLitModule(
                model=self._model,
                model_postprocess_fn=self._model_postprocess_fn,
                train_box_prompt_cfg=self._train_box_prompt_cfg,
                train_bbox_predictor=self._train_bbox_predictor,
                **optim_kwargs,
            )

    def _maybe_create_gspo_trainer(self):
        """
        Auto-create GSPO trainer if the config enables GSPO.
        The trainer implementation lives in examples/automm/Conv-LoRA/gspo_trainer.py.
        """
        if self._config is None:
            return None

        lora_cfg = getattr(self._config.optim, "lora", None)
        if not lora_cfg or not getattr(lora_cfg, "gspo_enabled", False):
            return None

        gspo_cfg = getattr(self._config.optim, "gspo", None)
        lambda_smooth = getattr(gspo_cfg, "lambda_smooth", 0.1) if gspo_cfg else 0.1
        lambda_boundary = getattr(gspo_cfg, "lambda_boundary", 0.3) if gspo_cfg else 0.3
        w_boundary = getattr(gspo_cfg, "w_boundary", 0.3) if gspo_cfg else 0.3
        w_smooth = getattr(gspo_cfg, "w_smooth", 0.1) if gspo_cfg else 0.1
        w_thin = getattr(gspo_cfg, "w_thin", 0.05) if gspo_cfg else 0.05

        # Dynamically load GSPOConvLoRATrainer from examples path.
        examples_root = Path(__file__).resolve().parents[3] / "examples" / "automm" / "Conv-LoRA" / "gspo_trainer.py"
        if not examples_root.exists():
            logger.warning("GSPO is enabled but gspo_trainer.py not found at %s; GSPO will be skipped.", examples_root)
            return None

        spec = importlib.util.spec_from_file_location("autogluon_multimodal.gspo_trainer", examples_root)
        if spec is None or spec.loader is None:
            logger.warning("Failed to load GSPO trainer spec from %s; GSPO will be skipped.", examples_root)
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        GSPOConvLoRATrainer = getattr(module, "GSPOConvLoRATrainer", None)
        if GSPOConvLoRATrainer is None:
            logger.warning("GSPOConvLoRATrainer not found in %s; GSPO will be skipped.", examples_root)
            return None

        try:
            return GSPOConvLoRATrainer(
                predictor=None,  # Not required by current implementation
                group_size=getattr(lora_cfg, "gspo_group_size", 3),
                warmup_epochs=getattr(lora_cfg, "gspo_warmup_epochs", 5),
                contrastive_weight=getattr(lora_cfg, "gspo_contrastive_weight", 0.1),
                quality_metric="iou",
                advantage_temperature=5.0,
                lambda_smooth=lambda_smooth,
                lambda_boundary=lambda_boundary,
                w_boundary=w_boundary,
                w_smooth=w_smooth,
                w_thin=w_thin,
            )
        except Exception as e:
            logger.warning("Failed to instantiate GSPOConvLoRATrainer: %s; GSPO will be skipped.", e)
            return None

    def on_predict_start(self, data: pd.DataFrame):
        data = self.data_to_df(data=data)
        if self._output_shape is None:  # for zero-shot evaluation/prediction
            self._output_shape = self.get_semantic_segmentation_class_num(data)
        self.ensure_predict_ready()
        return data
    
    def enable_tta(
        self,
        scales: List[float] = [0.75, 1.0, 1.25],
        flips: List[str] = ["none", "horizontal"],
        rotations: List[float] = [0],
        fusion_method: str = "mean",
        scale_weights: Optional[Dict[float, float]] = None,
        threshold: float = 0.5,
        min_area_ratio: float = 0.001,
        use_morphology: bool = False,
        resize_method: str = "bilinear",
        cache_dir: Optional[str] = None,
        resume_from_cache: bool = True,
        box_prompt_mode: str = "off",
    ):
        """
        Enable Test-Time Augmentation (TTA) for inference.
        
        Parameters
        ----------
        scales : List[float]
            List of scale factors for multi-scale testing.
            Recommended: [0.75, 1.0, 1.25] for fast (6 inferences).
            Or [0.5, 0.75, 1.0, 1.25, 1.5] for better results.
        flips : List[str]
            List of flip types: "none", "horizontal", "vertical".
            Recommended: ["none", "horizontal"] for most cases.
        rotations : List[float]
            List of rotation angles in degrees.
            Recommended: [0] for fast, or [-10, 0, 10] for better (18 inferences).
        fusion_method : str
            Method to fuse predictions: "mean" or "weighted_mean".
        scale_weights : Optional[Dict[float, float]]
            Weights for each scale when using weighted_mean.
        threshold : float
            Threshold for binary segmentation (can be tuned on validation set).
        min_area_ratio : float
            Remove connected components with area < min_area_ratio * image_area.
        use_morphology : bool
            Whether to apply morphological closing for smoothing.
        resize_method : str
            Resize interpolation method: "bilinear" or "bicubic".
            Recommended: "bilinear" for speed, "bicubic" for quality.
        cache_dir : Optional[str]
            Directory to cache TTA predictions for resume (default: None, no caching).
            Useful for long-running evaluations that might be interrupted.
        resume_from_cache : bool
            Whether to resume from cache if exists (default: True).
        
        Examples
        --------
        >>> # Basic TTA (6 inferences)
        >>> predictor.enable_tta()
        
        >>> # Advanced TTA (18 inferences)
        >>> predictor.enable_tta(
        ...     scales=[0.75, 1.0, 1.25],
        ...     flips=["none", "horizontal"],
        ...     rotations=[-10, 0, 10],
        ...     fusion_method="weighted_mean",
        ...     resize_method="bicubic"
        ... )
        """
        self._tta_predictor = TTAPredictor(
            scales=scales,
            flips=flips,
            rotations=rotations,
            fusion_method=fusion_method,
            scale_weights=scale_weights,
            threshold=threshold,
            min_area_ratio=min_area_ratio,
            use_morphology=use_morphology,
            resize_method=resize_method,
        )
        
        # Store cache configuration
        self._tta_cache_dir = cache_dir
        self._tta_resume_from_cache = resume_from_cache
        self._tta_box_prompt_mode = box_prompt_mode
        
        logger.info(f"TTA enabled with {len(self._tta_predictor.transforms)} augmentations")
        if cache_dir:
            logger.info(f"TTA caching enabled: {cache_dir} (resume={resume_from_cache})")
    
    def disable_tta(self):
        """Disable Test-Time Augmentation."""
        self._tta_predictor = None
        logger.info("TTA disabled")

    def evaluate(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        metrics: Optional[Union[str, List[str]]] = None,
        return_pred: Optional[bool] = False,
        realtime: Optional[bool] = False,
        **kwargs,
    ):
        """
        Evaluate model on a test dataset.

        Parameters
        ----------
        data
            A dataframe, containing the same columns as the training data.
            Or a str, that is a path of the annotation file for detection.
        metrics
            A list of metric names to report.
            If None, we only return the score for the stored `_eval_metric_name`.
        return_pred
            Whether to return the prediction result of each row.
        realtime
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        A dictionary with the metric names and their corresponding scores.
        Optionally return a dataframe of prediction results.
        """
        data = self.on_predict_start(data)
        return self.evaluate_semantic_segmentation(data, metrics, realtime)

    def predict(
        self,
        data: Union[pd.DataFrame, dict, list, str],
        realtime: Optional[bool] = False,
        save_results: Optional[bool] = None,
        **kwargs,
    ):
        """
        Predict values for the label column of new data.

        Parameters
        ----------
        data
            The data to make predictions for. Should contain same column names as training data and
            follow same format (except for the `label` column).
        realtime
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.
        save_results
            Whether to save the prediction results.

        Returns
        -------
        Array of predictions, one corresponding to each row in given dataset.
        When save_results is True, the output is a pandas dataframe containing the path of the predicted mask file for each input image.
        Otherwise, the output will have shape (#samples, height, width).
        """
        data = self.on_predict_start(data)
        if self._output_shape == 1:
            ret_type = LOGITS
        else:
            ret_type = SEMANTIC_MASK

        outputs = self.predict_per_run(
            data=data,
            realtime=realtime,
            requires_label=False,
        )

        logits = self.post_process_prediction(data, outputs, ret_type)

        pred = []
        for logit in logits:
            logit = logit.numpy()
            if ret_type == SEMANTIC_MASK:
                pred.append(logit.argmax(axis=1))
            else:
                pred.append((logit > 0.5).squeeze(axis=1))

        if save_results:
            self._save_path = setup_save_path(
                old_save_path=self._save_path,
                warn_if_exist=False,
            )
            pred = self.save_segmentation_result(
                pred=pred,
                data=data,
                result_path=self._save_path,
            )

        return pred

        # if (as_pandas is None and isinstance(data, pd.DataFrame)) or as_pandas is True:
        #     # TODO
        #     pred = self._as_pandas(data=data, to_be_converted=pred)

    def predict_proba(
        self,
        data: Union[pd.DataFrame, dict, list],
        as_pandas: Optional[bool] = None,
        as_multiclass: Optional[bool] = True,
        realtime: Optional[bool] = False,
        **kwargs,
    ):
        """
        Predict probabilities class probabilities rather than class labels.
        This is only for the classification. Calling it for regression will throw an exception.

        Parameters
        ----------
        data
            The data to make predictions for. Should contain same column names as training data and
              follow same format (except for the `label` column).
        as_pandas
            Whether to return the output as a pandas DataFrame(Series) (True) or numpy array (False).
        as_multiclass
            Whether to return the probability of all labels or
            just return the probability of the positive class for binary classification problems.
        realtime
            Whether to do realtime inference, which is efficient for small data (default False).
            If provided None, we would infer it on based on the data modalities
            and sample number.

        Returns
        -------
        Array of predicted class-probabilities, corresponding to each row in the given data.
        The output will always have shape (#samples, #classes, height, width).
        """
        assert (self._output_shape == 1 and as_multiclass == False) or (
            self._output_shape > 1 and as_multiclass == True
        )
        data = self.on_predict_start(data)

        outputs = self.predict_per_run(
            data=data,
            realtime=realtime,
            requires_label=False,
        )

        if as_multiclass:
            ret_type = SEMANTIC_MASK
        else:
            ret_type = LOGITS

        logits = self.post_process_prediction(data, outputs, ret_type)

        prob = []
        for logit in logits:
            logit = logit.numpy()
            if ret_type == SEMANTIC_MASK:
                prob.append(softmax(logit, axis=1))
            else:
                prob.append(logit)

        return prob

    def extract_embedding(
        self,
        data: Union[pd.DataFrame, dict, list],
        as_tensor: Optional[bool] = False,
        as_pandas: Optional[bool] = False,
        realtime: Optional[bool] = False,
        **kwargs,
    ):
        raise NotImplementedError("Semantic segmentation doesn't support calling `extract_embedding` yet.")

    def save_segmentation_result(self, pred: Iterable, data: Union[pd.DataFrame, Dict], result_path: str):
        """
        Saving segmentation results in pd.DataFrame format (per image)

        Parameters
        ----------
        pred
            List containing segmentation results for one image
        data
            Pandas data frame or dict containing the image information to be tested
        result_path
            Path to save result
        Returns
        -------
        The paths of the segmentation results as pandas DataFrame
        """

        def show_mask(mask, ax):
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)

        image_column_name = self.get_image_column_name(data)
        if isinstance(data, dict):
            image_names = data[image_column_name]
        else:
            image_names = data[image_column_name].to_list()
        results = []

        mask_path = os.path.join(result_path, "masks")
        txt_path = os.path.join(result_path, "result.txt")
        os.makedirs(mask_path, exist_ok=True)
        for image_pred, image_name in zip(pred, image_names):
            if self._output_shape == 1:
                mask = Image.fromarray(image_pred.squeeze(axis=0))
                per_mask_path = os.path.join(mask_path, os.path.basename(image_name))
                mask.save(per_mask_path)
            else:
                masks = []
                classes = np.unique(image_pred)
                for class_id in classes:
                    if class_id == 0:  # bg
                        continue
                    masks.append(image_pred == class_id)

                for mask in masks:
                    show_mask(mask, plt.gca())
                mask_name = ""
                for i in os.path.basename(image_name).split(".")[:-1]:
                    mask_name += i
                per_mask_path = os.path.join(mask_path, os.path.basename(image_name))
                plt.axis("off")
                plt.savefig(per_mask_path, bbox_inches="tight", dpi=300, pad_inches=0.0)

            results.append([image_name, per_mask_path])

        result_df = pd.DataFrame(results, columns=["image", "mask"])
        result_df.to_csv(txt_path, index=False)
        return result_df

    def post_process_prediction(self, data, outputs, ret_type):
        """
        Post-process segmentation results to match the size of original input images.

        Parameters
        ----------
        data
            Pandas data frame or dict containing the image information.
        outputs
            A list of segmentation output results.
        ret_type
            What kind of information to extract from model outputs.

        Returns
        -------
        A list of the post-processed segmentation results.
        """
        logits = [ele[ret_type] for ele in outputs]
        image_column_name = self.get_image_column_name(data)
        for idx in range(data.shape[0]):
            ori_image_size = Image.open(data[image_column_name][idx]).size  # width, height
            logits[idx] = F.interpolate(
                logits[idx].float(), (ori_image_size[1], ori_image_size[0]), mode="bilinear", align_corners=False
            )
        return logits

    def get_image_column_name(self, data: pd.DataFrame):
        if self.column_types is None:
            column_names = list(data.columns)
            if self._label_column in column_names:
                column_names.remove(self._label_column)
            assert (
                len(column_names) == 1
            ), f"More than one image columns {column_names} exist in the data. Make sure to provide data with one image column."
            return column_names[0]
        else:
            for k, v in self.column_types.items():
                if v == SEMANTIC_SEGMENTATION_IMG:
                    return k
        return None
