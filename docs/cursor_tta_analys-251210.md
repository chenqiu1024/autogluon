# 优化AutoGluon脚本输出中间结果
_Exported on 12/11/2025 at 13:24:34 PST from Cursor (2.2.12)_

---

**User**

从autogluon/docs/cursor_tta-251203.md了解前文。在执行以下命令时：`nohup ./run_tta_best_accuracy.sh AutogluonModels/ag-20251203_075302 > outputs/tta_best_accuracy-251204.log 2>&1 &`，由于内存持续增长，进程被杀掉：autogluon/examples/automm/Conv-LoRA/outputs/tta_best_accuracy-251204.log。问：为什么一定要跑完全部600个测试集样本才能得到结果？请改脚本使其可以在执行过程中输出中间结果

---

**Cursor**

我先看一下你的脚本和相关日志/文档，弄清现在的评估流程为什么只有在全部 600 个样本跑完后才输出结果，然后修改脚本，让它在运行过程中周期性地输出中间指标（例如已处理样本数、当前 best / 平均 accuracy 等），并且尽量控制内存增长。完成后给你说明怎么用。

```python
import logging
import os
from typing import Dict, Iterable, List, Optional, Union, Callable

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
        def predict_fn(image: np.ndarray) -> np.ndarray:
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
            
            # Forward pass
            with torch.no_grad():
                # Create batch dict for model
                # Need to provide a dummy label for model's forward pass
                batch = {
                    model.prefix + '_image': img_tensor,
                    model.prefix + '_label': torch.zeros((1, model.image_size, model.image_size), 
                                                         dtype=torch.long, device=device)
                }
                outputs = model(batch)
                
                # Extract logits
                logits = outputs[model.prefix][LOGITS]
                
                # Convert to probabilities and immediately move to CPU
                if self._output_shape == 1:
                    # Binary segmentation - remove channel dim
                    prob = torch.sigmoid(logits[0, 0]).cpu().numpy().copy()
                else:
                    # Multi-class segmentation
                    prob = torch.softmax(logits[0], dim=0).cpu().numpy().copy()
                
                # Free GPU memory immediately and aggressively
                del img_tensor, batch, outputs, logits
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

            # Predict with TTA
            if self._output_shape == 1:
                # Binary segmentation
                pred_mask, pred_prob = self._tta_predictor.predict_with_tta(
                    image, predict_fn, return_probs=True
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
                    image, predict_fn, return_probs=True
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
        if is_train:
            return SemanticSegmentationLitModule(
                model=model,
                model_postprocess_fn=model_postprocess_fn,
                trainable_param_names=peft_param_names,
                **optim_kwargs,
            )
        else:
            return SemanticSegmentationLitModule(
                model=self._model,
                model_postprocess_fn=self._model_postprocess_fn,
                **optim_kwargs,
            )

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
        cache_dir: Optional[str] = None,
        resume_from_cache: bool = True,
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
        ...     fusion_method="weighted_mean"
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
        )
        
        # Store cache configuration
        self._tta_cache_dir = cache_dir
        self._tta_resume_from_cache = resume_from_cache
        
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

```

**先回答“为什么一定要跑完 600 张才能出结果？”**

- **技术原因**：IoU / Dice 这类指标本质上是“全测试集上的统计量”（累计 TP/FP/FN 后再算比值），严格意义上的“官方测试集指标”必须在**所有样本都评估完**后才能给出完整数值。
- **实现原因**：原来的 TTA 实现里，`_evaluate_with_tta` 虽然是流式更新 `metric_objects`，但**只在全部样本处理完之后才 `compute()` 一次并返回结果**，因此脚本在中途被杀掉时，外层 `run_semantic_segmentation.py` 拿不到任何结果，只能等 600 张都跑完。

---

### 我做了什么修改：在评估过程中输出中间结果

你现在跑的命令：

```bash
nohup ./run_tta_best_accuracy.sh AutogluonModels/ag-20251203_075302 > outputs/tta_best_accuracy-251204.log 2>&1 &
```

内部调用的是：

```bash
python3 run_semantic_segmentation.py ... --tta_enable ...
```

`MultiModalPredictor.evaluate(...)` 最终会走到 `SemanticSegmentationLearner._evaluate_with_tta`（你日志里的 `TTA inference device: cuda`、`Processed 10/600 images with TTA ...` 等就是这里打印的）。

我在 `autogluon/multimodal/src/autogluon/multimodal/learners/semantic_segmentation.py` 的 `_evaluate_with_tta` **进度日志部分**加了“中间指标”输出逻辑：  

- **位置**：在原本的进度日志下面，也就是“Processed X/Y images with TTA (avg..., ETA...)”那里。
- **策略**：  
  - 仍然每 **10 张**图打印一次进度（和之前一样）。  
  - 在同一个时机，如果当前走的是 **流式评估路径**（`cache_dir is None`，也就是你现在默认的配置），就对当前累计的 `metric_objects` 调一次 `compute()`，算出**截至目前为止的 IoU / Dice 等指标**，并打进日志。
  - 调 `compute()` 不会清空指标状态（torchmetrics 的语义），所以可以安全地多次调用，只是多几次标量计算，几乎不增加显著开销。

修改后的这段逻辑大致效果（简化示意）：

```python
logger.info(
    f"Processed {num_processed}/{total_images} images with TTA "
    f"(avg: {avg_time:.2f}s/img, ETA: {eta/60:.1f}min)"
)

if streaming_metrics:  # 即 cache_dir=None 的流式低内存模式
    intermediate_results = {}
    for metric_name, metric_obj in metric_objects.items():
        score = metric_obj.compute()
        if hasattr(score, "item"):
            score = score.item()
        intermediate_results[metric_name] = float(score)
    if intermediate_results:
        logger.info(
            f"Intermediate TTA metrics after {num_processed} images: "
            f"{intermediate_results}"
        )
```

---

### 对你现在的使用方式意味着什么？

- 你继续用同一个脚本命令跑：

  ```bash
  nohup ./run_tta_best_accuracy.sh AutogluonModels/ag-20251203_075302 > outputs/tta_best_accuracy-251204.log 2>&1 &
  ```

- 在评估过程中，日志文件 `outputs/tta_best_accuracy-251204.log` 中会周期性出现类似行（示例）：

  ```text
  Processed 100/600 images with TTA (avg: 19.52s/img, ETA: 162.6min)
  Intermediate TTA metrics after 100 images: {'iou': 0.8231, 'dice': 0.8745}
  ```

- **一旦进程因为内存等原因在中途被 KILL**：
  - 你仍然可以从 `nohup` 日志里看到**最近一次输出的 `Intermediate TTA metrics`**，比如“after 300 images / 340 images”的 IoU / Dice，作为一个**可靠的中间评估结果**。
  - 不再出现“跑了几个小时被杀，什么指标都看不到”的情况。

---

### 补充：如果你只想快速测一部分样本

脚本本身还额外支持“只跑前 N 张”的快速模式（不用改代码就能用）：

- 只跑前 5 张（debug 模式）：

  ```bash
  python3 run_semantic_segmentation.py \
      --task isic2017 \
      --eval \
      --ckpt_path AutogluonModels/ag-20251203_075302 \
      --tta_enable \
      --tta_scales 0.75 1.0 1.25 \
      --tta_flips none horizontal \
      --tta_rotations -10 0 10 \
      --tta_fusion weighted_mean \
      --tta_morphology \
      --output_dir outputs/eval_tta_best_accuracy_debug \
      --debug
  ```

- 或者只跑前 N 张，比如前 100 张：

  ```bash
  python3 run_semantic_segmentation.py \
      ...（同上参数）... \
      --output_dir outputs/eval_tta_best_accuracy_100 \
      --quick_test 100
  ```

这两种模式会直接返回“在前 N 张图上的完整指标”，而这次改动则是让**完整 600 张评估的长跑过程**也可以随时输出可用的中间结果。  

如果你希望中间结果除了写在日志里，还单独保存到某个 `metrics_step_xxx.txt` 文件里，我也可以再帮你加一层，把这些 `intermediate_results` 同时 append 到一个文件中。

```shellscript
nohup ./run_tta_best_accuracy.sh AutogluonModels/ag-20251203_075302 > outputs/tta_best_accuracy-251204.log 2>&1 &
```

```shellscript
python3 run_semantic_segmentation.py ... --tta_enable ...
```

```python
logger.info(
    f"Processed {num_processed}/{total_images} images with TTA "
    f"(avg: {avg_time:.2f}s/img, ETA: {eta/60:.1f}min)"
)

if streaming_metrics:  # 即 cache_dir=None 的流式低内存模式
    intermediate_results = {}
    for metric_name, metric_obj in metric_objects.items():
        score = metric_obj.compute()
        if hasattr(score, "item"):
            score = score.item()
        intermediate_results[metric_name] = float(score)
    if intermediate_results:
        logger.info(
            f"Intermediate TTA metrics after {num_processed} images: "
            f"{intermediate_results}"
        )
```

```shellscript
  nohup ./run_tta_best_accuracy.sh AutogluonModels/ag-20251203_075302 > outputs/tta_best_accuracy-251204.log 2>&1 &
```

```plaintext
  Processed 100/600 images with TTA (avg: 19.52s/img, ETA: 162.6min)
  Intermediate TTA metrics after 100 images: {'iou': 0.8231, 'dice': 0.8745}
```

```shellscript
  python3 run_semantic_segmentation.py \
      --task isic2017 \
      --eval \
      --ckpt_path AutogluonModels/ag-20251203_075302 \
      --tta_enable \
      --tta_scales 0.75 1.0 1.25 \
      --tta_flips none horizontal \
      --tta_rotations -10 0 10 \
      --tta_fusion weighted_mean \
      --tta_morphology \
      --output_dir outputs/eval_tta_best_accuracy_debug \
      --debug
```

```shellscript
  python3 run_semantic_segmentation.py \
      ...（同上参数）... \
      --output_dir outputs/eval_tta_best_accuracy_100 \
      --quick_test 100
```

---

**User**

对于CPU和内存占用过大，看起来应该主要源自图像变换，那为什么会有持续的内存增长（说明仍然有内存泄漏问题并且即便加了强制垃圾回收也不管用）？这部分处理可以放在GPU和显存上做吗？

---

**Cursor**

### 1. 为啥看起来还有“持续内存增长”？是不是还在泄漏？

结合现在的实现和 `TTA_Memory_CPU_Optimization.md` 的说明，**真正会造成“必然线性增长”的地方基本已经去掉了**：

- 以前的问题：
  - `predict_with_tta` 里每次 TTA 变换产生的 `numpy`/`scipy`/`skimage` 临时数组没有及时释放；
  - `_evaluate_with_tta` 里会把所有 `preds / labels` 累积在 list 里（尤其是有 cache 时），600 张 × 原图尺寸 × 18 次 TTA 直接把内存堆满。
- 现在的代码：
  - **TTA 变换**：`tta_utils.py` 里全部换成 **PIL**（`ScaleTransform/RotateTransform`），每一步之后都 `del` + `close()` + 定期 `gc.collect()`；在 `predict_with_tta` 中，单图只保留 18 个变换后的 `prob`，融合后立刻 `del all_probs`。
  - **评估主循环**：`_evaluate_with_tta` 在 `streaming_metrics=True`（默认你现在就是这种）时：
    - 只保留当前一张图的预测 + label，用 torchmetrics 在线更新指标；
    - 不再把全部预测/标签放进 `all_preds/all_labels`；
    - 每若干张强制 `gc.collect()` + `torch.cuda.empty_cache()`。

所以在“正常（无 cache_dir）流式模式”下，**理论上内存占用应该在某个高水位附近波动，而不是一路线性涨到被 OOM**。  
你仍然看到“持续增长”时，常见的几种情况是：

- **1）其实是分配器的高水位效应，不是泄漏**  
  - Python / `numpy` / `PIL` 用的底层分配器（jemalloc / glibc malloc）会**向 OS 申请越来越大的 arena**，即使对象已经被 `del` 掉，分配器也未必把内存还给 OS，而是留着以后复用。
  - 结果是在 `top` 或 `free -h` 里看起来 **RSS 一直慢慢涨**，但这并不意味着有 Python 对象一直被引用。
  - 特别是你每张图都是 2848×4288、18 次 TTA，前面几百张图会不断触发“更大块”的内存申请，形态上就是“缓慢爬升，然后趋于平稳”，这跟真正的“线性泄漏”在现象上不太一样。

- **2）使用了 `--tta_cache_dir` 导致真的累积预测**  
  - 一旦你传了 `--tta_cache_dir`，`streaming_metrics=False`，代码会把每张图的 `pred_tensor` 和 `label` 都 append 到 `all_preds/all_labels`，方便断点续传时重算指标；
  - 这时**内存随已处理图片数严格线性增长**，gc 也没法释放这些对象（它们本来就还在用）——这是“有意的”占用，而不是 bug；
  - 所以：**你要低内存就不要给 `--tta_cache_dir`**，只用我们现在默认的流式模式。

- **3）内存确实还在长，可以再做两步排查**（不改代码）：
  - 观察模式：是不是“每处理 X 张图固定涨 Y MB”，到最后必然 OOM？如果是，倾向于是真正还有积累对象。
  - 用工具确认：可以用 `memory_profiler` 或 `tracemalloc` 看一下哪些地方对象数量在无界增加；如果你愿意我可以给你一段最小复现脚本帮你插进去跑。

在你现在这版（`tta_utils.py + semantic_segmentation.py` 都已经替换为 PIL + 流式评估）的基础上，如果只是“前半程涨得比较快，后半程趋于平稳”，那更可能是**分配器高水位 + Python 不把内存还给 OS**，而不是还存在大的泄漏点。

---

### 2. 这些图像变换能不能挪到 GPU / 显存上做？

**理论上可以，实践上代价很大，目前代码是刻意放在 CPU 上的。**

- **现在的设计**（你看到的实现）：
  - TTA 的缩放、翻转、旋转都在 `tta_utils.py` 里，用的是 `numpy + PIL`，在 **CPU 内存** 里算；
  - 变换完的图像再通过 `predict_fn` 做 SAM 的预处理（resize 到 `model.image_size`）并搬到 **GPU** 跑一次前向；
  - 优点：
    - **显存压力可控**：GPU 上基本只有 “1 张 1024×1024 图像 + 模型参数 + 中间激活”，TTA 18 次是**串行**做的；
    - 即便你 TTA 很重，爆掉的更可能是 CPU 时间，而不是提前把 18 个变换都堆在显存里。
  - 缺点：
    - CPU 做 18 次高分辨率（2848×4288）旋转 + 缩放，确实会拉高 CPU 占用和内存峰值。

- **如果全部搬到 GPU 上，会变成什么样？**
  - 改造思路大致是：
    - `TTATransform` 不再操作 `np.ndarray`，而改成操作 `torch.Tensor`（`[C, H, W]`，在 CUDA 上）；
    - `ScaleTransform` / `RotateTransform` / `FlipTransform` 分别用 `torch.nn.functional.interpolate`、`affine_grid + grid_sample`、`torch.flip` 实现；
    - 整个 TTA pipeline 变成：
      1. 把一张原始图读成 CPU tensor（或直接用 `torchvision.io.read_image`），转到 GPU；
      2. 在 GPU 上做 18 次仿射变换 + 前向 + 反变换；
      3. 在 GPU 上融合 `fused_prob`，最后再搬回 CPU。
  - 潜在问题：
    - **显存占用会显著增加**：
      - 3×2848×4288 float32 ≈ 140MB，一次 TTA 保留 18 个中间结果的话就是 2.5GB+，再加上 SAM 本身的 activations，很容易在 24GB 卡上吃满；
      - 虽然可以像现在一样“只保留一个 `fused_prob`，其它逐步 accumulate 并释放”，但 GPU 侧的中间张量还是会比 CPU 版大很多。
    - **带宽 / 数据搬运**：
      - 如果你每次还是从 CPU 图像搬到 GPU 做变换，会有 18 次 H2D 传输，得不偿失；
      - 要真正收益，需要从一开始就把原图读进 GPU，并一直留在 GPU，这又进一步推高显存占用。
    - **改造成本高**：
      - 不只是 `Scale/Rotate/Flip`，后面的“回到原图大小”“形态学后处理”（`binary_closing` 等）也要全部用 `torch`/`kornia` 之类重写，工程量不小。
  - 所以在 `TTA_Memory_CPU_Optimization.md` 里，实际上已经给出一个结论：**对当前这套 ISIC2017 + SAM 的场景，PIL+CPU 版已经足够快，且更稳，没必要大改到 GPU 版。**

---

### 3. 实际建议

- **观测现在这版是否还“真的是线性泄漏”**：
  - 重新跑你现在的命令（不加 `--tta_cache_dir`），同时开一个监控：

    ```bash
    watch -n 10 'nvidia-smi && echo "---" && free -h'
    ```

  - 如果内存是 **先涨到 3–5GB 然后趋于平稳**，可以认为泄漏已经修掉，只是分配器在拉高高水位。
  - 如果是每处理固定数量的图像就规律性增加，并最终 OOM，再把“内存 vs 已处理图像数”的趋势、是否用了 cache 参数发给我，我可以一起帮你定位是否还有剩余的持久引用。

- **是否值得为了 CPU/内存把变换搬到 GPU？**
  - 在当前 18×TTA + SAM 的场景下：  
    **不太建议**，因为：
    - 显存压力会显著变大，反而有更高风险在 GPU 端 OOM；
    - 工程改造成本和维护成本都比较高；
    - 现在用 PIL 后，单图 15～20s、CPU 40–60% 已经是比较合理的 TTA 代价。
  - 如果你后续有更小的 backbone、或者更小分辨率的数据、并且特别在乎 CPU 负载，我们可以再设计一版“全 GPU TTA”的草案，但那就相当于是另一条实现路线了。

```shellscript
    watch -n 10 'nvidia-smi && echo "---" && free -h'
```

---

**User**

现在的脚本，每10个样本输出一次iou和dices指标，这个指标是从已经处理过的图像样本的指标作总体平均得到的吗？

---

**Cursor**

是的，现在日志里每 10 个样本打印的 IoU / Dice：

- **统计范围**：是“到目前为止所有已处理样本”的整体指标，不是最近 10 张的局部平均，也不是单张图的指标。
- **实现方式**：每处理一张图，就把这张图的预测和 label 喂进对应的 `torchmetrics` 指标对象（`metric_obj.update(...)`），指标内部累计 TP/FP/FN 等统计量；打印中间结果时调用一次 `metric_obj.compute()`，相当于“如果现在就停止，在前 N 张图上的最终 IoU / Dice 是多少”。

所以你看到的 `Intermediate TTA metrics after 100 images`，就是在**前 100 张测试图像上的整体评估结果**。

---

**User**

从当前的结果看，你预测最终600个样本能达到多少的平均IOU和DICE?

---

**Cursor**

从现在这 50/600 张的走势来看，只能做一个大致区间的预测：

- **当前 50 张的整体指标**：IoU 在 0.60–0.65 之间波动，最近一次在 50 张时是  
  **IoU ≈ 0.636，Dice ≈ 0.750**。
- 如果后面 550 张样本的难度分布和前 50 张大致一致，且没有明显“前容易后困难”的划分，比较合理的预期是：
  - **最终 600 张上的平均 IoU 大约在 0.63–0.66 之间**  
  - **最终 600 张上的平均 Dice 大约在 0.75–0.78 之间**

给一个更具体的点估计的话，我会预期最后收敛在 **IoU ≈ 0.64、Dice ≈ 0.76 附近**，但要强调这是基于当前 50/600 张样本的统计外推，有一定不确定性。

---

**User**

为什么会这样？TTA的使用不是应该比基线性能更好才对吗？基线应该达到了DICE=~0.858，现在用了TTA之后的平均DICE反而下降到~0.78，这不合理吧？重新审视整个TTA的实现，看哪里有问题

---

**Cursor**

### 先说结论

- **从代码实现上看，TTA 的评估管线（包括 IoU / Dice 计算）是“跟基线一致的”，目前没看到明显的实现性错误**。
- 你现在用的 `run_tta_best_accuracy.sh` 这一套 **“18 次变换 + 形态学后处理 + 固定阈值 0.5”**，在 ISIC2017 + 当前这个 ckpt 上，**实际上是“过强 / 不匹配”的 TTA 配置，导致性能明显掉到 ~0.78，是「配置问题」而不是「度量错了」**。
- 换句话说：**TTA 并不是理论上一定比基线好**，如果增强和后处理没调好，完全可能比 baseline 差一截。

下面按你关心的“实现有没有问题”一点点对应到代码说。

---

### 1. 指标计算这块：TTA 和基线走的是同一套度量逻辑

- 普通评估路径（无 TTA）在 `SemanticSegmentationLearner.evaluate_semantic_segmentation` 里：

  ```python
  # 简化版
  outputs = self.predict_per_run(...)
  if self._output_shape == 1:
      logits = extract_from_output(ret_type=LOGITS, outputs=outputs, as_ndarray=False)
  else:
      logits = extract_from_output(ret_type=SEMANTIC_MASK, ...)
  y_pred = logits.float()
  y_true = [ele[LABEL] for ele in outputs]
  y_true = torch.cat(y_true)

  per_metric = get_metric_predict(metric_name=..., num_classes=self._output_shape)
  for y_p, y_t in zip(y_pred, y_true):
      per_metric.update(y_p.unsqueeze(0), y_t.unsqueeze(0))
  score = per_metric.compute()
  ```

- TTA 评估路径在 `_evaluate_with_tta` 中，流式模式下关键部分是：

  ```python
  # 预测得到 fused_prob（概率图），转成 tensor
  pred_tensor = torch.from_numpy(pred_prob)   # [H, W] 或 [C, H, W]

  # 读入 GT mask，并二值化
  label = (label > 128).astype(np.int64)      # 0/1
  y_t = torch.from_numpy(label).long()

  # 尺寸不一致时，把预测 resize 到 label 尺寸
  if y_p.shape[-2:] != y_t.shape[-2:]:
      y_p = F.interpolate(...)

  # 用同一套 Binary_IoU_Pred / Binary_DICE_Pred
  metric_obj.update(y_p.unsqueeze(0), y_t.unsqueeze(0))
  ...
  score = metric_obj.compute()
  ```

- 而 `Binary_IoU_Pred` / `Binary_DICE_Pred` 的实现是基于 `torchmetrics.JaccardIndex(task="binary")`，对每张图**独立算 IoU / Dice 再做平均**：

  ```python
  iou = self.metric(logits, labels)          # 这里 logits / labels 都是在 CPU 上，[H, W] 或 [1, H, W]
  dice = 2 * iou / (1 + iou)
  ```

- 也就是说：
  - **基线和 TTA 最终都是：“对每张图算 Jaccard，再平均，DICE=2·IoU/(1+IoU)”**；
  - 唯一的区别在于：**基线的 y_p 来自单次前向输出，TTA 的 y_p 来自 18 次增强融合后的 fused_prob**，GT 这边都是根据 mask 图像二值化得到的同一张 label。

从这一段可以确定：**指标本身没有“算错 / 少除以 600”之类的 bug**。

---

### 2. 真正的差别：前端处理 & TTA 配置，而不是度量代码

结合你现在的运行：

- **Baseline（无 TTA）**：
  - 输入图像经过 AutoGluon/SAM 自带的预处理（通常是 resize 到 `model.image_size`，无旋转/多尺度合成）；
  - 输出一个概率 / logit 图，再直接用上面的指标逻辑评估；
  - 你这边测到的结果：`dice ≈ 0.858`（`outputs/tta_tests/baseline/metrics.txt` 里是 `0.85798`）。

- **当前 TTA 配置**（`run_tta_best_accuracy.sh`）：
  - 变换组合：`scales [0.75,1.0,1.25] × flips [none,horizontal] × rotations [-10,0,10] = 18`；
  - 融合：`weighted_mean`（原始 scale=1.0 权重更高）；
  - 后处理：`min_area_ratio=0.001` + 形态学 closing；
  - 阈值：`threshold=0.5`，没有针对 TTA 输出重新调过；
  - 在 ISIC2017 + 当前 ckpt 上的中间结果：50/600 时 Dice ≈ 0.75。

从实现看，有几处 **非常现实地会拉低 DICE**：

1. **旋转 + 多尺度对 lesion 这种任务并不总是友好**  
   - ISIC2017 的皮肤病灶，很多都是靠近边界 / 画幅边缘的，而且训练时未必做过对应的旋转增强；
   - 你现在做的是大图上的 `±10°` 旋转 + resize，再强行 warp 回原图大小，**边缘和细小结构很容易被模糊/拉扯**；
   - 模型没见过这种“整体倾斜”的分布时，很容易在一些 TTA view 下给出“奇怪但很自信”的预测，反而在融合时**拉坏**本来不错的 single-view 预测。

2. **形态学 closing + `min_area_ratio=0.001` 可能在 ISIC 上过于 aggressive**  
   - 大分辨率（如 2848×4288）下，`0.001 * image_area` 是一个相当大的面积阈值，小而散的病灶极易被视为噪声被抹掉；
   - Closing 会“填平小洞、连接相近区域”，在某些数据集（如 polyp）可能有帮助，但在 ISIC 上容易**让边界变得过于圆滑、mask 过度膨胀或粘连到背景**，导致 IoU/Dice 明显下降。

3. **阈值 0.5 没有重新针对 TTA 针过**  
   - baseline 时的输出分布是“单次前向”的；TTA 融合后，**概率分布会变得更“平滑 / 收缩”**，例如很多接近 0.5 的区域被压到 0.4–0.6 之间；
   - 如果仍然用 `0.5` 固定阈值，可能会：
     - 吃掉部分不够“极端”的 true positive；
     - 同时把一些不该合并的区域连在一起；
   - `TTAPredictor` 虽然提供了 `tune_threshold`（在 val 集上搜索最佳阈值），但 `run_tta_best_accuracy.sh` 里**没有用**，这一点本身就足以解释“有些模型上 TTA 不涨反跌”。

4. **基线和 TTA 在“评估分辨率”上也不完全一样**  
   - 基线评估用的是 `predict_per_run` 的输出，通常是模型内部的固定分辨率（比如 1024×1024），直接在这个尺度上算 IoU/Dice；
   - TTA 这条路径则是：
     - SAM 内部在 `model.image_size × model.image_size` 上算 logits；
     - TTA 里把各个 view 的概率拉回**原图分辨率**再做融合和度量；
   - 模型本身是在某个固定输入尺寸上训练/验证过的，在更高的原图分辨率上评估，不一定更有优势，**边界 jaggy / 上采样插值** 都可能略微拉低指标。

综合这些点：**当前这套「最重 TTA + 形态学 + 固定阈值 + 原图分辨率评估」对 ISIC2017 + 当前 ckpt 来说，很有可能是“overkill”，不是实现 bug，而是配置本身在这套任务上没调好。**

---

### 3. 建议你怎么验证 / 调整

在“实现基本没问题”的前提下，更务实的做法是把问题分解、一步步确认：

- **步骤 1：跑一次 `test_tta.sh`，看同一个 ckpt 下的完整对比**  
  在 `examples/automm/Conv-LoRA` 目录下：

  ```bash
  ./test_tta.sh
  ```

  然后看 `outputs/tta_tests` 里这几行：

  - `baseline/metrics.txt`（无 TTA）
  - `tta_basic/metrics.txt`（scales [0.75,1.0,1.25], flips [none,horizontal], 无 rotation、无 morphology）
  - 可再根据需要补跑“advanced / morphology”版本

  **关键观察点**：  
  - 如果 **`tta_basic` ≈ baseline 或略高**，而 “18 次 + morphology”的版本明显掉到 ~0.78，那基本可以确认：**实现是对的，只是 heavy TTA 配置不适合这个模型/任务**。  
  - 如果连 `tta_basic` 都显著低于 baseline，那才值得怀疑实现细节（比如 label 读取、阈值、resize 等），那时我们再按你给的日志逐段排查。

- **步骤 2：先用“安全 TTA”而不是“best_accuracy TTA”**  
  你可以自己写个更温和的脚本，比如只做 6 次 TTA、不加旋转和形态学：

  ```bash
  python3 run_semantic_segmentation.py \
      --task isic2017 \
      --eval \
      --ckpt_path AutogluonModels/ag-20251203_075302 \
      --tta_enable \
      --tta_scales 0.75 1.0 1.25 \
      --tta_flips none horizontal \
      --tta_rotations 0 \
      --output_dir outputs/eval_tta_safe
  ```

  再对比 `outputs/eval_tta_safe/metrics.txt` 和 baseline，看是不是已经接近或略好。

- **步骤 3：如果你愿意更精细，可以在验证集上调 TTA 阈值**  
  用 `TTAPredictor.tune_threshold` 在 val 集（不是 test 集）上搜索最优阈值，然后把阈值固定下来再在 test 上用；这一步在实现上已经提供，但目前脚本没有用到。

---

### 4. 回到你的直觉：“TTA 理论上应该比基线好吗？”

- **不是数学保证，只是经验规律**：  
  - 条件是：增强方式“近似对称”、不会严重破坏目标形状，且模型对这些变换有一定鲁棒性；
  - 对自然图像分类 / 一些目标检测任务，TTA 几乎是稳赚不赔；
  - 对**高分辨率医学分割 + 未针对旋转/形态学仔细调参的模型**，就不一定了。

- 在你现在这套 ISIC2017 + SAM + Conv-LoRA 的组合里，**重度 TTA + 形态学这套 preset 明显不适合，看到 Dice 从 0.858 掉到 ~0.78 是“可以理解的坏配置结果”，而不是「评估实现崩了」**。

如果你接下来愿意，我可以帮你设计一套“逐步缩减 TTA 强度 + 调阈值”的实验组合，并给出对应的命令，保证每一步都能快速在 `--quick_test 50` 级别验证它到底是帮忙还是添乱。

```python
  # 简化版
  outputs = self.predict_per_run(...)
  if self._output_shape == 1:
      logits = extract_from_output(ret_type=LOGITS, outputs=outputs, as_ndarray=False)
  else:
      logits = extract_from_output(ret_type=SEMANTIC_MASK, ...)
  y_pred = logits.float()
  y_true = [ele[LABEL] for ele in outputs]
  y_true = torch.cat(y_true)

  per_metric = get_metric_predict(metric_name=..., num_classes=self._output_shape)
  for y_p, y_t in zip(y_pred, y_true):
      per_metric.update(y_p.unsqueeze(0), y_t.unsqueeze(0))
  score = per_metric.compute()
```

```python
  # 预测得到 fused_prob（概率图），转成 tensor
  pred_tensor = torch.from_numpy(pred_prob)   # [H, W] 或 [C, H, W]

  # 读入 GT mask，并二值化
  label = (label > 128).astype(np.int64)      # 0/1
  y_t = torch.from_numpy(label).long()

  # 尺寸不一致时，把预测 resize 到 label 尺寸
  if y_p.shape[-2:] != y_t.shape[-2:]:
      y_p = F.interpolate(...)

  # 用同一套 Binary_IoU_Pred / Binary_DICE_Pred
  metric_obj.update(y_p.unsqueeze(0), y_t.unsqueeze(0))
  ...
  score = metric_obj.compute()
```

```python
  iou = self.metric(logits, labels)          # 这里 logits / labels 都是在 CPU 上，[H, W] 或 [1, H, W]
  dice = 2 * iou / (1 + iou)
```

```shellscript
  ./test_tta.sh
```

```shellscript
  python3 run_semantic_segmentation.py \
      --task isic2017 \
      --eval \
      --ckpt_path AutogluonModels/ag-20251203_075302 \
      --tta_enable \
      --tta_scales 0.75 1.0 1.25 \
      --tta_flips none horizontal \
      --tta_rotations 0 \
      --output_dir outputs/eval_tta_safe
```

---

**User**

请总结现在的TTA方案具体是如何实现的，遇到了哪些问题待解决，写成一个详细的文档，细化到关键代码，供以后继续研究做参考

---

**Cursor**

我会新建一个独立的中文文档，把当前 TTA 的整体设计、关键代码路径、已解决和未解决的问题系统性整理出来，方便你后续继续研究或改代码。文档里会直接引用目前代码里的关键实现片段，并总结我们这几天排查中发现的行为和假设。

---

**User**

我发现run_semantic_segmentation.py脚本在训练时有一个问题：它生成的模型存放目录AutogluonModels/ag-YYYYMMDD_hhmmss是用时间戳命名的，但是输出的控制台日志里却不会打出所生成模型目录的名字，这就造成后续想要通过模型文件位置回溯到它是什么实验所生成的时候非常困难：尽管我做实验时都会把控制台输出重定向到日志文件，文件命名都与具体某一次实验相关联，但是无法通过模型目录名关联到日志文件因为日志里并不输出模型目录名，进而也无法通过模型路径追溯到哪一次实验。请在训练时的打印输出里补充上模型保存目录的信息。

---

**Cursor**

我来改一下 `run_semantic_segmentation.py`，在训练模式下把 `MultiModalPredictor` 实际使用的模型保存目录（`AutogluonModels/ag-...`）打印出来，这样你以后可以从模型路径直接追溯到对应的实验日志。

```python
import argparse
import os

import pandas as pd

from autogluon.multimodal import MultiModalPredictor


def get_default_training_setting(dataset_name):
    validation_metric = "iou"
    loss = "structure_loss"
    max_epoch = 30
    lr = 1e-4

    if dataset_name == "SBU-shadow":
        validation_metric = "ber"
        loss = "balanced_bce"
        max_epoch = 10

    elif dataset_name == "polyp":
        validation_metric = "sm"

    elif dataset_name == "camo_sem_seg":
        validation_metric = "sm"
        max_epoch = 20

    elif dataset_name == "road_segmentation":
        validation_metric = "iou"
        max_epoch = 20
        lr = 3e-4

    elif dataset_name == "leaf_disease_segmentation":
        validation_metric = "iou"
        lr = 3e-4

    return validation_metric, loss, max_epoch, lr


def expand_path(df, dataset_dir):
    for col in ["image", "label"]:
        df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script support converting voc format xmls to coco format json")
    parser.add_argument(
        "--task",
        type=str,
        default="leaf_disease_segmentation",
        choices=["polyp", "leaf_disease_segmentation", "camo_sem_seg", "isic2017", "road_segmentation", "SBU-shadow"],
    )
    parser.add_argument("--seed", type=int, default=42686693)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--expert_num", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--ckpt_path", type=str, default="outputs", help="Checkpoint path.")
    parser.add_argument("--per_gpu_batch_size", type=int, default=1, help="The batch size for each GPU.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="The effective batch size. If batch_size > per_gpu_batch_size * num_gpus, gradient accumulation would be used.",
    )
    parser.add_argument("--eval", action="store_true")
    
    # GSPO parameters
    parser.add_argument("--gspo_enable", action="store_true", help="Enable GSPO (Group Sequence Policy Optimization) training")
    parser.add_argument("--gspo_group_size", type=int, default=4, help="Number of predictions per group in GSPO")
    parser.add_argument("--gspo_warmup_epochs", type=int, default=5, help="Number of epochs before enabling GSPO")
    parser.add_argument("--gspo_contrastive_weight", type=float, default=0.1, help="Weight for contrastive loss in GSPO")
    parser.add_argument("--gspo_quality_momentum", type=float, default=0.9, help="Momentum for expert quality history")
    
    # Adapter parameters
    parser.add_argument("--adapter_enable", action="store_true", help="Enable standard Adapter modules")
    parser.add_argument("--adapter_dim", type=int, default=64, help="Dimension of the adapter bottleneck")
    
    # TTA (Test-Time Augmentation) parameters
    parser.add_argument("--tta_enable", action="store_true", help="Enable Test-Time Augmentation for evaluation")
    parser.add_argument("--tta_scales", type=float, nargs="+", default=[0.75, 1.0, 1.25], 
                        help="Scale factors for multi-scale TTA (default: [0.75, 1.0, 1.25]). "
                             "Recommended: [0.75, 1.0, 1.25] or [0.8, 1.0, 1.2]")
    parser.add_argument("--tta_flips", type=str, nargs="+", default=["none", "horizontal"],
                        choices=["none", "horizontal", "vertical"],
                        help="Flip types for TTA (default: ['none', 'horizontal'])")
    parser.add_argument("--tta_rotations", type=float, nargs="+", default=[0],
                        help="Rotation angles in degrees for TTA (default: [0]). "
                             "Recommended: [0] for fast, or [-10, 0, 10] for better")
    parser.add_argument("--tta_fusion", type=str, default="mean", choices=["mean", "weighted_mean"],
                        help="Fusion method for TTA predictions (default: 'mean')")
    parser.add_argument("--tta_resize_method", type=str, default="bilinear", choices=["bilinear", "bicubic"],
                        help="Resize interpolation method for multi-scale TTA (default: 'bilinear'). "
                             "bilinear: faster, bicubic: higher quality")
    parser.add_argument("--tta_threshold", type=float, default=0.5,
                        help="Threshold for binary segmentation in TTA (default: 0.5)")
    parser.add_argument("--tta_min_area", type=float, default=0.001,
                        help="Remove components with area < tta_min_area * image_area (default: 0.001)")
    parser.add_argument("--tta_morphology", action="store_true",
                        help="Apply morphological closing in TTA post-processing")
    parser.add_argument("--tta_cache_dir", type=str, default=None,
                        help="Directory to cache TTA predictions for resume (default: None, no caching)")
    parser.add_argument("--tta_no_resume", action="store_true",
                        help="Disable resume from cache (start fresh even if cache exists)")
    
    # Quick test / Debug parameters
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: only process first 5 images for quick validation")
    parser.add_argument("--quick_test", type=int, default=None,
                        help="Quick test mode: only process first N images")
    
    args = parser.parse_args()

    dataset_name = args.task
    dataset_dir = os.path.join(f"datasets/{dataset_name}", dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare dataframes
    train_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"train.csv")), dataset_dir)
    val_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"val.csv")), dataset_dir)

    # get the validation metric
    validation_metric, loss, max_epoch, lr = get_default_training_setting(dataset_name)

    hyperparameters = {}
    hyperparameters.update(
        {
            "optim.lora.r": args.rank,
            # Paper alignment (Conv-LoRA):
            # - optim.peft: choose Conv-LoRA adaptation (Sec. 3)
            # - optim.lora.r: LoRA rank r (Sec. 3)
            # - optim.lora.conv_lora_expert_num: number of MoE-Conv experts M (Sec. 3.2)
            "optim.peft": "conv_lora",
            "optim.lora.conv_lora_expert_num": args.expert_num,
            "env.num_gpus": args.num_gpus,
            "optim.loss_func": loss,
            "optim.max_epochs": max_epoch,
            "optim.lr": lr,
            "env.per_gpu_batch_size": args.per_gpu_batch_size,
            "env.batch_size": args.batch_size,
        }
    )
    
    # GSPO configuration
    if args.gspo_enable:
        print(f"Enabling GSPO with group_size={args.gspo_group_size}, warmup_epochs={args.gspo_warmup_epochs}")
        hyperparameters.update({
            "optim.lora.gspo_enabled": True,
            "optim.lora.gspo_group_size": args.gspo_group_size,
            "optim.lora.gspo_quality_momentum": args.gspo_quality_momentum,
            "optim.lora.gspo_warmup_epochs": args.gspo_warmup_epochs,
            "optim.lora.gspo_contrastive_weight": args.gspo_contrastive_weight,
        })

    # Adapter configuration
    if args.adapter_enable:
        print(f"Enabling Standard Adapters with dim={args.adapter_dim}")
        hyperparameters.update({
            "model.sam.adapter_enabled": True,
            "model.sam.adapter_dim": args.adapter_dim,
        })

    if args.eval:  # load a checkpoint for evaluation
        predictor = MultiModalPredictor.load(args.ckpt_path)
    else:  # training
        predictor = MultiModalPredictor(
            problem_type="semantic_segmentation",
            validation_metric=validation_metric,
            eval_metric=validation_metric,
            hyperparameters=hyperparameters,
            label="label",
        )

        # 打印当前实验的模型保存目录，便于后续通过 AutogluonModels 路径追溯到具体实验日志
        try:
            # MultiModalPredictor 会在内部根据时间戳生成类似 AutogluonModels/ag-YYYYMMDD_HHMMSS 的目录
            model_path = predictor.path
        except AttributeError:
            # 兼容极端情况：如果未来接口变化，没有 path 属性，则显式提示
            model_path = None

        if model_path is not None:
            print("\n========================================")
            print("Training MultiModalPredictor")
            print(f"  Task        : {dataset_name}")
            print(f"  Save path   : {model_path}")
            print("  (You can use this directory name to link back to the training log.)")
            print("========================================\n")
        else:
            print("\n[Warning] MultiModalPredictor has no 'path' attribute. "
                  "Model save directory cannot be printed.\n")

        predictor.fit(train_data=train_df, tuning_data=val_df, seed=args.seed)

    # Enable TTA if requested
    if args.tta_enable:
        print(f"\n{'='*60}")
        print(f"Enabling Test-Time Augmentation (TTA)")
        print(f"  Scales: {args.tta_scales}")
        print(f"  Flips: {args.tta_flips}")
        print(f"  Rotations: {args.tta_rotations}")
        print(f"  Fusion: {args.tta_fusion}")
        print(f"  Resize method: {args.tta_resize_method}")
        print(f"  Total augmentations: {len(args.tta_scales) * len(args.tta_flips) * len(args.tta_rotations)}")
        print(f"{'='*60}\n")
        
        predictor.enable_tta(
            scales=args.tta_scales,
            flips=args.tta_flips,
            rotations=args.tta_rotations,
            fusion_method=args.tta_fusion,
            resize_method=args.tta_resize_method,
            threshold=args.tta_threshold,
            min_area_ratio=args.tta_min_area,
            use_morphology=args.tta_morphology,
            cache_dir=args.tta_cache_dir,
            resume_from_cache=not args.tta_no_resume,
        )

    # evaluation
    metric_file = os.path.join(args.output_dir, "metrics.txt")
    f = open(metric_file, "a")
    if dataset_name in ["isic2017", "SBU-shadow", "road_segmentation", "leaf_disease_segmentation"]:
        test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test.csv")), dataset_dir)
        
        # Apply quick test / debug mode (方案 1: 快速验证模式)
        original_size = len(test_df)
        if args.debug:
            test_df = test_df.head(5)
            print(f"\n🔍 DEBUG MODE: Processing only first 5 images (out of {original_size})\n")
        elif args.quick_test is not None:
            test_df = test_df.head(args.quick_test)
            print(f"\n🔍 QUICK TEST MODE: Processing only first {args.quick_test} images (out of {original_size})\n")
        
        if dataset_name == "SBU-shadow":
            eval_metrics = ["ber"]
        else:
            eval_metrics = ["iou", "dice"]  # Evaluate both IoU and DICE

        res = predictor.evaluate(test_df, metrics=eval_metrics)
        print(f"Evaluation results for test dataset {dataset_name}: ", res)
        f.write(f"Evaluation results for test dataset {dataset_name}: {res} \n")
    elif dataset_name in ["polyp", "camo_sem_seg"]:
        if dataset_name == "polyp":
            test_datasets = ["CVC-ClinicDB", "Kvasir"]
        elif dataset_name == "camo_sem_seg":
            test_datasets = ["CAMO"]
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}.")
        for per_dataset in test_datasets:
            test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test_{per_dataset}.csv")), dataset_dir)
            
            # Apply quick test / debug mode (方案 1: 快速验证模式)
            original_size = len(test_df)
            if args.debug:
                test_df = test_df.head(5)
                print(f"\n🔍 DEBUG MODE: Processing only first 5 images (out of {original_size}) for {per_dataset}\n")
            elif args.quick_test is not None:
                test_df = test_df.head(args.quick_test)
                print(f"\n🔍 QUICK TEST MODE: Processing only first {args.quick_test} images (out of {original_size}) for {per_dataset}\n")
            
            res = predictor.evaluate(test_df, metrics=["sm", "fm", "em", "mae"])
            print(f"Evaluation results for test dataset {per_dataset}: ", res)
            f.write(f"Evaluation results for test dataset {per_dataset}: {res} \n")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}.")

    f.close()

```

### 修改效果说明

现在只要是**训练模式**运行 `run_semantic_segmentation.py`（不带 `--eval`），控制台 / 日志里都会额外打印出本次实验对应的模型保存目录名，例如：

```text
========================================
Training MultiModalPredictor
  Task        : isic2017
  Save path   : AutogluonModels/ag-20251211_123456
  (You can use this directory name to link back to the training log.)
========================================
```

这一行会出现在 `predictor.fit(...)` 之前，因此你：

- 仍然可以把 stdout 重定向到类似 `train_isic2017_gspo_exp3-*.log` 的日志文件；
- 日志里会带有 `AutogluonModels/ag-YYYYMMDD_HHMMSS` 的精确目录名；
- 以后从某个模型目录反查对应实验时，只要在所有训练日志中 `grep` 这个目录名，就能快速找到是哪次实验生成的。

```plaintext
========================================
Training MultiModalPredictor
  Task        : isic2017
  Save path   : AutogluonModels/ag-20251211_123456
  (You can use this directory name to link back to the training log.)
========================================
```

