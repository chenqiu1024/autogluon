"""
Test-Time Augmentation (TTA) utilities for semantic segmentation.

Implementation follows standard TTA practices for medical/natural image segmentation:
- Multi-scale testing
- Horizontal/vertical flips
- Small rotations
- Probability averaging
- Post-processing (threshold tuning, small component removal)

References:
- Standard practice in medical image segmentation competitions
- Commonly used in Kaggle, Grand Challenges, etc.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from scipy.ndimage import binary_closing, label
from skimage.transform import resize

logger = logging.getLogger(__name__)


class TTATransform:
    """Base class for TTA transformations."""
    
    def __init__(self):
        pass
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply transformation to image."""
        raise NotImplementedError
    
    def apply_inverse_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply inverse transformation to mask prediction."""
        raise NotImplementedError

    def transform_box(self, box: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """Transform a box [x1, y1, x2, y2] under this augmentation."""
        return box


class ScaleTransform(TTATransform):
    """Multi-scale transformation."""
    
    def __init__(self, scale: float, resize_method: str = "bilinear"):
        """
        Parameters
        ----------
        scale : float
            Scale factor for resizing
        resize_method : str
            Resize interpolation method: "bilinear" or "bicubic"
        """
        super().__init__()
        self.scale = scale
        self.resize_method = resize_method.lower()
        
        # Map resize method to PIL constant
        if self.resize_method == "bilinear":
            self.pil_resample = Image.BILINEAR
        elif self.resize_method == "bicubic":
            self.pil_resample = Image.BICUBIC
        else:
            raise ValueError(f"Unknown resize_method: {resize_method}. Use 'bilinear' or 'bicubic'")
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Scale image using PIL (memory efficient)."""
        if self.scale == 1.0:
            return image
        
        h, w = image.shape[:2]
        new_h, new_w = int(h * self.scale), int(w * self.scale)
        
        # Use PIL for image resizing (more memory efficient than skimage)
        needs_scale_back = False
        
        if len(image.shape) == 2:
            # Grayscale
            if image.dtype != np.uint8:
                image_uint8 = (image * 255).astype(np.uint8)
                needs_scale_back = True
            else:
                image_uint8 = image
            pil_img = Image.fromarray(image_uint8, mode='L')
        else:
            # RGB
            if image.dtype != np.uint8:
                image_uint8 = (image * 255).astype(np.uint8)
                needs_scale_back = True
            else:
                image_uint8 = image
            pil_img = Image.fromarray(image_uint8)
        
        # Resize with specified method
        scaled_pil = pil_img.resize((new_w, new_h), self.pil_resample)
        scaled = np.array(scaled_pil).copy()  # Explicit copy
        
        # Explicitly close PIL images to free memory IMMEDIATELY
        pil_img.close()
        scaled_pil.close()
        
        # Scale back if needed
        if needs_scale_back:
            scaled = scaled.astype(np.float32) / 255.0
        
        return scaled

    def transform_box(self, box: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box.astype(np.float32)
        return np.array([x1 * self.scale, y1 * self.scale, x2 * self.scale, y2 * self.scale], dtype=np.float32)
    
    def apply_inverse_mask(self, mask: np.ndarray) -> np.ndarray:
        """Resize mask back to original size (memory efficient)."""
        if self.scale == 1.0:
            return mask
        
        # mask shape: (H', W') or (C, H', W')
        if len(mask.shape) == 2:
            h, w = mask.shape
            new_h, new_w = int(h / self.scale), int(w / self.scale)
            
            # Use PIL for masks too (faster and more memory efficient)
            # Scale to 0-255 range for PIL
            mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
            mask_pil = Image.fromarray(mask_uint8, mode='L')
            resized_pil = mask_pil.resize((new_w, new_h), self.pil_resample)
            resized = np.array(resized_pil).astype(np.float32).copy()
            
            # Close PIL images IMMEDIATELY
            mask_pil.close()
            resized_pil.close()
            
            if mask.max() <= 1.0:
                resized = resized / 255.0
        else:  # (C, H, W)
            c, h, w = mask.shape
            new_h, new_w = int(h / self.scale), int(w / self.scale)
            resized = np.zeros((c, new_h, new_w), dtype=np.float32)
            
            for i in range(c):
                mask_ch = mask[i]
                mask_uint8 = (mask_ch * 255).astype(np.uint8) if mask_ch.max() <= 1.0 else mask_ch.astype(np.uint8)
                mask_pil = Image.fromarray(mask_uint8, mode='L')
                resized_pil = mask_pil.resize((new_w, new_h), self.pil_resample)
                resized[i] = np.array(resized_pil).astype(np.float32)
                
                # Close PIL images IMMEDIATELY
                mask_pil.close()
                resized_pil.close()
                
                if mask_ch.max() <= 1.0:
                    resized[i] = resized[i] / 255.0
        
        return resized


class FlipTransform(TTATransform):
    """Horizontal or vertical flip."""
    
    def __init__(self, flip_type: str = "horizontal"):
        """
        Parameters
        ----------
        flip_type : str
            "horizontal", "vertical", or "none"
        """
        super().__init__()
        assert flip_type in ["horizontal", "vertical", "none"]
        self.flip_type = flip_type
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Flip image."""
        if self.flip_type == "none":
            return image
        elif self.flip_type == "horizontal":
            return np.fliplr(image).copy()
        elif self.flip_type == "vertical":
            return np.flipud(image).copy()
    
    def apply_inverse_mask(self, mask: np.ndarray) -> np.ndarray:
        """Flip mask back."""
        if self.flip_type == "none":
            return mask
        elif self.flip_type == "horizontal":
            if len(mask.shape) == 2:
                return np.fliplr(mask).copy()
            else:  # (C, H, W)
                return np.flip(mask, axis=2).copy()
        elif self.flip_type == "vertical":
            if len(mask.shape) == 2:
                return np.flipud(mask).copy()
            else:  # (C, H, W)
                return np.flip(mask, axis=1).copy()

    def transform_box(self, box: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        h, w = image_shape
        x1, y1, x2, y2 = box.astype(np.float32)
        if self.flip_type == "horizontal":
            x1_new = (w - 1) - x2
            x2_new = (w - 1) - x1
            return np.array([x1_new, y1, x2_new, y2], dtype=np.float32)
        elif self.flip_type == "vertical":
            y1_new = (h - 1) - y2
            y2_new = (h - 1) - y1
            return np.array([x1, y1_new, x2, y2_new], dtype=np.float32)
        return box


class RotateTransform(TTATransform):
    """Small angle rotation using PIL (faster than scipy)."""
    
    def __init__(self, angle: float):
        """
        Parameters
        ----------
        angle : float
            Rotation angle in degrees (positive = counter-clockwise)
        """
        super().__init__()
        self.angle = angle
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Rotate image using PIL (much faster than scipy)."""
        if self.angle == 0:
            return image
        
        # PIL is faster than scipy.ndimage.rotate
        # Convert to PIL Image
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
            needs_scale_back = True
        else:
            image_uint8 = image.astype(np.uint8)
            needs_scale_back = False
        
        img_pil = Image.fromarray(image_uint8)
        
        # Rotate (PIL uses negative angle for counter-clockwise, opposite of scipy)
        rotated_pil = img_pil.rotate(-self.angle, resample=Image.BILINEAR, expand=False)
        
        # Convert back to numpy
        rotated = np.array(rotated_pil).copy()  # Explicit copy
        
        # Close PIL objects immediately
        img_pil.close()
        rotated_pil.close()
        
        if needs_scale_back:
            rotated = rotated.astype(np.float32) / 255.0
        
        return rotated

    def transform_box(self, box: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        if self.angle == 0:
            return box
        h, w = image_shape
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        x1, y1, x2, y2 = box.astype(np.float32)
        corners = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=np.float32)
        theta = np.deg2rad(self.angle)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
        shifted = corners - np.array([cx, cy], dtype=np.float32)
        rotated = (shifted @ rot.T) + np.array([cx, cy], dtype=np.float32)
        x_min, y_min = rotated[:, 0].min(), rotated[:, 1].min()
        x_max, y_max = rotated[:, 0].max(), rotated[:, 1].max()
        x_min = np.clip(x_min, 0, w - 1)
        y_min = np.clip(y_min, 0, h - 1)
        x_max = np.clip(x_max, 0, w - 1)
        y_max = np.clip(y_max, 0, h - 1)
        return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
    
    def apply_inverse_mask(self, mask: np.ndarray) -> np.ndarray:
        """Rotate mask back (inverse rotation) using PIL."""
        if self.angle == 0:
            return mask
        
        # For masks, use PIL which is faster
        if len(mask.shape) == 2:
            # Single channel mask
            # Scale to 0-255 for PIL
            mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
            mask_pil = Image.fromarray(mask_uint8, mode='L')
            
            # Rotate back (opposite direction)
            rotated_pil = mask_pil.rotate(self.angle, resample=Image.BILINEAR, expand=False)
            
            rotated = np.array(rotated_pil).astype(np.float32).copy()
            
            # Close immediately
            mask_pil.close()
            rotated_pil.close()
            
            if mask.max() <= 1.0:
                rotated = rotated / 255.0
        else:  # (C, H, W)
            c, h, w = mask.shape
            rotated = np.zeros_like(mask)
            for i in range(c):
                mask_ch = mask[i]
                mask_uint8 = (mask_ch * 255).astype(np.uint8) if mask_ch.max() <= 1.0 else mask_ch.astype(np.uint8)
                mask_pil = Image.fromarray(mask_uint8, mode='L')
                rotated_pil = mask_pil.rotate(self.angle, resample=Image.BILINEAR, expand=False)
                rotated[i] = np.array(rotated_pil).astype(np.float32)
                
                # Close immediately
                mask_pil.close()
                rotated_pil.close()
                
                if mask_ch.max() <= 1.0:
                    rotated[i] = rotated[i] / 255.0
        
        return rotated


class ComposedTransform(TTATransform):
    """Compose multiple transformations."""
    
    def __init__(self, transforms: List[TTATransform]):
        super().__init__()
        self.transforms = transforms
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply all transformations in order."""
        result = image
        for t in self.transforms:
            result = t.apply(result)
        return result
    
    def apply_inverse_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply inverse transformations in reverse order."""
        result = mask
        for t in reversed(self.transforms):
            result = t.apply_inverse_mask(result)
        return result

    def transform_box(self, box: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        cur_box = box
        cur_shape = image_shape
        for t in self.transforms:
            cur_box = t.transform_box(cur_box, cur_shape)
            if isinstance(t, ScaleTransform):
                h, w = cur_shape
                cur_shape = (int(h * t.scale), int(w * t.scale))
        return cur_box


class TTAPredictor:
    """
    Test-Time Augmentation wrapper for semantic segmentation models.
    
    Supports:
    - Multi-scale testing
    - Horizontal/vertical flips  
    - Small rotations
    - Probability averaging (mean or weighted mean)
    - Post-processing (threshold tuning, small component removal)
    """
    
    def __init__(
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
    ):
        """
        Parameters
        ----------
        scales : List[float]
            List of scale factors for multi-scale testing.
            Recommended: [0.75, 1.0, 1.25] for fast, or [0.5, 0.75, 1.0, 1.25, 1.5] for better.
        flips : List[str]
            List of flip types: "none", "horizontal", "vertical".
            Recommended: ["none", "horizontal"] for medical images.
        rotations : List[float]
            List of rotation angles in degrees.
            Recommended: [0] for fast, or [-10, 0, 10] for better.
        fusion_method : str
            Method to fuse predictions: "mean" or "weighted_mean".
        scale_weights : Optional[Dict[float, float]]
            Weights for each scale when using weighted_mean.
            If None and fusion_method="weighted_mean", scale=1.0 gets higher weight.
        threshold : float
            Threshold for binary segmentation (can be tuned on validation set).
        min_area_ratio : float
            Remove connected components with area < min_area_ratio * image_area.
        use_morphology : bool
            Whether to apply morphological closing for smoothing.
        resize_method : str
            Resize interpolation method: "bilinear" or "bicubic".
            Recommended: "bilinear" for speed, "bicubic" for quality.
        """
        self.scales = scales
        self.flips = flips
        self.rotations = rotations
        self.fusion_method = fusion_method
        self.threshold = threshold
        self.min_area_ratio = min_area_ratio
        self.use_morphology = use_morphology
        self.resize_method = resize_method
        
        # Setup scale weights
        if scale_weights is None and fusion_method == "weighted_mean":
            # Default: give scale=1.0 higher weight
            total_scales = len(scales)
            self.scale_weights = {s: 1.0 for s in scales}
            if 1.0 in scales:
                self.scale_weights[1.0] = 2.0  # Double weight for original scale
        else:
            self.scale_weights = scale_weights
        
        # Generate all transform combinations
        self.transforms = self._generate_transforms()
        
        logger.info(f"TTA initialized with {len(self.transforms)} augmentations")
        logger.info(f"  Scales: {scales}")
        logger.info(f"  Flips: {flips}")
        logger.info(f"  Rotations: {rotations}")
        logger.info(f"  Fusion: {fusion_method}")
        logger.info(f"  Total inferences per image: {len(self.transforms)}")
    
    def _generate_transforms(self) -> List[Tuple[ComposedTransform, float]]:
        """Generate all combinations of transforms with their weights."""
        transforms = []
        
        for scale in self.scales:
            for flip in self.flips:
                for rotate in self.rotations:
                    # Create composed transform
                    t = ComposedTransform([
                        ScaleTransform(scale, resize_method=self.resize_method),
                        FlipTransform(flip),
                        RotateTransform(rotate),
                    ])
                    
                    # Compute weight for this transform
                    if self.fusion_method == "weighted_mean" and self.scale_weights is not None:
                        weight = self.scale_weights.get(scale, 1.0)
                    else:
                        weight = 1.0
                    
                    transforms.append((t, weight))
        
        return transforms
    
    def predict_with_tta(
        self,
        image: np.ndarray,
        predict_fn,
        return_probs: bool = True,
        box_prompt: Optional[np.ndarray] = None,
        box_ref_shape: Optional[Tuple[int, int]] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Perform TTA prediction on a single image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image, shape (H, W, C) or (H, W)
        predict_fn : callable
            Function that takes an image and returns prediction.
            Should return soft probabilities or logits.
        return_probs : bool
            If True, return both binary mask and probability map.
            If False, only return binary mask.
        
        Returns
        -------
        mask : np.ndarray
            Binary segmentation mask (H, W)
        probs : np.ndarray (optional)
            Probability map (H, W) if return_probs=True
        """
        import gc
        
        original_shape = image.shape[:2]
        all_probs = []
        all_weights = []
        
        # Apply each transform and collect predictions
        for idx, (transform, weight) in enumerate(self.transforms):
            # Transform image & box together
            transformed_img = transform.apply(image)
            transformed_box = None
            if box_prompt is not None:
                transformed_box = transform.transform_box(box_prompt, image.shape[:2])
            
            # Get prediction (should be probability or logit)
            pred = predict_fn(transformed_img, transformed_box, transformed_img.shape[:2])
            
            # Free transformed image immediately
            del transformed_img
            
            # Apply inverse transform to prediction
            pred_original = transform.apply_inverse_mask(pred)
            
            # Free pred immediately
            del pred
            
            # Ensure correct shape - use PIL instead of skimage for speed
            if pred_original.shape[:2] != original_shape:
                # Use PIL for resizing (faster than skimage)
                if len(pred_original.shape) == 2:
                    pred_uint8 = (pred_original * 255).astype(np.uint8) if pred_original.max() <= 1.0 else pred_original.astype(np.uint8)
                    pred_pil = Image.fromarray(pred_uint8, mode='L')
                    resized_pil = pred_pil.resize((original_shape[1], original_shape[0]), Image.BILINEAR)
                    pred_resized = np.array(resized_pil).astype(np.float32).copy()
                    
                    # Close immediately
                    pred_pil.close()
                    resized_pil.close()
                    
                    if pred_original.max() <= 1.0:
                        pred_resized = pred_resized / 255.0
                    
                    pred_original = pred_resized
                else:
                    resized = np.zeros((pred_original.shape[0], original_shape[0], original_shape[1]), dtype=np.float32)
                    for i in range(pred_original.shape[0]):
                        pred_ch = pred_original[i]
                        pred_uint8 = (pred_ch * 255).astype(np.uint8) if pred_ch.max() <= 1.0 else pred_ch.astype(np.uint8)
                        pred_pil = Image.fromarray(pred_uint8, mode='L')
                        resized_pil = pred_pil.resize((original_shape[1], original_shape[0]), Image.BILINEAR)
                        resized[i] = np.array(resized_pil).astype(np.float32)
                        
                        # Close immediately
                        pred_pil.close()
                        resized_pil.close()
                        
                        if pred_ch.max() <= 1.0:
                            resized[i] = resized[i] / 255.0
                    
                    pred_original = resized
            
            all_probs.append(pred_original.copy())  # Explicit copy
            all_weights.append(weight)
            
            # Free memory explicitly
            del pred_original
            
            # More aggressive garbage collection
            if (idx + 1) % 2 == 0:  # Every 2 transforms
                gc.collect()
        
        # Fuse predictions
        if self.fusion_method == "mean":
            fused_prob = np.mean(all_probs, axis=0)
        elif self.fusion_method == "weighted_mean":
            weights = np.array(all_weights)
            weights = weights / weights.sum()
            fused_prob = np.average(all_probs, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Free the list of probabilities
        del all_probs, all_weights
        gc.collect()
        
        # Convert to 2D if needed (for binary segmentation)
        if len(fused_prob.shape) == 3 and fused_prob.shape[0] == 1:
            fused_prob = fused_prob[0]
        
        # Apply threshold
        binary_mask = (fused_prob > self.threshold).astype(np.uint8)
        
        # Post-processing
        binary_mask = self._post_process(binary_mask, original_shape)
        
        if return_probs:
            return binary_mask, fused_prob
        else:
            # Free fused_prob if not needed
            del fused_prob
            return binary_mask
    
    def _post_process(self, mask: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Post-process binary mask:
        1. Remove small connected components
        2. Optional morphological closing
        
        Parameters
        ----------
        mask : np.ndarray
            Binary mask (H, W)
        original_shape : Tuple[int, int]
            Original image shape (H, W)
        
        Returns
        -------
        processed_mask : np.ndarray
            Post-processed binary mask
        """
        # Remove small components
        if self.min_area_ratio > 0:
            image_area = original_shape[0] * original_shape[1]
            min_area = int(self.min_area_ratio * image_area)
            
            labeled, num_features = ndimage.label(mask)
            for i in range(1, num_features + 1):
                component = (labeled == i)
                if component.sum() < min_area:
                    mask[component] = 0
        
        # Morphological closing
        if self.use_morphology:
            # Use scipy's binary_closing with elliptical structuring element
            from scipy.ndimage import generate_binary_structure, iterate_structure
            # Create a disk-like structure (approximation of ellipse)
            struct = generate_binary_structure(2, 1)
            struct = iterate_structure(struct, 2)  # Expand to ~5x5
            mask = binary_closing(mask, structure=struct).astype(np.uint8)
        
        return mask
    
    def tune_threshold(
        self,
        val_images: List[np.ndarray],
        val_masks: List[np.ndarray],
        predict_fn,
        metric_fn,
        threshold_range: Tuple[float, float] = (0.3, 0.7),
        num_steps: int = 41,
    ) -> float:
        """
        Tune threshold on validation set to maximize a metric (e.g., Dice, IoU).
        
        Parameters
        ----------
        val_images : List[np.ndarray]
            Validation images
        val_masks : List[np.ndarray]
            Ground truth masks
        predict_fn : callable
            Prediction function
        metric_fn : callable
            Metric function that takes (pred_mask, gt_mask) and returns a score
        threshold_range : Tuple[float, float]
            Range of thresholds to search
        num_steps : int
            Number of threshold values to try
        
        Returns
        -------
        best_threshold : float
            Optimal threshold value
        """
        thresholds = np.linspace(threshold_range[0], threshold_range[1], num_steps)
        best_threshold = self.threshold
        best_score = -np.inf
        
        logger.info(f"Tuning threshold on {len(val_images)} validation images...")
        
        # First, get all probability predictions with current TTA settings
        all_probs = []
        for img in val_images:
            _, prob = self.predict_with_tta(img, predict_fn, return_probs=True)
            all_probs.append(prob)
        
        # Try each threshold
        for threshold in thresholds:
            scores = []
            for prob, gt_mask in zip(all_probs, val_masks):
                pred_mask = (prob > threshold).astype(np.uint8)
                pred_mask = self._post_process(pred_mask, gt_mask.shape)
                score = metric_fn(pred_mask, gt_mask)
                scores.append(score)
            
            avg_score = np.mean(scores)
            
            if avg_score > best_score:
                best_score = avg_score
                best_threshold = threshold
        
        logger.info(f"Best threshold: {best_threshold:.3f} (score: {best_score:.4f})")
        self.threshold = best_threshold
        
        return best_threshold


def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    """
    Calculate Dice coefficient for binary masks.
    
    Parameters
    ----------
    pred : np.ndarray
        Predicted binary mask
    target : np.ndarray
        Ground truth binary mask
    smooth : float
        Smoothing factor to avoid division by zero
    
    Returns
    -------
    dice : float
        Dice coefficient
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return dice


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    """
    Calculate IoU (Jaccard) score for binary masks.
    
    Parameters
    ----------
    pred : np.ndarray
        Predicted binary mask
    target : np.ndarray
        Ground truth binary mask
    smooth : float
        Smoothing factor to avoid division by zero
    
    Returns
    -------
    iou : float
        IoU score
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou

