"""
Lightweight bounding-box regression utilities for box-prompting SAM.

Task: given an input image, predict the minimal bounding box of the target
segmentation mask. Outputs normalized coordinates (x1, y1, x2, y2) in [0, 1],
which can be rescaled to original image pixels at inference time.
"""

import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import functional as TF


def compute_bbox_from_mask(mask: np.ndarray) -> Optional[np.ndarray]:
    """Compute (x1, y1, x2, y2) from a binary mask. Returns None if empty."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def giou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Generalized IoU loss for boxes in (x1, y1, x2, y2) format, normalized to [0,1].
    Returns 1 - GIoU.
    """
    # Intersection
    x1_int = torch.max(pred[:, 0], target[:, 0])
    y1_int = torch.max(pred[:, 1], target[:, 1])
    x2_int = torch.min(pred[:, 2], target[:, 2])
    y2_int = torch.min(pred[:, 3], target[:, 3])

    inter_w = torch.clamp(x2_int - x1_int, min=0)
    inter_h = torch.clamp(y2_int - y1_int, min=0)
    inter_area = inter_w * inter_h

    # Areas
    area_pred = torch.clamp(pred[:, 2] - pred[:, 0], min=0) * torch.clamp(
        pred[:, 3] - pred[:, 1], min=0
    )
    area_tgt = torch.clamp(target[:, 2] - target[:, 0], min=0) * torch.clamp(
        target[:, 3] - target[:, 1], min=0
    )

    union = area_pred + area_tgt - inter_area + eps
    iou = inter_area / union

    # Smallest enclosing box
    x1_c = torch.min(pred[:, 0], target[:, 0])
    y1_c = torch.min(pred[:, 1], target[:, 1])
    x2_c = torch.max(pred[:, 2], target[:, 2])
    y2_c = torch.max(pred[:, 3], target[:, 3])

    area_c = torch.clamp(x2_c - x1_c, min=0) * torch.clamp(y2_c - y1_c, min=0) + eps

    giou = iou - (area_c - union) / area_c
    return 1.0 - giou


@dataclass
class BBoxSample:
    image: torch.Tensor
    bbox: torch.Tensor  # normalized (4,)
    has_box: torch.Tensor  # float scalar
    orig_size: Tuple[int, int]
    image_path: str


class BBoxDataset(Dataset):
    """
    Dataset for bbox regression.

    Expected DataFrame columns:
      - image: path to RGB image
      - label: path to mask (binary)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 320,
        augment: bool = False,
        rotation_deg: float = 10.0,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.augment = augment
        self.rotation_deg = rotation_deg
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __len__(self) -> int:
        return len(self.df)

    def _apply_transforms(self, img: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if self.rotation_deg > 0:
                angle = random.uniform(-self.rotation_deg, self.rotation_deg)
                img = TF.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)
        # 固定方形缩放，确保 batch 内尺寸一致，避免 DataLoader stack 报错
        img = TF.resize(
            img,
            (self.image_size, self.image_size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        mask = TF.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        return img, mask

    def __getitem__(self, idx: int) -> BBoxSample:
        row = self.df.iloc[idx]
        image_path = row["image"]
        label_path = row["label"]

        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size
        mask = Image.open(label_path)
        if mask.mode != "L":
            mask = mask.convert("L")

        img, mask = self._apply_transforms(img, mask)

        mask_np = np.array(mask)
        bbox = compute_bbox_from_mask(mask_np)
        has_box = 1.0 if bbox is not None else 0.0
        if bbox is None:
            bbox = np.zeros(4, dtype=np.float32)
        # Normalize by transformed image size (keeps ratios w.r.t original)
        w, h = img.size
        bbox_norm = bbox / np.array([w, h, w, h], dtype=np.float32)
        bbox_norm = np.clip(bbox_norm, 0.0, 1.0)

        img_t = TF.to_tensor(img)
        img_t = self.normalize(img_t)

        return BBoxSample(
            image=img_t,
            bbox=torch.from_numpy(bbox_norm.astype(np.float32)),
            has_box=torch.tensor(has_box, dtype=torch.float32),
            orig_size=(orig_h, orig_w),
            image_path=image_path,
        )


class BBoxRegressor(nn.Module):
    """Simple ResNet-18 backbone + MLP head for bbox regression."""

    def __init__(self, pretrained: bool = True, dropout: float = 0.1):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            except Exception as e:
                # Fallback when weight download/check_hash fails (offline / mirror issue)
                print(f"[Warning] Failed to load ResNet18 pretrained weights ({e}); using random init.")
                weights = None

        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        logits = self.head(feat)
        # Sigmoid to keep coordinates in [0,1]
        return torch.sigmoid(logits)


@torch.no_grad()
def infer_single_bbox(
    model: nn.Module,
    image_path: str,
    device: torch.device,
    image_size: int = 320,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Predict bbox for one image. Returns pixel coordinates (x1,y1,x2,y2) on original image scale.
    """
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    img_resized = TF.resize(img, image_size, interpolation=transforms.InterpolationMode.BILINEAR)
    img_t = TF.to_tensor(img_resized)
    img_t = transforms.Normalize(mean=mean, std=std)(img_t)
    img_t = img_t.unsqueeze(0).to(device)

    pred_norm = model(img_t)[0].clamp(0.0, 1.0)
    pred_np = pred_norm.detach().cpu().numpy()
    x1, y1, x2, y2 = pred_np
    # Ensure ordering
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    bbox_px = np.array([x1 * orig_w, y1 * orig_h, x2 * orig_w, y2 * orig_h], dtype=np.float32)
    return bbox_px


def save_checkpoint(state: Dict, path: str):
    torch.save(state, path)


def load_checkpoint(model: nn.Module, path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint


def collate_fn(batch):
    images = torch.stack([b.image for b in batch], dim=0)
    bboxes = torch.stack([b.bbox for b in batch], dim=0)
    has_box = torch.stack([b.has_box for b in batch], dim=0)
    meta = {"orig_size": [b.orig_size for b in batch], "image_path": [b.image_path for b in batch]}
    return images, bboxes, has_box, meta


def bbox_metrics(pred: torch.Tensor, target: torch.Tensor, has_box: torch.Tensor) -> Dict[str, float]:
    """Compute L1 and IoU on samples with boxes."""
    mask = has_box > 0.5
    if mask.sum() == 0:
        return {"l1": 0.0, "iou": 0.0}
    pred = pred[mask]
    target = target[mask]

    l1 = F.l1_loss(pred, target, reduction="mean").item()

    # IoU
    def _iou(a, b):
        x1 = torch.max(a[:, 0], b[:, 0])
        y1 = torch.max(a[:, 1], b[:, 1])
        x2 = torch.min(a[:, 2], b[:, 2])
        y2 = torch.min(a[:, 3], b[:, 3])
        inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area_a = torch.clamp(a[:, 2] - a[:, 0], min=0) * torch.clamp(a[:, 3] - a[:, 1], min=0)
        area_b = torch.clamp(b[:, 2] - b[:, 0], min=0) * torch.clamp(b[:, 3] - b[:, 1], min=0)
        union = area_a + area_b - inter + 1e-6
        return (inter / union).mean().item()

    iou = _iou(pred, target)
    return {"l1": l1, "iou": iou}


def train_one_epoch(model, loader, optimizer, device, l1_weight=1.0, giou_weight=0.5):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, bboxes, has_box, _ in loader:
        images = images.to(device)
        bboxes = bboxes.to(device)
        has_box = has_box.to(device)

        preds = model(images)

        # Mask samples without bbox
        mask = has_box > 0.5
        if mask.sum() == 0:
            continue
        preds_masked = preds[mask]
        bboxes_masked = bboxes[mask]

        l1 = F.l1_loss(preds_masked, bboxes_masked)
        g = giou_loss(preds_masked, bboxes_masked).mean()
        loss = l1_weight * l1 + giou_weight * g

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = mask.sum().item()
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []
    all_has_box = []
    for images, bboxes, has_box, _ in loader:
        images = images.to(device)
        bboxes = bboxes.to(device)
        has_box = has_box.to(device)

        preds = model(images)

        mask = has_box > 0.5
        if mask.sum() > 0:
            preds_masked = preds[mask]
            bboxes_masked = bboxes[mask]
            l1 = F.l1_loss(preds_masked, bboxes_masked)
            g = giou_loss(preds_masked, bboxes_masked).mean()
            loss = l1 + 0.5 * g
            total_loss += loss.item() * mask.sum().item()
            total_samples += mask.sum().item()

        all_preds.append(preds.cpu())
        all_targets.append(bboxes.cpu())
        all_has_box.append(has_box.cpu())

    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    has_box_cat = torch.cat(all_has_box, dim=0)
    metrics = bbox_metrics(preds_cat, targets_cat, has_box_cat)
    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, metrics


class BBoxPromptPredictor:
    """
    Lightweight inference wrapper for bbox regression.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: Optional[torch.device] = None,
        image_size: int = 320,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.model = BBoxRegressor(pretrained=False).to(self.device)
        load_checkpoint(self.model, ckpt_path, self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image_path: str) -> Optional[np.ndarray]:
        bbox_px = infer_single_bbox(
            self.model,
            image_path=image_path,
            device=self.device,
            image_size=self.image_size,
            mean=self.mean,
            std=self.std,
        )
        # Return as numpy (x1,y1,x2,y2)
        return bbox_px


def export_pred_csv(df: pd.DataFrame, predictor: BBoxPromptPredictor, out_path: str):
    """
    Run inference on a DataFrame with 'image' column and save bbox predictions as CSV
    with columns: image, bbox_x1, bbox_y1, bbox_x2, bbox_y2.
    """
    records = []
    for _, row in df.iterrows():
        img_path = row["image"]
        bbox = predictor.predict(img_path)
        records.append(
            {
                "image": img_path,
                "bbox_x1": float(bbox[0]),
                "bbox_y1": float(bbox[1]),
                "bbox_x2": float(bbox[2]),
                "bbox_y2": float(bbox[3]),
            }
        )
    pd.DataFrame(records).to_csv(out_path, index=False)

