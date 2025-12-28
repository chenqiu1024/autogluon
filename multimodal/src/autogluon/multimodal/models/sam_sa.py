import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..constants import LOGITS, LABEL, IMAGE, IMAGE_VALID_NUM, MASK_LABEL, CLASS_LABEL, COLUMN, BOX

try:
    from segment_anything import sam_model_registry as sa_model_registry
    SA_AVAILABLE = True
except Exception:
    SA_AVAILABLE = False

logger = logging.getLogger(__name__)


class SAMForSemanticSegmentationSA(nn.Module):
    """
    Minimal segment_anything backend wrapper to align interface with HF SAM counterpart.
    Currently supports binary segmentation only, without Conv-LoRA/Adapter/GSPO injections.
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str,
        num_classes: int = 1,
        frozen_layers: Optional[list] = None,
        num_mask_tokens: int = 1,
        image_norm: Optional[str] = None,
    ):
        super().__init__()
        if not SA_AVAILABLE:
            raise ImportError(
                "segment_anything is not installed. Please install it or use --sam_backend hf."
            )
        self.prefix = prefix
        self.num_classes = num_classes
        self.frozen_layers = frozen_layers
        self.num_mask_tokens = num_mask_tokens
        self.image_norm = image_norm
        self.output_moe_loss = False

        self._load_checkpoint(checkpoint_name)

        # try to infer image size; default 1024 for SAM
        try:
            self.image_size = self.model.image_encoder.img_size
        except Exception:
            self.image_size = 1024

    def _load_checkpoint(self, checkpoint_name: str):
        model_type = "vit_h"
        lower_name = str(checkpoint_name).lower()
        if "vit_l" in lower_name:
            model_type = "vit_l"
        elif "vit_b" in lower_name:
            model_type = "vit_b"
        if model_type not in sa_model_registry:
            raise ValueError(f"[SA] Unsupported model type: {model_type}")
        try:
            self.model = sa_model_registry[model_type](checkpoint=checkpoint_name)
            logger.info(f"[SA] Loaded SAM ({model_type}) from {checkpoint_name}")
        except Exception as e:
            raise RuntimeError(f"[SA] Failed to load SAM from {checkpoint_name}: {e}") from e

    # key helpers to match HF counterpart
    @property
    def image_key(self):
        return f"{self.prefix}_{IMAGE}"

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    @property
    def mask_label_key(self):
        return f"{self.prefix}_{MASK_LABEL}"

    @property
    def class_label_key(self):
        return f"{self.prefix}_{CLASS_LABEL}"

    @property
    def box_key(self):
        return f"{self.prefix}_box"

    @property
    def image_column_prefix(self):
        return f"{self.image_key}_{COLUMN}"

    @property
    def image_path_key(self):
        return f"{self.prefix}_image_path"

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def forward(self, batch):
        """
        Binary segmentation only (num_classes=1). Multi-class not yet supported for SA backend.
        """
        if self.num_classes != 1:
            raise NotImplementedError("SA backend currently supports binary segmentation only.")

        images = batch[self.image_key]  # shape: [B, C, H, W]
        boxes = batch.get(self.box_key, None)
        B = images.shape[0]

        pred_masks_list = []
        for i in range(B):
            img = images[i : i + 1]
            box = None
            if boxes is not None:
                box = boxes[i : i + 1]

            # image encoder returns embedding and reg_loss (ignored)
            image_embeddings, _ = self.model.image_encoder(img)

            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=box,
                masks=None,
            )

            low_res_masks, _ = self.model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            # upsample to input image size
            up_mask = F.interpolate(
                low_res_masks,
                size=(img.shape[-2], img.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks_list.append(up_mask)

        pred_masks = torch.cat(pred_masks_list, dim=0)  # [B, 1, H, W]

        if self.training:
            rets_dict = {self.prefix: {LOGITS: pred_masks}}
        else:
            rets_dict = {self.prefix: {LOGITS: pred_masks, LABEL: batch[self.label_key]}}

        return rets_dict

