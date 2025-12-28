import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..constants import COLUMN, IMAGE, LABEL, LOGITS, MOE_LOSS
from .adaptation_layers import ConvLoRAConv2d
from .utils import image_mean_std

logger = logging.getLogger(__name__)


class ResEncUNetForSemanticSegmentation(nn.Module):
    """
    ResEnc U-Net（dynamic-network-architectures）包装成 AutoGluon semantic_segmentation 的统一接口。

    目标：
    - 走 AutoGluon 的 data processor（resize/normalize）与 Lightning 训练循环
    - 允许通过 optim.peft=conv_lora 注入 ConvLoRAConv2d，并把 MOE_LOSS 透传到训练 loss
    - 输出二值 logits（shape: [B, 1, H, W]）
    """

    def __init__(
        self,
        prefix: str,
        num_classes: int = 1,
        image_size: int = 512,
        image_norm: Optional[str] = "imagenet",
        pretrained: bool = False,
        # ResEnc UNet hyperparams（先给出合理默认，后续可用 config 覆盖）
        n_stages: int = 7,
        features_per_stage=(32, 64, 128, 256, 512, 512, 512),
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6, 6),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1, 1),
        deep_supervision: bool = False,
    ):
        super().__init__()
        if pretrained:
            logger.warning("ResEncUNetForSemanticSegmentation 当前不支持加载预训练权重，pretrained=True 将被忽略。")

        assert num_classes == 1, "当前选择的是二值 logits 形式，请将 num_classes 设为 1。"

        self.prefix = prefix
        self.num_classes = num_classes
        self.image_size = image_size
        self.image_mean, self.image_std = image_mean_std(image_norm)
        # name_to_id 用于 layerwise lr 等插件；简单全部设 0
        self.name_to_id = {n: 0 for n, _ in self.named_parameters()}

        from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

        self.model = ResidualEncoderUNet(
            input_channels=3,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=nn.Conv2d,
            kernel_sizes=3,
            strides=(1, 2, 2, 2, 2, 2, 2),
            n_blocks_per_stage=n_blocks_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=True,
            norm_op=nn.InstanceNorm2d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=deep_supervision,
        )

    @property
    def image_key(self):
        return f"{self.prefix}_{IMAGE}"

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    @property
    def image_column_prefix(self):
        return f"{self.image_key}_{COLUMN}"

    def _collect_moe_loss(self) -> Optional[torch.Tensor]:
        moe = None
        for m in self.modules():
            if isinstance(m, ConvLoRAConv2d) and getattr(m, "moe_loss", None) is not None:
                moe = m.moe_loss if moe is None else (moe + m.moe_loss)
        return moe

    def forward(self, batch):
        x = batch[self.image_key]
        # semantic seg processor 会产出 [B, N, C, H, W]，本任务默认 N=1
        if x.dim() == 5:
            x = x[:, 0]
        # 网络输出期望 [B, 1, H, W]
        logits = self.model(x)
        # deep_supervision=False 时 logits 是 Tensor；True 时为 list/tuple（暂不支持）
        if isinstance(logits, (list, tuple)):
            raise NotImplementedError("deep_supervision=True 暂未接入 AutoGluon loss，请先使用 deep_supervision=False。")

        # 对齐 label 的空间分辨率（保险起见）
        if logits.shape[-2:] != batch[self.label_key].shape[-2:]:
            logits = F.interpolate(logits, batch[self.label_key].shape[-2:], mode="bilinear", align_corners=False)

        if self.training:
            out = {self.prefix: {LOGITS: logits}}
        else:
            out = {self.prefix: {LOGITS: logits, LABEL: batch[self.label_key]}}

        moe_loss = self._collect_moe_loss()
        if moe_loss is not None:
            out[self.prefix][MOE_LOSS] = moe_loss
        return out

    def get_layer_ids(self):
        # 先给一个最简单的实现，确保 layerwise lr decay 的调用不报错
        return {n: 0 for n, _ in self.named_parameters()}


