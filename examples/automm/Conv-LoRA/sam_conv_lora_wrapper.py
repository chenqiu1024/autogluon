"""
SAM + Conv-LoRA 包装器。

目标：
- 从 AutoGluon 的 MultiModalPredictor / SemanticSegmentationLearner 中抽取底层模型；
- 冻结非 LoRA 参数，只保留 Conv-LoRA 参数可训练；
- 暴露一个相对简单的 forward 接口，后续 RLOO 训练脚本可以在此基础上实现自定义训练循环。

注意：
- 这里对 AutoGluon 内部结构做了一些假设，实际使用时如有不一致，请根据具体版本适当修改。
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import torch
from torch import nn

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.learners.semantic_segmentation import SemanticSegmentationLearner


class SAMConvLoRAWrapper(nn.Module):
    """
    一个轻量包装器，用于：
    - 访问并冻结 AutoGluon 内部的 SAM + Conv-LoRA 模型；
    - 只保留 Conv-LoRA 相关参数为可训练；
    - 暴露 `forward(batch)`，其中 batch 应该与 AutoGluon 训练时的 batch 字典保持一致。

    实际上，AutoGluon 在训练时会构造一个包含多模态字段的 batch 字典，
    这里我们不重新造轮子，而是直接在 RLOO 训练脚本中复用 DataModule
    所生成的 batch，并将其传入该 wrapper。
    """

    def __init__(self, predictor: MultiModalPredictor):
        super().__init__()
        self.predictor = predictor
        self.learner: SemanticSegmentationLearner = predictor._learner  # type: ignore
        if not isinstance(self.learner, SemanticSegmentationLearner):
            raise TypeError(
                "SAMConvLoRAWrapper 目前仅支持 'semantic_segmentation' 问题类型，"
                "请确保在构建 MultiModalPredictor 时 problem_type='semantic_segmentation'。"
            )

        # 底层 nn.Module（包含 SAM + Conv-LoRA）
        self.model: nn.Module = self.learner._model  # type: ignore[attr-defined]
        self.config = self.learner._config  # type: ignore[attr-defined]

        # 先全部冻结，再只打开 Conv-LoRA 参数
        for p in self.model.parameters():
            p.requires_grad = False

        # 使用 SemanticSegmentationLearner 提供的接口获取 PEFT 参数名
        peft_param_names = self.learner.get_peft_param_names_per_run(self.model, self.config)
        self.peft_param_names: Optional[List[str]] = peft_param_names

        trainable_names: List[str] = []
        if peft_param_names is None:
            # 兜底策略：如果配置中没有显式 PEFT 参数名，则默认所有包含 "lora" 字样的参数为可训练
            for name, p in self.model.named_parameters():
                if "lora" in name.lower():
                    p.requires_grad = True
                    trainable_names.append(name)
        else:
            peft_set = set(peft_param_names)
            for name, p in self.model.named_parameters():
                if name in peft_set:
                    p.requires_grad = True
                    trainable_names.append(name)

        self._trainable_param_names = trainable_names

    @property
    def trainable_param_names(self) -> List[str]:
        """
        返回当前仍处于可训练状态的参数名列表（主要是 Conv-LoRA 相关参数）。
        """
        return list(self._trainable_param_names)

    def forward(self, batch):
        """
        前向传播。

        参数
        ----
        batch:
            一个与 AutoGluon 训练阶段 DataModule 输出兼容的字典，
            通常包含图像张量、标签张量等键值。

        返回
        ----
        output:
            模型原始输出字典，通常包含 LOGITS / SEMANTIC_MASK 等键。
            RLOO 训练脚本可以在此基础上取出 logits，自行计算 Bernoulli 概率与 log_prob。
        """
        # 这里假设 self.model 是 AutoMMModel 类型，直接调用即可；
        # 实际上 AutoGluon 在训练中也是通过 run_model(self.model, batch) 来调用，
        # 因此如果你需要完全复用那一套逻辑，也可以在 RLOO 脚本中引入 run_model。
        return self.model(batch)


def freeze_non_lora_parameters(model: nn.Module, lora_keywords: Iterable[str] = ("lora",)) -> None:
    """
    一个通用的辅助函数：冻结除包含特定关键字（如 'lora'）以外的所有参数。
    在某些极简场景下，如果不想依赖 AutoGluon 的 peft 配置，可以直接调用本函数。
    """
    lora_keywords = tuple(k.lower() for k in lora_keywords)
    for name, p in model.named_parameters():
        if any(k in name.lower() for k in lora_keywords):
            p.requires_grad = True
        else:
            p.requires_grad = False


