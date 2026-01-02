# SAM框架的半弱监督方向
_Exported on 1/1/2026 at 18:55:45 PST from Cursor (2.2.44)_

---

**User**

从docs/cursor_gspo_v2-opus.md恢复对话上下文。如果我想把半/弱监督方法应用于当前的"SAM + Conv-LoRA + FC-Adapter + GSPO微调 + 推理时加预测box prompt增强" 的框架上，有什么方向可以考虑？

---

**Cursor**

我先从 `docs/cursor_gspo_v2-opus.md` 里把与你当前这套“SAM + Conv‑LoRA + FC‑Adapter + GSPO 微调 + 推理时 box prompt 增强”的关键信息（训练/损失/数据设定/推理流程）定位出来；然后在不改你整体框架的前提下，按**半监督**与**弱监督**两条线给出可以直接落地的方向（伪标签、教师‑学生一致性、box/点/涂鸦监督、MIL 等），并标注各自最适合插入到你现有 pipeline 的位置。

很好，`cursor_gspo_v2-opus.md` 里已经把你当前框架的关键点写得很清楚：**SAM 主干冻结**、主要训练 **Conv‑LoRA(MoE+门控)+Adapter**，以及 **GSPO 主要在 MoE 门控上做“质量驱动”的策略优化**。我接下来会把文档里与你推理“box prompt 增强”相关的段落也定位出来，再基于这些约束给半/弱监督的可插入点。

我已经从文档开头定位到了你这套框架的“训练哪些参数/GSPO作用域”核心结论，也找到了“训练/推理阶段注入预测框 box prompt”的实现位置。接下来我会把这两段原文关键片段截出来做引用，然后基于这些约束给出半监督与弱监督的可落地方向（并说明插到你 pipeline 的哪个环节最顺）。

### 已恢复的对话上下文（与你当前框架直接相关的要点）

- **GSPO 主要微调对象**：当前设计里，GSPO 强化学习微调的核心是 **Conv‑LoRA 的 MoE 门控（`w_gate`, `w_noise`）与 MoE‑Conv 专家**；FC Adapter 走标准监督，SAM 主干冻结。

```128:135:/root/autodl-tmp/works/autogluon/docs/cursor_gspo_v2-opus.md
| 模块 | 是否被 GSPO 微调 | 微调方式 |
|------|------------------|----------|
| **MoEGate (`w_gate`, `w_noise`)** | ✅ 是 | 质量偏置 + 优势加权梯度 |
| **MoE-Conv 专家 (`lora_moe_experts`)** | ✅ 是 | 优势加权梯度 |
| **LoRA 低秩矩阵 (`lora_A`, `lora_B`)** | ✅ 是 | 标准梯度（通过 GSPO 加权） |
| **Encoder-Adapter** | ❌ 否 | 标准监督学习 |
| **Decoder-Adapter** | ❌ 否 | 标准监督学习 |
| **SAM 主干参数** | ❌ 否 | 冻结 |
```

- **你提到的“推理时加预测 box prompt 增强”在代码里也有对应入口**：支持 `bbox_prompt_source=predict`，以及训练阶段也能 `train_box_prompt_mode=predict/mix/noisy` 注入框作为 prompt。

```5294:5311:/root/autodl-tmp/works/autogluon/docs/cursor_gspo_v2-opus.md
parser.add_argument("--tta_box_prompt_mode", type=str, default="off",
                    choices=["off", "add", "replace"],
                    help="Use GT box as SAM box prompt (off/add/replace); box not transformed with TTA")
# BBox prompt configuration
parser.add_argument("--bbox_prompt_source", type=str, default="none",
                    choices=["none", "gt", "predict"],
                    help="Box prompt source: none (no prompt), gt (from mask), predict (from bbox model).")
...
# Training-time box prompt mixing (default off)
parser.add_argument("--train_box_prompt_mode", type=str, default="off",
                    choices=["off", "gt", "noisy", "mix", "predict"],
                    help="Training-time box prompt injection: off/gt/noisy/mix/predict (default off). "
                         "predict: use bbox predictor model to generate box prompts from images.")
```

---

### 把半/弱监督“塞进”你这套框架的方向（按最贴合你现有组件来组织）

下面每个方向都尽量**复用你已具备的能力**：Conv‑LoRA+Adapter 的可训练性、GSPO 的“组内多次采样/质量加权”、以及 box prompt（含预测框/噪声框）的训练与推理入口。

## 半监督（有一部分像素级 GT + 大量无标签图）

- **方向 A：Teacher‑Student（EMA 教师）+ 一致性正则（强推）**
  - **怎么接入你现有框架**：把无标签图也走一遍前向；教师（EMA）在“弱增强/稳 prompt（例如预测框或较少抖动的框）”下出伪 mask；学生在“强增强 + prompt 扰动（你已有 noisy box）”下预测；做一致性损失（Dice/CE/KL）。
  - **你现成的杠杆点**：训练阶段已有 `train_box_prompt_mode=noisy/mix/predict`（可直接把“多视图”做成 prompt 视图 + 图像增强视图）。

- **方向 B：伪标签自训练（Pseudo‑labeling）但用“框 prompt”当可靠性约束**
  - **核心**：无标签样本先用当前模型（或 EMA 教师）在 **预测框 prompt** 下出 mask；只保留高置信/高稳定样本进入下一轮训练。
  - **关键技巧（适配 SAM + box）**：
    - 过滤规则别只看像素置信度：加上 **mask‑box 一致性**（mask 主要应落在 box 内、对 box 边界外泄惩罚、面积比例在合理区间）。
    - 用你已有的 **noisy box** 做稳定性测试：同一图像多次 jitter box，mask 形状一致才收。

- **方向 C：把 GSPO 的“质量信号”从有标签扩展到无标签（用代理 reward）**
  - **问题**：无标签没有 IoU 质量，GSPO 的 advantage 需要替代指标。
  - **可行的代理质量（reward）**（从最稳到最激进）：
    - **Teacher agreement**：学生预测与 EMA 教师预测的 IoU/Dice（把它当 reward）。
    - **Prompt‑jitter stability**：同图像同模型多次 jitter box，预测互相一致性（越一致 reward 越高）。
    - **Entropy/置信度**：mask 概率熵低（更确定）reward 高（要小心过度自信的错误）。
  - **好处**：仍能让 MoE 门控朝“稳定/可泛化的专家组合”偏置，而不是只对有标签样本过拟合。

## 弱监督（没有像素级 GT，只有 box/点/涂鸦/图像级标签等）

- **方向 D：纯 box 监督的分割训练（你框架天然匹配）**
  - **你已经在做的关键一步**：推理用 box prompt 增强；进一步可以把训练也改成“主要靠 box”。
  - **常用可落地损失（不依赖全像素 GT）**：
    - **Box‑inside / Box‑outside 约束**：box 外强负；box 内用“软正”（或仅对高置信区域算正），其余 ignore（部分标注损失）。
    - **投影/包围约束**：mask 在 x/y 方向投影应覆盖 box（弱但稳定）。
    - **边界/紧致先验**：TV/边界长度正则 + 面积先验，抑制 box 内全涂满。
  - **和你现有的 noisy box 特别搭**：把 box 抖动当数据增强，同时要求 mask 随 box 合理变化（不该完全崩）。

- **方向 E：点/涂鸦（scribble）监督 + SAM prompt（成本低、效果常比纯 box 好）**
  - **做法**：把少量点/涂鸦转成 prompt（正点/负点/稀疏线）；损失只在标注点附近监督，其它像素 ignore；再叠加一致性/先验。
  - **为什么适合你**：SAM 对点/框 prompt 很敏感；你只训练小参数（LoRA/Adapter），很适合快速吸收这种交互式弱信号。

- **方向 F：图像级标签（有无目标/类别）→ MIL/定位 → 框 → SAM 分割**
  - **链路**：先用 MIL/分类头或轻量检测器在图像级标签下产粗框（或热力图→框），再把框当 prompt 喂给 SAM，生成 mask 作为伪标签或直接做 box‑supervised 训练。
  - **你已有 `bbox_prompt_source=predict`**：如果你的 bbox predictor 能从弱标签训练出来，这条路几乎是“即插即用”。

---

### 我建议你优先走的 2 条“投入产出最高”的路线

- **首选（半监督）**：**EMA Teacher +（图像增强 + noisy box）一致性**  
  你已经具备 “训练时 box prompt 注入 + box jitter”，这是半监督里最值钱的“多视图构造器”。

- **首选（弱监督）**：**box 监督 + 稳定性过滤的伪标签循环**  
  把“预测框→mask”做成自动标注器，靠稳定性/box一致性筛掉垃圾伪标签，再用你的小参数微调吃进去。

---

如果你告诉我：你目前可用的弱标签形式（只有预测框？有人标 box？有点/涂鸦？还是纯无标签？）以及标注比例（比如 5% 像素级 GT + 95% 无标签），我可以把上面方向收敛成**一套具体训练配方**（loss 组合、采样比例、warmup、以及 GSPO 的 reward 如何改造）。

```plaintext
| 模块 | 是否被 GSPO 微调 | 微调方式 |
|------|------------------|----------|
| **MoEGate (`w_gate`, `w_noise`)** | ✅ 是 | 质量偏置 + 优势加权梯度 |
| **MoE-Conv 专家 (`lora_moe_experts`)** | ✅ 是 | 优势加权梯度 |
| **LoRA 低秩矩阵 (`lora_A`, `lora_B`)** | ✅ 是 | 标准梯度（通过 GSPO 加权） |
| **Encoder-Adapter** | ❌ 否 | 标准监督学习 |
| **Decoder-Adapter** | ❌ 否 | 标准监督学习 |
| **SAM 主干参数** | ❌ 否 | 冻结 |
```

```plaintext
parser.add_argument("--tta_box_prompt_mode", type=str, default="off",
                    choices=["off", "add", "replace"],
                    help="Use GT box as SAM box prompt (off/add/replace); box not transformed with TTA")
# BBox prompt configuration
parser.add_argument("--bbox_prompt_source", type=str, default="none",
                    choices=["none", "gt", "predict"],
                    help="Box prompt source: none (no prompt), gt (from mask), predict (from bbox model).")
...
# Training-time box prompt mixing (default off)
parser.add_argument("--train_box_prompt_mode", type=str, default="off",
                    choices=["off", "gt", "noisy", "mix", "predict"],
                    help="Training-time box prompt injection: off/gt/noisy/mix/predict (default off). "
                         "predict: use bbox predictor model to generate box prompts from images.")
```

---

**User**

原始的ISIC2017数据集中全部都有full mask，但是为了往半/弱监督上“凑”，可以只使用其中一小部分(比如10%是否合适？可以把比例作为一个可调参数)作为有GT mask的样本；其余用模拟人工标注的包围框作为弱标签，其生成方式可以是在GT mask的最小包围框基础上做一定的随机抖动，抖动应该绝大部分是向外扩而不是向内缩，因为从实际的使用场景来看，人类专家用交互工具手绘一个粗略的框来框选待分割的部分的话，应该是绝大部分情况下都只会比真值mask最小bounding box更大。并且要考虑提供参数选项使得可以选择随机抖动的程度（也要考虑按什么标准来定义为好，比如是按相对框尺寸的比例，还是按相对整张图片的比例，还是绝对像素大小？）。

---

**Cursor**

我会把需求拆成三块落地：先在 ISIC2017 的数据准备阶段加入“有 mask 的比例（可配）”，其余样本标记为弱监督；再把弱监督样本的 **box 生成+抖动（外扩为主、抖动幅度可配/可选定义标准）** 做成可复用函数；最后在训练里对“仅有 box 的样本”加一个仅依赖 box 的弱监督损失（至少约束 box 外为背景 + box 内熵最小/平滑），保证这些样本真的对学习有贡献。接下来我会先定位当前 ISIC2017 数据 CSV/数据加载与 `lit_semantic_seg.py` 的训练步实现位置。

Nice，核心入口都找到了：数据准备在 `examples/automm/Conv-LoRA/prepare_semantic_segmentation_datasets.py`，训练逻辑在 `multimodal/.../optim/lit_semantic_seg.py`，数据处理在 `multimodal/.../data/process_semantic_seg_img.py`。下一步我会先读数据准备脚本和数据处理器，搞清楚当前 CSV 需要哪些字段（image/mask/label 等），再决定把“弱监督仅 box”信息塞进哪一列最稳。

我已经确认两个关键约束：语义分割的 learner 在没显式 `column_types` 时会强制要求 DataFrame 只有“1 个图像列 + 1 个 label 列”，所以我们**不能靠在 CSV 里加 `is_labeled` 列**来做半监督；更稳的做法是用 `image_path` 做稳定哈希，按 `labeled_fraction` 决定该样本在训练中走“全监督”还是“仅 box 弱监督”。接下来我会直接改 `lit_semantic_seg.py`（实现按比例监督 + 外扩偏置 box jitter + 弱监督损失）以及 `run_semantic_segmentation.py`（加参数并把配置传进 learner）。

```python
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
            # Semi/weak-supervision simulation (optional)
            # NOTE: We avoid adding extra columns to dataframe because semantic segmentation learner
            # expects exactly one image column (+ label column) when column_types is not provided.
            # Instead, we deterministically assign "labeled" samples via stable hashing of image_path.
            "semi_labeled_fraction": 1.0,       # 1.0 => fully-supervised; e.g. 0.1 => only 10% uses GT mask loss
            "semi_labeled_seed": 0,
            # Weak box generation from GT (simulating human coarse boxes)
            "weak_box_jitter_mode": "box",      # box / image / pixel
            "weak_box_jitter_amount": 0.12,     # meaning depends on mode
            "weak_box_outward_only": True,      # mostly expand outward; True => never shrink vs GT bbox
            # Weak loss weights (box-only supervision)
            "weak_loss_outside_weight": 1.0,    # enforce outside-box background
            "weak_loss_entropy_weight": 0.05,   # entropy minimization inside box
            "weak_loss_tv_weight": 0.0,         # optional smoothness (total variation) inside box
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

        # Get GSPO-Adapter parameters (if available)
        sam_cfg = getattr(self._config.model, "sam", None)
        gspo_adapter_enabled = getattr(sam_cfg, "adapter_gspo_enabled", False) if sam_cfg else False
        gspo_adapter_momentum = getattr(sam_cfg, "adapter_gspo_momentum", 0.9) if sam_cfg else 0.9
        
        # Get GSPO-LoRA on Attention parameters (if available) - NEW
        gspo_lora_attention_enabled = getattr(sam_cfg, "gspo_lora_attention_enabled", False) if sam_cfg else False
        gspo_lora_attention_momentum = getattr(sam_cfg, "gspo_lora_attention_momentum", 0.9) if sam_cfg else 0.9

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
                # GSPO-Adapter extension
                gspo_adapter_enabled=gspo_adapter_enabled,
                gspo_adapter_momentum=gspo_adapter_momentum,
                # GSPO-LoRA on Attention extension (NEW)
                gspo_lora_attention_enabled=gspo_lora_attention_enabled,
                gspo_lora_attention_momentum=gspo_lora_attention_momentum,
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

```

```python
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
            # Only supported for binary segmentation with StructureLoss-like supervision.
            if isinstance(self.loss_func, Mask2FormerLoss):
                raise NotImplementedError("semi_labeled_fraction < 1.0 not supported for Mask2FormerLoss yet.")
            # pick logits for current model prefix (binary logits)
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
            
            # weak loss on unlabeled samples (requires boxes)
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
            
            # Keep any MoE loss from output dict
            moe_loss = output[self.model.prefix].get(MOE_LOSS, 0.0)
            loss = sup_loss + weak_loss + moe_loss
            # lightweight logging
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

```

```python
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
            # Only supported for binary segmentation with StructureLoss-like supervision.
            if isinstance(self.loss_func, Mask2FormerLoss):
                raise NotImplementedError("semi_labeled_fraction < 1.0 not supported for Mask2FormerLoss yet.")
            # pick logits for current model prefix (binary logits)
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
            
            # weak loss on unlabeled samples (requires boxes)
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
            
            # Keep any MoE loss from output dict
            moe_loss = output[self.model.prefix].get(MOE_LOSS, 0.0)
            loss = sup_loss + weak_loss + moe_loss
            # lightweight logging
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
        # Note: semi/weak supervision mixes labeled+unlabeled samples and uses custom loss;
        # current GSPO path assumes full GT masks for the whole batch.
        cfg = getattr(self, "train_box_prompt_cfg", {}) or {}
        semi_labeled_fraction = float(cfg.get("semi_labeled_fraction", 1.0))
        use_gspo = (
            self.gspo_trainer is not None and 
            self.gspo_trainer.is_gspo_active(self.current_epoch)
        )
        if semi_labeled_fraction < 1.0:
            use_gspo = False
        
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

```

```python
import argparse
import os

import numpy as np
import pandas as pd
import torch

from autogluon.multimodal import MultiModalPredictor
from bbox_prompt_model import BBoxPromptPredictor


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
    # GSPO reward / loss shaping hyper-parameters (可选，带默认值)
    parser.add_argument("--lambda_smooth", type=float, default=0.1, help="Smoothness loss weight (λ_smooth)")
    parser.add_argument("--lambda_boundary", type=float, default=0.3, help="Boundary loss weight (λ_boundary)")
    parser.add_argument("--w_boundary", type=float, default=0.3, help="Boundary reward weight (w_boundary)")
    parser.add_argument("--w_smooth", type=float, default=0.1, help="Smoothness reward weight (w_smooth)")
    parser.add_argument("--w_thin", type=float, default=0.05, help="Thin-structure reward weight (w_thin)")
    
    # Adapter parameters
    parser.add_argument("--adapter_enable", action="store_true", help="Enable standard Adapter modules")
    parser.add_argument("--adapter_dim", type=int, default=64, help="Dimension of the adapter bottleneck")
    
    # GSPO-Adapter extension parameters (Phase 1-2)
    parser.add_argument("--gspo_adapter_enable", action="store_true", 
                        help="Enable GSPO quality feedback for Encoder Adapters (Phase 1)")
    parser.add_argument("--gspo_adapter_momentum", type=float, default=0.9, 
                        help="Momentum for adapter quality history updates")
    parser.add_argument("--gspo_adapter_scale_adaptation", action="store_true",
                        help="Enable adaptive scale based on quality history (Phase 2)")
    
    # Decoder Attention LoRA parameters (LoRA on Attention)
    parser.add_argument("--decoder_attn_lora_enable", action="store_true", 
                        help="Enable LoRA on decoder attention Q/K/V projections")
    parser.add_argument("--decoder_attn_lora_r", type=int, default=8,
                        help="LoRA rank for decoder attention (default: 8)")
    parser.add_argument("--decoder_attn_lora_alpha", type=int, default=8,
                        help="LoRA alpha for decoder attention (default: 8, same as rank)")
    parser.add_argument("--decoder_attn_lora_dropout", type=float, default=0.0,
                        help="LoRA dropout for decoder attention (default: 0.0)")
    
    # GSPO-LoRA on Attention parameters (NEW)
    parser.add_argument("--gspo_lora_attention_enable", action="store_true",
                        help="Enable GSPO quality feedback for Decoder LoRA on Attention")
    parser.add_argument("--gspo_lora_attention_momentum", type=float, default=0.9,
                        help="Momentum for LoRA quality history updates (default: 0.9)")
    parser.add_argument("--gspo_lora_attention_scale_adaptation", action="store_true",
                        help="Enable adaptive scaling based on quality for LoRA")
    
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
    parser.add_argument("--tta_box_prompt_mode", type=str, default="off",
                        choices=["off", "add", "replace"],
                        help="Use GT box as SAM box prompt (off/add/replace); box not transformed with TTA")
    # BBox prompt configuration
    parser.add_argument("--bbox_prompt_source", type=str, default="none",
                        choices=["none", "gt", "predict"],
                        help="Box prompt source: none (no prompt), gt (from mask), predict (from bbox model).")
    parser.add_argument("--bbox_model_ckpt", type=str, default=None,
                        help="Checkpoint for bbox predictor when bbox_prompt_source=predict.")
    parser.add_argument("--bbox_model_image_size", type=int, default=320,
                        help="Input size for bbox predictor inference.")
    parser.add_argument("--bbox_preds_csv", type=str, default=None,
                        help="Optional CSV with columns image,bbox_x1,bbox_y1,bbox_x2,bbox_y2 to attach prompts.")
    # Training-time box prompt mixing (default off)
    parser.add_argument("--train_box_prompt_mode", type=str, default="off",
                        choices=["off", "gt", "noisy", "mix", "predict"],
                        help="Training-time box prompt injection: off/gt/noisy/mix/predict (default off). "
                             "predict: use bbox predictor model to generate box prompts from images.")
    parser.add_argument("--train_box_prob_no_prompt", type=float, default=0.2,
                        help="When mode=mix, probability of no prompt.")
    parser.add_argument("--train_box_prob_gt", type=float, default=0.3,
                        help="When mode=mix, probability of GT box prompt.")
    parser.add_argument("--train_box_prob_noisy", type=float, default=0.5,
                        help="When mode=mix, probability of noisy GT box prompt.")
    parser.add_argument("--train_box_noise_frac", type=float, default=0.12,
                        help="Noisy box jitter fraction relative to box size (e.g., 0.12 => ±12%).")
    parser.add_argument("--train_bbox_model_ckpt", type=str, default=None,
                        help="Checkpoint for bbox predictor when train_box_prompt_mode=predict.")
    parser.add_argument("--train_bbox_model_image_size", type=int, default=320,
                        help="Input size for bbox predictor during training (when train_box_prompt_mode=predict).")
    
    # Semi/weak supervision simulation (ISIC2017 has full masks; we can hide most masks and supervise with coarse boxes)
    parser.add_argument("--semi_labeled_fraction", type=float, default=1.0,
                        help="Fraction of samples to use GT mask loss (e.g., 0.1 => only 10% supervised, rest weak).")
    parser.add_argument("--semi_labeled_seed", type=int, default=0,
                        help="Seed for deterministic labeled/unlabeled assignment via hashing image paths.")
    parser.add_argument("--weak_box_jitter_mode", type=str, default="box",
                        choices=["box", "image", "pixel"],
                        help="How to define weak box jitter amount: "
                             "box=fraction of GT bbox size; image=fraction of full image size; pixel=absolute pixels.")
    parser.add_argument("--weak_box_jitter_amount", type=float, default=0.12,
                        help="Weak box jitter amount (meaning depends on --weak_box_jitter_mode). "
                             "Typical: 0.05~0.3 for box/image, or 5~30 for pixel.")
    parser.add_argument("--weak_box_outward_only", action="store_true",
                        help="If set, weak boxes are generated by outward expansion only (never shrink vs GT bbox).")
    parser.add_argument("--weak_loss_outside_weight", type=float, default=1.0,
                        help="Weight for outside-box background constraint on weak samples.")
    parser.add_argument("--weak_loss_entropy_weight", type=float, default=0.05,
                        help="Weight for inside-box entropy minimization on weak samples.")
    parser.add_argument("--weak_loss_tv_weight", type=float, default=0.0,
                        help="Weight for inside-box total variation smoothness on weak samples.")
    
    # Quick test / Debug parameters
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: only process first 5 images for quick validation")
    parser.add_argument("--quick_test", type=int, default=None,
                        help="Quick test mode: only process first N images")
    
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: auto based on task, typically 1e-4 or 3e-4)")
    
    # Decoder Adapter parameters
    parser.add_argument("--decoder_adapter_enable", action="store_true", help="Enable MLP-Adapter modules in the mask decoder")
    parser.add_argument("--decoder_adapter_dim", type=int, default=64, help="Dimension of the decoder adapter bottleneck")
    args = parser.parse_args()

    dataset_name = args.task
    dataset_dir = os.path.join(f"datasets/{dataset_name}", dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare dataframes
    train_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"train.csv")), dataset_dir)
    val_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"val.csv")), dataset_dir)

    # get the validation metric
    validation_metric, loss, max_epoch, lr = get_default_training_setting(dataset_name)
    
    # Override learning rate if user specified it via CLI
    if args.lr is not None:
        lr = args.lr
        print(f"Using custom learning rate: {lr}")

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
            # GSPO reward / loss shaping
            "optim.gspo.lambda_smooth": args.lambda_smooth,
            "optim.gspo.lambda_boundary": args.lambda_boundary,
            "optim.gspo.w_boundary": args.w_boundary,
            "optim.gspo.w_smooth": args.w_smooth,
            "optim.gspo.w_thin": args.w_thin,
        })

    # Adapter configuration
    if args.adapter_enable:
        print(f"Enabling Standard Encoder Adapters with dim={args.adapter_dim}")
        hyperparameters.update({
            "model.sam.adapter_enabled": True,
            "model.sam.adapter_dim": args.adapter_dim,
        })
        
        # GSPO-Adapter extension configuration (Phase 1-2)
        if args.gspo_adapter_enable:
            if not args.gspo_enable:
                print("Warning: --gspo_adapter_enable requires --gspo_enable. Enabling GSPO automatically.")
                args.gspo_enable = True
            print(f"Enabling GSPO-Adapter extension with momentum={args.gspo_adapter_momentum}")
            hyperparameters.update({
                "optim.gspo.adapter_enabled": True,
                "optim.gspo.adapter_momentum": args.gspo_adapter_momentum,
            })
            # Pass scale adaptation to model config
            if args.gspo_adapter_scale_adaptation:
                print("Enabling GSPO-Adapter scale adaptation (Phase 2)")
                hyperparameters.update({
                    "optim.gspo.adapter_scale_adaptation": True,
                    "model.sam.adapter_gspo_enabled": True,
                    "model.sam.adapter_gspo_scale_adaptation": True,
                })
    
    # Decoder Attention LoRA configuration
    if args.decoder_attn_lora_enable:
        gspo_lora_info = ""
        if args.gspo_lora_attention_enable:
            gspo_lora_info = f", GSPO enabled (momentum={args.gspo_lora_attention_momentum}"
            if args.gspo_lora_attention_scale_adaptation:
                gspo_lora_info += ", scale_adaptation=True"
            gspo_lora_info += ")"
        print(f"Enabling Decoder Attention LoRA: r={args.decoder_attn_lora_r}, "
              f"alpha={args.decoder_attn_lora_alpha}, dropout={args.decoder_attn_lora_dropout}{gspo_lora_info}")
        hyperparameters.update({
            "model.sam.decoder_attention_lora_r": args.decoder_attn_lora_r,
            "model.sam.decoder_attention_lora_alpha": args.decoder_attn_lora_alpha,
            "model.sam.decoder_attention_lora_dropout": args.decoder_attn_lora_dropout,
        })
        
        # GSPO-LoRA on Attention configuration (NEW)
        if args.gspo_lora_attention_enable:
            hyperparameters.update({
                "model.sam.gspo_lora_attention_enabled": True,
                "model.sam.gspo_lora_attention_momentum": args.gspo_lora_attention_momentum,
                "model.sam.gspo_lora_attention_scale_adaptation": args.gspo_lora_attention_scale_adaptation,
            })
        
    # Decoder Adapter configuration
    if args.decoder_adapter_enable:
        print(f"Enabling Decoder MLP-Adapters with dim={args.decoder_adapter_dim}")
        hyperparameters.update({
            "model.sam.decoder_adapter_enabled": True,
            "model.sam.decoder_adapter_dim": args.decoder_adapter_dim,
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

        # Configure training-time box prompt mixing
        predictor._learner._train_box_prompt_cfg = {
            "mode": args.train_box_prompt_mode,
            "p_no": args.train_box_prob_no_prompt,
            "p_gt": args.train_box_prob_gt,
            "p_noisy": args.train_box_prob_noisy,
            "noise_frac": args.train_box_noise_frac,
            # Semi/weak supervision simulation
            "semi_labeled_fraction": args.semi_labeled_fraction,
            "semi_labeled_seed": args.semi_labeled_seed,
            "weak_box_jitter_mode": args.weak_box_jitter_mode,
            "weak_box_jitter_amount": args.weak_box_jitter_amount,
            "weak_box_outward_only": bool(args.weak_box_outward_only),
            "weak_loss_outside_weight": args.weak_loss_outside_weight,
            "weak_loss_entropy_weight": args.weak_loss_entropy_weight,
            "weak_loss_tv_weight": args.weak_loss_tv_weight,
        }

        # Setup bbox predictor for training if mode=predict
        if args.train_box_prompt_mode == "predict":
            if not args.train_bbox_model_ckpt:
                raise ValueError("train_box_prompt_mode=predict requires --train_bbox_model_ckpt.")
            train_bbox_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_bbox_predictor = BBoxPromptPredictor(
                ckpt_path=args.train_bbox_model_ckpt,
                device=train_bbox_device,
                image_size=args.train_bbox_model_image_size,
            )
            predictor._learner._train_bbox_predictor = train_bbox_predictor
            print(f"\n[Info] Training-time bbox predictor loaded from: {args.train_bbox_model_ckpt}")
            print(f"       Input size: {args.train_bbox_model_image_size}\n")

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
            box_prompt_mode=args.tta_box_prompt_mode,
        )

    # Optional bbox predictor for predicted box prompts
    bbox_predictor = None
    if args.bbox_prompt_source == "predict":
        if not args.bbox_model_ckpt and not args.bbox_preds_csv:
            raise ValueError("bbox_prompt_source=predict requires --bbox_model_ckpt or --bbox_preds_csv.")
        if args.bbox_model_ckpt:
            bbox_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            bbox_predictor = BBoxPromptPredictor(
                ckpt_path=args.bbox_model_ckpt,
                device=bbox_device,
                image_size=args.bbox_model_image_size,
            )
        if args.tta_box_prompt_mode == "off":
            print("[Warning] bbox_prompt_source=predict set but tta_box_prompt_mode=off; box prompts will be ignored.")

    def attach_pred_boxes(df: pd.DataFrame) -> pd.DataFrame:
        """Attach predicted or precomputed boxes to dataframe."""
        df = df.copy()
        if args.bbox_preds_csv:
            preds = pd.read_csv(args.bbox_preds_csv)
            merged = df.merge(preds, on="image", how="left")
            if merged[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].isna().all().all():
                preds["image_basename"] = preds["image"].apply(lambda p: os.path.basename(p))
                merged["image_basename"] = merged["image"].apply(lambda p: os.path.basename(p))
                merged = merged.merge(
                    preds.drop(columns=["image"]), on="image_basename", how="left", suffixes=("", "_pred")
                )
                merged.drop(columns=["image_basename"], inplace=True)
                for col in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
                    pred_col = f"{col}_pred"
                    if pred_col in merged.columns:
                        merged[col] = merged[col].fillna(merged[pred_col])
                        merged.drop(columns=[pred_col], inplace=True)
            df = merged
        elif bbox_predictor is not None:
            boxes = []
            for _, r in df.iterrows():
                box = bbox_predictor.predict(r["image"])
                boxes.append(box)
            boxes = np.stack(boxes, axis=0)
            df["bbox_x1"] = boxes[:, 0]
            df["bbox_y1"] = boxes[:, 1]
            df["bbox_x2"] = boxes[:, 2]
            df["bbox_y2"] = boxes[:, 3]
        return df

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

        # Attach predicted bbox prompts if requested
        if args.bbox_prompt_source == "predict" and (args.bbox_preds_csv or bbox_predictor):
            test_df = attach_pred_boxes(test_df)

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
            
            if args.bbox_prompt_source == "predict" and (args.bbox_preds_csv or bbox_predictor):
                test_df = attach_pred_boxes(test_df)

            res = predictor.evaluate(test_df, metrics=["sm", "fm", "em", "mae"])
            print(f"Evaluation results for test dataset {per_dataset}: ", res)
            f.write(f"Evaluation results for test dataset {per_dataset}: {res} \n")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}.")

    f.close()

```

---

**User**

继续你刚才中断的工作

---

**Cursor**

我继续把“数据集准备 + quick_test 验证”这段跑通：先检查现在是否已有下载在跑；然后把 `prepare_semantic_segmentation_datasets.py` 改成支持只下载 `isic2017`（避免一次拉全套大包），再后台下载并重新执行 quick_test，确认半/弱监督（10% GT + 90% 外扩抖动框）训练能正常起跑。

```python
import argparse
import os

from autogluon.common.loaders import load_zip


def get_data_home_dir():
    return os.path.join(os.getcwd().replace("\\", "/"), "datasets")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=[
            "polyp",
            "leaf_disease_segmentation",
            "camo_sem_seg",
            "isic2017",
            "road_segmentation",
            "SBU-shadow",
        ],
        help="Which semantic segmentation datasets to download/unzip.",
    )
    args = parser.parse_args()

    base_dir = get_data_home_dir()
    for name in args.datasets:
        url = f"s3://automl-mm-bench/semantic_segmentation/{name}.zip"

        dataset_dir = os.path.join(base_dir, name)
        load_zip.unzip(url, unzip_dir=dataset_dir)

```

好的。我先把正在后台跑的 **ISIC2017 超大包（12.2G）下载**处理掉（它太慢，不适合做这次 smoke test），改用**自动生成一个极小的“ISIC2017 兼容 toy 数据集”**来验证你要的“10% GT + 90% 外扩抖动框 + 弱监督损失”训练路径能跑通；同时我会顺手加一个 `--max_epochs` 方便你后续快速迭代调参。

---

**User**

那就开始吧

---

**Cursor**

```python
import argparse
import os

import numpy as np
import pandas as pd
import torch

from autogluon.multimodal import MultiModalPredictor
from bbox_prompt_model import BBoxPromptPredictor


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
    # GSPO reward / loss shaping hyper-parameters (可选，带默认值)
    parser.add_argument("--lambda_smooth", type=float, default=0.1, help="Smoothness loss weight (λ_smooth)")
    parser.add_argument("--lambda_boundary", type=float, default=0.3, help="Boundary loss weight (λ_boundary)")
    parser.add_argument("--w_boundary", type=float, default=0.3, help="Boundary reward weight (w_boundary)")
    parser.add_argument("--w_smooth", type=float, default=0.1, help="Smoothness reward weight (w_smooth)")
    parser.add_argument("--w_thin", type=float, default=0.05, help="Thin-structure reward weight (w_thin)")
    
    # Adapter parameters
    parser.add_argument("--adapter_enable", action="store_true", help="Enable standard Adapter modules")
    parser.add_argument("--adapter_dim", type=int, default=64, help="Dimension of the adapter bottleneck")
    
    # GSPO-Adapter extension parameters (Phase 1-2)
    parser.add_argument("--gspo_adapter_enable", action="store_true", 
                        help="Enable GSPO quality feedback for Encoder Adapters (Phase 1)")
    parser.add_argument("--gspo_adapter_momentum", type=float, default=0.9, 
                        help="Momentum for adapter quality history updates")
    parser.add_argument("--gspo_adapter_scale_adaptation", action="store_true",
                        help="Enable adaptive scale based on quality history (Phase 2)")
    
    # Decoder Attention LoRA parameters (LoRA on Attention)
    parser.add_argument("--decoder_attn_lora_enable", action="store_true", 
                        help="Enable LoRA on decoder attention Q/K/V projections")
    parser.add_argument("--decoder_attn_lora_r", type=int, default=8,
                        help="LoRA rank for decoder attention (default: 8)")
    parser.add_argument("--decoder_attn_lora_alpha", type=int, default=8,
                        help="LoRA alpha for decoder attention (default: 8, same as rank)")
    parser.add_argument("--decoder_attn_lora_dropout", type=float, default=0.0,
                        help="LoRA dropout for decoder attention (default: 0.0)")
    
    # GSPO-LoRA on Attention parameters (NEW)
    parser.add_argument("--gspo_lora_attention_enable", action="store_true",
                        help="Enable GSPO quality feedback for Decoder LoRA on Attention")
    parser.add_argument("--gspo_lora_attention_momentum", type=float, default=0.9,
                        help="Momentum for LoRA quality history updates (default: 0.9)")
    parser.add_argument("--gspo_lora_attention_scale_adaptation", action="store_true",
                        help="Enable adaptive scaling based on quality for LoRA")
    
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
    parser.add_argument("--tta_box_prompt_mode", type=str, default="off",
                        choices=["off", "add", "replace"],
                        help="Use GT box as SAM box prompt (off/add/replace); box not transformed with TTA")
    # BBox prompt configuration
    parser.add_argument("--bbox_prompt_source", type=str, default="none",
                        choices=["none", "gt", "predict"],
                        help="Box prompt source: none (no prompt), gt (from mask), predict (from bbox model).")
    parser.add_argument("--bbox_model_ckpt", type=str, default=None,
                        help="Checkpoint for bbox predictor when bbox_prompt_source=predict.")
    parser.add_argument("--bbox_model_image_size", type=int, default=320,
                        help="Input size for bbox predictor inference.")
    parser.add_argument("--bbox_preds_csv", type=str, default=None,
                        help="Optional CSV with columns image,bbox_x1,bbox_y1,bbox_x2,bbox_y2 to attach prompts.")
    # Training-time box prompt mixing (default off)
    parser.add_argument("--train_box_prompt_mode", type=str, default="off",
                        choices=["off", "gt", "noisy", "mix", "predict"],
                        help="Training-time box prompt injection: off/gt/noisy/mix/predict (default off). "
                             "predict: use bbox predictor model to generate box prompts from images.")
    parser.add_argument("--train_box_prob_no_prompt", type=float, default=0.2,
                        help="When mode=mix, probability of no prompt.")
    parser.add_argument("--train_box_prob_gt", type=float, default=0.3,
                        help="When mode=mix, probability of GT box prompt.")
    parser.add_argument("--train_box_prob_noisy", type=float, default=0.5,
                        help="When mode=mix, probability of noisy GT box prompt.")
    parser.add_argument("--train_box_noise_frac", type=float, default=0.12,
                        help="Noisy box jitter fraction relative to box size (e.g., 0.12 => ±12%).")
    parser.add_argument("--train_bbox_model_ckpt", type=str, default=None,
                        help="Checkpoint for bbox predictor when train_box_prompt_mode=predict.")
    parser.add_argument("--train_bbox_model_image_size", type=int, default=320,
                        help="Input size for bbox predictor during training (when train_box_prompt_mode=predict).")
    
    # Semi/weak supervision simulation (ISIC2017 has full masks; we can hide most masks and supervise with coarse boxes)
    parser.add_argument("--semi_labeled_fraction", type=float, default=1.0,
                        help="Fraction of samples to use GT mask loss (e.g., 0.1 => only 10% supervised, rest weak).")
    parser.add_argument("--semi_labeled_seed", type=int, default=0,
                        help="Seed for deterministic labeled/unlabeled assignment via hashing image paths.")
    parser.add_argument("--weak_box_jitter_mode", type=str, default="box",
                        choices=["box", "image", "pixel"],
                        help="How to define weak box jitter amount: "
                             "box=fraction of GT bbox size; image=fraction of full image size; pixel=absolute pixels.")
    parser.add_argument("--weak_box_jitter_amount", type=float, default=0.12,
                        help="Weak box jitter amount (meaning depends on --weak_box_jitter_mode). "
                             "Typical: 0.05~0.3 for box/image, or 5~30 for pixel.")
    parser.add_argument("--weak_box_outward_only", action="store_true",
                        help="If set, weak boxes are generated by outward expansion only (never shrink vs GT bbox).")
    parser.add_argument("--weak_loss_outside_weight", type=float, default=1.0,
                        help="Weight for outside-box background constraint on weak samples.")
    parser.add_argument("--weak_loss_entropy_weight", type=float, default=0.05,
                        help="Weight for inside-box entropy minimization on weak samples.")
    parser.add_argument("--weak_loss_tv_weight", type=float, default=0.0,
                        help="Weight for inside-box total variation smoothness on weak samples.")
    
    # Quick test / Debug parameters
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: only process first 5 images for quick validation")
    parser.add_argument("--quick_test", type=int, default=None,
                        help="Quick test mode: only process first N images")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override default max_epoch for quick smoke tests.")
    
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: auto based on task, typically 1e-4 or 3e-4)")
    
    # Decoder Adapter parameters
    parser.add_argument("--decoder_adapter_enable", action="store_true", help="Enable MLP-Adapter modules in the mask decoder")
    parser.add_argument("--decoder_adapter_dim", type=int, default=64, help="Dimension of the decoder adapter bottleneck")
    args = parser.parse_args()

    dataset_name = args.task
    dataset_dir = os.path.join(f"datasets/{dataset_name}", dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare dataframes
    train_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"train.csv")), dataset_dir)
    val_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"val.csv")), dataset_dir)

    # get the validation metric
    validation_metric, loss, max_epoch, lr = get_default_training_setting(dataset_name)
    if args.max_epochs is not None:
        max_epoch = args.max_epochs
    
    # Override learning rate if user specified it via CLI
    if args.lr is not None:
        lr = args.lr
        print(f"Using custom learning rate: {lr}")

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
            # GSPO reward / loss shaping
            "optim.gspo.lambda_smooth": args.lambda_smooth,
            "optim.gspo.lambda_boundary": args.lambda_boundary,
            "optim.gspo.w_boundary": args.w_boundary,
            "optim.gspo.w_smooth": args.w_smooth,
            "optim.gspo.w_thin": args.w_thin,
        })

    # Adapter configuration
    if args.adapter_enable:
        print(f"Enabling Standard Encoder Adapters with dim={args.adapter_dim}")
        hyperparameters.update({
            "model.sam.adapter_enabled": True,
            "model.sam.adapter_dim": args.adapter_dim,
        })
        
        # GSPO-Adapter extension configuration (Phase 1-2)
        if args.gspo_adapter_enable:
            if not args.gspo_enable:
                print("Warning: --gspo_adapter_enable requires --gspo_enable. Enabling GSPO automatically.")
                args.gspo_enable = True
            print(f"Enabling GSPO-Adapter extension with momentum={args.gspo_adapter_momentum}")
            hyperparameters.update({
                "optim.gspo.adapter_enabled": True,
                "optim.gspo.adapter_momentum": args.gspo_adapter_momentum,
            })
            # Pass scale adaptation to model config
            if args.gspo_adapter_scale_adaptation:
                print("Enabling GSPO-Adapter scale adaptation (Phase 2)")
                hyperparameters.update({
                    "optim.gspo.adapter_scale_adaptation": True,
                    "model.sam.adapter_gspo_enabled": True,
                    "model.sam.adapter_gspo_scale_adaptation": True,
                })
    
    # Decoder Attention LoRA configuration
    if args.decoder_attn_lora_enable:
        gspo_lora_info = ""
        if args.gspo_lora_attention_enable:
            gspo_lora_info = f", GSPO enabled (momentum={args.gspo_lora_attention_momentum}"
            if args.gspo_lora_attention_scale_adaptation:
                gspo_lora_info += ", scale_adaptation=True"
            gspo_lora_info += ")"
        print(f"Enabling Decoder Attention LoRA: r={args.decoder_attn_lora_r}, "
              f"alpha={args.decoder_attn_lora_alpha}, dropout={args.decoder_attn_lora_dropout}{gspo_lora_info}")
        hyperparameters.update({
            "model.sam.decoder_attention_lora_r": args.decoder_attn_lora_r,
            "model.sam.decoder_attention_lora_alpha": args.decoder_attn_lora_alpha,
            "model.sam.decoder_attention_lora_dropout": args.decoder_attn_lora_dropout,
        })
        
        # GSPO-LoRA on Attention configuration (NEW)
        if args.gspo_lora_attention_enable:
            hyperparameters.update({
                "model.sam.gspo_lora_attention_enabled": True,
                "model.sam.gspo_lora_attention_momentum": args.gspo_lora_attention_momentum,
                "model.sam.gspo_lora_attention_scale_adaptation": args.gspo_lora_attention_scale_adaptation,
            })
        
    # Decoder Adapter configuration
    if args.decoder_adapter_enable:
        print(f"Enabling Decoder MLP-Adapters with dim={args.decoder_adapter_dim}")
        hyperparameters.update({
            "model.sam.decoder_adapter_enabled": True,
            "model.sam.decoder_adapter_dim": args.decoder_adapter_dim,
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

        # Configure training-time box prompt mixing
        predictor._learner._train_box_prompt_cfg = {
            "mode": args.train_box_prompt_mode,
            "p_no": args.train_box_prob_no_prompt,
            "p_gt": args.train_box_prob_gt,
            "p_noisy": args.train_box_prob_noisy,
            "noise_frac": args.train_box_noise_frac,
            # Semi/weak supervision simulation
            "semi_labeled_fraction": args.semi_labeled_fraction,
            "semi_labeled_seed": args.semi_labeled_seed,
            "weak_box_jitter_mode": args.weak_box_jitter_mode,
            "weak_box_jitter_amount": args.weak_box_jitter_amount,
            "weak_box_outward_only": bool(args.weak_box_outward_only),
            "weak_loss_outside_weight": args.weak_loss_outside_weight,
            "weak_loss_entropy_weight": args.weak_loss_entropy_weight,
            "weak_loss_tv_weight": args.weak_loss_tv_weight,
        }

        # Setup bbox predictor for training if mode=predict
        if args.train_box_prompt_mode == "predict":
            if not args.train_bbox_model_ckpt:
                raise ValueError("train_box_prompt_mode=predict requires --train_bbox_model_ckpt.")
            train_bbox_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_bbox_predictor = BBoxPromptPredictor(
                ckpt_path=args.train_bbox_model_ckpt,
                device=train_bbox_device,
                image_size=args.train_bbox_model_image_size,
            )
            predictor._learner._train_bbox_predictor = train_bbox_predictor
            print(f"\n[Info] Training-time bbox predictor loaded from: {args.train_bbox_model_ckpt}")
            print(f"       Input size: {args.train_bbox_model_image_size}\n")

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
            box_prompt_mode=args.tta_box_prompt_mode,
        )

    # Optional bbox predictor for predicted box prompts
    bbox_predictor = None
    if args.bbox_prompt_source == "predict":
        if not args.bbox_model_ckpt and not args.bbox_preds_csv:
            raise ValueError("bbox_prompt_source=predict requires --bbox_model_ckpt or --bbox_preds_csv.")
        if args.bbox_model_ckpt:
            bbox_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            bbox_predictor = BBoxPromptPredictor(
                ckpt_path=args.bbox_model_ckpt,
                device=bbox_device,
                image_size=args.bbox_model_image_size,
            )
        if args.tta_box_prompt_mode == "off":
            print("[Warning] bbox_prompt_source=predict set but tta_box_prompt_mode=off; box prompts will be ignored.")

    def attach_pred_boxes(df: pd.DataFrame) -> pd.DataFrame:
        """Attach predicted or precomputed boxes to dataframe."""
        df = df.copy()
        if args.bbox_preds_csv:
            preds = pd.read_csv(args.bbox_preds_csv)
            merged = df.merge(preds, on="image", how="left")
            if merged[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].isna().all().all():
                preds["image_basename"] = preds["image"].apply(lambda p: os.path.basename(p))
                merged["image_basename"] = merged["image"].apply(lambda p: os.path.basename(p))
                merged = merged.merge(
                    preds.drop(columns=["image"]), on="image_basename", how="left", suffixes=("", "_pred")
                )
                merged.drop(columns=["image_basename"], inplace=True)
                for col in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
                    pred_col = f"{col}_pred"
                    if pred_col in merged.columns:
                        merged[col] = merged[col].fillna(merged[pred_col])
                        merged.drop(columns=[pred_col], inplace=True)
            df = merged
        elif bbox_predictor is not None:
            boxes = []
            for _, r in df.iterrows():
                box = bbox_predictor.predict(r["image"])
                boxes.append(box)
            boxes = np.stack(boxes, axis=0)
            df["bbox_x1"] = boxes[:, 0]
            df["bbox_y1"] = boxes[:, 1]
            df["bbox_x2"] = boxes[:, 2]
            df["bbox_y2"] = boxes[:, 3]
        return df

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

        # Attach predicted bbox prompts if requested
        if args.bbox_prompt_source == "predict" and (args.bbox_preds_csv or bbox_predictor):
            test_df = attach_pred_boxes(test_df)

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
            
            if args.bbox_prompt_source == "predict" and (args.bbox_preds_csv or bbox_predictor):
                test_df = attach_pred_boxes(test_df)

            res = predictor.evaluate(test_df, metrics=["sm", "fm", "em", "mae"])
            print(f"Evaluation results for test dataset {per_dataset}: ", res)
            f.write(f"Evaluation results for test dataset {per_dataset}: {res} \n")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}.")

    f.close()

```

变更与落地点（已修改代码）：
- `lit_semantic_seg.py`：新增半/弱监督路径。按 `semi_labeled_fraction` 用图像路径哈希稳定采样有标样本；其余样本自动从 GT mask 生成最小框并“外扩偏置抖动”（支持 box/image/pixel 三种尺度定义，默认外扩不收缩）。弱监督损失包含：框外背景 BCE、框内熵最小化、可选 TV 平滑；与有标样本的结构损失混合。半监督时自动禁用 GSPO。训练时 box prompt 注入也会为未标注样本写入外扩框。
- `run_semantic_segmentation.py`：补充 CLI 参数（半/弱监督相关、外扩抖动控制、弱损失权重、`--max_epochs` 覆盖默认 epoch）。这些参数会传到 `_train_box_prompt_cfg` 供训练使用。
- `prepare_semantic_segmentation_datasets.py`：允许 `--datasets` 只下载指定数据集，避免一次性拉全量。
- 生成了极小玩具数据集 `datasets/isic2017/isic2017`（6 张，train 4/val 2，含 image/label 列的 CSV），可用于快速验证代码路径。

测试尝试：
- 在容器内用玩具集跑 quick test 两次（CPU/GPU 都尝试，LoRA/专家降小、batch=1、quick_test≤2、max_epochs=1），均被系统 `Killed`，推测是加载 SAM 权重时 OOM/受限。建议在有足够显存的 GPU 环境重试，或切换到较小的 SAM 变体（若已有可选项），或进一步减小输入尺寸/禁用不必要的模块。

如何自行验证（建议命令）：
1) 下载正式 ISIC2017（12G）时，可用单数据集模式：
```
python examples/automm/Conv-LoRA/prepare_semantic_segmentation_datasets.py --datasets isic2017
```
2) 运行半/弱监督（示例：10% 有标，外扩抖动 20% 以框尺寸计，弱损失权重默认）：
```
python examples/automm/Conv-LoRA/run_semantic_segmentation.py \
  --task isic2017 --num_gpus 1 --per_gpu_batch_size 1 --batch_size 1 \
  --adapter_enable --rank 3 --expert_num 4 \
  --semi_labeled_fraction 0.1 --semi_labeled_seed 123 \
  --weak_box_outward_only --weak_box_jitter_mode box --weak_box_jitter_amount 0.2 \
  --weak_loss_outside_weight 1.0 --weak_loss_entropy_weight 0.05 --weak_loss_tv_weight 0.0 \
  --train_box_prompt_mode off --max_epochs 1 --quick_test 4
```
（如仍遇 OOM，可进一步减小 `rank/expert_num`、降低输入分辨率，或使用更小的 SAM/backbone。）

当前状态：
- 代码与 CLI 已支持“10% 有标 + 外扩抖动框弱监督”路径。
- 玩具集已就绪，但在本机因资源限制未能跑通训练；需要更充足的 GPU/内存环境验证。

```plaintext
python examples/automm/Conv-LoRA/prepare_semantic_segmentation_datasets.py --datasets isic2017
```

```plaintext
python examples/automm/Conv-LoRA/run_semantic_segmentation.py \
  --task isic2017 --num_gpus 1 --per_gpu_batch_size 1 --batch_size 1 \
  --adapter_enable --rank 3 --expert_num 4 \
  --semi_labeled_fraction 0.1 --semi_labeled_seed 123 \
  --weak_box_outward_only --weak_box_jitter_mode box --weak_box_jitter_amount 0.2 \
  --weak_loss_outside_weight 1.0 --weak_loss_entropy_weight 0.05 --weak_loss_tv_weight 0.0 \
  --train_box_prompt_mode off --max_epochs 1 --quick_test 4
```

---

**User**

稍等，为什么“半监督时自动禁用 GSPO”? 不能同时采用吗？

---

**Cursor**

原因（当前代码行为）：
- 在 `lit_semantic_seg.py` 我为了快速打通“半监督（部分无 GT）”路径，直接在 `training_step` 里当 `semi_labeled_fraction < 1.0` 时强制 `use_gspo=False`。动机是：GSPO 目前的优势函数/奖励依赖像素级 GT 质量（IoU/Dice）；无标签样本没有可靠质量信号，担心混入噪声。

是否可以同时用 GSPO？可以，但要给无标签样本定义可用的“质量/奖励”：
- 简单策略：只对“有 GT 的子批次”启用 GSPO（把 unlabeled 剔出 GSPO 组），依旧用 IoU 做优势；未标注样本走弱监督损失但不参与 GSPO 质量统计。
- 更积极策略：为无标签样本定义代理质量，用于 GSPO 组内优势：
  - 一致性：同图像多次 box 抖动/多视图预测的互 IoU（稳定性越高质量越高）。
  - 教师一致：学生与 EMA 教师预测 IoU。
  - 置信度/熵：预测熵低、分数高。
  - Box 覆盖度：预测 mask 与弱框的 IoU/覆盖率（惩罚外泄）。

如果要改回“半监督也用 GSPO”的最小改动思路：
1) 在 `training_step` 不再全局禁用 GSPO；改为在 `_gspo_training_step` 中仅对“有 GT 的样本”构造 GSPO 组，unlabeled 跳过 GSPO 或用代理质量。
2) 在 `gspo_trainer.gspo_group_training_step` 中，当无标签时改用代理质量函数（如上）计算 advantage。
3) 可增加 CLI 开关：`--gspo_allow_semisup`（默认关），`--gspo_quality_unlabeled` 选择代理质量类型（consistency/teacher/entropy/box_iou）。

如果你希望我提供具体代码补丁，请切到 agent 模式，我会给出最小修改版本（带 IoU 质量仅用于有 GT，未标注样本用 box-一致性代理质量）。

---

**User**

先简要解释一下，你现在方案里半监督和弱监督具体的含义和区别是什么？

---

**Cursor**

### 你当前方案里，“半监督”和“弱监督”各指什么？

在我实现的这套“ISIC2017 造半/弱监督”的方案里，是**同一个训练 DataFrame（都有 full mask）**里人为“遮掉一部分强标签”，从而在训练时把样本分成两类：

- **半监督（semi-supervised）**：指**训练集同时包含**
  - **有强标签**的样本：仍然用 **GT full mask** 算标准分割损失（结构损失/IoU 等），这部分比例由 `semi_labeled_fraction` 控制（比如 0.1 = 10%）。
  - **无强标签**的样本：不再使用 full mask 做监督，只用弱信号（这里是“模拟人工框”）来约束模型。
  
  换句话说：**半监督是“数据层面的混合”**——强标签 + 弱/无标签一起训练。

- **弱监督（weakly-supervised）**：指对那部分“无强标签”的样本，训练时**仍然有监督信号，但更弱**——这里用的是**box 监督**：
  - box 的来源：从 GT mask 的最小外接框出发，再做**偏外扩**的随机抖动（模拟医生画粗框）。
  - 损失形式：只用“框外是背景 + 框内鼓励更确定/更平滑”等 **不依赖 full mask** 的弱监督损失。

  换句话说：**弱监督是“标签类型/约束形式”**——用框/点/涂鸦等弱标注，而不是像素级 mask。

### 两者的区别一句话总结

- **半监督**：强调**训练数据里强标签样本占比**（你用 `semi_labeled_fraction` 控制）。
- **弱监督**：强调**无强标签样本到底用什么弱标签来监督**（你这里是“外扩抖动框 + 弱监督损失”）。

在你的设定里：**训练总体是半监督（混合强+弱），而弱监督是半监督中那部分“无强标签样本”的监督方式（box-supervised）**。

---

**User**

那请设计和实现“半监督也用 GSPO”的方案

---

**Cursor**

```python
import argparse
import os

import numpy as np
import pandas as pd
import torch

from autogluon.multimodal import MultiModalPredictor
from bbox_prompt_model import BBoxPromptPredictor


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
    # GSPO reward / loss shaping hyper-parameters (可选，带默认值)
    parser.add_argument("--lambda_smooth", type=float, default=0.1, help="Smoothness loss weight (λ_smooth)")
    parser.add_argument("--lambda_boundary", type=float, default=0.3, help="Boundary loss weight (λ_boundary)")
    parser.add_argument("--w_boundary", type=float, default=0.3, help="Boundary reward weight (w_boundary)")
    parser.add_argument("--w_smooth", type=float, default=0.1, help="Smoothness reward weight (w_smooth)")
    parser.add_argument("--w_thin", type=float, default=0.05, help="Thin-structure reward weight (w_thin)")
    
    # Adapter parameters
    parser.add_argument("--adapter_enable", action="store_true", help="Enable standard Adapter modules")
    parser.add_argument("--adapter_dim", type=int, default=64, help="Dimension of the adapter bottleneck")
    
    # GSPO-Adapter extension parameters (Phase 1-2)
    parser.add_argument("--gspo_adapter_enable", action="store_true", 
                        help="Enable GSPO quality feedback for Encoder Adapters (Phase 1)")
    parser.add_argument("--gspo_adapter_momentum", type=float, default=0.9, 
                        help="Momentum for adapter quality history updates")
    parser.add_argument("--gspo_adapter_scale_adaptation", action="store_true",
                        help="Enable adaptive scale based on quality history (Phase 2)")
    
    # Decoder Attention LoRA parameters (LoRA on Attention)
    parser.add_argument("--decoder_attn_lora_enable", action="store_true", 
                        help="Enable LoRA on decoder attention Q/K/V projections")
    parser.add_argument("--decoder_attn_lora_r", type=int, default=8,
                        help="LoRA rank for decoder attention (default: 8)")
    parser.add_argument("--decoder_attn_lora_alpha", type=int, default=8,
                        help="LoRA alpha for decoder attention (default: 8, same as rank)")
    parser.add_argument("--decoder_attn_lora_dropout", type=float, default=0.0,
                        help="LoRA dropout for decoder attention (default: 0.0)")
    
    # GSPO-LoRA on Attention parameters (NEW)
    parser.add_argument("--gspo_lora_attention_enable", action="store_true",
                        help="Enable GSPO quality feedback for Decoder LoRA on Attention")
    parser.add_argument("--gspo_lora_attention_momentum", type=float, default=0.9,
                        help="Momentum for LoRA quality history updates (default: 0.9)")
    parser.add_argument("--gspo_lora_attention_scale_adaptation", action="store_true",
                        help="Enable adaptive scaling based on quality for LoRA")
    parser.add_argument("--gspo_allow_semisup", action="store_true",
                        help="Allow GSPO on labeled subset when semi_labeled_fraction<1.0 (unlabeled uses weak loss).")
    
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
    parser.add_argument("--tta_box_prompt_mode", type=str, default="off",
                        choices=["off", "add", "replace"],
                        help="Use GT box as SAM box prompt (off/add/replace); box not transformed with TTA")
    # BBox prompt configuration
    parser.add_argument("--bbox_prompt_source", type=str, default="none",
                        choices=["none", "gt", "predict"],
                        help="Box prompt source: none (no prompt), gt (from mask), predict (from bbox model).")
    parser.add_argument("--bbox_model_ckpt", type=str, default=None,
                        help="Checkpoint for bbox predictor when bbox_prompt_source=predict.")
    parser.add_argument("--bbox_model_image_size", type=int, default=320,
                        help="Input size for bbox predictor inference.")
    parser.add_argument("--bbox_preds_csv", type=str, default=None,
                        help="Optional CSV with columns image,bbox_x1,bbox_y1,bbox_x2,bbox_y2 to attach prompts.")
    # Training-time box prompt mixing (default off)
    parser.add_argument("--train_box_prompt_mode", type=str, default="off",
                        choices=["off", "gt", "noisy", "mix", "predict"],
                        help="Training-time box prompt injection: off/gt/noisy/mix/predict (default off). "
                             "predict: use bbox predictor model to generate box prompts from images.")
    parser.add_argument("--train_box_prob_no_prompt", type=float, default=0.2,
                        help="When mode=mix, probability of no prompt.")
    parser.add_argument("--train_box_prob_gt", type=float, default=0.3,
                        help="When mode=mix, probability of GT box prompt.")
    parser.add_argument("--train_box_prob_noisy", type=float, default=0.5,
                        help="When mode=mix, probability of noisy GT box prompt.")
    parser.add_argument("--train_box_noise_frac", type=float, default=0.12,
                        help="Noisy box jitter fraction relative to box size (e.g., 0.12 => ±12%).")
    parser.add_argument("--train_bbox_model_ckpt", type=str, default=None,
                        help="Checkpoint for bbox predictor when train_box_prompt_mode=predict.")
    parser.add_argument("--train_bbox_model_image_size", type=int, default=320,
                        help="Input size for bbox predictor during training (when train_box_prompt_mode=predict).")
    
    # Semi/weak supervision simulation (ISIC2017 has full masks; we can hide most masks and supervise with coarse boxes)
    parser.add_argument("--semi_labeled_fraction", type=float, default=1.0,
                        help="Fraction of samples to use GT mask loss (e.g., 0.1 => only 10% supervised, rest weak).")
    parser.add_argument("--semi_labeled_seed", type=int, default=0,
                        help="Seed for deterministic labeled/unlabeled assignment via hashing image paths.")
    parser.add_argument("--weak_box_jitter_mode", type=str, default="box",
                        choices=["box", "image", "pixel"],
                        help="How to define weak box jitter amount: "
                             "box=fraction of GT bbox size; image=fraction of full image size; pixel=absolute pixels.")
    parser.add_argument("--weak_box_jitter_amount", type=float, default=0.12,
                        help="Weak box jitter amount (meaning depends on --weak_box_jitter_mode). "
                             "Typical: 0.05~0.3 for box/image, or 5~30 for pixel.")
    parser.add_argument("--weak_box_outward_only", action="store_true",
                        help="If set, weak boxes are generated by outward expansion only (never shrink vs GT bbox).")
    parser.add_argument("--weak_loss_outside_weight", type=float, default=1.0,
                        help="Weight for outside-box background constraint on weak samples.")
    parser.add_argument("--weak_loss_entropy_weight", type=float, default=0.05,
                        help="Weight for inside-box entropy minimization on weak samples.")
    parser.add_argument("--weak_loss_tv_weight", type=float, default=0.0,
                        help="Weight for inside-box total variation smoothness on weak samples.")
    
    # Quick test / Debug parameters
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: only process first 5 images for quick validation")
    parser.add_argument("--quick_test", type=int, default=None,
                        help="Quick test mode: only process first N images")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override default max_epoch for quick smoke tests.")
    
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: auto based on task, typically 1e-4 or 3e-4)")
    
    # Decoder Adapter parameters
    parser.add_argument("--decoder_adapter_enable", action="store_true", help="Enable MLP-Adapter modules in the mask decoder")
    parser.add_argument("--decoder_adapter_dim", type=int, default=64, help="Dimension of the decoder adapter bottleneck")
    args = parser.parse_args()

    dataset_name = args.task
    dataset_dir = os.path.join(f"datasets/{dataset_name}", dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare dataframes
    train_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"train.csv")), dataset_dir)
    val_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"val.csv")), dataset_dir)

    # get the validation metric
    validation_metric, loss, max_epoch, lr = get_default_training_setting(dataset_name)
    if args.max_epochs is not None:
        max_epoch = args.max_epochs
    
    # Override learning rate if user specified it via CLI
    if args.lr is not None:
        lr = args.lr
        print(f"Using custom learning rate: {lr}")

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
            # GSPO reward / loss shaping
            "optim.gspo.lambda_smooth": args.lambda_smooth,
            "optim.gspo.lambda_boundary": args.lambda_boundary,
            "optim.gspo.w_boundary": args.w_boundary,
            "optim.gspo.w_smooth": args.w_smooth,
            "optim.gspo.w_thin": args.w_thin,
        })

    # Adapter configuration
    if args.adapter_enable:
        print(f"Enabling Standard Encoder Adapters with dim={args.adapter_dim}")
        hyperparameters.update({
            "model.sam.adapter_enabled": True,
            "model.sam.adapter_dim": args.adapter_dim,
        })
        
        # GSPO-Adapter extension configuration (Phase 1-2)
        if args.gspo_adapter_enable:
            if not args.gspo_enable:
                print("Warning: --gspo_adapter_enable requires --gspo_enable. Enabling GSPO automatically.")
                args.gspo_enable = True
            print(f"Enabling GSPO-Adapter extension with momentum={args.gspo_adapter_momentum}")
            hyperparameters.update({
                "optim.gspo.adapter_enabled": True,
                "optim.gspo.adapter_momentum": args.gspo_adapter_momentum,
            })
            # Pass scale adaptation to model config
            if args.gspo_adapter_scale_adaptation:
                print("Enabling GSPO-Adapter scale adaptation (Phase 2)")
                hyperparameters.update({
                    "optim.gspo.adapter_scale_adaptation": True,
                    "model.sam.adapter_gspo_enabled": True,
                    "model.sam.adapter_gspo_scale_adaptation": True,
                })
    
    # Decoder Attention LoRA configuration
    if args.decoder_attn_lora_enable:
        gspo_lora_info = ""
        if args.gspo_lora_attention_enable:
            gspo_lora_info = f", GSPO enabled (momentum={args.gspo_lora_attention_momentum}"
            if args.gspo_lora_attention_scale_adaptation:
                gspo_lora_info += ", scale_adaptation=True"
            gspo_lora_info += ")"
        print(f"Enabling Decoder Attention LoRA: r={args.decoder_attn_lora_r}, "
              f"alpha={args.decoder_attn_lora_alpha}, dropout={args.decoder_attn_lora_dropout}{gspo_lora_info}")
        hyperparameters.update({
            "model.sam.decoder_attention_lora_r": args.decoder_attn_lora_r,
            "model.sam.decoder_attention_lora_alpha": args.decoder_attn_lora_alpha,
            "model.sam.decoder_attention_lora_dropout": args.decoder_attn_lora_dropout,
        })
        
        # GSPO-LoRA on Attention configuration (NEW)
        if args.gspo_lora_attention_enable:
            hyperparameters.update({
                "model.sam.gspo_lora_attention_enabled": True,
                "model.sam.gspo_lora_attention_momentum": args.gspo_lora_attention_momentum,
                "model.sam.gspo_lora_attention_scale_adaptation": args.gspo_lora_attention_scale_adaptation,
            })
        
    # Decoder Adapter configuration
    if args.decoder_adapter_enable:
        print(f"Enabling Decoder MLP-Adapters with dim={args.decoder_adapter_dim}")
        hyperparameters.update({
            "model.sam.decoder_adapter_enabled": True,
            "model.sam.decoder_adapter_dim": args.decoder_adapter_dim,
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

        # Configure training-time box prompt mixing
        predictor._learner._train_box_prompt_cfg = {
            "mode": args.train_box_prompt_mode,
            "p_no": args.train_box_prob_no_prompt,
            "p_gt": args.train_box_prob_gt,
            "p_noisy": args.train_box_prob_noisy,
            "noise_frac": args.train_box_noise_frac,
            # Semi/weak supervision simulation
            "semi_labeled_fraction": args.semi_labeled_fraction,
            "semi_labeled_seed": args.semi_labeled_seed,
            "weak_box_jitter_mode": args.weak_box_jitter_mode,
            "weak_box_jitter_amount": args.weak_box_jitter_amount,
            "weak_box_outward_only": bool(args.weak_box_outward_only),
            "weak_loss_outside_weight": args.weak_loss_outside_weight,
            "weak_loss_entropy_weight": args.weak_loss_entropy_weight,
            "weak_loss_tv_weight": args.weak_loss_tv_weight,
        }

        # Setup bbox predictor for training if mode=predict
        if args.train_box_prompt_mode == "predict":
            if not args.train_bbox_model_ckpt:
                raise ValueError("train_box_prompt_mode=predict requires --train_bbox_model_ckpt.")
            train_bbox_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_bbox_predictor = BBoxPromptPredictor(
                ckpt_path=args.train_bbox_model_ckpt,
                device=train_bbox_device,
                image_size=args.train_bbox_model_image_size,
            )
            predictor._learner._train_bbox_predictor = train_bbox_predictor
            print(f"\n[Info] Training-time bbox predictor loaded from: {args.train_bbox_model_ckpt}")
            print(f"       Input size: {args.train_bbox_model_image_size}\n")

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
            box_prompt_mode=args.tta_box_prompt_mode,
        )

    # Optional bbox predictor for predicted box prompts
    bbox_predictor = None
    if args.bbox_prompt_source == "predict":
        if not args.bbox_model_ckpt and not args.bbox_preds_csv:
            raise ValueError("bbox_prompt_source=predict requires --bbox_model_ckpt or --bbox_preds_csv.")
        if args.bbox_model_ckpt:
            bbox_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            bbox_predictor = BBoxPromptPredictor(
                ckpt_path=args.bbox_model_ckpt,
                device=bbox_device,
                image_size=args.bbox_model_image_size,
            )
        if args.tta_box_prompt_mode == "off":
            print("[Warning] bbox_prompt_source=predict set but tta_box_prompt_mode=off; box prompts will be ignored.")

    def attach_pred_boxes(df: pd.DataFrame) -> pd.DataFrame:
        """Attach predicted or precomputed boxes to dataframe."""
        df = df.copy()
        if args.bbox_preds_csv:
            preds = pd.read_csv(args.bbox_preds_csv)
            merged = df.merge(preds, on="image", how="left")
            if merged[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].isna().all().all():
                preds["image_basename"] = preds["image"].apply(lambda p: os.path.basename(p))
                merged["image_basename"] = merged["image"].apply(lambda p: os.path.basename(p))
                merged = merged.merge(
                    preds.drop(columns=["image"]), on="image_basename", how="left", suffixes=("", "_pred")
                )
                merged.drop(columns=["image_basename"], inplace=True)
                for col in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
                    pred_col = f"{col}_pred"
                    if pred_col in merged.columns:
                        merged[col] = merged[col].fillna(merged[pred_col])
                        merged.drop(columns=[pred_col], inplace=True)
            df = merged
        elif bbox_predictor is not None:
            boxes = []
            for _, r in df.iterrows():
                box = bbox_predictor.predict(r["image"])
                boxes.append(box)
            boxes = np.stack(boxes, axis=0)
            df["bbox_x1"] = boxes[:, 0]
            df["bbox_y1"] = boxes[:, 1]
            df["bbox_x2"] = boxes[:, 2]
            df["bbox_y2"] = boxes[:, 3]
        return df

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

        # Attach predicted bbox prompts if requested
        if args.bbox_prompt_source == "predict" and (args.bbox_preds_csv or bbox_predictor):
            test_df = attach_pred_boxes(test_df)

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
            
            if args.bbox_prompt_source == "predict" and (args.bbox_preds_csv or bbox_predictor):
                test_df = attach_pred_boxes(test_df)

            res = predictor.evaluate(test_df, metrics=["sm", "fm", "em", "mae"])
            print(f"Evaluation results for test dataset {per_dataset}: ", res)
            f.write(f"Evaluation results for test dataset {per_dataset}: {res} \n")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}.")

    f.close()

```

```python
import argparse
import os

import numpy as np
import pandas as pd
import torch

from autogluon.multimodal import MultiModalPredictor
from bbox_prompt_model import BBoxPromptPredictor


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
    # GSPO reward / loss shaping hyper-parameters (可选，带默认值)
    parser.add_argument("--lambda_smooth", type=float, default=0.1, help="Smoothness loss weight (λ_smooth)")
    parser.add_argument("--lambda_boundary", type=float, default=0.3, help="Boundary loss weight (λ_boundary)")
    parser.add_argument("--w_boundary", type=float, default=0.3, help="Boundary reward weight (w_boundary)")
    parser.add_argument("--w_smooth", type=float, default=0.1, help="Smoothness reward weight (w_smooth)")
    parser.add_argument("--w_thin", type=float, default=0.05, help="Thin-structure reward weight (w_thin)")
    
    # Adapter parameters
    parser.add_argument("--adapter_enable", action="store_true", help="Enable standard Adapter modules")
    parser.add_argument("--adapter_dim", type=int, default=64, help="Dimension of the adapter bottleneck")
    
    # GSPO-Adapter extension parameters (Phase 1-2)
    parser.add_argument("--gspo_adapter_enable", action="store_true", 
                        help="Enable GSPO quality feedback for Encoder Adapters (Phase 1)")
    parser.add_argument("--gspo_adapter_momentum", type=float, default=0.9, 
                        help="Momentum for adapter quality history updates")
    parser.add_argument("--gspo_adapter_scale_adaptation", action="store_true",
                        help="Enable adaptive scale based on quality history (Phase 2)")
    
    # Decoder Attention LoRA parameters (LoRA on Attention)
    parser.add_argument("--decoder_attn_lora_enable", action="store_true", 
                        help="Enable LoRA on decoder attention Q/K/V projections")
    parser.add_argument("--decoder_attn_lora_r", type=int, default=8,
                        help="LoRA rank for decoder attention (default: 8)")
    parser.add_argument("--decoder_attn_lora_alpha", type=int, default=8,
                        help="LoRA alpha for decoder attention (default: 8, same as rank)")
    parser.add_argument("--decoder_attn_lora_dropout", type=float, default=0.0,
                        help="LoRA dropout for decoder attention (default: 0.0)")
    
    # GSPO-LoRA on Attention parameters (NEW)
    parser.add_argument("--gspo_lora_attention_enable", action="store_true",
                        help="Enable GSPO quality feedback for Decoder LoRA on Attention")
    parser.add_argument("--gspo_lora_attention_momentum", type=float, default=0.9,
                        help="Momentum for LoRA quality history updates (default: 0.9)")
    parser.add_argument("--gspo_lora_attention_scale_adaptation", action="store_true",
                        help="Enable adaptive scaling based on quality for LoRA")
    parser.add_argument("--gspo_allow_semisup", action="store_true",
                        help="Allow GSPO on labeled subset when semi_labeled_fraction<1.0 (unlabeled uses weak loss).")
    
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
    parser.add_argument("--tta_box_prompt_mode", type=str, default="off",
                        choices=["off", "add", "replace"],
                        help="Use GT box as SAM box prompt (off/add/replace); box not transformed with TTA")
    # BBox prompt configuration
    parser.add_argument("--bbox_prompt_source", type=str, default="none",
                        choices=["none", "gt", "predict"],
                        help="Box prompt source: none (no prompt), gt (from mask), predict (from bbox model).")
    parser.add_argument("--bbox_model_ckpt", type=str, default=None,
                        help="Checkpoint for bbox predictor when bbox_prompt_source=predict.")
    parser.add_argument("--bbox_model_image_size", type=int, default=320,
                        help="Input size for bbox predictor inference.")
    parser.add_argument("--bbox_preds_csv", type=str, default=None,
                        help="Optional CSV with columns image,bbox_x1,bbox_y1,bbox_x2,bbox_y2 to attach prompts.")
    # Training-time box prompt mixing (default off)
    parser.add_argument("--train_box_prompt_mode", type=str, default="off",
                        choices=["off", "gt", "noisy", "mix", "predict"],
                        help="Training-time box prompt injection: off/gt/noisy/mix/predict (default off). "
                             "predict: use bbox predictor model to generate box prompts from images.")
    parser.add_argument("--train_box_prob_no_prompt", type=float, default=0.2,
                        help="When mode=mix, probability of no prompt.")
    parser.add_argument("--train_box_prob_gt", type=float, default=0.3,
                        help="When mode=mix, probability of GT box prompt.")
    parser.add_argument("--train_box_prob_noisy", type=float, default=0.5,
                        help="When mode=mix, probability of noisy GT box prompt.")
    parser.add_argument("--train_box_noise_frac", type=float, default=0.12,
                        help="Noisy box jitter fraction relative to box size (e.g., 0.12 => ±12%).")
    parser.add_argument("--train_bbox_model_ckpt", type=str, default=None,
                        help="Checkpoint for bbox predictor when train_box_prompt_mode=predict.")
    parser.add_argument("--train_bbox_model_image_size", type=int, default=320,
                        help="Input size for bbox predictor during training (when train_box_prompt_mode=predict).")
    
    # Semi/weak supervision simulation (ISIC2017 has full masks; we can hide most masks and supervise with coarse boxes)
    parser.add_argument("--semi_labeled_fraction", type=float, default=1.0,
                        help="Fraction of samples to use GT mask loss (e.g., 0.1 => only 10% supervised, rest weak).")
    parser.add_argument("--semi_labeled_seed", type=int, default=0,
                        help="Seed for deterministic labeled/unlabeled assignment via hashing image paths.")
    parser.add_argument("--weak_box_jitter_mode", type=str, default="box",
                        choices=["box", "image", "pixel"],
                        help="How to define weak box jitter amount: "
                             "box=fraction of GT bbox size; image=fraction of full image size; pixel=absolute pixels.")
    parser.add_argument("--weak_box_jitter_amount", type=float, default=0.12,
                        help="Weak box jitter amount (meaning depends on --weak_box_jitter_mode). "
                             "Typical: 0.05~0.3 for box/image, or 5~30 for pixel.")
    parser.add_argument("--weak_box_outward_only", action="store_true",
                        help="If set, weak boxes are generated by outward expansion only (never shrink vs GT bbox).")
    parser.add_argument("--weak_loss_outside_weight", type=float, default=1.0,
                        help="Weight for outside-box background constraint on weak samples.")
    parser.add_argument("--weak_loss_entropy_weight", type=float, default=0.05,
                        help="Weight for inside-box entropy minimization on weak samples.")
    parser.add_argument("--weak_loss_tv_weight", type=float, default=0.0,
                        help="Weight for inside-box total variation smoothness on weak samples.")
    
    # Quick test / Debug parameters
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: only process first 5 images for quick validation")
    parser.add_argument("--quick_test", type=int, default=None,
                        help="Quick test mode: only process first N images")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override default max_epoch for quick smoke tests.")
    
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: auto based on task, typically 1e-4 or 3e-4)")
    
    # Decoder Adapter parameters
    parser.add_argument("--decoder_adapter_enable", action="store_true", help="Enable MLP-Adapter modules in the mask decoder")
    parser.add_argument("--decoder_adapter_dim", type=int, default=64, help="Dimension of the decoder adapter bottleneck")
    args = parser.parse_args()

    dataset_name = args.task
    dataset_dir = os.path.join(f"datasets/{dataset_name}", dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare dataframes
    train_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"train.csv")), dataset_dir)
    val_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"val.csv")), dataset_dir)

    # get the validation metric
    validation_metric, loss, max_epoch, lr = get_default_training_setting(dataset_name)
    if args.max_epochs is not None:
        max_epoch = args.max_epochs
    
    # Override learning rate if user specified it via CLI
    if args.lr is not None:
        lr = args.lr
        print(f"Using custom learning rate: {lr}")

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
            # GSPO reward / loss shaping
            "optim.gspo.lambda_smooth": args.lambda_smooth,
            "optim.gspo.lambda_boundary": args.lambda_boundary,
            "optim.gspo.w_boundary": args.w_boundary,
            "optim.gspo.w_smooth": args.w_smooth,
            "optim.gspo.w_thin": args.w_thin,
            "optim.gspo.allow_semisup": bool(args.gspo_allow_semisup),
        })

    # Adapter configuration
    if args.adapter_enable:
        print(f"Enabling Standard Encoder Adapters with dim={args.adapter_dim}")
        hyperparameters.update({
            "model.sam.adapter_enabled": True,
            "model.sam.adapter_dim": args.adapter_dim,
        })
        
        # GSPO-Adapter extension configuration (Phase 1-2)
        if args.gspo_adapter_enable:
            if not args.gspo_enable:
                print("Warning: --gspo_adapter_enable requires --gspo_enable. Enabling GSPO automatically.")
                args.gspo_enable = True
            print(f"Enabling GSPO-Adapter extension with momentum={args.gspo_adapter_momentum}")
            hyperparameters.update({
                "optim.gspo.adapter_enabled": True,
                "optim.gspo.adapter_momentum": args.gspo_adapter_momentum,
            })
            # Pass scale adaptation to model config
            if args.gspo_adapter_scale_adaptation:
                print("Enabling GSPO-Adapter scale adaptation (Phase 2)")
                hyperparameters.update({
                    "optim.gspo.adapter_scale_adaptation": True,
                    "model.sam.adapter_gspo_enabled": True,
                    "model.sam.adapter_gspo_scale_adaptation": True,
                })
    
    # Decoder Attention LoRA configuration
    if args.decoder_attn_lora_enable:
        gspo_lora_info = ""
        if args.gspo_lora_attention_enable:
            gspo_lora_info = f", GSPO enabled (momentum={args.gspo_lora_attention_momentum}"
            if args.gspo_lora_attention_scale_adaptation:
                gspo_lora_info += ", scale_adaptation=True"
            gspo_lora_info += ")"
        print(f"Enabling Decoder Attention LoRA: r={args.decoder_attn_lora_r}, "
              f"alpha={args.decoder_attn_lora_alpha}, dropout={args.decoder_attn_lora_dropout}{gspo_lora_info}")
        hyperparameters.update({
            "model.sam.decoder_attention_lora_r": args.decoder_attn_lora_r,
            "model.sam.decoder_attention_lora_alpha": args.decoder_attn_lora_alpha,
            "model.sam.decoder_attention_lora_dropout": args.decoder_attn_lora_dropout,
        })
        
        # GSPO-LoRA on Attention configuration (NEW)
        if args.gspo_lora_attention_enable:
            hyperparameters.update({
                "model.sam.gspo_lora_attention_enabled": True,
                "model.sam.gspo_lora_attention_momentum": args.gspo_lora_attention_momentum,
                "model.sam.gspo_lora_attention_scale_adaptation": args.gspo_lora_attention_scale_adaptation,
            })
        
    # Decoder Adapter configuration
    if args.decoder_adapter_enable:
        print(f"Enabling Decoder MLP-Adapters with dim={args.decoder_adapter_dim}")
        hyperparameters.update({
            "model.sam.decoder_adapter_enabled": True,
            "model.sam.decoder_adapter_dim": args.decoder_adapter_dim,
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

        # Configure training-time box prompt mixing
        predictor._learner._train_box_prompt_cfg = {
            "mode": args.train_box_prompt_mode,
            "p_no": args.train_box_prob_no_prompt,
            "p_gt": args.train_box_prob_gt,
            "p_noisy": args.train_box_prob_noisy,
            "noise_frac": args.train_box_noise_frac,
            # Semi/weak supervision simulation
            "semi_labeled_fraction": args.semi_labeled_fraction,
            "semi_labeled_seed": args.semi_labeled_seed,
            "weak_box_jitter_mode": args.weak_box_jitter_mode,
            "weak_box_jitter_amount": args.weak_box_jitter_amount,
            "weak_box_outward_only": bool(args.weak_box_outward_only),
            "weak_loss_outside_weight": args.weak_loss_outside_weight,
            "weak_loss_entropy_weight": args.weak_loss_entropy_weight,
            "weak_loss_tv_weight": args.weak_loss_tv_weight,
        }

        # Setup bbox predictor for training if mode=predict
        if args.train_box_prompt_mode == "predict":
            if not args.train_bbox_model_ckpt:
                raise ValueError("train_box_prompt_mode=predict requires --train_bbox_model_ckpt.")
            train_bbox_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_bbox_predictor = BBoxPromptPredictor(
                ckpt_path=args.train_bbox_model_ckpt,
                device=train_bbox_device,
                image_size=args.train_bbox_model_image_size,
            )
            predictor._learner._train_bbox_predictor = train_bbox_predictor
            print(f"\n[Info] Training-time bbox predictor loaded from: {args.train_bbox_model_ckpt}")
            print(f"       Input size: {args.train_bbox_model_image_size}\n")

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
            box_prompt_mode=args.tta_box_prompt_mode,
        )

    # Optional bbox predictor for predicted box prompts
    bbox_predictor = None
    if args.bbox_prompt_source == "predict":
        if not args.bbox_model_ckpt and not args.bbox_preds_csv:
            raise ValueError("bbox_prompt_source=predict requires --bbox_model_ckpt or --bbox_preds_csv.")
        if args.bbox_model_ckpt:
            bbox_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            bbox_predictor = BBoxPromptPredictor(
                ckpt_path=args.bbox_model_ckpt,
                device=bbox_device,
                image_size=args.bbox_model_image_size,
            )
        if args.tta_box_prompt_mode == "off":
            print("[Warning] bbox_prompt_source=predict set but tta_box_prompt_mode=off; box prompts will be ignored.")

    def attach_pred_boxes(df: pd.DataFrame) -> pd.DataFrame:
        """Attach predicted or precomputed boxes to dataframe."""
        df = df.copy()
        if args.bbox_preds_csv:
            preds = pd.read_csv(args.bbox_preds_csv)
            merged = df.merge(preds, on="image", how="left")
            if merged[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].isna().all().all():
                preds["image_basename"] = preds["image"].apply(lambda p: os.path.basename(p))
                merged["image_basename"] = merged["image"].apply(lambda p: os.path.basename(p))
                merged = merged.merge(
                    preds.drop(columns=["image"]), on="image_basename", how="left", suffixes=("", "_pred")
                )
                merged.drop(columns=["image_basename"], inplace=True)
                for col in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
                    pred_col = f"{col}_pred"
                    if pred_col in merged.columns:
                        merged[col] = merged[col].fillna(merged[pred_col])
                        merged.drop(columns=[pred_col], inplace=True)
            df = merged
        elif bbox_predictor is not None:
            boxes = []
            for _, r in df.iterrows():
                box = bbox_predictor.predict(r["image"])
                boxes.append(box)
            boxes = np.stack(boxes, axis=0)
            df["bbox_x1"] = boxes[:, 0]
            df["bbox_y1"] = boxes[:, 1]
            df["bbox_x2"] = boxes[:, 2]
            df["bbox_y2"] = boxes[:, 3]
        return df

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

        # Attach predicted bbox prompts if requested
        if args.bbox_prompt_source == "predict" and (args.bbox_preds_csv or bbox_predictor):
            test_df = attach_pred_boxes(test_df)

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
            
            if args.bbox_prompt_source == "predict" and (args.bbox_preds_csv or bbox_predictor):
                test_df = attach_pred_boxes(test_df)

            res = predictor.evaluate(test_df, metrics=["sm", "fm", "em", "mae"])
            print(f"Evaluation results for test dataset {per_dataset}: ", res)
            f.write(f"Evaluation results for test dataset {per_dataset}: {res} \n")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}.")

    f.close()

```

```python
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
            # Only supported for binary segmentation with StructureLoss-like supervision.
            if isinstance(self.loss_func, Mask2FormerLoss):
                raise NotImplementedError("semi_labeled_fraction < 1.0 not supported for Mask2FormerLoss yet.")
            # pick logits for current model prefix (binary logits)
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
            
            # weak loss on unlabeled samples (requires boxes)
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
            
            # Keep any MoE loss from output dict
            moe_loss = output[self.model.prefix].get(MOE_LOSS, 0.0)
            loss = sup_loss + weak_loss + moe_loss
            # lightweight logging
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
        # Note: semi/weak supervision mixes labeled+unlabeled samples and uses custom loss;
        # current GSPO path assumes full GT masks for the whole batch.
        cfg = getattr(self, "train_box_prompt_cfg", {}) or {}
        semi_labeled_fraction = float(cfg.get("semi_labeled_fraction", 1.0))
        use_gspo = (
            self.gspo_trainer is not None and 
            self.gspo_trainer.is_gspo_active(self.current_epoch)
        )
        if semi_labeled_fraction < 1.0:
            use_gspo = False
        
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

```

```python
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
            # Only supported for binary segmentation with StructureLoss-like supervision.
            if isinstance(self.loss_func, Mask2FormerLoss):
                raise NotImplementedError("semi_labeled_fraction < 1.0 not supported for Mask2FormerLoss yet.")
            # pick logits for current model prefix (binary logits)
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
            
            # weak loss on unlabeled samples (requires boxes)
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
            
            # Keep any MoE loss from output dict
            moe_loss = output[self.model.prefix].get(MOE_LOSS, 0.0)
            loss = sup_loss + weak_loss + moe_loss
            # lightweight logging
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
        # Note: semi/weak supervision mixes labeled+unlabeled samples and uses custom loss;
        # current GSPO path assumes full GT masks for the whole batch.
        cfg = getattr(self, "train_box_prompt_cfg", {}) or {}
        semi_labeled_fraction = float(cfg.get("semi_labeled_fraction", 1.0))
        use_gspo = (
            self.gspo_trainer is not None and 
            self.gspo_trainer.is_gspo_active(self.current_epoch)
        )
        if semi_labeled_fraction < 1.0:
            use_gspo = False
        
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

```

```python
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
            # Only supported for binary segmentation with StructureLoss-like supervision.
            if isinstance(self.loss_func, Mask2FormerLoss):
                raise NotImplementedError("semi_labeled_fraction < 1.0 not supported for Mask2FormerLoss yet.")
            # pick logits for current model prefix (binary logits)
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
            
            # weak loss on unlabeled samples (requires boxes)
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
            
            # Keep any MoE loss from output dict
            moe_loss = output[self.model.prefix].get(MOE_LOSS, 0.0)
            loss = sup_loss + weak_loss + moe_loss
            # lightweight logging
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

```

```python
"""
GSPO (Group Sequence Policy Optimization) Trainer for Conv-LoRA, Adapters, and Decoder LoRA

This module implements the GSPO training strategy that enhances Conv-LoRA's
MoE mechanism through group-level optimization and quality-aware feedback.

Extended to support:
1. Encoder Adapters (Phase 1-3):
   - Phase 1: Unified advantage weighting (adapters receive same quality-weighted gradients)
   - Phase 2: Adapter scale adaptation (contribution score based on quality history)
   - Phase 3: Adapter MoE architecture (optional, multiple adapter variants)

2. Decoder LoRA on Attention (NEW):
   - Phase 1: Quality history tracking for LoRA layers
   - Phase 2: Dynamic scaling adaptation based on quality feedback

Key components:
1. Group-level sampling: Generate multiple predictions per image
2. Advantage function: Compute relative quality within groups
3. Contrastive loss: Pull high-quality predictions together, push low-quality apart
4. Quality feedback: Update expert selection, adapter quality, AND LoRA quality based on actual performance
"""

import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np


class GSPOConvLoRATrainer:
    """
    GSPO-enhanced trainer for Conv-LoRA semantic segmentation.
    
    This trainer implements group-level policy optimization inspired by GSPO,
    adapted for the Conv-LoRA MoE architecture in semantic segmentation tasks.
    
    Args:
        predictor: The MultiModalPredictor instance
        group_size: Number of predictions to generate per image in a group
        warmup_epochs: Number of epochs to train without GSPO before enabling it
        contrastive_weight: Weight for the contrastive loss component
        quality_metric: Metric to use for quality evaluation ('iou', 'dice', 'both')
        advantage_temperature: Temperature for computing advantage weights
    """
    
    def __init__(
        self,
        predictor,
        group_size: int = 4,
        warmup_epochs: int = 5,
        contrastive_weight: float = 0.1,
        quality_metric: str = 'iou',
        advantage_temperature: float = 5.0,
        lambda_smooth: float = 0.1,
        lambda_boundary: float = 0.3,
        w_boundary: float = 0.3,
        w_smooth: float = 0.1,
        w_thin: float = 0.05,
        # GSPO-Adapter extension parameters
        gspo_adapter_enabled: bool = False,
        gspo_adapter_momentum: float = 0.9,
        # GSPO-LoRA on Attention extension parameters (NEW)
        gspo_lora_attention_enabled: bool = False,
        gspo_lora_attention_momentum: float = 0.9,
        allow_semisup: bool = False,
    ):
        self.predictor = predictor
        self.group_size = group_size
        self.warmup_epochs = warmup_epochs
        self.contrastive_weight = contrastive_weight
        self.quality_metric = quality_metric
        self.advantage_temperature = advantage_temperature
        # Reward / loss shaping hyper-parameters
        self.lambda_smooth = lambda_smooth
        self.lambda_boundary = lambda_boundary
        self.w_boundary = w_boundary
        self.w_smooth = w_smooth
        self.w_thin = w_thin
        
        # GSPO-Adapter extension (Phase 1-2)
        self.gspo_adapter_enabled = gspo_adapter_enabled
        self.gspo_adapter_momentum = gspo_adapter_momentum
        
        # GSPO-LoRA on Attention extension (NEW)
        self.gspo_lora_attention_enabled = gspo_lora_attention_enabled
        self.gspo_lora_attention_momentum = gspo_lora_attention_momentum
        self.allow_semisup = allow_semisup
        
        # Statistics tracking
        self.expert_performance_log = defaultdict(list)
        self.group_quality_history = []
        self.adapter_quality_log = []  # Track adapter quality over time
        self.lora_quality_log = []  # Track LoRA quality over time (NEW)
        self.current_epoch = 0
        
    def is_gspo_active(self, epoch: int) -> bool:
        """Check if GSPO should be active for the given epoch."""
        return epoch >= self.warmup_epochs
    
    def compute_segmentation_quality(
        self,
        pred_masks: torch.Tensor,
        gt_masks: torch.Tensor,
        metric: str = 'iou'
    ) -> torch.Tensor:
        """
        Compute segmentation quality metrics.
        
        Args:
            pred_masks: Predicted masks [B, C, H, W] or [B, H, W]
            gt_masks: Ground truth masks [B, C, H, W] or [B, H, W]
            metric: 'iou', 'dice', or 'both'
            
        Returns:
            quality_scores: Tensor of shape [B] containing quality scores
        """
        # Ensure masks are in the right format
        if pred_masks.dim() == 4 and pred_masks.shape[1] > 1:
            pred_masks = torch.argmax(pred_masks, dim=1)
        if gt_masks.dim() == 4 and gt_masks.shape[1] > 1:
            gt_masks = torch.argmax(gt_masks, dim=1)
        
        # Flatten spatial dimensions
        pred_flat = pred_masks.view(pred_masks.shape[0], -1).float()
        gt_flat = gt_masks.view(gt_masks.shape[0], -1).float()
        
        # Binarize if needed
        pred_flat = (pred_flat > 0.5).float()
        gt_flat = (gt_flat > 0.5).float()
        
        # Compute intersection and union
        intersection = (pred_flat * gt_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1) - intersection
        
        # IoU
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        if metric == 'iou':
            return iou
        
        # DICE
        dice = (2 * intersection + 1e-6) / (pred_flat.sum(dim=1) + gt_flat.sum(dim=1) + 1e-6)
        
        if metric == 'dice':
            return dice
        
        # Both: average of IoU and DICE
        return (iou + dice) / 2.0

    def _prepare_binary_masks(
        self, pred_masks: torch.Tensor, gt_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert logits/probabilities to single-channel probability maps for
        boundary/smoothness computation.
        """
        # Ensure channel dimension
        if pred_masks.dim() == 3:
            pred_masks = pred_masks.unsqueeze(1)
        if gt_masks.dim() == 3:
            gt_masks = gt_masks.unsqueeze(1)

        if pred_masks.shape[1] > 1:
            # Use the maximum class probability as the foreground probability
            pred_probs = torch.softmax(pred_masks, dim=1).max(dim=1, keepdim=True).values
        else:
            pred_probs = torch.sigmoid(pred_masks)

        if gt_masks.shape[1] > 1:
            gt_probs = torch.argmax(gt_masks, dim=1, keepdim=True).float()
        else:
            gt_probs = gt_masks.float()

        return pred_probs, gt_probs

    def _boundary_map(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Approximate boundary map via spatial gradients.
        """
        grad_x = torch.abs(masks[..., :, 1:] - masks[..., :, :-1])
        grad_y = torch.abs(masks[..., 1:, :] - masks[..., :-1, :])
        grad_x = torch.nn.functional.pad(grad_x, (0, 1, 0, 0))
        grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1))
        return torch.clamp(grad_x + grad_y, 0.0, 1.0)

    def compute_smoothness_loss(self, pred_probs: torch.Tensor) -> torch.Tensor:
        """
        Laplacian smoothness loss (per-sample).
        """
        lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=pred_probs.device, dtype=pred_probs.dtype)
        lap_kernel = lap_kernel.view(1, 1, 3, 3)
        lap = torch.nn.functional.conv2d(pred_probs, lap_kernel, padding=1)
        return (lap ** 2).mean(dim=(1, 2, 3))

    def compute_boundary_loss(self, pred_probs: torch.Tensor, gt_probs: torch.Tensor) -> torch.Tensor:
        """
        L1 difference between predicted and GT boundary maps (per-sample).
        """
        pred_edge = self._boundary_map(pred_probs)
        gt_edge = self._boundary_map(gt_probs)
        return torch.mean(torch.abs(pred_edge - gt_edge), dim=(1, 2, 3))

    def compute_thin_reward(self, pred_probs: torch.Tensor, gt_probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Dice-style reward on boundary skeletons (per-sample).
        """
        pred_edge = self._boundary_map(pred_probs)
        gt_edge = self._boundary_map(gt_probs)
        intersection = (pred_edge * gt_edge).flatten(1).sum(dim=1)
        union = pred_edge.flatten(1).sum(dim=1) + gt_edge.flatten(1).sum(dim=1)
        return (2 * intersection + eps) / (union + eps)
    
    def gspo_group_training_step(
        self,
        images: torch.Tensor,
        masks_gt: torch.Tensor,
        forward_fn,
        loss_fn,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        GSPO core training step: generate group predictions and compute weighted loss.
        
        Args:
            images: Input images [B, C, H, W]
            masks_gt: Ground truth masks [B, C, H, W] or [B, H, W]
            forward_fn: Forward function that returns (predictions, moe_loss, selected_experts)
            loss_fn: Loss function for segmentation
            
        Returns:
            total_loss: Combined loss for backpropagation
            metrics: Dictionary of metrics for logging
        """
        B = images.shape[0]
        G = self.group_size
        
        group_predictions = []
        group_quality_scores = []
        group_base_quality_scores = []
        group_smooth_losses = []
        group_boundary_losses = []
        group_thin_rewards = []
        group_moe_losses = []
        group_selected_experts = []
        group_seg_losses = []
        
        # 1. Generate group predictions with variations
        for g in range(G):
            # Apply slight variations via dropout to create diversity
            # (dropout is automatically different across forward passes)
            with torch.set_grad_enabled(True):
                pred_masks, moe_loss, selected_experts = forward_fn(images)
            
            pred_probs, gt_probs = self._prepare_binary_masks(pred_masks, masks_gt)
            smooth_loss = self.compute_smoothness_loss(pred_probs)
            boundary_loss = self.compute_boundary_loss(pred_probs, gt_probs)
            thin_reward = self.compute_thin_reward(pred_probs, gt_probs)

            # Compute quality scores
            quality = self.compute_segmentation_quality(
                pred_masks, masks_gt, metric=self.quality_metric
            )
            shaped_quality = (
                quality
                + self.w_boundary * (1.0 - boundary_loss.detach())
                - self.w_smooth * smooth_loss.detach()
                + self.w_thin * thin_reward.detach()
            )
            
            # Compute segmentation loss
            seg_loss = loss_fn(pred_masks, masks_gt)
            seg_loss = (
                seg_loss
                + self.lambda_smooth * smooth_loss.mean()
                + self.lambda_boundary * boundary_loss.mean()
            )
            
            group_predictions.append(pred_masks)
            group_quality_scores.append(shaped_quality)
            group_base_quality_scores.append(quality)
            group_smooth_losses.append(smooth_loss.detach())
            group_boundary_losses.append(boundary_loss.detach())
            group_thin_rewards.append(thin_reward.detach())
            group_moe_losses.append(moe_loss)
            group_selected_experts.append(selected_experts)
            group_seg_losses.append(seg_loss)
        
        # 2. Compute group-level advantage function (GSPO core)
        quality_tensor = torch.stack(group_quality_scores)  # [G, B]
        baseline = quality_tensor.mean(dim=0, keepdim=True)  # [1, B]
        advantages = quality_tensor - baseline  # [G, B]
        
        # 3. GSPO weighted loss
        total_seg_loss = 0
        total_moe_loss = 0
        
        for g in range(G):
            advantage = advantages[g]  # [B]
            
            # Compute adaptive weights based on advantage
            # Higher advantage -> higher weight (stronger gradient)
            weight = torch.sigmoid(advantage * self.advantage_temperature)
            
            # Weighted segmentation loss
            seg_loss = group_seg_losses[g]
            if seg_loss.dim() == 0:  # scalar loss
                weighted_seg_loss = seg_loss * weight.mean()
            else:  # per-sample loss
                weighted_seg_loss = (seg_loss * weight).mean()
            
            total_seg_loss += weighted_seg_loss
            total_moe_loss += group_moe_losses[g]
        
        total_seg_loss /= G
        total_moe_loss /= G
        
        # 4. GSPO contrastive loss
        contrastive_loss = self.compute_group_contrastive_loss(
            group_predictions, group_quality_scores
        )
        
        # 5. Combine losses
        total_loss = total_seg_loss + 1e-2 * total_moe_loss + self.contrastive_weight * contrastive_loss
        
        # 6. Collect metrics
        metrics = {
            'loss': total_loss.item(),
            'seg_loss': total_seg_loss.item(),
            'moe_loss': total_moe_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'avg_quality': quality_tensor.mean().item(),
            'avg_quality_base': torch.stack(group_base_quality_scores).mean().item(),
            'avg_smooth_loss': torch.stack(group_smooth_losses).mean().item(),
            'avg_boundary_loss': torch.stack(group_boundary_losses).mean().item(),
            'avg_thin_reward': torch.stack(group_thin_rewards).mean().item(),
            'quality_variance': quality_tensor.var().item(),
            'max_advantage': advantages.max().item(),
            'min_advantage': advantages.min().item(),
        }
        
        # 7. Store group quality for analysis
        self.group_quality_history.append({
            'epoch': self.current_epoch,
            'qualities': quality_tensor.detach().cpu().numpy(),
            'advantages': advantages.detach().cpu().numpy(),
        })
        
        return total_loss, metrics, group_selected_experts
    
    def compute_group_contrastive_loss(
        self,
        predictions: List[torch.Tensor],
        quality_scores: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        GSPO-inspired contrastive loss.
        
        The idea is to make predictions with similar quality have similar features,
        while predictions with different quality should have different features.
        
        Args:
            predictions: List of G prediction tensors [B, C, H, W]
            quality_scores: List of G quality tensors [B]
            
        Returns:
            contrastive_loss: Scalar loss
        """
        G = len(predictions)
        
        if G < 2:
            return torch.tensor(0.0, device=predictions[0].device)
        
        quality_tensor = torch.stack(quality_scores)  # [G, B]
        
        # Extract features from predictions (global average pooling)
        features = []
        for pred in predictions:
            # Apply global average pooling
            feat = F.adaptive_avg_pool2d(pred, (1, 1)).flatten(1)  # [B, C]
            feat = F.normalize(feat, dim=1)  # Normalize for cosine similarity
            features.append(feat)
        features = torch.stack(features)  # [G, B, C]
        
        # Compute pairwise cosine similarities
        # Reshape for batch matrix multiplication
        feat_flat = features.view(G, -1)  # [G, B*C]
        similarities = torch.mm(feat_flat, feat_flat.t())  # [G, G]
        
        # Normalize by number of features
        similarities = similarities / (features.shape[1] * features.shape[2])
        
        # Compute quality differences
        quality_mean = quality_tensor.mean(dim=1)  # [G]
        quality_diff = torch.abs(
            quality_mean.unsqueeze(0) - quality_mean.unsqueeze(1)
        )  # [G, G]
        
        # Contrastive objective:
        # - If quality is similar (small diff), encourage high similarity
        # - If quality is different (large diff), encourage low similarity
        target_similarity = 1.0 - quality_diff / (quality_diff.max() + 1e-6)
        
        # MSE loss between actual and target similarities
        contrastive_loss = F.mse_loss(similarities, target_similarity)
        
        return contrastive_loss
    
    def update_expert_feedback(
        self,
        selected_experts_groups: List[Optional[torch.Tensor]],
        quality_scores: List[torch.Tensor],
        model
    ):
        """
        Update expert quality history in the model based on GSPO feedback.
        
        Args:
            selected_experts_groups: List of selected expert tensors from each group
            quality_scores: List of quality score tensors from each group
            model: The model containing MoEGate modules
        """
        if not selected_experts_groups or selected_experts_groups[0] is None:
            return
        
        # Collect all MoEGate modules from the model
        moe_gates = []
        for module in model.modules():
            if hasattr(module, 'update_quality_history'):
                moe_gates.append(module)
        
        if not moe_gates:
            return
        
        # Update each MoE gate with quality feedback
        for g_idx, (selected_experts, quality) in enumerate(zip(selected_experts_groups, quality_scores)):
            if selected_experts is not None:
                for moe_gate in moe_gates:
                    moe_gate.update_quality_history(selected_experts, quality)
    
    def update_adapter_feedback(
        self,
        quality_scores: List[torch.Tensor],
        model
    ):
        """
        GSPO-Adapter Extension: Update adapter quality history based on GSPO feedback.
        
        This method is called after each GSPO training step to provide quality feedback
        to all AdapterLayer modules in the model.
        
        Args:
            quality_scores: List of quality score tensors from each group [G] of [B]
            model: The model containing AdapterLayer modules
        """
        if not self.gspo_adapter_enabled:
            return
        
        # Compute average quality across groups
        if not quality_scores:
            return
            
        avg_quality = torch.stack([q.mean() for q in quality_scores]).mean()
        
        # Collect all AdapterLayer modules from the model
        adapters = []
        for name, module in model.named_modules():
            # Check if this is an AdapterLayer with GSPO enabled
            if hasattr(module, 'update_quality_feedback') and hasattr(module, 'gspo_enabled'):
                if module.gspo_enabled:
                    adapters.append((name, module))
        
        if not adapters:
            return
        
        # Update each adapter with quality feedback
        for name, adapter in adapters:
            adapter.update_quality_feedback(avg_quality)
        
        # Log adapter quality for analysis
        self.adapter_quality_log.append({
            'epoch': self.current_epoch,
            'avg_quality': avg_quality.item(),
            'num_adapters': len(adapters),
        })
    
    def get_adapter_stats(self, model) -> Dict:
        """
        Get GSPO statistics from all adapters in the model.
        
        Args:
            model: The model containing AdapterLayer modules
            
        Returns:
            Dictionary with adapter statistics
        """
        stats = {}
        for name, module in model.named_modules():
            if hasattr(module, 'get_gspo_stats'):
                adapter_stats = module.get_gspo_stats()
                if adapter_stats:
                    stats[name] = adapter_stats
        return stats
    
    def update_lora_attention_feedback(
        self,
        quality_scores: List[torch.Tensor],
        model
    ):
        """
        GSPO-LoRA Extension: Update LoRA layer quality history based on GSPO feedback.
        
        This method is called after each GSPO training step to provide quality feedback
        to all LoRALinear modules (in Decoder Attention) that have GSPO enabled.
        
        The feedback mechanism allows LoRA layers to dynamically adjust their contribution
        based on the quality of predictions, following the GSPO principle of
        quality-aware optimization.
        
        Args:
            quality_scores: List of quality score tensors from each group [G] of [B]
            model: The model containing LoRALinear modules (with gspo_enabled=True)
        """
        if not self.gspo_lora_attention_enabled:
            return
        
        # Compute average quality across groups
        if not quality_scores:
            return
            
        avg_quality = torch.stack([q.mean() for q in quality_scores]).mean()
        
        # Import LoRALinear to check module type
        # Note: We check for the method instead of type to avoid circular imports
        lora_layers = []
        for name, module in model.named_modules():
            # Check if this is a LoRALinear with GSPO enabled
            # LoRALinear has: update_quality_feedback, gspo_enabled, r > 0
            if (hasattr(module, 'update_quality_feedback') and 
                hasattr(module, 'gspo_enabled') and 
                hasattr(module, 'r') and
                module.gspo_enabled and
                module.r > 0):
                lora_layers.append((name, module))
        
        if not lora_layers:
            return
        
        # Update each LoRA layer with quality feedback
        for name, lora in lora_layers:
            lora.update_quality_feedback(avg_quality)
        
        # Log LoRA quality for analysis
        self.lora_quality_log.append({
            'epoch': self.current_epoch,
            'avg_quality': avg_quality.item(),
            'num_lora_layers': len(lora_layers),
        })
    
    def get_lora_stats(self, model) -> Dict:
        """
        Get GSPO statistics from all LoRA layers in the model.
        
        Args:
            model: The model containing LoRALinear modules
            
        Returns:
            Dictionary with LoRA statistics (quality_history, contribution_score, etc.)
        """
        stats = {}
        for name, module in model.named_modules():
            if (hasattr(module, 'get_gspo_stats') and 
                hasattr(module, 'r') and 
                hasattr(module, 'gspo_enabled')):
                if module.gspo_enabled and module.r > 0:
                    lora_stats = module.get_gspo_stats()
                    if lora_stats:
                        stats[name] = lora_stats
        return stats
    
    def get_statistics(self) -> Dict:
        """Get training statistics for analysis."""
        return {
            'expert_performance': dict(self.expert_performance_log),
            'group_quality_history': self.group_quality_history,
            'adapter_quality_log': self.adapter_quality_log,
            'lora_quality_log': self.lora_quality_log,  # NEW
        }
    
    def reset_statistics(self):
        """Reset accumulated statistics."""
        self.expert_performance_log.clear()
        self.group_quality_history.clear()
        self.adapter_quality_log.clear()
        self.lora_quality_log.clear()  # NEW


```

```python
*** Begin Patch
*** Update File: multimodal/src/autogluon/multimodal/learners/semantic_segmentation.py
@@
         gspo_adapter_enabled = getattr(sam_cfg, "adapter_gspo_enabled", False) if sam_cfg else False
         gspo_adapter_momentum = getattr(gspo_cfg, "adapter_momentum", 0.9) if gspo_cfg else 0.9
         gspo_adapter_scale_adaptation = getattr(gspo_cfg, "adapter_scale_adaptation", False) if gspo_cfg else False
+        allow_semisup = getattr(gspo_cfg, "allow_semisup", False) if gspo_cfg else False
@@
         GSPOConvLoRATrainer = getattr(module, "GSPOConvLoRATrainer", None)
@@
         trainer = GSPOConvLoRATrainer(
             predictor=self,
             group_size=getattr(lora_cfg, "gspo_group_size", 4),
             warmup_epochs=getattr(lora_cfg, "gspo_warmup_epochs", 5),
             contrastive_weight=getattr(lora_cfg, "gspo_contrastive_weight", 0.1),
             quality_metric=getattr(lora_cfg, "gspo_quality_metric", "iou"),
             advantage_temperature=getattr(lora_cfg, "gspo_advantage_temperature", 5.0),
             lambda_smooth=lambda_smooth,
             lambda_boundary=lambda_boundary,
             w_boundary=w_boundary,
             w_smooth=w_smooth,
             w_thin=w_thin,
             # Adapter extension
             gspo_adapter_enabled=gspo_adapter_enabled,
             gspo_adapter_momentum=gspo_adapter_momentum,
             # LoRA on attention extension
             gspo_lora_attention_enabled=gspo_lora_attention_enabled,
             gspo_lora_attention_momentum=gspo_lora_attention_momentum,
+            allow_semisup=allow_semisup,
         )
*** End Patch
```

```python
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
            # Semi/weak-supervision simulation (optional)
            # NOTE: We avoid adding extra columns to dataframe because semantic segmentation learner
            # expects exactly one image column (+ label column) when column_types is not provided.
            # Instead, we deterministically assign "labeled" samples via stable hashing of image_path.
            "semi_labeled_fraction": 1.0,       # 1.0 => fully-supervised; e.g. 0.1 => only 10% uses GT mask loss
            "semi_labeled_seed": 0,
            # Weak box generation from GT (simulating human coarse boxes)
            "weak_box_jitter_mode": "box",      # box / image / pixel
            "weak_box_jitter_amount": 0.12,     # meaning depends on mode
            "weak_box_outward_only": True,      # mostly expand outward; True => never shrink vs GT bbox
            # Weak loss weights (box-only supervision)
            "weak_loss_outside_weight": 1.0,    # enforce outside-box background
            "weak_loss_entropy_weight": 0.05,   # entropy minimization inside box
            "weak_loss_tv_weight": 0.0,         # optional smoothness (total variation) inside box
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

        # Get GSPO-Adapter parameters (if available)
        sam_cfg = getattr(self._config.model, "sam", None)
        gspo_adapter_enabled = getattr(sam_cfg, "adapter_gspo_enabled", False) if sam_cfg else False
        gspo_adapter_momentum = getattr(sam_cfg, "adapter_gspo_momentum", 0.9) if sam_cfg else 0.9
        
        # Get GSPO-LoRA on Attention parameters (if available) - NEW
        gspo_lora_attention_enabled = getattr(sam_cfg, "gspo_lora_attention_enabled", False) if sam_cfg else False
        gspo_lora_attention_momentum = getattr(sam_cfg, "gspo_lora_attention_momentum", 0.9) if sam_cfg else 0.9
        allow_semisup = getattr(gspo_cfg, "allow_semisup", False) if gspo_cfg else False

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
                # GSPO-Adapter extension
                gspo_adapter_enabled=gspo_adapter_enabled,
                gspo_adapter_momentum=gspo_adapter_momentum,
                # GSPO-LoRA on Attention extension (NEW)
                gspo_lora_attention_enabled=gspo_lora_attention_enabled,
                gspo_lora_attention_momentum=gspo_lora_attention_momentum,
                allow_semisup=allow_semisup,
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

```

已实现“半监督也能用 GSPO（仅作用于有 GT 子批次，弱标签样本走弱监督损失）”的代码改动：

- 新增开关：`--gspo_allow_semisup`（默认关）。开启后，当 `semi_labeled_fraction < 1` 时：
  - 有 GT 的子批次：仍走 GSPO 组级训练（优势函数基于真值质量）。
  - 无 GT 的子批次：使用外扩抖动框的弱监督损失（框外背景约束 + 框内熵最小化 + 可选 TV），不参与 GSPO。
- 训练时 box prompt 注入被抽成 `_apply_train_box_prompts`，GSPO/非 GSPO 路径共享，保证弱监督子批次有框。
- GSPO trainer 支持 `allow_semisup` 参数（通过 hyperparameters `optim.gspo.allow_semisup` 注入）。
- CLI 已更新（`run_semantic_segmentation.py`），并把配置透传给 trainer；`gspo_trainer.py`/`lit_semantic_seg.py`/`semantic_segmentation.py` 做了相应对接。

当前未实现：无标签样本的“代理质量”直接参与 GSPO（仍是只对有标签子批次算优势）；TODO `t3` 保留。

使用示例（半监督+GSPO on labeled）：
```
python examples/automm/Conv-LoRA/run_semantic_segmentation.py \
  --task isic2017 --num_gpus 1 --per_gpu_batch_size 1 --batch_size 1 \
  --adapter_enable --rank 3 --expert_num 4 \
  --semi_labeled_fraction 0.1 --semi_labeled_seed 123 \
  --weak_box_outward_only --weak_box_jitter_mode box --weak_box_jitter_amount 0.2 \
  --weak_loss_outside_weight 1.0 --weak_loss_entropy_weight 0.05 --weak_loss_tv_weight 0.0 \
  --gspo_enable --gspo_allow_semisup \
  --train_box_prompt_mode off --max_epochs 1 --quick_test 4
```
（如资源紧张可进一步减小 rank/专家数、输入分辨率，或使用较小 SAM/backbone。）

说明：
- GSPO 只对“有标签子批次”启用，避免无标签样本缺少可靠优势信号。
- 无标签样本仍获得弱监督梯度，且 MoE/门控参数也会被这些弱损失更新。  
- 未对无标签样本的代理质量（稳定性/教师一致性等）进入 GSPO 做优势，若需要可在后续实现。

```plaintext
python examples/automm/Conv-LoRA/run_semantic_segmentation.py \
  --task isic2017 --num_gpus 1 --per_gpu_batch_size 1 --batch_size 1 \
  --adapter_enable --rank 3 --expert_num 4 \
  --semi_labeled_fraction 0.1 --semi_labeled_seed 123 \
  --weak_box_outward_only --weak_box_jitter_mode box --weak_box_jitter_amount 0.2 \
  --weak_loss_outside_weight 1.0 --weak_loss_entropy_weight 0.05 --weak_loss_tv_weight 0.0 \
  --gspo_enable --gspo_allow_semisup \
  --train_box_prompt_mode off --max_epochs 1 --quick_test 4
```

---

**User**

执行`nohup python run_semantic_segmentation.py \
  --task isic2017 --num_gpus 1 --per_gpu_batch_size 1 --batch_size 1 \
  --adapter_enable --rank 3 --expert_num 4 \
  --semi_labeled_fraction 0.1 --semi_labeled_seed 123 \
  --weak_box_outward_only --weak_box_jitter_mode box --weak_box_jitter_amount 0.2 \
  --weak_loss_outside_weight 1.0 --weak_loss_entropy_weight 0.05 --weak_loss_tv_weight 0.0 \
  --gspo_enable --gspo_allow_semisup \
  --train_box_prompt_mode off --max_epochs 1 --quick_test 4 \
  --output_dir outputs/semisup_gspo_gpt-260101 \
> outputs/semisup_gspo_gpt-260101.log 2>&1 &`报错，看日志文件examples/automm/Conv-LoRA/outputs/semisup_gspo_gpt-260101.log的最后部分

---

**Cursor**

```yaml
optim:
  optim_type: "adamw"
  lr: 1.0e-4
  weight_decay: 0.001
  lr_choice: "layerwise_decay"
  lr_decay: 0.9
  lr_schedule: "cosine_decay"
  max_epochs: 20
  max_steps: -1
  warmup_steps: 0.1
  end_lr: 0
  lr_mult: 1 # multiply lr for downstream heads
  patience: 10
  val_check_interval: 0.5 # Check validation score a fraction of epoch if float and every n steps if integer with n < total step in epoch.
  check_val_every_n_epoch: 1 # Check validation score every n epochs.
  skip_final_val: False # Flag to skip the last validation
  gradient_clip_val: 1
  gradient_clip_algorithm: "norm"
  track_grad_norm: -1 # Whether to check gradient norm. We can set it to 2 to check for gradient norm.
  log_every_n_steps: 10
  label_smoothing: 0
  top_k: 3
  top_k_average_method:
    "greedy_soup" # We support averaging method described in https://arxiv.org/pdf/2203.05482.pdf.
    # Currently support "uniform_soup", "greedy_soup", and "best".
  peft: null # Can be 'bit_fit' (only finetune bias), 'norm_fit' (finetune the normalization terms + bias terms), lora (LoRA Adaptations only), lora_bias (LoRA Adaptation + bit_fit), lora_norm (LoRA Adaptation + norm_fit), or null
  lora:
    module_filter: null # Specify which module (if any) to adapt(e.g. ".*EncDecAttention|.*DenseReluDense"). Default consider all modules in a model (i.e. empty filter).
    filter: # Specify which layer in a module to adapt. Default fine-tune only query and value attention weights, recommended in https://arxiv.org/abs/2106.09685
      - "query"
      - "value"
      - "^q$" # The filter can also be a regex. We will use re.match(filter, name) to determine if LoRA should be applied.
      - "^v$"
      - "^k$"
      - "^o$"
    r: 8
    alpha: 8
    conv_lora_expert_num: 8 # default setting for Conv-LoRA
    gspo_enabled: False # Enable GSPO (Group Sequence Policy Optimization) enhancements
    gspo_group_size: 3 # Number of experts to select in GSPO mode (TopK)
    gspo_quality_momentum: 0.9 # Momentum coefficient for expert quality history updates
    gspo_warmup_epochs: 5 # Number of epochs before enabling full GSPO training
    gspo_contrastive_weight: 0.1 # Weight for contrastive loss in GSPO training
  loss_func:
    "auto" # The replaced loss for regression. Can only support loss function in torch.nn.
    # example
    # "BCEWithLogitsLoss" or "nn.BCEWithLogitsloss"
  focal_loss:
    alpha: null
    gamma: 2.0
    reduction: "mean"
  mask2former_loss:
    loss_cross_entropy_weight: 10.0
    loss_mask_weight: 5.0
    loss_dice_weight: 5.0
  extra_trainable_params: []
  cross_modal_align: null
  cross_modal_align_weight: 0
  automatic_optimization: True
  lemda:
    turn_on: False
    arch_type: "mlp_vae"
    z_dim: 8
    num_layers: 6
    kld_weight: 0.1
    mse_weight: 0.1
    adv_weight: 1.0e-4
    consist_weight: 0.01
    consist_threshold: 0.5
    lr: 1.0e-4
    optim_type: "adamw"
    weight_decay: 1.0e-5
  gspo:
    lambda_smooth: 0.1
    lambda_boundary: 0.3
    w_boundary: 0.3
    w_smooth: 0.1
    w_thin: 0.05
    allow_semisup: False  # Allow GSPO on labeled subset when semi-supervised
    # GSPO-Adapter extension parameters
    adapter_enabled: False  # Enable GSPO quality feedback for Encoder Adapters
    adapter_momentum: 0.9   # Momentum for adapter quality history updates
    adapter_scale_adaptation: False  # Enable adaptive scale based on quality (Phase 2)

```

问题原因：`optim.gspo.allow_semisup` 没在默认配置里注册，Omegaconf 校验报 KeyError。

已修复：
- 在 `multimodal/src/autogluon/multimodal/configs/optim/default.yaml` 增加 `allow_semisup: False`（GSPO 配置块）。
- 其它代码保持不变。

接下来请重新运行（示例，同你之前的命令即可）：
```
nohup python run_semantic_segmentation.py \
  --task isic2017 --num_gpus 1 --per_gpu_batch_size 1 --batch_size 1 \
  --adapter_enable --rank 3 --expert_num 4 \
  --semi_labeled_fraction 0.1 --semi_labeled_seed 123 \
  --weak_box_outward_only --weak_box_jitter_mode box --weak_box_jitter_amount 0.2 \
  --weak_loss_outside_weight 1.0 --weak_loss_entropy_weight 0.05 --weak_loss_tv_weight 0.0 \
  --gspo_enable --gspo_allow_semisup \
  --train_box_prompt_mode off --max_epochs 1 --quick_test 4 \
  --output_dir outputs/semisup_gspo_gpt-260101 \
> outputs/semisup_gspo_gpt-260101.log 2>&1 &
```

如再有报错，把新日志尾部发给我即可。

```plaintext
nohup python run_semantic_segmentation.py \
  --task isic2017 --num_gpus 1 --per_gpu_batch_size 1 --batch_size 1 \
  --adapter_enable --rank 3 --expert_num 4 \
  --semi_labeled_fraction 0.1 --semi_labeled_seed 123 \
  --weak_box_outward_only --weak_box_jitter_mode box --weak_box_jitter_amount 0.2 \
  --weak_loss_outside_weight 1.0 --weak_loss_entropy_weight 0.05 --weak_loss_tv_weight 0.0 \
  --gspo_enable --gspo_allow_semisup \
  --train_box_prompt_mode off --max_epochs 1 --quick_test 4 \
  --output_dir outputs/semisup_gspo_gpt-260101 \
> outputs/semisup_gspo_gpt-260101.log 2>&1 &
```

