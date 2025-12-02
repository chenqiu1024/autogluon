# 探讨网络结构中适配器的优化
_Exported on 12/1/2025 at 22:29:57 PST from Cursor (2.1.44)_

---

**User**

阅读docs/cursor_gspo_adapter_convlora-251129.md以了解上下文。接下来请继续思考：在本实验的网络结构中，还有哪里可以添加其它类型的Adapter(s)，进而有可能进一步提升性能？

---

**Cursor**

我先快速通读你提到的文档，把当前 Conv-LoRA / Adapter 的放置位置、实验设定和瓶颈整理出来，然后结合 SAM/ViT/Decoder 等模块的结构，系统性思考还能在哪些层或分支插入其它类型的 Adapter 以可能提升性能。接着我会按模块分别给出建议，并简单分析每种插点的潜在收益与代价。

### 先简单回顾一下你当前的结构

- **Conv-LoRA**：挂在 `SamVisionLayer` 的注意力模块上（`SamVisionAttention.qkv`），主要做**空间维度**的适配 + GSPO 负责 MoE 策略优化。  
- **标准 Adapter**：只插在 `SamVisionLayer` 的 **MLP 之后、与 MLP 并行** 的通道瓶颈（`AdapterLayer`）。  
- **未改动/未加 Adapter 的部分**：`SamPatchEmbeddings`、`SamVisionNeck`、`SamPromptEncoder`（点/框/掩码编码）、`SamMaskDecoder`（Two-Way Transformer）、上采样和分类 MLP 等。

在这个前提下，**还能加 Adapter 的关键位置**主要有三类：**(1) Mask Decoder, (2) Prompt Encoder, (3) Vision Neck & 输出头**。

---

### 1. 在 Mask Decoder（Two-Way Transformer）里加 “解码器 Adapter”

Mask Decoder 是直接负责从 `image_embeddings + prompts` 生成 mask 的“任务头”，目前完全是原版结构，没有 PEFT 模块。

- **插入位置建议**（类比你在 Vision Encoder 里的做法）：
  - 在 `SamTwoWayAttentionBlock` 里，对 **queries 的 MLP 之后** 加并行 Adapter：
    - 现在：`queries = queries + self.mlp(queries); queries = layer_norm3(queries)`
    - 可以改成：`queries = residual + mlp_out (+ adapter_out)`
  - 也可以在 **token->image cross-attn 或 image->token cross-attn 之后** 加一个小 Adapter，用来专门调整“解码时的注意力模式”。

- **潜在收益**：
  - 目前 Conv-LoRA+Adapter 只改造 **视觉编码特征**，Mask Decoder 仍是“通用 SAM 风格”的解码器。
  - 在 Decoder 里加 Adapter，可以让模型**直接在“预测空间”里学习医学特有的边界形状、数目（多病灶）、类别间混淆模式**，往往对 Dice/IoU 有额外提升，尤其是小目标和细边界。

---

### 2. 在 Prompt Encoder 里加 “提示 Adapter”（视你的任务而定）

在当前 semantic segmentation pipeline 里，你大概率是 **不依赖交互点/框提示**（多为全图分割，`no_mask_embed` 为主）。因此：

- **如果未来要用点/框/粗 mask 作为提示**：
  - 可以在 `SamPromptEncoder` 里，对以下几处加极小的 Adapter：
    - `SamMaskEmbedding` 的卷积后（`dense_embeddings`）加一个 Conv/MLP Adapter，用于适配“粗掩码 → 精细组合”的风格。
    - `_embed_points` / `_embed_boxes` 后，加一个瓶颈 Adapter，专门学习“医生点击/框选行为 ↔ 真正目标”的偏差。

- **什么时候值得做**：
  - 当你希望做 **交互式医学分割 / 少点提示分割** 时，这类 Adapter 能显著提升“点击 → 分割结果”的鲁棒性。
  - 若当前实验只做 **纯语义分割，无提示输入**，Prompt Encoder 几乎“不工作”，这里加 Adapter 贡献会很有限，可以先放在后面阶段再做。

---

### 3. 在 Vision Neck + 输出头部分加 “任务头 Adapter / Conv-Adapter”

当前 Vision Encoder 输出后，经过 `SamVisionNeck` 的两层卷积 + LayerNorm 再交给 Mask Decoder，这一段也是可以加 PEFT 的高价值位置：

- **Vision Neck (`SamVisionNeck`)**：
  - 在 `conv1` / `conv2` 之后加一个极小的 **Conv-Adapter**（例如 depthwise separable conv + 1x1 瓶颈），专门调节：
    - 医学图像特有的对比度、纹理、噪声模式；
    - 小结构（病灶、毛细结构）的增强。
  - 这类 Adapter 对**空间局部模式**很敏感，能和前面 Attention/Conv-LoRA 的全局建模形成互补。

- **多类输出头 (`output_classifier_mlps` 等)**：
  - 现在多类分类是一个线性层，可以：
    - 在 `class_predictions` 前加一个小的 bottleneck Adapter，
    - 或者做“类别组 Adapter”（例如器官类 vs 肿瘤类共用一个小 Adapter）。
  - 对极不均衡、细分多类的医学任务，有时能改善**长尾类**的表现。

---

### 4. “其它类型 Adapter” 的设计方向（在上述位置中任选几处尝试）

在以上这些新插点里，你可以尝试 **不同风格的 Adapter**，而不仅仅是线性瓶颈：

- **Conv-Adapter / Spatial-Adapter**：在 Mask Decoder 或 Vision Neck 中使用 3×3 depthwise conv + 1×1 瓶颈，专门强化局部形状和纹理。
- **MoE-Adapter + GSPO**：在 Decoder 的 MLP 后用一个小型 MoE-Adapter，并把 GSPO 的策略扩展到这些 Adapter 上，让策略不仅选 Conv-LoRA 专家，还选择“解码器 Adapter 专家”。
- **层选择 / 层稀疏 Adapter**：只在 Vision Encoder 或 Two-Way Transformer 的后几层加大一点的 Adapter（前面不加或加很小），在参数开销不涨太多的前提下提升表达力。

---

### 简短建议

- **优先级推荐**：
  1. **优先尝试：Mask Decoder 里的 MLP-Adapter**（对最终 mask 直接生效，最有希望带来稳定提升）。
  2. 其次尝试：**Vision Neck 的 Conv-Adapter**（小参数、实现简单）。
  3. 若未来做交互式/提示式分割，再考虑：**Prompt Encoder 的提示 Adapter**。

- 每加一种 Adapter，建议都单独做 **消融组合**（仅 Conv-LoRA+Encoder Adapter vs +Decoder Adapter vs +Neck Adapter），看哪一块带来的收益最大，再决定是否一起启用。

---

**User**

好的，请规划设计 “Mask Decoder 里加 MLP-Adapter“ 的方案

请回答为什么对 Decoder-Adapter采用 配置驱动的方式，而与Vision Encoder加Adapter的方式不一样？

---

**Cursor**

### 核心原因：历史包袱 + 解耦性 + 复用性

**1. Vision Encoder 的 Adapter 是“事后外挂”的历史产物**

- 最初我们只做 **Conv-LoRA**，完全复用 HuggingFace 的 `SamVisionEncoder`/`SamVisionLayer`，**配置里没有任何 Adapter 字段**。  
- 后来在 AutoGluon 这层才加上“标准 Adapter”，最简单、侵入性最小的做法就是在 `SAMForSemanticSegmentation.__init__` 里遍历：
  - `for layer in self.model.vision_encoder.layers: layer.adapter = AdapterLayer(...)`  
- 也就是说，Encoder-Adapter 更像是对 **已存在的 Encoder 实例做一次“猴子打补丁”**，不去大改原始 HF config/构造流程，这是一个**兼顾稳定性和工作量的折中实现**。

---

**2. Decoder-Adapter 是全新设计，可以一开始就“干净地做在 config 里”**

- Mask Decoder（`SamMaskDecoder` + `SamTwoWayTransformer` + `SamTwoWayAttentionBlock`）这套代码我们已经自己 fork 了一份（`modeling_sam_for_conv_lora.py`），并且：
  - 解码器的 block 本来就严格依赖 `SamMaskDecoderConfig` 来确定 hidden_size、层数等。
- 现在新增 Decoder-Adapter，是一个**从 0 开始设计的新能力**，没有历史兼容包袱，因此更合理的方式是：
  - 在 `SamMaskDecoderConfig` / `SamConfig` 里加 `decoder_adapter_enabled` / `decoder_adapter_dim`；
  - 构造 `SamTwoWayAttentionBlock` 时根据 config 自动创建 `adapter_queries`。
- 这样带来的好处：
  - Decoder 的行为完全由 **自身 config 决定**，更符合 HuggingFace 模型的一贯风格；
  - 以后如果别人直接用这份 `SamModel`（不走 AutoGluon 包装），只要改 config 就能开关 Decoder-Adapter，**复用性更好**；
  - 不需要在外层再写一段很深的遍历逻辑（`model.mask_decoder.transformer.layers[...]`）去“手动挂 adapter”，代码结构更清晰、维护成本更低。

---

**3. 总结一句**

- **Encoder-Adapter**：是为了尽量不动 HuggingFace 原始构造逻辑，在 AutoGluon 外层“外挂”上的方案，属于历史折中。  
- **Decoder-Adapter**：是我们现在新设计的功能，没有历史包袱，因此可以直接走 **config 驱动 + 构造时自动挂载** 的“更干净、更通用”的实现方式。  
- 长远来看，其实也可以考虑逐步把 Encoder-Adapter 也收敛到同样的 config 驱动风格，减少两种实现方式的不一致，这只是优先级问题。

---

**User**

Implement the plan as specified, it is attached for your reference. Do NOT edit the plan file itself.

To-do's from the plan have already been created. Do not create them again. Mark them as in_progress as you work, starting with the first one. Don't stop until you have completed all the to-dos.

---

**Cursor**

我会按你给的 plan，一步步在配置、训练脚本和模型代码里实现 Decoder-Adapter（MLP-Adapter），每完成一个阶段就更新 todo 状态，并在最后简单总结变更点和如何使用。

先在 YAML 的 `sam` 配置和训练脚本里加 decoder_adapter 配置/CLI 参数，接着会修改 `sam.py` 的加载逻辑和 `modeling_sam_for_conv_lora.py` 的 Two-Way Block。

```yaml
model:
  names:
  categorical_mlp:
    hidden_size: 64
    activation: "leaky_relu"
    num_layers: 1
    dropout: 0.1
    normalization: "layer_norm"
    data_types:
      - "categorical"

  numerical_mlp:
    hidden_size: 128
    activation: "leaky_relu"
    num_layers: 1
    dropout: 0.1
    normalization: "layer_norm"
    token_dim: 8
    embedding_arch:
    data_types:
      - "numerical"
    merge: "concat"

  hf_text:
    checkpoint_name: "google/electra-base-discriminator"
    gradient_checkpointing: False
    pooling_mode: 'cls'  # The pooling mode, can be 'cls' or 'mean'
    data_types:
      - "text"
    tokenizer_name: "hf_auto"
    use_fast: True  # Use a fast Rust-based tokenizer if it is supported for a given model. If a fast tokenizer is not available for a given model, a normal Python-based tokenizer is returned instead.
    max_text_len: 512  # If None or <=0, then use the max length of pretrained models.
    insert_sep: True
    low_cpu_mem_usage: False
    text_segment_num: 2
    stochastic_chunk: False
    text_aug_detect_length: 10                # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.1            # augmentation magnitude randomly drawn from [0, text_trivial_aug_maxscale]
    text_train_augment_types:        # specify augmentation space manually, will randomly select one from the following and identity
      # - "random_swap(0.05)"          # less than 0.1 based on eda paper
      # - "random_delete(0.05)"        # less than 0.1 based on eda paper
      # - "syn_replacement(0.05)"  # less than 0.1 based on eda paper
      # - "insert_punc(0.05)"

  ner_text:
    checkpoint_name: "bert-base-cased"
    max_text_len: 512
    gradient_checkpointing: False
    low_cpu_mem_usage: False
    data_types:
      - "text_ner"
    tokenizer_name: "hf_auto"
    insert_sep: False
    text_segment_num: 2
    stochastic_chunk: False
    special_tags: 
      - X # CLS, SEP, and non-first tokens of a word will be labelled as X
      - O # Outside of a named entity

  document_transformer:
    checkpoint_name: "microsoft/layoutlmv3-base" # document foundation models
    gradient_checkpointing: False
    pooling_mode: 'cls'  # The pooling mode, can be 'cls' or 'mean'
    data_types:
      - "document"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: 224
    tokenizer_name: "hf_auto"
    max_text_len: 512  # If None or <=0, then use the max length of pretrained models.
    insert_sep: True
    low_cpu_mem_usage: False
    text_segment_num: 2
    stochastic_chunk: False
    text_aug_detect_length: 10                # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.0            # augmentation magnitude randomly drawn from [0, text_trivial_aug_maxscale]

  t_few:
    checkpoint_name: "t5-small" #"bigscience/T0_3B"
    gradient_checkpointing: False
    data_types:
      - "text"
    tokenizer_name: "hf_auto"
    length_norm: 1.0 # Normalizes length to adjust for length bias in target template
    unlikely_loss: 1.0 # Adds loss term that lowers probability of incorrect outputs
    mc_loss: 1.0 # Adds multiple choice cross entropy loss
    max_text_len: 512  # If None or <=0, then use the max length of pretrained models.
    text_segment_num: 2
    insert_sep: True
    low_cpu_mem_usage: False
    stochastic_chunk: False
    text_aug_detect_length: 10                # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.0            # augmentation magnititude randomly drawn from [0, text_trivial_aug_maxscale]
    text_train_augment_types:

  timm_image:
    checkpoint_name: "swin_base_patch4_window7_224"
    mix_choice: "all_logits"
    data_types:
      - "image"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: null
    image_chan_num: 3
    use_learnable_image: False
    max_image_num_per_column: 1

  mmdet_image:
    checkpoint_name: "yolov3_mobilenetv2_8xb24-320-300e_coco"
    config_file: ""
    data_types:
      - "image"
    max_img_num_per_col: 1
    output_bbox_format: "xyxy"  # now support xyxy or xywh, for bbox format details see https://keras.io/api/keras_cv/bounding_box/formats/
    frozen_layers: null
    coco_root: null

  mmocr_text_detection:
    checkpoint_name: "TextSnake"
    data_types:
      - "image"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: 224
    max_img_num_per_col: 2

  mmocr_text_recognition:
    checkpoint_name: "ABINet"
    data_types:
      - "image"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: 224
    max_img_num_per_col: 2

  clip:
    checkpoint_name: "openai/clip-vit-base-patch32"
    data_types:
      - "image"
      - "text"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "clip"
    image_size: 224
    image_chan_num: 3
    use_learnable_image: False
    max_image_num_per_column: 1
    tokenizer_name: "clip"
    max_text_len: 77  # The maximum possible length.
    insert_sep: False
    text_segment_num: 1
    stochastic_chunk: False
    text_aug_detect_length: 10                     # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.0                      # scale randomly drawn from [0, text_trivial_aug_maxscale]
    text_train_augment_types:        # specify augmentation space manually, will randomly select one from the following and identity
      # - "random_swap(0.05)"          # less than 0.1 based on eda paper
      # - "random_delete(0.05)"        # less than 0.1 based on eda paper
      # - "syn_replacement(0.05)"  # less than 0.1 based on eda paper
      # - "insert_punc(0.05)"

  fusion_mlp:
    aux_loss_weight:
    adapt_in_features: "max"
    hidden_sizes:
      - 128
    activation: "leaky_relu"
    dropout: 0.1
    normalization: "layer_norm"
    data_types:

  fusion_ner:
    weight:
    adapt_in_features: "max"
    hidden_sizes:
      - 128
    activation: "leaky_relu"
    drop_rate: 0.1
    normalization: "layer_norm"
    data_types:

  fusion_transformer:
    aux_loss_weight:
    hidden_size: 192
    num_blocks: 3
    attention_num_heads: 8
    adapt_in_features: "max"
    attention_dropout: 0.2
    residual_dropout: 0.0
    ffn_dropout: 0.1
    ffn_hidden_size: 192
    normalization: "layer_norm"
    ffn_activation: "geglu"
    head_activation: "relu"
    data_types:
    additive_attention: False # Whether to use lightweight additive attention, can be True, False or "auto"
    share_qv_weights: False # Whether to share weight for query and value, can be True, False or "auto"

  ft_transformer:
    data_types:
      - "categorical"
      - "numerical"
    embedding_arch:
      - 'linear'
    token_dim: 192
    hidden_size: 192
    num_blocks: 3
    attention_num_heads: 8
    attention_dropout: 0.2
    residual_dropout: 0.0
    ffn_dropout: 0.1
    ffn_hidden_size: 192
    ffn_activation: "geglu"
    head_activation: "relu"
    normalization: "layer_norm"
    merge: "concat"
    requires_all_dtypes: False
    additive_attention: False # Whether to use lightweight additive attention, can be True, False or "auto"
    share_qv_weights: False # Whether to share weight for query and value, can be True, False or "auto"
    pooling_mode: "cls"
    checkpoint_name: null

  sam:
    checkpoint_name: "facebook/sam-vit-huge"
    image_norm: "imagenet"
    data_types:
      - "semantic_segmentation_img"
    train_transforms:
      - "random_horizontal_flip"
    val_transforms: []
    img_transforms:
      - "resize_to_square"
    gt_transforms:
      - "resize_gt_to_square"
    max_img_num_per_col: 1
    frozen_layers: ["mask_decoder.iou_prediction_head", "prompt_encoder"]
    num_mask_tokens: 1
    ignore_label: 255
    adapter_enabled: False
    adapter_dim: 64
    decoder_adapter_enabled: False
    decoder_adapter_dim: 64

  meta_transformer:
    data_types:
      - "image"
      - "text"
      - "categorical"
      - "numerical"
    checkpoint_path: null
    model_version: "base"
    requires_all_dtypes: False
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: 224
    image_chan_num: 3
    use_learnable_image: False
    max_image_num_per_column: 1
    tokenizer_name: "hf_auto"
    max_text_len: 512  # If None or <=0, then use the max length of pretrained models.
    insert_sep: True
    text_segment_num: 2
    stochastic_chunk: False
    text_aug_detect_length: 10     # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.1  # augmentation magnitude randomly drawn from [0, text_trivial_aug_maxscale]
    text_train_augment_types:
    merge: "concat"

```

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
    
    # Encoder Adapter parameters
    parser.add_argument("--adapter_enable", action="store_true", help="Enable standard Adapter modules in the vision encoder")
    parser.add_argument("--adapter_dim", type=int, default=64, help="Dimension of the encoder adapter bottleneck")
    
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
        print(f"Enabling Standard Encoder Adapters with dim={args.adapter_dim}")
        hyperparameters.update({
            "model.sam.adapter_enabled": True,
            "model.sam.adapter_dim": args.adapter_dim,
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
        predictor.fit(train_data=train_df, tuning_data=val_df, seed=args.seed)

    # evaluation
    metric_file = os.path.join(args.output_dir, "metrics.txt")
    f = open(metric_file, "a")
    if dataset_name in ["isic2017", "SBU-shadow", "road_segmentation", "leaf_disease_segmentation"]:
        test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test.csv")), dataset_dir)
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
            res = predictor.evaluate(test_df, metrics=["sm", "fm", "em", "mae"])
            print(f"Evaluation results for test dataset {per_dataset}: ", res)
            f.write(f"Evaluation results for test dataset {per_dataset}: {res} \n")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}.")

    f.close()

```

```python
# coding=utf-8
# Copyright 2023 The Meta AI Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch SAM model."""

import collections
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.sam.configuration_sam import (
    SamConfig,
    SamMaskDecoderConfig,
    SamPromptEncoderConfig,
    SamVisionConfig,
)
from transformers.utils import ModelOutput, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from ..adaptation_layers import AdapterLayer

logger = logging.get_logger(__name__)


SAM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/sam-vit-huge",
    "facebook/sam-vit-large",
    "facebook/sam-vit-base",
    # See all SAM models at https://huggingface.co/models?filter=sam
]


@dataclass
class SamVisionEncoderOutput(ModelOutput):
    """
    Base class for sam vision model's outputs that also contains image embeddings obtained by applying the projection
    layer to the pooler_output.

    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    moe_loss: Optional[torch.FloatTensor] = None


@dataclass
class SamImageSegmentationOutput(ModelOutput):
    """
    Base class for Segment-Anything model's output

    Args:
        iou_scores (`torch.FloatTensor` of shape `(batch_size, num_masks)`):
            The iou scores of the predicted masks.
        pred_masks (`torch.FloatTensor` of shape `(batch_size, num_masks, height, width)`):
            The predicted low resolutions masks. Needs to be post-processed by the processor
        vision_hidden_states  (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the vision model at the output of each layer plus the optional initial embedding outputs.
        vision_attentions  (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        mask_decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    iou_scores: torch.FloatTensor = None
    pred_masks: torch.FloatTensor = None
    vision_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    vision_attentions: Optional[Tuple[torch.FloatTensor]] = None
    mask_decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    vision_moe_loss: Optional[torch.FloatTensor] = None


class SamPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        embeddings = self.projection(pixel_values).permute(0, 2, 3, 1)
        return embeddings


class SamMLPBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lin1 = nn.Linear(config.hidden_size, config.mlp_dim)
        self.lin2 = nn.Linear(config.mlp_dim, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.lin1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.lin2(hidden_states)
        return hidden_states


# Copied from transformers.models.convnext.modeling_convnext.ConvNextLayerNorm with ConvNext->Sam
class SamLayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height,
    width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            x = torch.nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            input_dtype = x.dtype
            x = x.float()
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = x.to(dtype=input_dtype)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SamAttention(nn.Module):
    """
    SAM's attention layer that allows for downscaling the size of the embedding after projection to queries, keys, and
    values.
    """

    def __init__(self, config, downsample_rate=None):
        super().__init__()
        self.hidden_size = config.hidden_size

        downsample_rate = config.attention_downsample_rate if downsample_rate is None else downsample_rate

        self.internal_dim = config.hidden_size // downsample_rate
        self.num_attention_heads = config.num_attention_heads
        if self.internal_dim % config.num_attention_heads != 0:
            raise ValueError("num_attention_heads must divide hidden_size.")

        self.q_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, self.hidden_size)

    def _separate_heads(self, hidden_states: Tensor, num_attention_heads: int) -> Tensor:
        batch, point_batch_size, n_tokens, channel = hidden_states.shape
        c_per_head = channel // num_attention_heads
        hidden_states = hidden_states.reshape(batch * point_batch_size, n_tokens, num_attention_heads, c_per_head)
        return hidden_states.transpose(1, 2)

    def _recombine_heads(self, hidden_states: Tensor, point_batch_size: int) -> Tensor:
        batch, n_heads, n_tokens, c_per_head = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states.reshape(batch // point_batch_size, point_batch_size, n_tokens, n_heads * c_per_head)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attention_similarity: Tensor = None) -> Tensor:
        # Input projections
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        point_batch_size = query.shape[1]
        # Separate into heads
        query = self._separate_heads(query, self.num_attention_heads)
        key = self._separate_heads(key, self.num_attention_heads)
        value = self._separate_heads(value, self.num_attention_heads)

        # SamAttention
        _, _, _, c_per_head = query.shape
        attn = query @ key.permute(0, 1, 3, 2)  # batch_size * point_batch_size  x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        if attention_similarity is not None:
            attn = attn + attention_similarity
            attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ value
        out = self._recombine_heads(out, point_batch_size)
        out = self.out_proj(out)

        return out


class SamTwoWayAttentionBlock(nn.Module):
    def __init__(self, config, attention_downsample_rate: int = 2, skip_first_layer_pe: bool = False):
        """
        A transformer block with four layers:
            (1) self-attention of sparse inputs (2) cross attention of sparse inputs -> dense inputs (3) mlp block on
            sparse inputs (4) cross attention of dense inputs -> sparse inputs

        Arguments:
            config (`SamMaskDecoderConfig`):
                The configuration file used to instantiate the block
            attention_downsample_rate (*optionalk*, int, defaults to 2):
                The downsample ratio of the block used to reduce the inner dim of the attention.
            skip_first_layer_pe (*optional*, bool, defaults to `False`):
                Whether or not to skip the addition of the query_point_embedding on the first layer.
        """
        super().__init__()

        self.hidden_size = config.hidden_size
        self.layer_norm_eps = config.layer_norm_eps

        self.self_attn = SamAttention(config, downsample_rate=1)
        self.layer_norm1 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.cross_attn_token_to_image = SamAttention(config, downsample_rate=attention_downsample_rate)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.mlp = SamMLPBlock(config)
        self.layer_norm3 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        
        # Decoder MLP-Adapter on queries (optional, config-driven)
        self.adapter_queries = None
        # Prefer explicit decoder-specific flags, but fall back to generic adapter flags if provided
        decoder_adapter_enabled = getattr(config, "decoder_adapter_enabled", False)
        decoder_adapter_dim = getattr(config, "decoder_adapter_dim", getattr(config, "adapter_dim", 64))
        if decoder_adapter_enabled:
            self.adapter_queries = AdapterLayer(self.hidden_size, decoder_adapter_dim)

        self.layer_norm4 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.cross_attn_image_to_token = SamAttention(config, downsample_rate=attention_downsample_rate)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_point_embedding: Tensor,
        key_point_embedding: Tensor,
        attention_similarity: Tensor,
        output_attentions: bool = False,
    ):
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(query=queries, key=queries, value=queries)
        else:
            query = queries + query_point_embedding
            attn_out = self.self_attn(query=query, key=query, value=queries)
            queries = queries + attn_out
        queries = self.layer_norm1(queries)

        # Cross attention block, tokens attending to image embedding
        query = queries + query_point_embedding
        key = keys + key_point_embedding

        attn_out = self.cross_attn_token_to_image(
            query=query, key=key, value=keys, attention_similarity=attention_similarity
        )
        queries = queries + attn_out

        queries = self.layer_norm2(queries)

        # MLP block + optional parallel Adapter on queries
        mlp_out = self.mlp(queries)
        if self.adapter_queries is not None:
            adapter_out = self.adapter_queries(queries)
            queries = queries + mlp_out + adapter_out
        else:
            queries = queries + mlp_out
        queries = self.layer_norm3(queries)

        # Cross attention block, image embedding attending to tokens
        query = queries + query_point_embedding
        key = keys + key_point_embedding

        attn_out = self.cross_attn_image_to_token(query=key, key=query, value=queries)
        keys = keys + attn_out

        keys = self.layer_norm4(keys)

        outputs = (queries, keys)

        if output_attentions:
            outputs = outputs + (attn_out,)
        else:
            outputs = outputs + (None,)

        return outputs


class SamTwoWayTransformer(nn.Module):
    def __init__(self, config: SamMaskDecoderConfig):
        super().__init__()
        self.config = config

        self.num_hidden_layers = config.num_hidden_layers
        self.layers = nn.ModuleList()

        for i in range(self.num_hidden_layers):
            self.layers.append(SamTwoWayAttentionBlock(config, skip_first_layer_pe=(i == 0)))

        self.final_attn_token_to_image = SamAttention(config)
        self.layer_norm_final_attn = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        point_embeddings: Tensor,
        image_embeddings: Tensor,
        image_positional_embeddings: Tensor,
        attention_similarity: Tensor,
        target_embedding=None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        all_attentions = ()

        if image_embeddings is None:
            raise ValueError("You have to specify an image_embedding")

        image_embeddings = image_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)
        image_positional_embeddings = image_positional_embeddings.flatten(2).permute(0, 2, 1).unsqueeze(1)

        # Prepare queries
        queries = point_embeddings
        keys = image_embeddings

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            if target_embedding is not None:
                queries += target_embedding

            queries, keys, attention_outputs = layer(
                queries=queries,
                keys=keys,
                query_point_embedding=point_embeddings,
                key_point_embedding=image_positional_embeddings,
                attention_similarity=attention_similarity,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_attentions = all_attentions + (attention_outputs,)

        # Apply the final attenion layer from the points to the image
        query = queries + point_embeddings
        key = keys + image_positional_embeddings

        attn_out = self.final_attn_token_to_image(query=query, key=key, value=keys)

        queries = queries + attn_out
        queries = self.layer_norm_final_attn(queries)
        return queries, keys, all_attentions


class SamFeedForward(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = False
    ):
        super().__init__()
        self.num_layers = num_layers
        self.activation = nn.ReLU()
        self.proj_in = nn.Linear(input_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, output_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)])
        self.sigmoid_output = sigmoid_output

    def forward(self, hidden_states):
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.activation(hidden_states)
        for layer in self.layers:
            hidden_states = self.activation(layer(hidden_states))

        hidden_states = self.proj_out(hidden_states)
        if self.sigmoid_output:
            hidden_states = F.sigmoid(hidden_states)
        return hidden_states


class SamMaskDecoder(nn.Module):
    def __init__(self, config: SamMaskDecoderConfig):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.num_multimask_outputs = config.num_multimask_outputs
        self.num_mask_tokens = config.num_multimask_outputs + 1

        self.iou_token = nn.Embedding(1, self.hidden_size)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.hidden_size)

        self.transformer = SamTwoWayTransformer(config)

        # should we create a new class for this?
        self.upscale_conv1 = nn.ConvTranspose2d(self.hidden_size, self.hidden_size // 4, kernel_size=2, stride=2)
        self.upscale_conv2 = nn.ConvTranspose2d(self.hidden_size // 4, self.hidden_size // 8, kernel_size=2, stride=2)
        self.upscale_layer_norm = SamLayerNorm(self.hidden_size // 4, data_format="channels_first")
        self.activation = nn.GELU()

        mlps_list = []
        for _ in range(self.num_mask_tokens):
            mlps_list += [SamFeedForward(self.hidden_size, self.hidden_size, self.hidden_size // 8, 3)]
        self.output_hypernetworks_mlps = nn.ModuleList(mlps_list)

        self.iou_prediction_head = SamFeedForward(
            self.hidden_size, config.iou_head_hidden_dim, self.num_mask_tokens, config.iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_positional_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        output_attentions: Optional[bool] = None,
        attention_similarity: torch.Tensor = None,
        target_embedding: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (`torch.Tensor`):
                the embeddings from the image encoder
            image_positional_embedding (`torch.Tensor`):
                positional encoding with the shape of image_embeddings
            sparse_prompt_embeddings (`torch.Tensor`):
                The embeddings of the points and boxes
            dense_prompt_embeddings (`torch.Tensor`):
                the embeddings of the mask inputs
            multimask_output (bool):
                Whether to return multiple masks or a single mask.
            output_attentions (bool, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        batch_size, num_channels, height, width = image_embeddings.shape
        point_batch_size = sparse_prompt_embeddings.shape[1]
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

        if sparse_prompt_embeddings.sum().item() != 0:
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        else:
            tokens = output_tokens
        point_embeddings = tokens.to(self.iou_token.weight.dtype)

        # Expand per-image data in batch direction to be per-point
        image_embeddings = image_embeddings + dense_prompt_embeddings
        image_embeddings = image_embeddings.repeat(point_batch_size, 1, 1, 1)
        image_positional_embeddings = image_positional_embeddings.repeat(point_batch_size, 1, 1, 1)

        # Run the transformer, image_positional_embedding are consumed
        point_embedding, image_embeddings, attentions = self.transformer(
            point_embeddings=point_embeddings,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )
        iou_token_out = point_embedding[:, :, 0, :]
        mask_tokens_out = point_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        image_embeddings = image_embeddings.transpose(2, 3).reshape(
            batch_size * point_batch_size, num_channels, height, width
        )

        upscaled_embedding = self.upscale_conv1(image_embeddings)
        upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
        upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            current_mlp = self.output_hypernetworks_mlps[i]
            hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
        hyper_in = torch.stack(hyper_in_list, dim=2)

        _, num_channels, height, width = upscaled_embedding.shape
        upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
        masks = (hyper_in @ upscaled_embedding).reshape(batch_size, point_batch_size, -1, height, width)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, :, mask_slice, :, :]
        iou_pred = iou_pred[:, :, mask_slice]

        outputs = (masks, iou_pred)

        if output_attentions:
            outputs = outputs + (attentions,)
        else:
            outputs = outputs + (None,)

        return outputs


class SamPositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale = config.hidden_size // 2
        self.register_buffer("positional_embedding", self.scale * torch.randn((2, config.num_pos_feats)))

    def forward(self, input_coords, input_shape=None):
        """Positionally encode points that are normalized to [0,1]."""
        coordinates = input_coords.clone()

        if input_shape is not None:
            coordinates[:, :, :, 0] = coordinates[:, :, :, 0] / input_shape[1]
            coordinates[:, :, :, 1] = coordinates[:, :, :, 1] / input_shape[0]

        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coordinates = 2 * coordinates - 1
        coordinates = coordinates.to(self.positional_embedding.dtype)
        coordinates = coordinates @ self.positional_embedding
        coordinates = 2 * np.pi * coordinates
        # outputs d_1 x ... x d_n x channel shape
        return torch.cat([torch.sin(coordinates), torch.cos(coordinates)], dim=-1)


class SamMaskEmbedding(nn.Module):
    def __init__(self, config: SamPromptEncoderConfig):
        super().__init__()
        self.mask_input_channels = config.mask_input_channels // 4
        self.activation = ACT2FN[config.hidden_act]
        self.conv1 = nn.Conv2d(1, self.mask_input_channels, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.mask_input_channels, config.mask_input_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(config.mask_input_channels, config.hidden_size, kernel_size=1)
        self.layer_norm1 = SamLayerNorm(
            self.mask_input_channels, eps=config.layer_norm_eps, data_format="channels_first"
        )
        self.layer_norm2 = SamLayerNorm(
            self.mask_input_channels * 4, eps=config.layer_norm_eps, data_format="channels_first"
        )

    def forward(self, masks):
        hidden_states = self.conv1(masks)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.activation(hidden_states)
        dense_embeddings = self.conv3(hidden_states)
        return dense_embeddings


class SamPromptEncoder(nn.Module):
    def __init__(self, config: SamPromptEncoderConfig, shared_patch_embedding):
        super().__init__()
        self.shared_embedding = shared_patch_embedding
        self.mask_embed = SamMaskEmbedding(config)
        self.no_mask_embed = nn.Embedding(1, config.hidden_size)

        self.image_embedding_size = (config.image_embedding_size, config.image_embedding_size)
        self.input_image_size = config.image_size

        self.point_embed = nn.ModuleList(
            [nn.Embedding(1, config.hidden_size) for i in range(config.num_point_embeddings)]
        )
        self.hidden_size = config.hidden_size
        self.not_a_point_embed = nn.Embedding(1, config.hidden_size)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            target_point_shape = (points.shape[0], points.shape[1], 1, points.shape[-1])
            target_labels_shape = (points.shape[0], points.shape[1], 1)
            padding_point = torch.zeros(target_point_shape, device=points.device)
            padding_label = -torch.ones(target_labels_shape, device=labels.device)
            points = torch.cat([points, padding_point], dim=2)
            labels = torch.cat([labels, padding_label], dim=2)
        input_shape = (self.input_image_size, self.input_image_size)
        point_embedding = self.shared_embedding(points, input_shape)

        # torch.where and expanding the labels tensor is required by the ONNX export
        point_embedding = torch.where(labels[..., None] == -1, self.not_a_point_embed.weight, point_embedding)

        # This is required for the ONNX export. The dtype, device need to be explicitly
        # specified as otherwise torch.onnx.export interprets as double
        point_embedding = torch.where(
            labels[..., None] != -10,
            point_embedding,
            torch.tensor(0.0, dtype=point_embedding.dtype, device=point_embedding.device),
        )

        point_embedding = torch.where(
            (labels == 0)[:, :, :, None],
            point_embedding + self.point_embed[0].weight[None, None, :, :],
            point_embedding,
        )

        point_embedding = torch.where(
            (labels == 1)[:, :, :, None],
            point_embedding + self.point_embed[1].weight[None, None, :, :],
            point_embedding,
        )

        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        batch_size, nb_boxes = boxes.shape[:2]
        coords = boxes.reshape(batch_size, nb_boxes, 2, 2)
        input_shape = (self.input_image_size, self.input_image_size)
        corner_embedding = self.shared_embedding(coords, input_shape)
        corner_embedding[:, :, 0, :] += self.point_embed[2].weight
        corner_embedding[:, :, 1, :] += self.point_embed[3].weight
        return corner_embedding

    def forward(
        self,
        input_points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        input_labels: Optional[torch.Tensor],
        input_boxes: Optional[torch.Tensor],
        input_masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense embeddings.

        Args:
            points (`torch.Tensor`, *optional*):
                point coordinates and labels to embed.
            boxes (`torch.Tensor`, *optional*):
                boxes to embed
            masks (`torch.Tensor`, *optional*):
                masks to embed
        """
        sparse_embeddings = None
        batch_size = 1
        target_device = self.shared_embedding.positional_embedding.device
        if input_points is not None:
            batch_size, point_batch_size = input_points.shape[:2]
            if input_labels is None:
                raise ValueError("If points are provided, labels must also be provided.")
            point_embeddings = self._embed_points(input_points, input_labels, pad=(input_boxes is None))
            sparse_embeddings = point_embeddings
        if input_boxes is not None:
            batch_size = input_boxes.shape[0]
            box_embeddings = self._embed_boxes(input_boxes)
            if sparse_embeddings is None:
                sparse_embeddings = box_embeddings
            else:
                sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=2)
        if input_masks is not None:
            dense_embeddings = self.mask_embed(input_masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch_size, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        if sparse_embeddings is None:
            sparse_embeddings = torch.zeros((batch_size, 1, 1, self.hidden_size), device=target_device)

        return sparse_embeddings, dense_embeddings


class SamVisionAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(self, config, window_size):
        super().__init__()
        input_size = (
            (config.image_size // config.patch_size, config.image_size // config.patch_size)
            if window_size == 0
            else (window_size, window_size)
        )

        self.num_attention_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.scale = head_dim**-0.5
        self.dropout = config.attention_dropout

        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.use_rel_pos = config.use_rel_pos
        if self.use_rel_pos:
            if input_size is None:
                raise ValueError("Input size must be provided if using relative positional encoding.")

            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        """
        Get relative positional embeddings according to the relative positions of
            query and key sizes.

        Args:
            q_size (int):
                size of the query.
            k_size (int):
                size of key k.
            rel_pos (`torch.Tensor`):
                relative position embeddings (L, channel).

        Returns:
            Extracted positional embeddings according to relative positions.
        """
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)

        # Scale the coords with short length if shapes for q and k are different.
        q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

        return rel_pos_resized[relative_coords.long()]

    def add_decomposed_rel_pos(
        self,
        attn: torch.Tensor,
        query: torch.Tensor,
        rel_pos_h: torch.Tensor,
        rel_pos_w: torch.Tensor,
        q_size: Tuple[int, int],
        k_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
        https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py

        Args:
            attn (`torch.Tensor`):
                attention map.
            query (`torch.Tensor`):
                query q in the attention layer with shape (batch_size, query_height * query_width, channel).
            rel_pos_h (`torch.Tensor`):
                relative position embeddings (Lh, channel) for height axis.
            rel_pos_w (`torch.Tensor`):
                relative position embeddings (Lw, channel) for width axis.
            q_size (tuple):
                spatial sequence size of query q with (query_height, query_width).
            k_size (tuple):
                spatial sequence size of key k with (key_height, key_width).

        Returns:
            attn (`torch.Tensor`):
                attention map with added relative positional embeddings.
        """
        query_height, query_width = q_size
        key_height, key_width = k_size
        relative_position_height = self.get_rel_pos(query_height, key_height, rel_pos_h)
        relative_position_width = self.get_rel_pos(query_width, key_width, rel_pos_w)

        batch_size, _, dim = query.shape
        reshaped_query = query.reshape(batch_size, query_height, query_width, dim)
        rel_h = torch.einsum("bhwc,hkc->bhwk", reshaped_query, relative_position_height)
        rel_w = torch.einsum("bhwc,wkc->bhwk", reshaped_query, relative_position_width)
        attn = attn.reshape(batch_size, query_height, query_width, key_height, key_width)
        attn = attn + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
        attn = attn.reshape(batch_size, query_height * query_width, key_height * key_width)
        return attn

    def forward(self, hidden_states: torch.Tensor, output_attentions=False, output_moe_loss=False) -> torch.Tensor:
        batch_size, height, width, _ = hidden_states.shape
        # qkv with shape (3, batch_size, nHead, height * width, channel)
        ##### modify here for conv-lora
        # Paper alignment:
        # - Sec. 3.3 (Integration): Inject Conv-LoRA at attention projections (q/k/v).
        #   The adapted linear layer (possibly Conv-LoRA) may return (tensor, moe_loss, selected_experts) when output_moe_loss=True.
        qkv = self.qkv(hidden_states)
        if output_moe_loss:
            # Handle both 2-value and 3-value returns for backward compatibility
            if isinstance(qkv, tuple) and len(qkv) == 3:
                qkv, moe_loss, selected_experts = qkv
            else:
                qkv, moe_loss = qkv
        qkv = qkv.reshape(batch_size, height * width, 3, self.num_attention_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (batch_size * nHead, height * width, channel)
        query, key, value = qkv.reshape(3, batch_size * self.num_attention_heads, height * width, -1).unbind(0)

        attn_weights = (query * self.scale) @ key.transpose(-2, -1)

        if self.use_rel_pos:
            attn_weights = self.add_decomposed_rel_pos(
                attn_weights, query, self.rel_pos_h, self.rel_pos_w, (height, width), (height, width)
            )

        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = (attn_probs @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)

        attn_output = self.proj(attn_output)

        if output_attentions:
            outputs = (attn_output, attn_weights)
        else:
            outputs = (attn_output, None)
        if output_moe_loss:
            # Sec. 4: propagate MoE load-balancing loss upward for aggregation
            outputs += (moe_loss,)
        return outputs


class SamVisionLayer(nn.Module):
    def __init__(self, config, window_size):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = SamVisionAttention(config, window_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SamMLPBlock(config)
        self.window_size = window_size
        
        # Adapter Integration
        self.adapter = None
        if hasattr(config, "adapter_enabled") and config.adapter_enabled:
            adapter_dim = getattr(config, "adapter_dim", 64)
            self.adapter = AdapterLayer(config.hidden_size, adapter_dim)


    def window_partition(self, hidden_states: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Args:
        Partition into non-overlapping windows with padding if needed.
            hidden_states (tensor): input tokens with [batch_size, height, width, channel]. window_size (int): window
            size.

        Returns:
            windows: windows after partition with [batch_size * num_windows, window_size, window_size, channel].
            (pad_height, pad_width): padded height and width before partition
        """
        batch_size, height, width, channel = hidden_states.shape

        pad_h = (window_size - height % window_size) % window_size
        pad_w = (window_size - width % window_size) % window_size
        hidden_states = F.pad(hidden_states, (0, 0, 0, pad_w, 0, pad_h))
        pad_height, pad_width = height + pad_h, width + pad_w

        hidden_states = hidden_states.reshape(
            batch_size, pad_height // window_size, window_size, pad_width // window_size, window_size, channel
        )
        windows = hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(-1, window_size, window_size, channel)
        return windows, (pad_height, pad_width)

    def window_unpartition(
        self, windows: torch.Tensor, window_size: int, padding_shape: Tuple[int, int], original_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Args:
        Window unpartition into original sequences and removing padding.
            hidden_states (tensor):
                input tokens with [batch_size * num_windows, window_size, window_size, channel].
            window_size (int):
                window size.
            padding_shape (Tuple):
                padded height and width (pad_height, pad_width).
            original_shape (Tuple): original height and width (height, width) before padding.

        Returns:
            hidden_states: unpartitioned sequences with [batch_size, height, width, channel].
        """
        pad_height, pad_width = padding_shape
        height, width = original_shape
        batch_size = windows.shape[0] // (pad_height * pad_width // window_size // window_size)
        hidden_states = windows.reshape(
            batch_size, pad_height // window_size, pad_width // window_size, window_size, window_size, -1
        )
        hidden_states = (
            hidden_states.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(batch_size, pad_height, pad_width, -1)
        )

        hidden_states = hidden_states[:, :height, :width, :].contiguous()
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: Optional[bool] = False,
        output_moe_loss: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        # Window partition
        if self.window_size > 0:
            height, width = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, padding_shape = self.window_partition(hidden_states, self.window_size)

        ### modify here
        attn_outputs = self.attn(
            hidden_states=hidden_states, output_attentions=output_attentions, output_moe_loss=output_moe_loss
        )
        if output_moe_loss:
            hidden_states, attn_weights, moe_loss = attn_outputs
        else:
            hidden_states, attn_weights = attn_outputs
        # Reverse window partition
        if self.window_size > 0:
            hidden_states = self.window_unpartition(hidden_states, self.window_size, padding_shape, (height, width))


        hidden_states = residual + hidden_states
        
        # MLP Block + Parallel Adapter
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        mlp_out = self.mlp(hidden_states)
        
        if self.adapter is not None:
            adapter_out = self.adapter(hidden_states)
            hidden_states = residual + mlp_out + adapter_out
        else:
            hidden_states = residual + mlp_out


        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        else:
            outputs += (None,)  # keep attn_weights on the same place
        if output_moe_loss:
            # Sec. 4: return per-layer MoE loss to be summed in encoder
            outputs += (moe_loss,)

        return outputs


class SamVisionNeck(nn.Module):
    def __init__(self, config: SamVisionConfig):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv2d(config.hidden_size, config.output_channels, kernel_size=1, bias=False)
        self.layer_norm1 = SamLayerNorm(config.output_channels, data_format="channels_first")
        self.conv2 = nn.Conv2d(config.output_channels, config.output_channels, kernel_size=3, padding=1, bias=False)
        self.layer_norm2 = SamLayerNorm(config.output_channels, data_format="channels_first")

    def forward(self, hidden_states):
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.layer_norm1(hidden_states)

        hidden_states = self.conv2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        return hidden_states


class SamVisionEncoder(nn.Module):
    def __init__(self, config: SamVisionConfig):
        super().__init__()
        self.config = config
        self.image_size = config.image_size

        self.patch_embed = SamPatchEmbeddings(config)

        self.pos_embed = None
        if config.use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1,
                    config.image_size // config.patch_size,
                    config.image_size // config.patch_size,
                    config.hidden_size,
                )
            )

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = SamVisionLayer(
                config,
                window_size=config.window_size if i not in config.global_attn_indexes else 0,
            )
            self.layers.append(layer)

        self.neck = SamVisionNeck(config)

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.patch_embed

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_moe_loss: Optional[bool] = None,  # MoE loss for Conv-LoRA
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SamVisionEncoderOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        hidden_states = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_moe_loss = 0.0 if output_moe_loss else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, output_attentions=output_attentions, output_moe_loss=output_moe_loss
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if output_moe_loss:
                # Accumulate MoE load-balancing losses across layers (Sec. 4)
                all_moe_loss = all_moe_loss + layer_outputs[2]

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.neck(hidden_states)

        if not return_dict:
            outputs = (
                hidden_states,
                all_hidden_states,
                all_self_attentions,
                all_moe_loss,
            )
            # if output_hidden_states:
            #     outputs = outputs + (all_hidden_states,)
            # if output_attentions:
            #     outputs = outputs + (all_self_attentions,)
            # if output_moe_loss:
            #     outputs = outputs + (all_moe_loss,)
            return outputs

        return SamVisionEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            moe_loss=all_moe_loss,  # modify here
        )


class SamPreTrainedModel(PreTrainedModel):
    config_class = SamConfig
    base_model_prefix = "sam"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


SAM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`SamConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


SAM_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`SamProcessor`]. See [`SamProcessor.__call__`] for
            details.
        input_points (`torch.FloatTensor` of shape `(batch_size, num_points, 2)`):
            Input 2D spatial points, this is used by the prompt encoder to encode the prompt. Generally yields to much
            better results. The points can be obtained by passing a list of list of list to the processor that will
            create corresponding `torch` tensors of dimension 4. The first dimension is the image batch size, the
            second dimension is the point batch size (i.e. how many segmentation masks do we want the model to predict
            per input point), the third dimension is the number of points per segmentation mask (it is possible to pass
            multiple points for a single mask), and the last dimension is the x (vertical) and y (horizontal)
            coordinates of the point. If a different number of points is passed either for each image, or for each
            mask, the processor will create "PAD" points that will correspond to the (0, 0) coordinate, and the
            computation of the embedding will be skipped for these points using the labels.
        input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points)`):
            Input labels for the points, this is used by the prompt encoder to encode the prompt. According to the
            official implementation, there are 3 types of labels

            - `1`: the point is a point that contains the object of interest
            - `0`: the point is a point that does not contain the object of interest
            - `-1`: the point corresponds to the background

            We added the label:

            - `-10`: the point is a padding point, thus should be ignored by the prompt encoder

            The padding labels should be automatically done by the processor.
        input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes, 4)`):
            Input boxes for the points, this is used by the prompt encoder to encode the prompt. Generally yields to
            much better generated masks. The boxes can be obtained by passing a list of list of list to the processor,
            that will generate a `torch` tensor, with each dimension corresponding respectively to the image batch
            size, the number of boxes per image and the coordinates of the top left and bottom right point of the box.
            In the order (`x1`, `y1`, `x2`, `y2`):

            - `x1`: the x coordinate of the top left point of the input box
            - `y1`: the y coordinate of the top left point of the input box
            - `x2`: the x coordinate of the bottom right point of the input box
            - `y2`: the y coordinate of the bottom right point of the input box

        input_masks (`torch.FloatTensor` of shape `(batch_size, image_size, image_size)`):
            SAM model also accepts segmentation masks as input. The mask will be embedded by the prompt encoder to
            generate a corresponding embedding, that will be fed later on to the mask decoder. These masks needs to be
            manually fed by the user, and they need to be of shape (`batch_size`, `image_size`, `image_size`).

        image_embeddings (`torch.FloatTensor` of shape `(batch_size, output_channels, window_size, window_size)`):
            Image embeddings, this is used by the mask decder to generate masks and iou scores. For more memory
            efficient computation, users can first retrieve the image embeddings using the `get_image_embeddings`
            method, and then feed them to the `forward` method instead of feeding the `pixel_values`.
        multimask_output (`bool`, *optional*):
            In the original implementation and paper, the model always outputs 3 masks per image (or per point / per
            bounding box if relevant). However, it is possible to just output a single mask, that corresponds to the
            "best" mask, by specifying `multimask_output=False`.
        attention_similarity (`torch.FloatTensor`, *optional*):
            Attention similarity tensor, to be provided to the mask decoder for target-guided attention in case the
            model is used for personalization as introduced in [PerSAM](https://arxiv.org/abs/2305.03048).
        target_embedding (`torch.FloatTensor`, *optional*):
            Embedding of the target concept, to be provided to the mask decoder for target-semantic prompting in case
            the model is used for personalization as introduced in [PerSAM](https://arxiv.org/abs/2305.03048).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "Segment Anything Model (SAM) for generating segmentation masks, given an input image and ",
    " optional 2D location and bounding boxes.",
    SAM_START_DOCSTRING,
)
class SamModel(SamPreTrainedModel):
    _tied_weights_keys = ["prompt_encoder.shared_embedding.positional_embedding"]

    def __init__(self, config):
        super().__init__(config)
        self.shared_image_embedding = SamPositionalEmbedding(config.vision_config)

        self.vision_encoder = SamVisionEncoder(config.vision_config)
        self.prompt_encoder = SamPromptEncoder(config.prompt_encoder_config, self.shared_image_embedding)
        self.mask_decoder = SamMaskDecoder(config.mask_decoder_config)

        self.post_init()

    def get_input_embeddings(self):
        return self.vision_encoder.get_input_embeddings()

    def get_image_wide_positional_embeddings(self):
        size = self.config.prompt_encoder_config.image_embedding_size
        target_device = self.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(torch.stack([x_embed, y_embed], dim=-1))
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)  # channel x height x width

    @torch.no_grad()
    def get_image_embeddings(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns the image embeddings by passing the pixel values through the vision encoder.

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Input pixel values
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        """
        vision_output = self.vision_encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeddings = vision_output[0]
        return image_embeddings

    @torch.no_grad()
    def get_prompt_embeddings(
        self,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
    ):
        r"""
        Returns the prompt embeddings by passing the input points, labels, boxes and masks through the prompt encoder.

        Args:
            input_points (`torch.FloatTensor` of shape `(batch_size, point_batch_size, num_points_per_image, 2)`):
                Optional input points for the prompt encoder. The padding of the point is automatically done by the
                processor. `point_batch_size` refers to the number of masks that we want the model to predict per
                point. The model will output `point_batch_size` times 3 masks in total.
            input_labels (`torch.LongTensor` of shape `(batch_size, point_batch_size, num_points_per_image)`):
                Optional input labels for the prompt encoder. The padding of the labels is automatically done by the
                processor, or can be fed by the user.
            input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes_per_image, 4)`):
                Optional input boxes for the prompt encoder. The padding of the boxes is automatically done by the
                processor. users can also pass manually the input boxes.
            input_masks (`torch.LongTensor` of shape `(batch_size, image_size, image_size)`):
                Optional input masks for the prompt encoder.
        """
        prompt_output = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )
        return prompt_output

    @add_start_docstrings_to_model_forward(SAM_INPUTS_DOCSTRING)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_points: Optional[torch.FloatTensor] = None,
        input_labels: Optional[torch.LongTensor] = None,
        input_boxes: Optional[torch.FloatTensor] = None,
        input_masks: Optional[torch.LongTensor] = None,
        image_embeddings: Optional[torch.FloatTensor] = None,
        multimask_output: bool = True,
        attention_similarity: Optional[torch.FloatTensor] = None,
        target_embedding: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_moe_loss: Optional[bool] = None,  # MoE loss for Conv-LoRA
        return_dict=None,
        **kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        r"""
        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoModel, AutoProcessor

        >>> model = AutoModel.from_pretrained("facebook/sam-vit-base")
        >>> processor = AutoProcessor.from_pretrained("facebook/sam-vit-base")

        >>> img_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-car.png"
        >>> raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        >>> input_points = [[[400, 650]]]  # 2D location of a window on the car
        >>> inputs = processor(images=raw_image, input_points=input_points, return_tensors="pt")

        >>> # Get segmentation mask
        >>> outputs = model(**inputs)

        >>> # Postprocess masks
        >>> masks = processor.post_process_masks(
        ...     outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        ... )
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and image_embeddings is None:
            raise ValueError("Either pixel_values or image_embeddings must be provided.")

        if pixel_values is not None and image_embeddings is not None:
            raise ValueError("Only one of pixel_values and image_embeddings can be provided.")

        if input_points is not None and len(input_points.shape) != 4:
            raise ValueError(
                "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.",
                " got {}.".format(input_points.shape),
            )
        if input_boxes is not None and len(input_boxes.shape) != 3:
            raise ValueError(
                "The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.",
                " got {}.".format(input_boxes.shape),
            )
        if input_points is not None and input_boxes is not None:
            point_batch_size = input_points.shape[1]
            box_batch_size = input_boxes.shape[1]
            if point_batch_size != box_batch_size:
                raise ValueError(
                    "You should provide as many bounding boxes as input points per box. Got {} and {}.".format(
                        point_batch_size, box_batch_size
                    )
                )

        image_positional_embeddings = self.get_image_wide_positional_embeddings()
        # repeat with batch size
        batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

        vision_attentions = None
        vision_hidden_states = None
        vision_moe_loss = None

        if pixel_values is not None:
            vision_outputs = self.vision_encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                output_moe_loss=output_moe_loss,
                return_dict=return_dict,
            )
            image_embeddings = vision_outputs[0]

            if output_hidden_states:
                vision_hidden_states = vision_outputs[1]
            if output_attentions:
                # vision_attentions = vision_outputs[-1]
                vision_attentions = vision_outputs[2]
            if output_moe_loss:
                vision_moe_loss = vision_outputs[-1]

        if input_points is not None and input_labels is None:
            input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

        if input_points is not None and image_embeddings.shape[0] != input_points.shape[0]:
            raise ValueError(
                "The batch size of the image embeddings and the input points must be the same. ",
                "Got {} and {} respectively.".format(image_embeddings.shape[0], input_points.shape[0]),
                " if you want to pass multiple points for the same image, make sure that you passed ",
                " input_points of shape (batch_size, point_batch_size, num_points_per_image, 3) and ",
                " input_labels of shape (batch_size, point_batch_size, num_points_per_image)",
            )

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )

        low_res_masks, iou_predictions, mask_decoder_attentions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            attention_similarity=attention_similarity,
            target_embedding=target_embedding,
            output_attentions=output_attentions,
        )

        if not return_dict:
            # output = (iou_predictions, low_res_masks)
            # if output_hidden_states:
            #     output = output + (vision_hidden_states,)

            # if output_attentions:
            #     output = output + (vision_attentions, mask_decoder_attentions)
            output = (
                iou_predictions,
                low_res_masks,
                vision_hidden_states,
                vision_attentions,
                mask_decoder_attentions,
                vision_moe_loss,
            )
            return output

        return SamImageSegmentationOutput(
            iou_scores=iou_predictions,
            pred_masks=low_res_masks,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
            mask_decoder_attentions=mask_decoder_attentions,
            vision_moe_loss=vision_moe_loss,
        )

```

Decoder 侧的 Adapter 模块已经挂在 Two-Way Block 的 MLP 上了，接下来会把超参数从 YAML/CLI 正确传进 SAM 模型构造，并跑一次单测验证。

```python
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import SamConfig

from ..constants import CLASS_LABEL, CLASS_LOGITS, COLUMN, IMAGE, IMAGE_VALID_NUM, LABEL, LOGITS, MASK_LABEL, MOE_LOSS
from .adaptation_layers import ConvLoRALinear, AdapterLayer
from .custom_hf_models.modeling_sam_for_conv_lora import SamImageSegmentationOutput, SamModel
from .utils import assign_layer_ids, freeze_model_layers, image_mean_std

logger = logging.getLogger(__name__)


def multi_class_mask_decoder_forward(
    self,
    image_embeddings: torch.Tensor,
    image_positional_embeddings: torch.Tensor,
    sparse_prompt_embeddings: torch.Tensor,
    dense_prompt_embeddings: torch.Tensor,
    multimask_output: bool,
    output_attentions: Optional[bool] = None,
    attention_similarity: torch.Tensor = None,
    target_embedding: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Modify the forward method of SamMaskDecoder for multi-class semantic segmentation
    based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/modeling_sam.py#L468.

    Args:
        image_embeddings (`torch.Tensor`):
            the embeddings from the image encoder
        image_positional_embedding (`torch.Tensor`):
            positional encoding with the shape of image_embeddings
        sparse_prompt_embeddings (`torch.Tensor`):
            The embeddings of the points and boxes
        dense_prompt_embeddings (`torch.Tensor`):
            the embeddings of the mask inputs
        multimask_output (bool):
            Whether to return multiple masks or a single mask.
        output_attentions (bool, *optional*):
            Whether or not to return the attentions tensors of all attention layers.
    """
    batch_size, num_channels, height, width = image_embeddings.shape
    point_batch_size = sparse_prompt_embeddings.shape[1]
    # Concatenate output tokens
    output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
    output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

    if sparse_prompt_embeddings.sum().item() != 0:
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
    else:
        tokens = output_tokens
    point_embeddings = tokens.to(self.iou_token.weight.dtype)

    # Expand per-image data in batch direction to be per-point
    image_embeddings = image_embeddings + dense_prompt_embeddings
    image_embeddings = image_embeddings.repeat(point_batch_size, 1, 1, 1)
    image_positional_embeddings = image_positional_embeddings.repeat(point_batch_size, 1, 1, 1)

    # Run the transformer, image_positional_embedding are consumed
    point_embedding, image_embeddings, attentions = self.transformer(
        point_embeddings=point_embeddings,
        image_embeddings=image_embeddings,
        image_positional_embeddings=image_positional_embeddings,
        attention_similarity=attention_similarity,
        target_embedding=target_embedding,
        output_attentions=output_attentions,
    )
    iou_token_out = point_embedding[:, :, 0, :]
    mask_tokens_out = point_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

    # Upscale mask embeddings and predict masks using the mask tokens
    image_embeddings = image_embeddings.transpose(2, 3).reshape(
        batch_size * point_batch_size, num_channels, height, width
    )

    upscaled_embedding = self.upscale_conv1(image_embeddings)
    upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
    upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

    ################ Modify the original code. We aim at returning a single mask for each object.
    ################ Original logic is to return multiple masks.
    ################ So we use an MLP network to process all the mask proposals instead of multiple networks.
    # hyper_in_list = []
    # for i in range(self.num_mask_tokens):
    #     current_mlp = self.output_hypernetworks_mlps[i]
    #     hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
    # hyper_in = torch.stack(hyper_in_list, dim=2)
    hyper_in = self.output_hypernetworks_mlps[0](mask_tokens_out)
    ################

    _, num_channels, height, width = upscaled_embedding.shape
    upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
    masks = (hyper_in @ upscaled_embedding).reshape(
        batch_size, point_batch_size, -1, height, width
    )  # bs, 1, num_tokens, h, w

    ################ New added class prediction logic.
    class_predictions = self.output_classifier_mlps(mask_tokens_out).reshape(
        batch_size, point_batch_size, -1, self.num_classes + 1
    )
    ################

    # Generate mask quality predictions
    iou_pred = self.iou_prediction_head(iou_token_out)

    outputs = (masks, iou_pred)

    if output_attentions:
        outputs = outputs + (attentions,)
    else:
        outputs = outputs + (None,)

    return outputs + (class_predictions,)


def multi_class_sam_model_forward(
    self,
    pixel_values: Optional[torch.FloatTensor] = None,
    input_points: Optional[torch.FloatTensor] = None,
    input_labels: Optional[torch.LongTensor] = None,
    input_boxes: Optional[torch.FloatTensor] = None,
    input_masks: Optional[torch.LongTensor] = None,
    image_embeddings: Optional[torch.FloatTensor] = None,
    multimask_output: bool = True,
    attention_similarity: Optional[torch.FloatTensor] = None,
    target_embedding: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_moe_loss: Optional[bool] = None,  # MoE loss for Conv-LoRA
    return_dict=None,
    **kwargs,
) -> List[Dict[str, torch.Tensor]]:
    r"""
    Modify the forward method of SamModel for multi-class semantic segmentation
    based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/modeling_sam.py#L1279.

    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if pixel_values is None and image_embeddings is None:
        raise ValueError("Either pixel_values or image_embeddings must be provided.")

    if pixel_values is not None and image_embeddings is not None:
        raise ValueError("Only one of pixel_values and image_embeddings can be provided.")

    if input_points is not None and len(input_points.shape) != 4:
        raise ValueError(
            "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.",
            " got {}.".format(input_points.shape),
        )
    if input_boxes is not None and len(input_boxes.shape) != 3:
        raise ValueError(
            "The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.",
            " got {}.".format(input_boxes.shape),
        )
    if input_points is not None and input_boxes is not None:
        point_batch_size = input_points.shape[1]
        box_batch_size = input_boxes.shape[1]
        if point_batch_size != box_batch_size:
            raise ValueError(
                "You should provide as many bounding boxes as input points per box. Got {} and {}.".format(
                    point_batch_size, box_batch_size
                )
            )

    image_positional_embeddings = self.get_image_wide_positional_embeddings()
    # repeat with batch size
    batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]
    image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

    vision_attentions = None
    vision_hidden_states = None
    vision_moe_loss = None

    if pixel_values is not None:
        vision_outputs = self.vision_encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_moe_loss=output_moe_loss,
            return_dict=return_dict,
        )
        image_embeddings = vision_outputs[0]

        if output_hidden_states:
            vision_hidden_states = vision_outputs[1]
        if output_attentions:
            # vision_attentions = vision_outputs[-1]
            vision_attentions = vision_outputs[2]
        if output_moe_loss:
            vision_moe_loss = vision_outputs[-1]

    if input_points is not None and input_labels is None:
        input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

    if input_points is not None and image_embeddings.shape[0] != input_points.shape[0]:
        raise ValueError(
            "The batch size of the image embeddings and the input points must be the same. ",
            "Got {} and {} respectively.".format(image_embeddings.shape[0], input_points.shape[0]),
            " if you want to pass multiple points for the same image, make sure that you passed ",
            " input_points of shape (batch_size, point_batch_size, num_points_per_image, 3) and ",
            " input_labels of shape (batch_size, point_batch_size, num_points_per_image)",
        )

    sparse_embeddings, dense_embeddings = self.prompt_encoder(
        input_points=input_points,
        input_labels=input_labels,
        input_boxes=input_boxes,
        input_masks=input_masks,
    )

    low_res_masks, iou_predictions, mask_decoder_attentions, class_predictions = self.mask_decoder(
        image_embeddings=image_embeddings,
        image_positional_embeddings=image_positional_embeddings,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
        attention_similarity=attention_similarity,
        target_embedding=target_embedding,
        output_attentions=output_attentions,
    )

    if not return_dict:
        output = (
            iou_predictions,
            low_res_masks,
            vision_hidden_states,
            vision_attentions,
            mask_decoder_attentions,
            vision_moe_loss,
        )
        return output

    return (
        SamImageSegmentationOutput(
            iou_scores=iou_predictions,
            pred_masks=low_res_masks,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
            mask_decoder_attentions=mask_decoder_attentions,
            vision_moe_loss=vision_moe_loss,
        ),
        ################ New added. Return class predictions as well.
        class_predictions,
        ################
    )


class SAMForSemanticSegmentation(nn.Module):
    """
    Support SAM for semantic segmentation.
    Refer to https://huggingface.co/docs/transformers/main/model_doc/sam
    Paper alignment (Conv-LoRA, ICLR 2024):
    - Sec. 4 (Training): When Conv-LoRA is active, expose/aggregate MoE load-balancing loss produced inside
      attention projections (q/k/v). This module toggles and collects that loss when trainable Conv-LoRA exists.
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str,
        num_classes: int = 1,
        pretrained: Optional[bool] = True,
        frozen_layers: Optional[list] = None,
        num_mask_tokens: int = 1,
        image_norm: Optional[str] = None,
        adapter_enabled: bool = False,
        adapter_dim: int = 64,
        decoder_adapter_enabled: bool = False,
        decoder_adapter_dim: int = 64,
    ):
        """
        Load a pretrained Segment Anything Model (SAM).

        Parameters
        ----------
        prefix
            The prefix of the SAMForSemanticSegmentation model.
        checkpoint_name
            Name of the SAM checkpoint.
        num_classes
            The number of classes
        pretrained
            Whether using the pretrained SAM models. If pretrained=True, download the pretrained model.
        frozen_layers
            A list of substrings of frozen layers' names.
        num_mask_tokens
            The number of mask proposals.
        image_norm
            How to normalize an image. We now support:
            - inception
                Normalize image by IMAGENET_INCEPTION_MEAN and IMAGENET_INCEPTION_STD from timm
            - imagenet
                Normalize image by IMAGENET_DEFAULT_MEAN and IMAGENET_DEFAULT_STD from timm
            - clip
                Normalize image by mean (0.48145466, 0.4578275, 0.40821073) and
                std (0.26862954, 0.26130258, 0.27577711), used for CLIP.
        adapter_enabled
            Whether to enable standard Adapter modules in the vision encoder.
        adapter_dim
            The dimension of the adapter bottleneck.
        """

        super().__init__()
        self.prefix = prefix
        self.pretrained = pretrained
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes
        self.frozen_layers = frozen_layers

        self.device = None
        self.name_to_id = {}

        # Store adapter configs
        self.adapter_enabled = adapter_enabled
        self.adapter_dim = adapter_dim
        self.decoder_adapter_enabled = decoder_adapter_enabled
        self.decoder_adapter_dim = decoder_adapter_dim

        self._load_checkpoint(checkpoint_name)

        freeze_model_layers(self.model, self.frozen_layers)

        self.image_size = self.model.vision_encoder.image_size
        self.config = self.model.config
        self.image_mean, self.image_std = image_mean_std(image_norm)

        # Inject Adapters into vision encoder if enabled
        if self.adapter_enabled:
            logger.info(f"Injecting Adapters with dim={self.adapter_dim} into Vision Encoder")
            for layer in self.model.vision_encoder.layers:
                # Manually create and assign adapter since config didn't have it at init time
                layer.adapter = AdapterLayer(self.config.hidden_size, self.adapter_dim)


        self.model.mask_decoder.num_mask_tokens = num_mask_tokens
        mask_token_data = self.model.mask_decoder.mask_tokens.weight.data[0]
        self.model.mask_decoder.mask_tokens = nn.Embedding(num_mask_tokens, self.model.mask_decoder.hidden_size)
        for i in range(num_mask_tokens):
            self.model.mask_decoder.mask_tokens.weight.data[i] = mask_token_data
        hyper_mlps = self.model.mask_decoder.output_hypernetworks_mlps[0]
        self.model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList([hyper_mlps])
        if num_classes > 1:
            self.model.mask_decoder.num_classes = num_classes
            self.model.mask_decoder.output_classifier_mlps = nn.Linear(
                self.model.mask_decoder.hidden_size, num_classes + 1
            )

            mask_decoder_forward = multi_class_mask_decoder_forward.__get__(
                self.model.mask_decoder, self.model.mask_decoder.__class__
            )
            setattr(self.model.mask_decoder, "forward", mask_decoder_forward)

            sam_model_forward = multi_class_sam_model_forward.__get__(self.model, self.model.__class__)
            setattr(self.model, "forward", sam_model_forward)

        # for Conv-LoRA
        self.output_moe_loss = False

    def _load_checkpoint(self, checkpoint_name):
        if self.pretrained:
            # Try to load from local cache first to avoid network issues
            try:
                self.model = SamModel.from_pretrained(checkpoint_name, local_files_only=True)
                logger.info(f"Loaded model from local cache: {checkpoint_name}")
            except Exception as e:
                logger.info(f"Local cache not found, downloading from Hugging Face: {checkpoint_name}")
            self.model = SamModel.from_pretrained(checkpoint_name)
        else:
            configuration = SamConfig(name_or_path=checkpoint_name)
            self.model = SamModel(configuration)

    def save(self, save_path: str = "./"):
        self.model.save_pretrained(save_path)
        logger.info(f"Model weights for {self.prefix} is saved to {save_path}.")

    @property
    def image_key(self):
        return f"{self.prefix}_{IMAGE}"

    @property
    def image_valid_num_key(self):
        return f"{self.prefix}_{IMAGE_VALID_NUM}"

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    @property
    def image_column_prefix(self):
        return f"{self.image_key}_{COLUMN}"

    @property
    def image_feature_dim(self):
        return self.model.num_features

    @property
    def mask_label_key(self):
        return f"{self.prefix}_{MASK_LABEL}"

    @property
    def class_label_key(self):
        return f"{self.prefix}_{CLASS_LABEL}"

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, ConvLoRALinear):
                # If Conv-LoRA is present and trainable, enable returning moe_loss from the attention blocks (Sec. 4)
                self.output_moe_loss = True
                return self

        return self

    def forward(
        self,
        batch,
    ):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with mask predictions.
        """
        # binary
        if self.num_classes == 1:
            rets = self.model(batch[self.image_key], multimask_output=False, output_moe_loss=self.output_moe_loss)
            pred_masks = rets.pred_masks[:, 0, :, :, :]
            pred_masks = F.interpolate(
                pred_masks, (self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
            if self.training:
                rets_dict = {self.prefix: {LOGITS: pred_masks}}
            else:
                rets_dict = {self.prefix: {LOGITS: pred_masks, LABEL: batch[self.label_key]}}
        # multi-class
        else:
            rets = self.model(batch[self.image_key], multimask_output=False, output_moe_loss=self.output_moe_loss)
            rets, class_predictions = rets
            pred_masks = rets.pred_masks[:, 0, :, :, :]
            pred_classes = class_predictions[:, 0, :, :]
            pred_masks = F.interpolate(
                pred_masks, (self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
            if self.training:
                rets_dict = {self.prefix: {LOGITS: pred_masks, CLASS_LOGITS: pred_classes}}
            else:
                rets_dict = {
                    self.prefix: {
                        LOGITS: pred_masks,
                        CLASS_LOGITS: pred_classes,
                        LABEL: batch[self.label_key],
                    }
                }
        if self.output_moe_loss:
            rets_dict[self.prefix].update({MOE_LOSS: rets.vision_moe_loss})
        return rets_dict

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        In the AutoModel scenario, this function may not always return the correct result.
        Thus, you can use "print(json.dumps(name_to_id, indent=2))" to manually check whether
        the layer ids are reasonable.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = "model"
        pre_encoder_patterns = (
            "vision_encoder",
            "prompt_encoder",
        )
        post_encoder_patterns = ("mask_decoder",)

        names = [n for n, _ in self.named_parameters()]

        name_to_id, names = assign_layer_ids(
            names=names,
            pre_encoder_patterns=pre_encoder_patterns,
            post_encoder_patterns=post_encoder_patterns,
            model_pre=model_prefix,
        )
        if len(names) > 0:
            logger.debug(f"outer layers are treated as head: {names}")
        for n in names:
            assert n not in name_to_id
            name_to_id[n] = 0

        return name_to_id

```

```python
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import SamConfig

from ..constants import CLASS_LABEL, CLASS_LOGITS, COLUMN, IMAGE, IMAGE_VALID_NUM, LABEL, LOGITS, MASK_LABEL, MOE_LOSS
from .adaptation_layers import ConvLoRALinear, AdapterLayer
from .custom_hf_models.modeling_sam_for_conv_lora import SamImageSegmentationOutput, SamModel
from .utils import assign_layer_ids, freeze_model_layers, image_mean_std

logger = logging.getLogger(__name__)


def multi_class_mask_decoder_forward(
    self,
    image_embeddings: torch.Tensor,
    image_positional_embeddings: torch.Tensor,
    sparse_prompt_embeddings: torch.Tensor,
    dense_prompt_embeddings: torch.Tensor,
    multimask_output: bool,
    output_attentions: Optional[bool] = None,
    attention_similarity: torch.Tensor = None,
    target_embedding: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Modify the forward method of SamMaskDecoder for multi-class semantic segmentation
    based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/modeling_sam.py#L468.

    Args:
        image_embeddings (`torch.Tensor`):
            the embeddings from the image encoder
        image_positional_embedding (`torch.Tensor`):
            positional encoding with the shape of image_embeddings
        sparse_prompt_embeddings (`torch.Tensor`):
            The embeddings of the points and boxes
        dense_prompt_embeddings (`torch.Tensor`):
            the embeddings of the mask inputs
        multimask_output (bool):
            Whether to return multiple masks or a single mask.
        output_attentions (bool, *optional*):
            Whether or not to return the attentions tensors of all attention layers.
    """
    batch_size, num_channels, height, width = image_embeddings.shape
    point_batch_size = sparse_prompt_embeddings.shape[1]
    # Concatenate output tokens
    output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
    output_tokens = output_tokens.repeat(batch_size, point_batch_size, 1, 1)

    if sparse_prompt_embeddings.sum().item() != 0:
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
    else:
        tokens = output_tokens
    point_embeddings = tokens.to(self.iou_token.weight.dtype)

    # Expand per-image data in batch direction to be per-point
    image_embeddings = image_embeddings + dense_prompt_embeddings
    image_embeddings = image_embeddings.repeat(point_batch_size, 1, 1, 1)
    image_positional_embeddings = image_positional_embeddings.repeat(point_batch_size, 1, 1, 1)

    # Run the transformer, image_positional_embedding are consumed
    point_embedding, image_embeddings, attentions = self.transformer(
        point_embeddings=point_embeddings,
        image_embeddings=image_embeddings,
        image_positional_embeddings=image_positional_embeddings,
        attention_similarity=attention_similarity,
        target_embedding=target_embedding,
        output_attentions=output_attentions,
    )
    iou_token_out = point_embedding[:, :, 0, :]
    mask_tokens_out = point_embedding[:, :, 1 : (1 + self.num_mask_tokens), :]

    # Upscale mask embeddings and predict masks using the mask tokens
    image_embeddings = image_embeddings.transpose(2, 3).reshape(
        batch_size * point_batch_size, num_channels, height, width
    )

    upscaled_embedding = self.upscale_conv1(image_embeddings)
    upscaled_embedding = self.activation(self.upscale_layer_norm(upscaled_embedding))
    upscaled_embedding = self.activation(self.upscale_conv2(upscaled_embedding))

    ################ Modify the original code. We aim at returning a single mask for each object.
    ################ Original logic is to return multiple masks.
    ################ So we use an MLP network to process all the mask proposals instead of multiple networks.
    # hyper_in_list = []
    # for i in range(self.num_mask_tokens):
    #     current_mlp = self.output_hypernetworks_mlps[i]
    #     hyper_in_list += [current_mlp(mask_tokens_out[:, :, i, :])]
    # hyper_in = torch.stack(hyper_in_list, dim=2)
    hyper_in = self.output_hypernetworks_mlps[0](mask_tokens_out)
    ################

    _, num_channels, height, width = upscaled_embedding.shape
    upscaled_embedding = upscaled_embedding.reshape(batch_size, point_batch_size, num_channels, height * width)
    masks = (hyper_in @ upscaled_embedding).reshape(
        batch_size, point_batch_size, -1, height, width
    )  # bs, 1, num_tokens, h, w

    ################ New added class prediction logic.
    class_predictions = self.output_classifier_mlps(mask_tokens_out).reshape(
        batch_size, point_batch_size, -1, self.num_classes + 1
    )
    ################

    # Generate mask quality predictions
    iou_pred = self.iou_prediction_head(iou_token_out)

    outputs = (masks, iou_pred)

    if output_attentions:
        outputs = outputs + (attentions,)
    else:
        outputs = outputs + (None,)

    return outputs + (class_predictions,)


def multi_class_sam_model_forward(
    self,
    pixel_values: Optional[torch.FloatTensor] = None,
    input_points: Optional[torch.FloatTensor] = None,
    input_labels: Optional[torch.LongTensor] = None,
    input_boxes: Optional[torch.FloatTensor] = None,
    input_masks: Optional[torch.LongTensor] = None,
    image_embeddings: Optional[torch.FloatTensor] = None,
    multimask_output: bool = True,
    attention_similarity: Optional[torch.FloatTensor] = None,
    target_embedding: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_moe_loss: Optional[bool] = None,  # MoE loss for Conv-LoRA
    return_dict=None,
    **kwargs,
) -> List[Dict[str, torch.Tensor]]:
    r"""
    Modify the forward method of SamModel for multi-class semantic segmentation
    based on https://github.com/huggingface/transformers/blob/main/src/transformers/models/sam/modeling_sam.py#L1279.

    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if pixel_values is None and image_embeddings is None:
        raise ValueError("Either pixel_values or image_embeddings must be provided.")

    if pixel_values is not None and image_embeddings is not None:
        raise ValueError("Only one of pixel_values and image_embeddings can be provided.")

    if input_points is not None and len(input_points.shape) != 4:
        raise ValueError(
            "The input_points must be a 4D tensor. Of shape `batch_size`, `point_batch_size`, `nb_points_per_image`, `2`.",
            " got {}.".format(input_points.shape),
        )
    if input_boxes is not None and len(input_boxes.shape) != 3:
        raise ValueError(
            "The input_points must be a 3D tensor. Of shape `batch_size`, `nb_boxes`, `4`.",
            " got {}.".format(input_boxes.shape),
        )
    if input_points is not None and input_boxes is not None:
        point_batch_size = input_points.shape[1]
        box_batch_size = input_boxes.shape[1]
        if point_batch_size != box_batch_size:
            raise ValueError(
                "You should provide as many bounding boxes as input points per box. Got {} and {}.".format(
                    point_batch_size, box_batch_size
                )
            )

    image_positional_embeddings = self.get_image_wide_positional_embeddings()
    # repeat with batch size
    batch_size = pixel_values.shape[0] if pixel_values is not None else image_embeddings.shape[0]
    image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)

    vision_attentions = None
    vision_hidden_states = None
    vision_moe_loss = None

    if pixel_values is not None:
        vision_outputs = self.vision_encoder(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_moe_loss=output_moe_loss,
            return_dict=return_dict,
        )
        image_embeddings = vision_outputs[0]

        if output_hidden_states:
            vision_hidden_states = vision_outputs[1]
        if output_attentions:
            # vision_attentions = vision_outputs[-1]
            vision_attentions = vision_outputs[2]
        if output_moe_loss:
            vision_moe_loss = vision_outputs[-1]

    if input_points is not None and input_labels is None:
        input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int, device=input_points.device)

    if input_points is not None and image_embeddings.shape[0] != input_points.shape[0]:
        raise ValueError(
            "The batch size of the image embeddings and the input points must be the same. ",
            "Got {} and {} respectively.".format(image_embeddings.shape[0], input_points.shape[0]),
            " if you want to pass multiple points for the same image, make sure that you passed ",
            " input_points of shape (batch_size, point_batch_size, num_points_per_image, 3) and ",
            " input_labels of shape (batch_size, point_batch_size, num_points_per_image)",
        )

    sparse_embeddings, dense_embeddings = self.prompt_encoder(
        input_points=input_points,
        input_labels=input_labels,
        input_boxes=input_boxes,
        input_masks=input_masks,
    )

    low_res_masks, iou_predictions, mask_decoder_attentions, class_predictions = self.mask_decoder(
        image_embeddings=image_embeddings,
        image_positional_embeddings=image_positional_embeddings,
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
        attention_similarity=attention_similarity,
        target_embedding=target_embedding,
        output_attentions=output_attentions,
    )

    if not return_dict:
        output = (
            iou_predictions,
            low_res_masks,
            vision_hidden_states,
            vision_attentions,
            mask_decoder_attentions,
            vision_moe_loss,
        )
        return output

    return (
        SamImageSegmentationOutput(
            iou_scores=iou_predictions,
            pred_masks=low_res_masks,
            vision_hidden_states=vision_hidden_states,
            vision_attentions=vision_attentions,
            mask_decoder_attentions=mask_decoder_attentions,
            vision_moe_loss=vision_moe_loss,
        ),
        ################ New added. Return class predictions as well.
        class_predictions,
        ################
    )


class SAMForSemanticSegmentation(nn.Module):
    """
    Support SAM for semantic segmentation.
    Refer to https://huggingface.co/docs/transformers/main/model_doc/sam
    Paper alignment (Conv-LoRA, ICLR 2024):
    - Sec. 4 (Training): When Conv-LoRA is active, expose/aggregate MoE load-balancing loss produced inside
      attention projections (q/k/v). This module toggles and collects that loss when trainable Conv-LoRA exists.
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str,
        num_classes: int = 1,
        pretrained: Optional[bool] = True,
        frozen_layers: Optional[list] = None,
        num_mask_tokens: int = 1,
        image_norm: Optional[str] = None,
        adapter_enabled: bool = False,
        adapter_dim: int = 64,
        decoder_adapter_enabled: bool = False,
        decoder_adapter_dim: int = 64,
    ):
        """
        Load a pretrained Segment Anything Model (SAM).

        Parameters
        ----------
        prefix
            The prefix of the SAMForSemanticSegmentation model.
        checkpoint_name
            Name of the SAM checkpoint.
        num_classes
            The number of classes
        pretrained
            Whether using the pretrained SAM models. If pretrained=True, download the pretrained model.
        frozen_layers
            A list of substrings of frozen layers' names.
        num_mask_tokens
            The number of mask proposals.
        image_norm
            How to normalize an image. We now support:
            - inception
                Normalize image by IMAGENET_INCEPTION_MEAN and IMAGENET_INCEPTION_STD from timm
            - imagenet
                Normalize image by IMAGENET_DEFAULT_MEAN and IMAGENET_DEFAULT_STD from timm
            - clip
                Normalize image by mean (0.48145466, 0.4578275, 0.40821073) and
                std (0.26862954, 0.26130258, 0.27577711), used for CLIP.
        adapter_enabled
            Whether to enable standard Adapter modules in the vision encoder.
        adapter_dim
            The dimension of the adapter bottleneck.
        """

        super().__init__()
        self.prefix = prefix
        self.pretrained = pretrained
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes
        self.frozen_layers = frozen_layers

        self.device = None
        self.name_to_id = {}

        # Store adapter configs
        self.adapter_enabled = adapter_enabled
        self.adapter_dim = adapter_dim
        self.decoder_adapter_enabled = decoder_adapter_enabled
        self.decoder_adapter_dim = decoder_adapter_dim

        self._load_checkpoint(checkpoint_name)

        freeze_model_layers(self.model, self.frozen_layers)

        self.image_size = self.model.vision_encoder.image_size
        self.config = self.model.config
        self.image_mean, self.image_std = image_mean_std(image_norm)

        # Inject Adapters into vision encoder if enabled
        if self.adapter_enabled:
            logger.info(f"Injecting Adapters with dim={self.adapter_dim} into Vision Encoder")
            for layer in self.model.vision_encoder.layers:
                # Manually create and assign adapter since config didn't have it at init time
                layer.adapter = AdapterLayer(self.config.hidden_size, self.adapter_dim)


        self.model.mask_decoder.num_mask_tokens = num_mask_tokens
        mask_token_data = self.model.mask_decoder.mask_tokens.weight.data[0]
        self.model.mask_decoder.mask_tokens = nn.Embedding(num_mask_tokens, self.model.mask_decoder.hidden_size)
        for i in range(num_mask_tokens):
            self.model.mask_decoder.mask_tokens.weight.data[i] = mask_token_data
        hyper_mlps = self.model.mask_decoder.output_hypernetworks_mlps[0]
        self.model.mask_decoder.output_hypernetworks_mlps = nn.ModuleList([hyper_mlps])
        if num_classes > 1:
            self.model.mask_decoder.num_classes = num_classes
            self.model.mask_decoder.output_classifier_mlps = nn.Linear(
                self.model.mask_decoder.hidden_size, num_classes + 1
            )

            mask_decoder_forward = multi_class_mask_decoder_forward.__get__(
                self.model.mask_decoder, self.model.mask_decoder.__class__
            )
            setattr(self.model.mask_decoder, "forward", mask_decoder_forward)

            sam_model_forward = multi_class_sam_model_forward.__get__(self.model, self.model.__class__)
            setattr(self.model, "forward", sam_model_forward)

        # for Conv-LoRA
        self.output_moe_loss = False

    def _load_checkpoint(self, checkpoint_name):
        if self.pretrained:
            # Prepare configuration so that decoder adapter flags are available during model construction
            configuration = SamConfig.from_pretrained(checkpoint_name)
            # Propagate decoder adapter settings into mask decoder config (if present)
            if hasattr(configuration, "mask_decoder_config") and configuration.mask_decoder_config is not None:
                configuration.mask_decoder_config.decoder_adapter_enabled = self.decoder_adapter_enabled
                configuration.mask_decoder_config.decoder_adapter_dim = self.decoder_adapter_dim
            # Try to load from local cache first to avoid network issues
            try:
                self.model = SamModel.from_pretrained(checkpoint_name, config=configuration, local_files_only=True)
                logger.info(f"Loaded model from local cache with custom config: {checkpoint_name}")
            except Exception as e:
                logger.info(f"Local cache not found, downloading from Hugging Face with custom config: {checkpoint_name}")
                self.model = SamModel.from_pretrained(checkpoint_name, config=configuration)
        else:
            configuration = SamConfig(name_or_path=checkpoint_name)
            if hasattr(configuration, "mask_decoder_config") and configuration.mask_decoder_config is not None:
                configuration.mask_decoder_config.decoder_adapter_enabled = self.decoder_adapter_enabled
                configuration.mask_decoder_config.decoder_adapter_dim = self.decoder_adapter_dim
            self.model = SamModel(configuration)

    def save(self, save_path: str = "./"):
        self.model.save_pretrained(save_path)
        logger.info(f"Model weights for {self.prefix} is saved to {save_path}.")

    @property
    def image_key(self):
        return f"{self.prefix}_{IMAGE}"

    @property
    def image_valid_num_key(self):
        return f"{self.prefix}_{IMAGE_VALID_NUM}"

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    @property
    def image_column_prefix(self):
        return f"{self.image_key}_{COLUMN}"

    @property
    def image_feature_dim(self):
        return self.model.num_features

    @property
    def mask_label_key(self):
        return f"{self.prefix}_{MASK_LABEL}"

    @property
    def class_label_key(self):
        return f"{self.prefix}_{CLASS_LABEL}"

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, ConvLoRALinear):
                # If Conv-LoRA is present and trainable, enable returning moe_loss from the attention blocks (Sec. 4)
                self.output_moe_loss = True
                return self

        return self

    def forward(
        self,
        batch,
    ):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with mask predictions.
        """
        # binary
        if self.num_classes == 1:
            rets = self.model(batch[self.image_key], multimask_output=False, output_moe_loss=self.output_moe_loss)
            pred_masks = rets.pred_masks[:, 0, :, :, :]
            pred_masks = F.interpolate(
                pred_masks, (self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
            if self.training:
                rets_dict = {self.prefix: {LOGITS: pred_masks}}
            else:
                rets_dict = {self.prefix: {LOGITS: pred_masks, LABEL: batch[self.label_key]}}
        # multi-class
        else:
            rets = self.model(batch[self.image_key], multimask_output=False, output_moe_loss=self.output_moe_loss)
            rets, class_predictions = rets
            pred_masks = rets.pred_masks[:, 0, :, :, :]
            pred_classes = class_predictions[:, 0, :, :]
            pred_masks = F.interpolate(
                pred_masks, (self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
            if self.training:
                rets_dict = {self.prefix: {LOGITS: pred_masks, CLASS_LOGITS: pred_classes}}
            else:
                rets_dict = {
                    self.prefix: {
                        LOGITS: pred_masks,
                        CLASS_LOGITS: pred_classes,
                        LABEL: batch[self.label_key],
                    }
                }
        if self.output_moe_loss:
            rets_dict[self.prefix].update({MOE_LOSS: rets.vision_moe_loss})
        return rets_dict

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        In the AutoModel scenario, this function may not always return the correct result.
        Thus, you can use "print(json.dumps(name_to_id, indent=2))" to manually check whether
        the layer ids are reasonable.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = "model"
        pre_encoder_patterns = (
            "vision_encoder",
            "prompt_encoder",
        )
        post_encoder_patterns = ("mask_decoder",)

        names = [n for n, _ in self.named_parameters()]

        name_to_id, names = assign_layer_ids(
            names=names,
            pre_encoder_patterns=pre_encoder_patterns,
            post_encoder_patterns=post_encoder_patterns,
            model_pre=model_prefix,
        )
        if len(names) > 0:
            logger.debug(f"outer layers are treated as head: {names}")
        for n in names:
            assert n not in name_to_id
            name_to_id[n] = 0

        return name_to_id

```

### 实现结果概览

- **配置层面**：在默认模型配置中给 SAM 增加了 Decoder-Adapter 开关与维度：
  
```250:268:multimodal/src/autogluon/multimodal/configs/model/default.yaml
sam:
  ...
  adapter_enabled: False
  adapter_dim: 64
  decoder_adapter_enabled: False
  decoder_adapter_dim: 64
```

- **训练脚本 (`run_semantic_segmentation.py`)**：  
  - 新增 CLI 参数，用于单独控制 Decoder-Adapter：
    - `--decoder_adapter_enable`：启用 Mask Decoder 中的 MLP-Adapter
    - `--decoder_adapter_dim`：Decoder-Adapter bottleneck 维度（默认 64）
  - 将其写入 hyperparameters：
  
```91:127:examples/automm/Conv-LoRA/run_semantic_segmentation.py
# Encoder Adapter parameters
parser.add_argument("--adapter_enable", action="store_true", help="Enable standard Adapter modules in the vision encoder")
parser.add_argument("--adapter_dim", type=int, default=64, help="Dimension of the encoder adapter bottleneck")

# Decoder Adapter parameters
parser.add_argument("--decoder_adapter_enable", action="store_true", help="Enable MLP-Adapter modules in the mask decoder")
parser.add_argument("--decoder_adapter_dim", type=int, default=64, help="Dimension of the decoder adapter bottleneck")
...
# Adapter configuration
if args.adapter_enable:
    print(f"Enabling Standard Encoder Adapters with dim={args.adapter_dim}")
    hyperparameters.update({
        "model.sam.adapter_enabled": True,
        "model.sam.adapter_dim": args.adapter_dim,
    })

# Decoder Adapter configuration
if args.decoder_adapter_enable:
    print(f"Enabling Decoder MLP-Adapters with dim={args.decoder_adapter_dim}")
    hyperparameters.update({
        "model.sam.decoder_adapter_enabled": True,
        "model.sam.decoder_adapter_dim": args.decoder_adapter_dim,
    })
```

- **Mask Decoder 内部：在 Two-Way Block 的 MLP 上挂 MLP-Adapter**  
  在 `SamTwoWayAttentionBlock` 中增加 `adapter_queries`，并在 MLP 分支后并行加入：

```278:327:multimodal/src/autogluon/multimodal/models/custom_hf_models/modeling_sam_for_conv_lora.py
self.mlp = SamMLPBlock(config)
self.layer_norm3 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

# Decoder MLP-Adapter on queries (optional, config-driven)
self.adapter_queries = None
# Prefer explicit decoder-specific flags, but fall back to generic adapter flags if provided
decoder_adapter_enabled = getattr(config, "decoder_adapter_enabled", False)
decoder_adapter_dim = getattr(config, "decoder_adapter_dim", getattr(config, "adapter_dim", 64))
if decoder_adapter_enabled:
    self.adapter_queries = AdapterLayer(self.hidden_size, decoder_adapter_dim)
...
# MLP block + optional parallel Adapter on queries
mlp_out = self.mlp(queries)
if self.adapter_queries is not None:
    adapter_out = self.adapter_queries(queries)
    queries = queries + mlp_out + adapter_out
else:
    queries = queries + mlp_out
queries = self.layer_norm3(queries)
```

含义：  
- `queries` 仍然是点/掩码 token；  
- MLP 和 Adapter 共用同一输入 `queries`，输出残差相加后再 `LayerNorm3`；  
- 当 `decoder_adapter_enabled=False` 时，逻辑完全退化成原始的 `queries + mlp_out`，不会改变行为。

- **SAM 封装：将 YAML/CLI 的 decoder_adapter_* 注入到 Mask Decoder 配置**  

在 `SAMForSemanticSegmentation` 中：

1. 构造函数保存 encoder/decoder adapter 配置，并继续用 encoder adapter 注入 Vision Encoder：

```266:333:multimodal/src/autogluon/multimodal/models/sam.py
def __init__(..., adapter_enabled: bool = False, adapter_dim: int = 64,
             decoder_adapter_enabled: bool = False, decoder_adapter_dim: int = 64):
    ...
    # Store adapter configs
    self.adapter_enabled = adapter_enabled
    self.adapter_dim = adapter_dim
    self.decoder_adapter_enabled = decoder_adapter_enabled
    self.decoder_adapter_dim = decoder_adapter_dim

    self._load_checkpoint(checkpoint_name)
    ...
    # Inject Adapters into vision encoder if enabled
    if self.adapter_enabled:
        logger.info(f"Injecting Adapters with dim={self.adapter_dim} into Vision Encoder")
        for layer in self.model.vision_encoder.layers:
            layer.adapter = AdapterLayer(self.config.hidden_size, self.adapter_dim)
```

2. 在 `_load_checkpoint` 里，在 **构建模型之前** 把 Decoder-Adapter 配置写进 `SamConfig.mask_decoder_config`，这样 `SamMaskDecoder` / `SamTwoWayAttentionBlock` 在构造时就能读到：

```360:371:multimodal/src/autogluon/multimodal/models/sam.py
def _load_checkpoint(self, checkpoint_name):
    if self.pretrained:
        # Prepare configuration so that decoder adapter flags are available during model construction
        configuration = SamConfig.from_pretrained(checkpoint_name)
        # Propagate decoder adapter settings into mask decoder config (if present)
        if hasattr(configuration, "mask_decoder_config") and configuration.mask_decoder_config is not None:
            configuration.mask_decoder_config.decoder_adapter_enabled = self.decoder_adapter_enabled
            configuration.mask_decoder_config.decoder_adapter_dim = self.decoder_adapter_dim
        # Try to load from local cache first to avoid network issues
        try:
            self.model = SamModel.from_pretrained(checkpoint_name, config=configuration, local_files_only=True)
            logger.info(f"Loaded model from local cache with custom config: {checkpoint_name}")
        except Exception as e:
            logger.info(f"Local cache not found, downloading from Hugging Face with custom config: {checkpoint_name}")
            self.model = SamModel.from_pretrained(checkpoint_name, config=configuration)
    else:
        configuration = SamConfig(name_or_path=checkpoint_name)
        if hasattr(configuration, "mask_decoder_config") and configuration.mask_decoder_config is not None:
            configuration.mask_decoder_config.decoder_adapter_enabled = self.decoder_adapter_enabled
            configuration.mask_decoder_config.decoder_adapter_dim = self.decoder_adapter_dim
        self.model = SamModel(configuration)
```

这样就实现了你计划中强调的“Decoder 端用 **config 驱动** 创建 Adapter，而不是事后手动遍历挂载”。

> 说明：我尝试在 sandbox 中跑 `tests/unittests/models/test_sam_adapter.py` 做回归测试，但环境里缺少 `torch`（`ModuleNotFoundError`），所以暂时只能依赖静态检查；改动本身保持了向后兼容（默认 decoder_adapter_enabled=False 时行为不变）。

---

### 使用方式示例

- **完整混合架构（Conv-LoRA + GSPO + Encoder-Adapter + Decoder-Adapter）**：

```bash
cd /Users/domqiu/Works/autogluon/examples/automm/Conv-LoRA

python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --gspo_enable \
    --adapter_enable \
    --adapter_dim 64 \
    --decoder_adapter_enable \
    --decoder_adapter_dim 64 \
    --output_dir outputs/gspo_hybrid_full_decoder_adapter
```

- **只开 Decoder-Adapter 做消融**（关闭 Encoder-Adapter）：

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --gspo_enable \
    --decoder_adapter_enable \
    --decoder_adapter_dim 64 \
    --output_dir outputs/gspo_decoder_adapter_only
```

你可以按计划对比：  
- 仅 Conv-LoRA  
- Conv-LoRA + Encoder-Adapter  
- Conv-LoRA + Decoder-Adapter  
- Conv-LoRA + Encoder-Adapter + Decoder-Adapter  
再叠加/不叠加 GSPO，来量化 Decoder-Adapter 的额外贡献。

```yaml
sam:
  ...
  adapter_enabled: False
  adapter_dim: 64
  decoder_adapter_enabled: False
  decoder_adapter_dim: 64
```

```python
# Encoder Adapter parameters
parser.add_argument("--adapter_enable", action="store_true", help="Enable standard Adapter modules in the vision encoder")
parser.add_argument("--adapter_dim", type=int, default=64, help="Dimension of the encoder adapter bottleneck")

# Decoder Adapter parameters
parser.add_argument("--decoder_adapter_enable", action="store_true", help="Enable MLP-Adapter modules in the mask decoder")
parser.add_argument("--decoder_adapter_dim", type=int, default=64, help="Dimension of the decoder adapter bottleneck")
...
# Adapter configuration
if args.adapter_enable:
    print(f"Enabling Standard Encoder Adapters with dim={args.adapter_dim}")
    hyperparameters.update({
        "model.sam.adapter_enabled": True,
        "model.sam.adapter_dim": args.adapter_dim,
    })

# Decoder Adapter configuration
if args.decoder_adapter_enable:
    print(f"Enabling Decoder MLP-Adapters with dim={args.decoder_adapter_dim}")
    hyperparameters.update({
        "model.sam.decoder_adapter_enabled": True,
        "model.sam.decoder_adapter_dim": args.decoder_adapter_dim,
    })
```

```python
self.mlp = SamMLPBlock(config)
self.layer_norm3 = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

# Decoder MLP-Adapter on queries (optional, config-driven)
self.adapter_queries = None
# Prefer explicit decoder-specific flags, but fall back to generic adapter flags if provided
decoder_adapter_enabled = getattr(config, "decoder_adapter_enabled", False)
decoder_adapter_dim = getattr(config, "decoder_adapter_dim", getattr(config, "adapter_dim", 64))
if decoder_adapter_enabled:
    self.adapter_queries = AdapterLayer(self.hidden_size, decoder_adapter_dim)
...
# MLP block + optional parallel Adapter on queries
mlp_out = self.mlp(queries)
if self.adapter_queries is not None:
    adapter_out = self.adapter_queries(queries)
    queries = queries + mlp_out + adapter_out
else:
    queries = queries + mlp_out
queries = self.layer_norm3(queries)
```

```python
def __init__(..., adapter_enabled: bool = False, adapter_dim: int = 64,
             decoder_adapter_enabled: bool = False, decoder_adapter_dim: int = 64):
    ...
    # Store adapter configs
    self.adapter_enabled = adapter_enabled
    self.adapter_dim = adapter_dim
    self.decoder_adapter_enabled = decoder_adapter_enabled
    self.decoder_adapter_dim = decoder_adapter_dim

    self._load_checkpoint(checkpoint_name)
    ...
    # Inject Adapters into vision encoder if enabled
    if self.adapter_enabled:
        logger.info(f"Injecting Adapters with dim={self.adapter_dim} into Vision Encoder")
        for layer in self.model.vision_encoder.layers:
            layer.adapter = AdapterLayer(self.config.hidden_size, self.adapter_dim)
```

```python
def _load_checkpoint(self, checkpoint_name):
    if self.pretrained:
        # Prepare configuration so that decoder adapter flags are available during model construction
        configuration = SamConfig.from_pretrained(checkpoint_name)
        # Propagate decoder adapter settings into mask decoder config (if present)
        if hasattr(configuration, "mask_decoder_config") and configuration.mask_decoder_config is not None:
            configuration.mask_decoder_config.decoder_adapter_enabled = self.decoder_adapter_enabled
            configuration.mask_decoder_config.decoder_adapter_dim = self.decoder_adapter_dim
        # Try to load from local cache first to avoid network issues
        try:
            self.model = SamModel.from_pretrained(checkpoint_name, config=configuration, local_files_only=True)
            logger.info(f"Loaded model from local cache with custom config: {checkpoint_name}")
        except Exception as e:
            logger.info(f"Local cache not found, downloading from Hugging Face with custom config: {checkpoint_name}")
            self.model = SamModel.from_pretrained(checkpoint_name, config=configuration)
    else:
        configuration = SamConfig(name_or_path=checkpoint_name)
        if hasattr(configuration, "mask_decoder_config") and configuration.mask_decoder_config is not None:
            configuration.mask_decoder_config.decoder_adapter_enabled = self.decoder_adapter_enabled
            configuration.mask_decoder_config.decoder_adapter_dim = self.decoder_adapter_dim
        self.model = SamModel(configuration)
```

```shellscript
cd /Users/domqiu/Works/autogluon/examples/automm/Conv-LoRA

python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --gspo_enable \
    --adapter_enable \
    --adapter_dim 64 \
    --decoder_adapter_enable \
    --decoder_adapter_dim 64 \
    --output_dir outputs/gspo_hybrid_full_decoder_adapter
```

```shellscript
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --gspo_enable \
    --decoder_adapter_enable \
    --decoder_adapter_dim 64 \
    --output_dir outputs/gspo_decoder_adapter_only
```

---

**User**

请把你实现的详细方案写在文档里，要包括但不限于 模型结构图、实验从头到尾所有需要调用的脚本、参数说明、消融实验的设计，等等

---

**Cursor**

```markdown
# 设计实验Adapter部分的创新与遵循
_Exported on 11/30/2025 at 20:14:26 PST from Cursor (2.1.41)_

---

**User**

阅读docs/cursor_gspo_adapter_convlora-251129.md以了解上下文。接着回答：为什么在你设计本实验的Adapter部分时，只遵循了部分的原始设计，而创新/修改了另外一些部分？你这样做的理由是什么？“✅ 遵循的部分

瓶颈架构 - 来自 Houlsby et al., 2019 ✓

GeLU 激活 - 来自现代 Transformer 实践 ✓

插入 ViT 层 - 来自 Medical SAM Adapter ✓

🆕 创新/修改的部分

并行集成 - 不同于原始串行设计

外部残差 - 与 MLP 共享残差基准

可学习缩放 - 借鉴 LoRA 的思想

零初始化 - 借鉴 LoRA 的初始化策略✅ 遵循的部分

瓶颈架构 - 来自 Houlsby et al., 2019 ✓

GeLU 激活 - 来自现代 Transformer 实践 ✓

插入 ViT 层 - 来自 Medical SAM Adapter ✓

🆕 创新/修改的部分

并行集成 - 不同于原始串行设计

外部残差 - 与 MLP 共享残差基准

可学习缩放 - 借鉴 LoRA 的思想

零初始化 - 借鉴 LoRA 的初始化策略”

---

**Cursor**

我先快速阅读指定的 `docs/cursor_gspo_adapter_convlora-251129.md`，抓住你当前实验中 Adapter 设计的整体思路和上下文，然后基于其中的论证来解释：哪些部分为什么保持原始设计，哪些地方为什么要改、改动的具体动机和权衡。接下来我会用中文总结成一段清晰的设计 rationale。

我会分段读取这篇较长的文档，重点搜索和 Adapter 设计相关的小节（尤其是“并行集成”、“外部残差”、“可学习缩放”等关键词），然后结合上下文给出一段简洁但有逻辑的设计理由说明。

### 总体思路：为什么“部分继承 + 部分改造”

**简要回答**：  
我只保留了 Adapter 设计中**真正承载“思想”的那几部分**（瓶颈结构、激活函数、插在 ViT 层的思路），而在那些**更偏“实现细节/工程权衡”的地方**（串行/并行、残差位置、缩放、初始化）做了改造，以便：

- 更好地适配当前这个 **“SAM + Conv-LoRA + GSPO + 冻结主干”** 的复杂场景  
- 降低训练不稳定 / 性能退化的风险  
- 减少对已有 Conv-LoRA 实现的侵入式修改，方便维护与消融实验  

下面按你列的几点逐条说明。

---

### ✅ 为什么这些部分保持了原始设计？

- **瓶颈架构（Houlsby Adapter 的核心）**
  - 这是 Adapter 的**本质思想**：用 `d → r → d` 的低秩/瓶颈结构，在参数很少的情况下，对通道维度进行可学习的偏移。
  - 无论是 Houlsby 原论文还是 Medical SAM Adapter，本质都是这个思路，所以这里基本**原样继承**，只是把维度换成适合 SAM 的 `1280 → 64 → 1280`。
  - 这一点决定了“Adapter 是谁”、起什么作用，因此没有理由去推翻。

- **GeLU 激活**
  - Houlsby 原文里 ReLU/GeLU 都常见；现代 Transformer（包括 ViT 和 SAM）内部大量使用 GeLU。
  - 为了与 SAM 主干/Conv-LoRA 的数值行为保持一致，直接采用 GeLU 是一种**与主干对齐的工程常识**，不是有必要去“另起炉灶”的地方。

- **插入 ViT 层（对 vision encoder 做轻量改造）**
  - 无论是 NLP Adapter 还是 Medical SAM Adapter，核心共识是：**贴在 Transformer Block 里做小改动、冻结大部分主干**。
  - 当前实验目标也是：在**不破坏 SAM 大结构**的前提下，用少量参数做领域适配，所以继续沿用“ViT 内插 Adapter”的整体策略是自然选择。

---

### 🆕 为什么这些地方要“创新 / 改造”？

#### 1. **并行集成（而不是原始的串行堆叠）**

- 原始 Houlsby 设计更偏向：在 Attention/FFN 前后插多个 Adapter，形成**串行结构**，路径更长。
- 在本实验中：
  - ViT 的 MLP 本身就很强 + Conv-LoRA 已经在 Attention 路径上改动很大；
  - 如果再在 FFN 路径上做**多层串行堆叠**，容易：
    - 过度改变预训练分布（和原 SAM 差异巨大）  
    - 加深“有效深度”，带来优化不稳定、收敛慢的问题。
- 因此改成：**MLP 和 Adapter 并行，最后在同一个残差上做“三路相加”**：
  - 把 Adapter 明确当作一个“**额外的校正项（correction term）**”，而不是再造一个新的深网络分支。
  - 与 Conv-LoRA 在 Attention 路径的改造形成**一主（原 MLP）两辅（Conv-LoRA + Adapter）**的格局，更容易控制各路的贡献。

#### 2. **外部残差（与 MLP 共享同一个残差基准）**

- 原始 Adapter/Medical SAM Adapter多是：  
  `output = x + Adapter(x)`（残差在 Adapter 模块内部完成）。
- 你的实现是：  
  `hidden_states = residual + mlp_out + adapter_out`  
  即 **MLP 和 Adapter 共用同一个 residual 基准**，残差逻辑在外层统一处理。
- 这样改的原因：
  - 从实现上：  
    - 与现有 SAM / Conv-LoRA 的 `residual + mlp_out` 代码**最小改动地对接**，只需要在末尾多加一项 `+ adapter_out`。  
    - 减少对原有 block 结构的侵入修改，有利于调试和后续替换/关闭 Adapter。
  - 从数值行为上：  
    - 把 Adapter 明确定位为“在与 MLP 同一尺度上的附加修正”，而不是新开一条残差支路，可以更直接控制三者的相对大小（结合可学习缩放和零初始化）。
    - 更容易做到：“初始时退化为原模型，训练过程中逐渐长出有用的偏移”。

#### 3. **可学习缩放（借鉴 LoRA 的思想）**

- LoRA 里有类似的 `α / r` 或可学习 scale，用来：
  - 控制低秩增量 `ΔW` 的有效幅度；
  - 避免一开始梯度过大、直接把预训练权重“冲散”。
- 在本实验中：
  - Adapter 作为**第二个 PEFT 分支**（第一个是 Conv-LoRA），如果没有额外的尺度控制，一旦学习率或梯度稍大，很可能出现：
    - Adapter 输出幅度远大于 MLP/Conv-LoRA，强行“盖过”原有表征；
    - 或者反过来几乎学不到东西。
  - 加上**可学习 scale 参数**，一开始可以设得很小，让 Adapter 从“几乎 0 贡献”开始，训练中根据梯度自动放大/缩小自己：
    - 这在**多 PEFT 模块共存**时尤其重要，相当于每个模块都有自己的“音量旋钮”。

#### 4. **零初始化（借鉴 LoRA 的初始化策略）**

- LoRA 的常见做法：低秩分解 `A @ B` 里，把其中一层（通常是上投影）**零初始化**，保证初始时 `ΔW ≈ 0`，模型行为 ≈ 预训练模型。
- 你这里对 Adapter 的 `up_proj` 做零初始化，本质上是在做同一件事：
  - **初始阶段**：
    - `adapter_out ≈ 0`  
    - 整个 block 输出近似于 `residual + mlp_out`，和“只有 Conv-LoRA 时”的行为几乎一样。
  - **好处**：
    - 避免在加入 Adapter 的那一刻，模型分布发生剧烈跳变；
    - 对于已经比较“精细”的 Conv-LoRA + GSPO 训练过程，只是“温柔地”添加了一个额外自由度，而不是重启一套新动力学。
- 在有 Conv-LoRA + GSPO + 冻结主干的组合下，训练本身就相对脆弱，**零初始化是非常关键的稳定器**，这一点是原始 Adapter 论文在“单一 Adapter + 传统训练”场景下不那么迫切的，但在你的场景中就变得很重要，因此是有意识地从 LoRA 实践中引入的改造。

---

### 小结：为什么“该保留的保留，该改的敢改”

- **保留的**（瓶颈、GeLU、插入 ViT 层）：  
  这些是 Adapter 思想的**核心共识**，跨 NLP / Vision / Medical 都是通用的，继续沿用可以保证你做的是“真正的 Adapter”，而不是换了个名字的别的模块。
- **改造的**（并行集成、外部残差、可学习缩放、零初始化）：  
  这些更多是“**在特定系统里，怎么把 Adapter 嵌进去更稳、更好控**”的工程/优化细节：
  - 要和 **Conv-LoRA + GSPO** 共存；
  - 要尽量**不破坏 SAM 预训练分布**；
  - 要能**一键关/开、方便做消融**；
  - 要保证**初始行为接近原模型，训练逐渐生效**。
  
因此，整体策略就是：  
**在“概念层面”尽量忠实于原始 Adapter 设计，在“实现与训练动力学层面”根据你这个实验的特殊场景做有针对性的改造。**


## Mask Decoder 中 MLP-Adapter 详细设计与实验说明

### 1. 整体模型结构与适配模块分布

- **整体数据流**（语义分割）：

```text
输入图像
  ↓
SAM Vision Encoder（ViT 主干，冻结）
  ├─ Conv-LoRA（Attention 中 qkv 线性层，含 MoE-Conv + GSPO）
  └─ Encoder-Adapter（可选，插在每个 SamVisionLayer 的 MLP 之后）
  ↓
image_embeddings  (B, C, H', W')
  ↓
SAM Mask Decoder（Two-Way Transformer）
  ├─ Token Self-Attn & Cross-Attn (queries ↔ image_embeddings)
  ├─ Decoder MLP-Adapter（本次新增，插在 queries 的 MLP 之后）
  └─ Upsampling + Hypernetworks + IoU Head
  ↓
低分辨率 mask + IoU 评分 (+ 多类 logits)
  ↓
上采样到原图尺寸 → 语义分割输出
```

- **PEFT 模块分工**：
  - **Conv-LoRA**：作用在 Vision Encoder 的 Attention（空间维度适配，多专家 + GSPO）。
  - **Encoder-Adapter**：作用在 Vision Encoder 的 MLP（通道维度适配）。
  - **Decoder-Adapter（本次新增）**：作用在 Mask Decoder Two-Way Block 的 MLP（解码阶段的语义/边界适配）。

### 2. Decoder-Adapter 的网络结构与插入位置

- **子模块类型：AdapterLayer**
  - 结构：`Linear(C → r) → GELU → Linear(r → C)`，其中：
    - **C**：Two-Way Block 中 token 的 hidden_size；
    - **r**：`decoder_adapter_dim`（默认 64）。
  - 与 Vision Encoder 中使用的 Adapter 保持同一风格（bottleneck + GELU）。

- **插入位置：SamTwoWayAttentionBlock 中的 MLP 路径**

简化前后对比（只画 queries 分支）：

```text
原始 Two-Way Block (queries 路径):

queries
  ↓ Self-Attn (+PE)
  ↓ Cross-Attn token→image
  ↓ LayerNorm2
  ↓
MLP(queries)
  ↓
queries = queries + mlp_out
  ↓
LayerNorm3
  ↓
Cross-Attn image→token

加入 Decoder-Adapter 后:

queries
  ↓ Self-Attn (+PE)
  ↓ Cross-Attn token→image
  ↓ LayerNorm2
  ↓
mlp_out = MLP(queries)
if adapter_queries is not None:
    adapter_out = AdapterLayer(queries)
    queries = queries + mlp_out + adapter_out
else:
    queries = queries + mlp_out
  ↓
LayerNorm3
  ↓
Cross-Attn image→token
```

- **作用方式**：
  - 与 Encoder-Adapter 类似，Decoder-Adapter 也是**并行集成**在 MLP 上；
  - 输入为当前层的 `queries`，即点 token / mask token 的隐表示；
  - 输出是对 MLP 输出的**通道方向校正项**，直接影响后续 Cross-Attn image→token 以及最终 mask/head 的表示。

### 3. 配置与脚本：从零开始跑完整实验

#### 3.1 配置项（YAML）

在 `model.default.yaml` 中，SAM 配置新增：

```yaml
model:
  sam:
    ...
    adapter_enabled: False
    adapter_dim: 64
    decoder_adapter_enabled: False
    decoder_adapter_dim: 64
```

- **含义**：
  - `adapter_enabled` / `adapter_dim`：控制 Vision Encoder 中的标准 Adapter（已存在）。
  - `decoder_adapter_enabled` / `decoder_adapter_dim`：控制 Mask Decoder 中的 MLP-Adapter（本次新增）。
  - 默认均为 `False` 和 `64`，保证在不显式开启时行为与旧版完全一致。

#### 3.2 训练脚本与命令行参数

训练入口仍然是：

- `examples/automm/Conv-LoRA/run_semantic_segmentation.py`

新增/更新的关键 CLI 参数：

- **基础模型与 Conv-LoRA / GSPO**：
  - **`--task`**：数据集名称（`polyp`, `leaf_disease_segmentation`, `camo_sem_seg`, `isic2017`, `road_segmentation`, `SBU-shadow`）
  - **`--rank`**：Conv-LoRA 低秩 r（如 3）
  - **`--expert_num`**：Conv-LoRA MoE-Conv 专家数（如 8）
  - **`--gspo_enable`**：是否启用 GSPO 训练
  - 其他如 `--gspo_group_size`, `--gspo_warmup_epochs`, `--gspo_contrastive_weight`, `--gspo_quality_momentum` 对应论文中的 GSPO 超参。

- **Encoder-Adapter（已存在）**：
  - **`--adapter_enable`**：启用 Vision Encoder 中的标准 Adapter
  - **`--adapter_dim`**：Encoder-Adapter bottleneck 维度（默认 64）

- **Decoder-Adapter（本次新增）**：
  - **`--decoder_adapter_enable`**：
    - 启用 Mask Decoder Two-Way Block 中的 MLP-Adapter；
    - 内部等价于配置 `model.sam.decoder_adapter_enabled = True`。
  - **`--decoder_adapter_dim`**：
    - Decoder-Adapter bottleneck 维度（默认 64）；
    - 内部等价于 `model.sam.decoder_adapter_dim = 64`。

脚本会把上述参数写入 `hyperparameters`，最终由 `MultiModalPredictor` 构建含 Conv-LoRA + GSPO + 各类 Adapter 的 SAM 语义分割模型。

#### 3.3 从训练到评估的完整流程示例（以 ISIC2017 为例）

- **1）准备数据集 CSV**（遵循现有 Conv-LoRA 示例的约定）：
  - `datasets/isic2017/isic2017/train.csv`
  - `datasets/isic2017/isic2017/val.csv`
  - `datasets/isic2017/isic2017/test.csv`
  - 每个 CSV 至少包含 `image`, `label` 两列（文件名相对路径）。

- **2）训练 + 自动评估（单条命令完成）**

- **基线：Conv-LoRA + GSPO（无任何 Adapter）**：

```bash
cd examples/automm/Conv-LoRA

python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --gspo_enable \
    --output_dir outputs/baseline_conv_lora_gspo
```

- **只启用 Decoder-Adapter（验证 Decoder 的额外贡献）**：

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --gspo_enable \
    --decoder_adapter_enable \
    --decoder_adapter_dim 64 \
    --output_dir outputs/decoder_adapter_only
```

- **Encoder-Adapter + Decoder-Adapter（双路径通道适配）**：

```bash
python3 run_semantic_segmentation.py \
    --task isic2017 \
    --rank 3 \
    --expert_num 8 \
    --gspo_enable \
    --adapter_enable \
    --adapter_dim 64 \
    --decoder_adapter_enable \
    --decoder_adapter_dim 64 \
    --output_dir outputs/encoder_decoder_adapter
```

脚本会在训练完成后自动：

- 在 `outputs/.../metrics.txt` 中写入测试集评估结果；
- 对于 ISIC2017 这类任务，默认评估 `iou` 和 `dice`。

### 4. Decoder-Adapter 相关参数说明（总结版）

- **结构相关**：
  - **`decoder_adapter_enabled` / `--decoder_adapter_enable`**：
    - `False`：Mask Decoder 使用原始结构；
    - `True`：在每个 Two-Way Block 的 MLP 上并行挂载 MLP-Adapter。
  - **`decoder_adapter_dim` / `--decoder_adapter_dim`**：
    - Bottleneck 维度 r，控制 Decoder-Adapter 容量与参数量；
    - 推荐初始值 64，可在 32–128 范围内做调参。

- **与其它 PEFT 模块的关系**：
  - **Conv-LoRA**：
    - 控制 Attention 中 qkv 的卷积型低秩适配（空间维度）；
    - 通过 `--rank`, `--expert_num`, `--gspo_*` 等参数配置。
  - **Encoder-Adapter**：
    - 控制 Vision Encoder 中 MLP 的通道适配（特征编码阶段）；
    - 通过 `--adapter_enable`, `--adapter_dim`。
  - **Decoder-Adapter**（本次新增）：
    - 控制 Mask Decoder 中 MLP 的通道适配（解码/预测阶段）；
    - 通过 `--decoder_adapter_enable`, `--decoder_adapter_dim`。

### 5. 消融实验设计（推荐表）

- **基础组合（单任务，如 ISIC2017）**：

- **1）仅 Conv-LoRA + GSPO（无 Adapter）**：
  - 目标：获得论文 Conv-LoRA + GSPO 的强基线；
  - 命令：只加 `--gspo_enable`，不加任何 `adapter_*` 参数。

- **2）Conv-LoRA + Encoder-Adapter**：
  - 目标：验证在编码阶段加通道 Adapter 的收益；
  - 命令：`--adapter_enable --adapter_dim 64`，不加 `--decoder_adapter_enable`。

- **3）Conv-LoRA + Decoder-Adapter（本次核心）**：
  - 目标：验证只在解码阶段加 Adapter 的收益；
  - 命令：`--decoder_adapter_enable --decoder_adapter_dim 64`，不加 `--adapter_enable`。

- **4）Conv-LoRA + Encoder-Adapter + Decoder-Adapter（全量）**：
  - 目标：验证双路径 Adapter 与 Conv-LoRA 的协同增益；
  - 命令：同时加 `--adapter_enable` 与 `--decoder_adapter_enable`。

- **指标与期望现象**：
  - **指标**：
    - 主：IoU、Dice；
    - 辅：可关注小目标、边界区域的局部可视化。
  - **理想趋势**（示意）：
    - 3）> 1）：证明 Decoder-Adapter 在解码阶段带来额外收益；
    - 2）> 1）：证明 Encoder-Adapter 在编码阶段有效；
    - 4）≥ max(2,3)：证明双路径 Adapter 与 Conv-LoRA / GSPO 有协同效果。

### 6. 实践建议与注意事项

- **调参顺序**：
  - 先在固定 Conv-LoRA + GSPO 配置下，只调节 `decoder_adapter_dim`（如 32 / 64 / 96）；
  - 若收益稳定，再考虑联合调节 Encoder-Adapter 与 Decoder-Adapter 的维度。

- **稳定性**：
  - Decoder-Adapter 内部沿用零初始化和残差并行设计；
  - 在不开启时，行为与旧模型严格一致，适合逐步打开消融。

整体而言，Decoder-Adapter 把 Adapter 的“通道适配”能力从 **特征编码阶段（Vision Encoder）扩展到了解码预测阶段（Mask Decoder）**，为医学语义分割提供了一个更完整的、端到端的 PEFT 适配路径。  

```

### 已在文档中补充的内容摘要

- **模型结构图**：在 `cursor_gspo_adapter_convlora_followingups-251130.md` 里新增了 “Mask Decoder 中 MLP-Adapter 详细设计与实验说明” 大节，给出了从输入图像 → Vision Encoder（Conv-LoRA + Encoder-Adapter）→ Mask Decoder（新增 Decoder-Adapter）→ 输出分割的整体结构示意，以及 Two-Way Block 内部加入 MLP-Adapter 前后的对比流程图。  
- **脚本与参数说明**：详细列出了 `run_semantic_segmentation.py` 所需的所有关键参数（Conv-LoRA、GSPO、Encoder-Adapter、Decoder-Adapter），并给出从数据准备、训练到自动评估（`metrics.txt`）的完整命令示例，包括仅 Decoder-Adapter、Encoder+Decoder 双 Adapter 和全量混合配置的调用方式。  
- **Decoder-Adapter 设计细节**：说明了 Decoder-Adapter 的网络结构（`Linear(C→r) → GELU → Linear(r→C)`）、在 `SamTwoWayAttentionBlock` 中的具体插入位置和残差形式，以及与 Encoder-Adapter / Conv-LoRA 的分工关系。  
- **消融实验设计**：规划了 4 组核心实验组合（无 Adapter、仅 Encoder-Adapter、仅 Decoder-Adapter、双 Adapter），明确了对应命令、对比指标（IoU / Dice）和期望现象，并补充了调参与稳定性方面的实践建议。

---

**User**

请确认：我们在本实验中使用的强化学习方法是什么

---

**Cursor**

**本实验中使用的强化学习方法是 GSPO（Group Sequence Policy Optimization）**。  
它是一种基于策略梯度 / PPO 风格的 **组级序列策略优化方法**，用来在 Conv-LoRA 的 MoE-Conv 专家之间进行策略优化与选择，而不是 DQN、SAC 等其他 RL 算法。

---

**User**

该方法体现在代码中的哪里

---

**Cursor**

### 简明回答

**GSPO 在代码中主要体现在两个地方：入口开关 + Conv-LoRA 的 MoE gating 逻辑里。**

---

### 1. 入口：脚本里把 GSPO 超参写进 LoRA 配置

在语义分割入口脚本中，通过命令行参数打开 GSPO，并把相关超参写入 `hyperparameters`：

```114:122:examples/automm/Conv-LoRA/run_semantic_segmentation.py
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
```

这些字段最终会传到 Conv-LoRA 的适配层里。

---

### 2. 核心：Conv-LoRA 的 MoE 门控和质量更新（强化学习部分）

GSPO 的具体实现嵌在 Conv-LoRA 的线性层和 MoE 门控中：

1. **ConvLoRALinear 中开启 GSPO，并把参数传给门控：**

```678:711:multimodal/src/autogluon/multimodal/models/adaptation_layers.py
class ConvLoRALinear(nn.Linear, LoRALayer):
    def __init__(..., conv_lora_expert_num: Optional[int] = None,
                 gspo_enabled: bool = False,
                 gspo_group_size: int = 3,
                 gspo_quality_momentum: float = 0.9, ...):
        ...
        self.gspo_enabled = gspo_enabled
        ...
        self.lora_moe_gating = MoEGate(
            M=conv_lora_expert_num,
            d=self.r,
            K=topk,
            gspo_enabled=gspo_enabled,
            gspo_group_size=gspo_group_size,
            gspo_quality_momentum=gspo_quality_momentum
        )
...
        if self.gspo_enabled:
            gates, moe_loss, selected_experts = self.lora_moe_gating(lora_res)
        else:
            ...
```

2. **`MoEGate`：这里就是 GSPO 的“策略 + 价值更新”逻辑：**

- **质量感知 bias + 组级 Top-K 选择（Group Sequence Policy Optimization 的核心）：**

```802:821:multimodal/src/autogluon/multimodal/models/adaptation_layers.py
class MoEGate(nn.Module):
    def __init__(..., gspo_enabled=False, gspo_group_size=3, gspo_quality_momentum=0.9):
        ...
        self.gspo_enabled = gspo_enabled
        self.gspo_group_size = gspo_group_size
        self.gspo_quality_momentum = gspo_quality_momentum
        # GSPO: group_size 作为 K
        self.k = gspo_group_size if gspo_enabled else K
        ...
        if gspo_enabled:
            self.register_buffer("expert_quality_history", torch.zeros(M))
            self.register_buffer("expert_usage_count", torch.zeros(M))
...
    def forward(self, feats, loss_coef=1e-2, noise_epsilon=1e-2):
        ...
        clean_logits = feats_S @ self.w_gate
        # GSPO: 按历史质量给 logits 加 bias（利用 + 探索）
        if self.gspo_enabled and self.training:
            quality_bias = self._compute_quality_bias()
            clean_logits = clean_logits + quality_bias.unsqueeze(0)
        ...
        top_logits, top_indices = logits.topk(min(self.k + 1, self.M), dim=1)
        top_k_indices = top_indices[:, : self.k]
        ...
        return gates, loss, top_k_indices
```

- **根据分割质量（如 IoU/Dice）更新专家质量历史，形成 RL 式反馈回路：**

```979:1003:multimodal/src/autogluon/multimodal/models/adaptation_layers.py
def update_quality_history(self, selected_experts, quality_scores):
    """
    GSPO: Update expert quality history based on actual performance.
    """
    if not self.gspo_enabled or not hasattr(self, 'expert_quality_history'):
        return
    
    with torch.no_grad():
        for b in range(selected_experts.shape[0]):
            quality = quality_scores[b].item() if quality_scores[b].numel() == 1 else quality_scores[b].mean().item()
            for k in range(selected_experts.shape[1]):
                expert_id = selected_experts[b, k].item()
                # 动量更新
                current_quality = self.expert_quality_history[expert_id]
                self.expert_quality_history[expert_id] = (
                    self.gspo_quality_momentum * current_quality +
                    (1 - self.gspo_quality_momentum) * quality
                )
                self.expert_usage_count[expert_id] += 1
```

**总结一下：**

- 命令行里的 `--gspo_enable` 只是打开 GSPO；
- 真正的“强化学习策略优化”体现在 `ConvLoRALinear` + `MoEGate`：
  - 使用质量历史 `expert_quality_history` 对门控 logits 施加偏置（策略更新）；
  - 使用 `update_quality_history` 按训练过程中观察到的分割质量（IoU/Dice）更新每个专家的“价值”，形成**基于专家质量的组级策略优化**——这就是 GSPO 在代码中的具体落地。

```python
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
```

```python
class ConvLoRALinear(nn.Linear, LoRALayer):
    def __init__(..., conv_lora_expert_num: Optional[int] = None,
                 gspo_enabled: bool = False,
                 gspo_group_size: int = 3,
                 gspo_quality_momentum: float = 0.9, ...):
        ...
        self.gspo_enabled = gspo_enabled
        ...
        self.lora_moe_gating = MoEGate(
            M=conv_lora_expert_num,
            d=self.r,
            K=topk,
            gspo_enabled=gspo_enabled,
            gspo_group_size=gspo_group_size,
            gspo_quality_momentum=gspo_quality_momentum
        )
...
        if self.gspo_enabled:
            gates, moe_loss, selected_experts = self.lora_moe_gating(lora_res)
        else:
            ...
```

```python
class MoEGate(nn.Module):
    def __init__(..., gspo_enabled=False, gspo_group_size=3, gspo_quality_momentum=0.9):
        ...
        self.gspo_enabled = gspo_enabled
        self.gspo_group_size = gspo_group_size
        self.gspo_quality_momentum = gspo_quality_momentum
        # GSPO: group_size 作为 K
        self.k = gspo_group_size if gspo_enabled else K
        ...
        if gspo_enabled:
            self.register_buffer("expert_quality_history", torch.zeros(M))
            self.register_buffer("expert_usage_count", torch.zeros(M))
...
    def forward(self, feats, loss_coef=1e-2, noise_epsilon=1e-2):
        ...
        clean_logits = feats_S @ self.w_gate
        # GSPO: 按历史质量给 logits 加 bias（利用 + 探索）
        if self.gspo_enabled and self.training:
            quality_bias = self._compute_quality_bias()
            clean_logits = clean_logits + quality_bias.unsqueeze(0)
        ...
        top_logits, top_indices = logits.topk(min(self.k + 1, self.M), dim=1)
        top_k_indices = top_indices[:, : self.k]
        ...
        return gates, loss, top_k_indices
```

```python
def update_quality_history(self, selected_experts, quality_scores):
    """
    GSPO: Update expert quality history based on actual performance.
    """
    if not self.gspo_enabled or not hasattr(self, 'expert_quality_history'):
        return
    
    with torch.no_grad():
        for b in range(selected_experts.shape[0]):
            quality = quality_scores[b].item() if quality_scores[b].numel() == 1 else quality_scores[b].mean().item()
            for k in range(selected_experts.shape[1]):
                expert_id = selected_experts[b, k].item()
                # 动量更新
                current_quality = self.expert_quality_history[expert_id]
                self.expert_quality_history[expert_id] = (
                    self.gspo_quality_momentum * current_quality +
                    (1 - self.gspo_quality_momentum) * quality
                )
                self.expert_usage_count[expert_id] += 1
```

---

**User**

你说”GSPO 的具体实现嵌在 Conv-LoRA 的线性层和 MoE 门控中“，那么Encoder-Adapter和Decoder-Adapter会在训练中参与到GSPO强化学习的流程中吗？

---

**Cursor**

**不会直接参与 GSPO 的强化学习流程。**

- **GSPO 显式作用的唯一位置**：  
  在 `ConvLoRALinear` 和 `MoEGate` 里，通过  
  - `gspo_enabled / gspo_group_size / gspo_quality_momentum`  
  - `expert_quality_history` + `update_quality_history`  
  来优化 **Conv-LoRA 的 MoE-Conv 专家选择策略**。这一套逻辑只挂在 LoRA 的 qkv 线性层上。

- **Encoder-Adapter / Decoder-Adapter 的角色**：  
  - 都是普通的 `AdapterLayer`，只接受来自分割损失（IoU/Dice/structure loss）的标准反向梯度更新；  
  - **不参与** GSPO 的门控、Top-K 选择、quality history 更新等 RL 机制；  
  - 也不会出现在 `selected_experts`、`expert_quality_history` 这些 GSPO 相关状态里。

- **间接关系**：  
  - GSPO 会改变 Conv-LoRA 专家选择 → 影响主干特征分布；  
  - Encoder/Decoder-Adapter 在这个特征分布之上继续做通道适配，但本身是“普通监督学习”，不是 RL agent。

