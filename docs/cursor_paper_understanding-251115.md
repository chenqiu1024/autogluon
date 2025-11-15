# 查找数据集下载位置
_Exported on 11/15/2025 at 13:47:56 PST from Cursor (2.0.77)_

---

**User**

prepare_semantic_segmentation_datasets.py把数据集下载到哪里了？

---

**Cursor**

I'll search the workspace for the script and any references to its name to locate it and inspect where it writes downloads. Then I'll open the file to read its arguments and default paths.

Now I'll open `examples/automm/Conv-LoRA/prepare_semantic_segmentation_datasets.py` to check its download/output directory logic and defaults.

**结论**：下载并解压到「运行脚本时的当前工作目录」下的 `datasets/<数据集名>` 目录中。

依据代码：

```6:16:/root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA/prepare_semantic_segmentation_datasets.py
def get_data_home_dir():
    return os.path.join(os.getcwd().replace("\\", "/"), "datasets")

if __name__ == "__main__":
    base_dir = get_data_home_dir()
    for name in ["polyp", "leaf_disease_segmentation", "camo_sem_seg", "isic2017", "road_segmentation", "SBU-shadow"]:
        url = f"s3://automl-mm-bench/semantic_segmentation/{name}.zip"

        dataset_dir = os.path.join(base_dir, name)
        load_zip.unzip(url, unzip_dir=dataset_dir)
```

- **目录规则**：`<当前工作目录>/datasets/<name>`
- **示例**：如果你在脚本所在目录执行
  `/root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA`,
  则数据会在：
  `/root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA/datasets/polyp` 等子目录里。

```python
def get_data_home_dir():
    return os.path.join(os.getcwd().replace("\\", "/"), "datasets")

if __name__ == "__main__":
    base_dir = get_data_home_dir()
    for name in ["polyp", "leaf_disease_segmentation", "camo_sem_seg", "isic2017", "road_segmentation", "SBU-shadow"]:
        url = f"s3://automl-mm-bench/semantic_segmentation/{name}.zip"

        dataset_dir = os.path.join(base_dir, name)
        load_zip.unzip(url, unzip_dir=dataset_dir)
```

---

**User**

@docs docs/Convolution Meets LoRA- Parameter Efficient Finetuning for Segment Anything Model.pdf 这篇文章里的Conv-LoRA方法，其对SAM的prompt是如何生成的？请结合文章和源码详细说明，并在源码中找到相应的地方

---

**Cursor**

I'll scan the repository for Conv-LoRA and SAM prompt generation implementation and open the key files to cite exact locations.

- 结论（结合论文与代码）：Conv‑LoRA并不额外“生成”交互式 prompt（点/框/参考掩码）。在本仓库的语义分割训练里，SAM始终以“无显式提示”的方式工作：不传入点/框/掩码，`PromptEncoder` 返回
  - 稀疏提示：全零占位
  - 稠密提示：一个固定的“no‑mask”嵌入张量
  该稠密提示直接加到图像编码特征上，由掩码解码器与“输出 tokens”（IoU token + mask tokens）一起生成掩码。Conv‑LoRA的核心是在视觉编码器注意力的 qkv 投影处注入带卷积 MoE 的LoRA分支，提升特征表达，但不改变 SAM 的 prompt 机制。

### 源码依据（关键位置）

1) 训练调用不传入任何 prompt（点/框/掩码），即“无提示”推理
```391:420:/root/autodl-tmp/works/autogluon/multimodal/src/autogluon/multimodal/models/sam.py
# binary
if self.num_classes == 1:
    rets = self.model(batch[self.image_key], multimask_output=False, output_moe_loss=self.output_moe_loss)
...
# multi-class
else:
    rets = self.model(batch[self.image_key], multimask_output=False, output_moe_loss=self.output_moe_loss)
```

2) Prompt 的生成逻辑：未提供点/框/掩码时
- 稀疏提示 `sparse_embeddings` 用零填充
- 稠密提示 `dense_embeddings` 用 `no_mask_embed`
```618:727:/root/autodl-tmp/works/autogluon/multimodal/src/autogluon/multimodal/models/custom_hf_models/modeling_sam_for_conv_lora.py
class SamPromptEncoder(nn.Module):
    ...
    def forward(...):
        ...
        if input_masks is not None:
            dense_embeddings = self.mask_embed(input_masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch_size, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        if sparse_embeddings is None:
            sparse_embeddings = torch.zeros((batch_size, 1, 1, self.hidden_size), device=target_device)
        return sparse_embeddings, dense_embeddings
```

3) 掩码解码器如何使用 prompt：稠密提示直接加到图像特征上；若稀疏提示为零，仅使用输出 tokens
```505:516:/root/autodl-tmp/works/autogluon/multimodal/src/autogluon/multimodal/models/custom_hf_models/modeling_sam_for_conv_lora.py
if sparse_prompt_embeddings.sum().item() != 0:
    tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
else:
    tokens = output_tokens
...
image_embeddings = image_embeddings + dense_prompt_embeddings
```

4) Conv‑LoRA 注入位置：只改视觉编码器注意力的线性层（qkv），不触碰 prompt 逻辑
- 预设将LoRA仅注入 `vision_encoder.*attn` 且筛选 `q`/`v`
```409:415:/root/autodl-tmp/works/autogluon/multimodal/src/autogluon/multimodal/utils/presets.py
"optim.peft": "lora",
"optim.lora.module_filter": [".*vision_encoder.*attn"],
"optim.lora.filter": ["q", "v"],
```
- Conv‑LoRA 实现（MoE‑Conv门控 + 多专家卷积分支）
```636:673:/root/autodl-tmp/works/autogluon/multimodal/src/autogluon/multimodal/models/adaptation_layers.py
class ConvLoRALinear(nn.Linear, LoRALayer):
    ...
    self.lora_moe_gating = MoEGate(M=conv_lora_expert_num, d=self.r, K=topk)
    self.lora_moe_experts = nn.ModuleList([... nn.Conv2d(..., kernel_size=3, ...) ...])
```
- qkv处启用Conv‑LoRA输出（保留接口以累积MoE损失）
```838:846:/root/autodl-tmp/works/autogluon/multimodal/src/autogluon/multimodal/models/custom_hf_models/modeling_sam_for_conv_lora.py
# qkv with shape ...
##### modify here for conv-lora
qkv = self.qkv(hidden_states)
if output_moe_loss:
    qkv, moe_loss = qkv
```

### 与论文的一致性（概述）
- 论文提出用卷积专家的LoRA分支替代纯全连接LoRA，直接作用于ViT注意力的qkv投影，增强密集特征表达以适配下游语义分割。
- Prompt侧并未设计新的生成器；本实现沿用SAM原生的PromptEncoder。语义分割训练阶段采用“无交互式提示”的默认设置，即固定的`no_mask`稠密提示 + 零稀疏提示。

- 小结
  - Conv‑LoRA不生成新的SAM prompt；训练时不提供点/框/掩码。
  - `PromptEncoder`回退为固定`no_mask_embed`稠密提示，直接加到图像特征上；稀疏提示为零。
  - Conv‑LoRA只改视觉编码器的注意力qkv投影（注入卷积MoE LoRA），与prompt机制解耦。

```python
# binary
if self.num_classes == 1:
    rets = self.model(batch[self.image_key], multimask_output=False, output_moe_loss=self.output_moe_loss)
...
# multi-class
else:
    rets = self.model(batch[self.image_key], multimask_output=False, output_moe_loss=self.output_moe_loss)
```

```python
class SamPromptEncoder(nn.Module):
    ...
    def forward(...):
        ...
        if input_masks is not None:
            dense_embeddings = self.mask_embed(input_masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                batch_size, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        if sparse_embeddings is None:
            sparse_embeddings = torch.zeros((batch_size, 1, 1, self.hidden_size), device=target_device)
        return sparse_embeddings, dense_embeddings
```

```python
if sparse_prompt_embeddings.sum().item() != 0:
    tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
else:
    tokens = output_tokens
...
image_embeddings = image_embeddings + dense_prompt_embeddings
```

```python
"optim.peft": "lora",
"optim.lora.module_filter": [".*vision_encoder.*attn"],
"optim.lora.filter": ["q", "v"],
```

```python
class ConvLoRALinear(nn.Linear, LoRALayer):
    ...
    self.lora_moe_gating = MoEGate(M=conv_lora_expert_num, d=self.r, K=topk)
    self.lora_moe_experts = nn.ModuleList([... nn.Conv2d(..., kernel_size=3, ...) ...])
```

```python
# qkv with shape ...
##### modify here for conv-lora
qkv = self.qkv(hidden_states)
if output_moe_loss:
    qkv, moe_loss = qkv
```

---

**User**

请精读整篇论文，深入理解Conv-LoRA的实现细节，找到 源码中每个与Conv-LoRA有关的步骤 与 论文中章节段落的对应关系，然后在代码里加入详细的注释说明，以便研究者可以方便地理解源码中每个步骤的作用和快速找到其对应于论文中的依据。

---

**Cursor**

I'll add a research-oriented annotation pass across Conv-LoRA related modules, mapping each implementation step to the paper’s sections, then run a lint check to ensure no issues. Starting with `adaptation_layers.py`, then `modeling_sam_for_conv_lora.py`, `sam.py`, `models/utils.py`, and a brief note in the example script.

```python
*** Begin Patch
*** Update File: /root/autodl-tmp/works/autogluon/multimodal/src/autogluon/multimodal/models/adaptation_layers.py
@@
 class ConvLoRALinear(nn.Linear, LoRALayer):
     """
-    Conv-LoRA incorporated in Linear Layer. Weights of linear layer are set to be frozen per default.
+    Conv-LoRA incorporated in Linear Layer. Weights of linear layer are set to be frozen per default.
+
+    Paper alignment (Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model, ICLR 2024):
+    - Sec. 3 (Method): Replace the standard LoRA rank-r linear branch with a spatial, convolutional expert branch
+      applied to the intermediate LoRA features with MoE-style routing (MoE-Conv).
+    - Sec. 3.2 (MoE-Conv & Gating): Use Noisy-TopK gating on globally pooled spatial features to select experts and
+      encourage balanced expert utilization via an auxiliary load-balancing loss.
+    - Sec. 3.3 (Integration into ViT/SAM): Inject Conv-LoRA at attention projections (q/k/v). Here this module is a
+      drop-in replacement for nn.Linear that augments the frozen base with trainable LoRA + MoE-Conv residual.
+    - Sec. 4 (Training): Return the MoE load-balancing loss (moe_loss) to be aggregated with task loss.
     """
 
     def __init__(
         self,
         in_features: int,
@@
         self.fan_in_fan_out = fan_in_fan_out
         # Actual trainable parameters
         if r > 0:
+            # Sec. 3: classic LoRA low-rank factors (A, B) around a frozen base linear layer
             self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
             self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
             self.scaling = self.lora_alpha / self.r
             # Freezing the pre-trained weight matrix
             self.weight.requires_grad = False
 
-            # MoE-Conv
+            # Sec. 3.2: MoE-Conv experts over spatial LoRA features
+            # topk experts are selected via Noisy-TopK gating (typically K=1)
             topk = 1
             self.lora_moe_gating = MoEGate(M=conv_lora_expert_num, d=self.r, K=topk)
             self.lora_moe_experts = nn.ModuleList([])
             self.upsample_ratios = list(range(1, conv_lora_expert_num + 1))
             for upsample_ratio in self.upsample_ratios:
                 expert = nn.Conv2d(in_channels=r, out_channels=r, kernel_size=3, stride=1, padding=1, bias=True)
                 expert.bias.data.zero_()
                 self.lora_moe_experts.append(nn.Sequential(expert, nn.GELU()))
             self.num_experts = conv_lora_expert_num
             self.multiply_by_gates = False
 
         self.reset_parameters()
         if fan_in_fan_out:
             self.weight.data = self.weight.data.T
 
     def reset_parameters(self):
         nn.Linear.reset_parameters(self)
         if hasattr(self, "lora_A"):
             # initialize A the same way as the default for nn.Linear and B to zero
             nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
             nn.init.zeros_(self.lora_B)
 
     def T(self, w):
         return w.T if self.fan_in_fan_out else w
 
     def forward(self, x: torch.Tensor):
+        # Base frozen pathway (Eq. 1-style residualization common to LoRA)
         result = F.linear(x, self.T(self.weight), bias=self.bias)
         if self.r > 0:
+            # 1) Low-rank LoRA "A" projection to r-dim features
             lora_res = self.lora_dropout(x) @ self.lora_A.T
             dim = lora_res.dim()
             if dim == 3:
                 B, L, C = lora_res.size()
                 H = W = int(math.sqrt(L))
                 lora_res = lora_res.reshape(B, H, W, C)
             else:
                 H, W = lora_res.size()[1:3]
 
-            # Calculate the gating values.
+            # 2) Sec. 3.2: MoE gating over spatial LoRA features (Noisy-TopK). Returns gates and auxiliary moe_loss.
             lora_res = lora_res.permute(0, 3, 1, 2).contiguous()
             gates, moe_loss = self.lora_moe_gating(lora_res)
 
-            # Distribute data samples to experts.
+            # 3) Dispatch spatial features to selected experts (sparse combine)
             dispatcher = SparseDispatcher(self.num_experts, gates)
             expert_inputs = dispatcher.dispatch(lora_res)
             expert_outputs = []
             for i in range(self.num_experts):
                 if len(expert_inputs[i]) == 0:
                     continue
                 upsample_ratio = self.upsample_ratios[i]
                 cur_res = expert_inputs[i]
+                # Sec. 3.2: per-expert optional up/down-sampling to diversify receptive fields
                 if upsample_ratio != 1:
                     cur_res = F.interpolate(cur_res, scale_factor=upsample_ratio, mode="bicubic")
                 cur_res = self.lora_moe_experts[i](cur_res)
                 if upsample_ratio != 1:
                     cur_res = F.interpolate(cur_res, size=(int(H), int(W)), mode="bicubic")
                 expert_outputs.append(cur_res)
 
-            # Combine data samples after processing by each expert.
+            # 4) Combine expert outputs weighted by gates (sparse add in log-space then back to tensor space)
             temp_lora_res = dispatcher.combine(expert_outputs, multiply_by_gates=self.multiply_by_gates)
             lora_res = lora_res + temp_lora_res
 
             lora_res = lora_res.permute(0, 2, 3, 1).contiguous()
             if dim == 3:
                 lora_res = lora_res.reshape(B, L, C)
+            # 5) Low-rank "B" projection back to out_features and scale (final LoRA residual)
             result += (lora_res @ self.lora_B.T) * self.scaling
 
         return result, moe_loss
 
 
 class MoEGate(nn.Module):
     def __init__(self, d, M=4, K=1, noisy_gating=True):
         """Constructor
         Args:
             d: input channel dimensionality.
             M: the number of experts.
             K: the number of chosen experts for each forward pass.
         """
         super(MoEGate, self).__init__()
         self.M = M
         self.k = K
-        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # global average pooling
+        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # global average pooling (Sec. 3.2: gating uses GAP features)
 
         self.noisy_gating = noisy_gating
 
         self.w_gate = nn.Parameter(torch.zeros(d, M), requires_grad=True)
         self.w_noise = nn.Parameter(torch.zeros(d, M), requires_grad=True)
@@
         assert self.k <= self.M
 
     def forward(self, feats, loss_coef=1e-2, noise_epsilon=1e-2):
+        """
+        Paper alignment:
+        - Sec. 3.2: Noisy-TopK gating with load-balancing objective; compute per-expert assignment probabilities
+          and an auxiliary loss encouraging balanced load and importance.
+        - Implementation details: GAP -> linear logits -> optional noise -> TopK -> Softmax weights; compute
+          importance/load statistics and CV^2-based regularization (scaled by loss_coef).
+        """
         batch_size = feats.shape[0]
 
         feats_S = self.gap(feats).view(batch_size, -1)
 
         clean_logits = feats_S @ self.w_gate
         if self.noisy_gating and self.training:
             raw_noise_stddev = feats_S @ self.w_noise
             noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
             noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
             logits = noisy_logits
         else:
             logits = clean_logits
 
         top_logits, top_indices = logits.topk(min(self.k + 1, self.M), dim=1)
         top_k_logits = top_logits[:, : self.k]
         top_k_indices = top_indices[:, : self.k]
         top_k_gates = self.softmax(top_k_logits)
         zeros = torch.zeros_like(logits, requires_grad=True).float()
         gates = zeros.scatter(1, top_k_indices, top_k_gates).to(logits.dtype)
 
         if self.noisy_gating and self.k < self.M and self.training:
             load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
         else:
             load = self._gates_to_load(gates)
 
         importance = gates.sum(0)
-        loss = self.cv_squared(importance) + self.cv_squared(load)
+        # Sec. 3.2: load-balancing loss = CV^2(importance) + CV^2(load), scaled
+        loss = self.cv_squared(importance) + self.cv_squared(load)
         loss *= loss_coef
 
         return gates, loss
@@
 class SparseDispatcher(object):
     """Helper for implementing a mixture of experts.
     The purpose of this class is to create input minibatches for the
     experts and to combine the results of the experts to form a unified
     output tensor.
+    Paper alignment:
+    - Sec. 3.2: After Noisy-TopK gating, route spatial tokens to chosen experts; combine the expert outputs
+      according to gate weights back to the original batch layout.
 
     There are two functions:
     dispatch - take an input Tensor and create input Tensors for each expert.
     combine - take output Tensors from each expert and form a combined output
       Tensor.  Outputs from different experts for the same batch element are
*** End Patch
```

```python
# Modified based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def identity(x):
    return x


class LoRALayer:
    """
    Abstract LoRA Layer.

    Parameters
    ----------
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = identity
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class IA3LoRALinear(nn.Linear, LoRALayer):
    """
    LoRA (low-rank adaptation) followed by (IA)^3 (weight rescaling) incorporated in a Linear Layer. Weights of Linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r=8,
        lora_alpha=8,
        lora_dropout: float = 0.0,
        fan_in_fan_out=False,
        merge_weights=False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        # In essence the $b$ parameter of LoRA.
        self.lora_b = nn.Parameter(torch.ones(out_features, 1))
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        self.fan_in_fan_out = fan_in_fan_out
        self.weight.requires_grad = False

        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.T(self.weight), bias=self.bias)
        if self.r > 0:
            result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

        hidden = result * self.lora_b.flatten()
        return hidden

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data /= self.lora_b.flatten()
                self.weight.data -= self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        # def T(w):
        #     return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += self.T(self.lora_B @ self.lora_A) * self.scaling
                self.weight.data *= self.lora_b.flatten()
            self.merged = True
        return hidden

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class IA3Linear(nn.Linear, LoRALayer):
    """
    (IA)^3 incorporated in a Linear Layer. Weights of Linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    scaling_rank
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Liu, Haokun and Tam, Derek and Muqeeth, Mohammed and Mohta, Jay and Huang, Tenghao and Bansal, Mohit and Raffel, Colin,
    "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning", 2022
    https://arxiv.org/pdf/2205.05638.pdf
    2. Code: https://github.com/r-three/t-few
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        merge_weights: False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self, r=4, lora_alpha=4, lora_dropout=0.0, merge_weights=merge_weights
        )  # Default arguments, only
        # In essence the $b$ parameter of LoRA.
        self.lora_b = nn.Parameter(torch.ones(out_features, 1))
        self.weight.requires_grad = False

    def forward(self, x: torch.Tensor):
        hidden = F.linear(x, self.weight, self.bias)
        hidden = hidden * self.lora_b.flatten()
        return hidden

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data /= self.lora_b.flatten()
            self.merged = False

    def eval(self):
        # def T(w):
        #     return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data *= self.lora_b.flatten()
            self.merged = True
        return hidden

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class LoRALinear(nn.Linear, LoRALayer):
    """
    LoRA incorporated in Linear Layer. Weights of linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        # def T(w):
        #     return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, self.T(self.weight), bias=self.bias)


class LoRAEmbedding(nn.Embedding, LoRALayer):
    """
    LoRA incorporated in Embedding Layer. Weights of embedding layer are set to be frozen per default.

    Parameters
    ----------
    num_embeddings
        size of the dictionary of embeddings.
    embedding_dim
         the size of each embedding vector.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0, merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(
                    x,
                    self.lora_A.T,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (after_A @ self.lora_B.T) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class LoRAMergedLinear(nn.Linear, LoRALayer):
    """
    LoRA where single nn.Linear represents more than one layer (used in some implementations of attention query/key/value projections). Weights of linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing
    r
        rank r of the low-rank decomposition
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout rate for LoRA
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out)
    merge_weights
        Merging weights during inference to reduce latency

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, "The length of enable_lora must divide out_features"
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(-1, self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
            self.merged = False

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data += self.zero_pad(T(delta_w * self.scaling))
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(
                    after_A.transpose(-2, -1), self.lora_B.unsqueeze(-1), groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result


class LoRAConv2d(nn.Conv2d, LoRALayer):
    """
    LoRA incorporated in 2d-Convolutional Layer. Weights of convolutional layer are set to be frozen per default.

    Parameters
    ----------
    in_channels
         Number of channels in the input image.
    out_channels
        Number of channels produced by the convolution.
    kernel_size
        Size of the convolving kernel.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Adding dropout to LoRA.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert type(kernel_size) is int
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * kernel_size, in_channels * kernel_size)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_channels * kernel_size, r * kernel_size)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = False

    def eval(self):
        nn.Conv2d.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.conv2d(
                x,
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return nn.Conv2d.forward(self, x)


class ConvLoRALinear(nn.Linear, LoRALayer):
    """
    Conv-LoRA incorporated in Linear Layer. Weights of linear layer are set to be frozen per default.

    Paper alignment (Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model, ICLR 2024):
    - Sec. 3 (Method): Replace standard LoRA's linear residual with a spatial, convolutional expert branch (MoE-Conv).
    - Sec. 3.2 (MoE-Conv & Gating): Noisy-TopK gating over GAP features, plus load-balancing auxiliary loss.
    - Sec. 3.3 (Integration): Inject at attention projections (q/k/v) of the vision encoder.
    - Sec. 4 (Training): Return moe_loss for aggregation with task loss.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.
    conv_lora_expert_num
        The number of experts in MoE-Conv.

    References
    ----------
    1. Zihan Zhong, Zhiqiang Tang, Tong He, Haoyang Fang, Chun Yuan,
    "Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model", 2024
    https://arxiv.org/abs/2401.17868
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        conv_lora_expert_num: Optional[int] = None,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            # Sec. 3: Low-rank LoRA factors around a frozen base linear
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            # Sec. 3.2: MoE-Conv experts with Noisy-TopK gating
            topk = 1
            self.lora_moe_gating = MoEGate(M=conv_lora_expert_num, d=self.r, K=topk)
            self.lora_moe_experts = nn.ModuleList([])
            self.upsample_ratios = list(range(1, conv_lora_expert_num + 1))
            for upsample_ratio in self.upsample_ratios:
                expert = nn.Conv2d(in_channels=r, out_channels=r, kernel_size=3, stride=1, padding=1, bias=True)
                expert.bias.data.zero_()
                self.lora_moe_experts.append(nn.Sequential(expert, nn.GELU()))
            self.num_experts = conv_lora_expert_num
            self.multiply_by_gates = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def forward(self, x: torch.Tensor):
        # Base frozen path + residual (LoRA principle)
        result = F.linear(x, self.T(self.weight), bias=self.bias)
        if self.r > 0:
            # 1) Project to rank-r LoRA space
            lora_res = self.lora_dropout(x) @ self.lora_A.T
            dim = lora_res.dim()
            if dim == 3:
                B, L, C = lora_res.size()
                H = W = int(math.sqrt(L))
                lora_res = lora_res.reshape(B, H, W, C)
            else:
                H, W = lora_res.size()[1:3]

            # 2) Sec. 3.2: Compute Noisy-TopK gates over spatial features; also returns auxiliary moe_loss.
            lora_res = lora_res.permute(0, 3, 1, 2).contiguous()
            gates, moe_loss = self.lora_moe_gating(lora_res)

            # 3) Route tokens to experts (SparseDispatcher)
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(lora_res)
            expert_outputs = []
            for i in range(self.num_experts):
                if len(expert_inputs[i]) == 0:
                    continue
                upsample_ratio = self.upsample_ratios[i]
                cur_res = expert_inputs[i]
                # Per-expert resolution change to diversify receptive fields (paper Sec. 3.2)
                if upsample_ratio != 1:
                    cur_res = F.interpolate(cur_res, scale_factor=upsample_ratio, mode="bicubic")
                cur_res = self.lora_moe_experts[i](cur_res)
                if upsample_ratio != 1:
                    cur_res = F.interpolate(cur_res, size=(int(H), int(W)), mode="bicubic")
                expert_outputs.append(cur_res)

            # 4) Combine expert outputs (weighted by gates) back to the batch layout
            temp_lora_res = dispatcher.combine(expert_outputs, multiply_by_gates=self.multiply_by_gates)
            lora_res = lora_res + temp_lora_res

            lora_res = lora_res.permute(0, 2, 3, 1).contiguous()
            if dim == 3:
                lora_res = lora_res.reshape(B, L, C)
            # 5) Project back to output dim with LoRA scale (final residual add)
            result += (lora_res @ self.lora_B.T) * self.scaling

        return result, moe_loss


class MoEGate(nn.Module):
    def __init__(self, d, M=4, K=1, noisy_gating=True):
        """Constructor
        Args:
            d: input channel dimensionality.
            M: the number of experts.
            K: the number of chosen experts for each forward pass.
        """
        super(MoEGate, self).__init__()
        self.M = M
        self.k = K
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # global average pooling

        self.noisy_gating = noisy_gating

        self.w_gate = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(d, M), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.M

    def forward(self, feats, loss_coef=1e-2, noise_epsilon=1e-2):
        """
        Paper alignment (Sec. 3.2):
        - GAP over spatial features -> linear logits (+ noise during training) -> TopK expert selection
        - Softmax over TopK logits yields gates; compute importance/load stats and CV^2-based load-balancing loss
        """
        batch_size = feats.shape[0]

        feats_S = self.gap(feats).view(batch_size, -1)

        clean_logits = feats_S @ self.w_gate
        if self.noisy_gating and self.training:
            raw_noise_stddev = feats_S @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.M), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True).float()
        gates = zeros.scatter(1, top_k_indices, top_k_gates).to(logits.dtype)

        if self.noisy_gating and self.k < self.M and self.training:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)

        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        return gates, loss

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.

    References
    ----------
    1. Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean,
    "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", 2017
    https://arxiv.org/abs/1701.06538
    2. Code: https://github.com/davidmrau/mixture-of-experts/blob/master/moe.py
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)  # [bs * num_of chosen experts, dim]
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            # stitched = stitched.mul(self._nonzero_gates)
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(-1).unsqueeze(-1))
        # zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size()[1],
            expert_out[-1].size()[2],
            expert_out[-1].size()[3],
            requires_grad=True,
            device=stitched.device,
        )
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

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

        # MLP block
        mlp_out = self.mlp(queries)
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
        #   The adapted linear layer (possibly Conv-LoRA) may return (tensor, moe_loss) when output_moe_loss=True.
        qkv = self.qkv(hidden_states)
        if output_moe_loss:
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
        layernorm_output = self.layer_norm2(hidden_states)
        hidden_states = hidden_states + self.mlp(layernorm_output)

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

```python
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import SamConfig

from ..constants import CLASS_LABEL, CLASS_LOGITS, COLUMN, IMAGE, IMAGE_VALID_NUM, LABEL, LOGITS, MASK_LABEL, MOE_LOSS
from .adaptation_layers import ConvLoRALinear
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
        """

        super().__init__()
        self.prefix = prefix
        self.pretrained = pretrained
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes
        self.frozen_layers = frozen_layers

        self.device = None
        self.name_to_id = {}

        self._load_checkpoint(checkpoint_name)

        freeze_model_layers(self.model, self.frozen_layers)

        self.image_size = self.model.vision_encoder.image_size
        self.config = self.model.config
        self.image_mean, self.image_std = image_mean_std(image_norm)

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
import functools
import logging
import re
import warnings
from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch._dynamo
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from torch import nn
from torch.nn.modules.loss import _Loss
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertTokenizer, CLIPTokenizer, ElectraTokenizer
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss

from ..constants import (
    ALL_MODALITIES,
    CATEGORICAL,
    CATEGORICAL_MLP,
    CLASS_LOGITS,
    CLIP,
    CLIP_IMAGE_MEAN,
    CLIP_IMAGE_STD,
    DOCUMENT,
    DOCUMENT_TRANSFORMER,
    FT_TRANSFORMER,
    FUSION_MLP,
    FUSION_NER,
    FUSION_TRANSFORMER,
    HF_TEXT,
    IMAGE,
    LOGITS,
    META_TRANSFORMER,
    MMDET_IMAGE,
    MMOCR_TEXT_DET,
    MMOCR_TEXT_RECOG,
    NER_TEXT,
    NUMERICAL,
    NUMERICAL_MLP,
    OCR,
    PEFT_ADDITIVE_STRATEGIES,
    REGRESSION,
    SAM,
    SEMANTIC_MASK,
    SEMANTIC_SEGMENTATION,
    SEMANTIC_SEGMENTATION_IMG,
    T_FEW,
    TEXT,
    TEXT_NER,
    TIMM_IMAGE,
)
from .adaptation_layers import ConvLoRALinear, IA3Linear, IA3LoRALinear, LoRALinear

logger = logging.getLogger(__name__)


ALL_TOKENIZERS = {
    "bert": BertTokenizer,
    "clip": CLIPTokenizer,
    "electra": ElectraTokenizer,
    "hf_auto": AutoTokenizer,
}


class DummyLayer(nn.Module):
    """
    DummyLayer to ensure that the gradient checkpointing will assign output layer as require_grad=True.
    Reference: https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
    """

    def __init__(self):
        super().__init__()
        self.dummy_bias = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def forward(self, x):
        return x + self.dummy_bias.to(x) - self.dummy_bias.to(x)


def init_weights(module: nn.Module):
    """
    Initialize one module. It uses xavier_norm to initialize nn.Embedding
    and xavier_uniform to initialize nn.Linear's weight.

    Parameters
    ----------
    module
        A Pytorch nn.Module.
    """
    if isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def assign_encoder_layer_ids(
    encoder_names: List[List[str]],
):
    """
    Assign ids to encoder layers. The encoder may contain several blocks e.g., block1 and block2.
    This function iterates through all the layers of each block from the input end towards the output end.
    It increases 1 on the layer id when the detected digit in a layer name changes.

    Parameters
    ----------
    encoder_names
        Encoder layer names.

    Returns
    -------
    name_to_id
        The encoder layer-to-id mapping.
    encoder_layer_num
        The encoder layer number.
    """
    name_to_id = {}
    cur_id = 0
    for i, group_names in enumerate(encoder_names):
        last_inferred_id = -1
        for n in group_names:
            detect_id = False
            n_splits = n.split(".")

            for split in n_splits:
                # the first digit encountered is used to infer layer id
                if split.isdigit():
                    inferred_id = int(split)
                    # increase at most 1 one time
                    if inferred_id != last_inferred_id:
                        cur_id += 1  # layer.0 -> layer_id 1
                        last_inferred_id = inferred_id

                    name_to_id[n] = cur_id
                    detect_id = True
                    break

            if detect_id is False:
                raise ValueError(f"parameter name: {n} not has no id inside")

    if len(name_to_id) > 0:
        encoder_layer_num = max(name_to_id.values())
    else:
        encoder_layer_num = 0
    return name_to_id, encoder_layer_num


def assign_non_encoder_layer_ids(
    non_encoder_names: List[str],
    layer_id: int,
):
    """
    Assign the provided id to non-encoder layers.

    Parameters
    ----------
    non_encoder_names
        Names layers not belonging to an encoder.
    layer_id
        provided id.

    Returns
    -------
    A dictionary mapping the layer names (keys) to their ids (values).
    """
    name_to_id = {}
    for n in non_encoder_names:
        name_to_id[n] = layer_id
    return name_to_id


def split_encoder_non_encoder(names: List[str], post_encoder_patterns: Tuple[str, ...]):
    """
    Group layer names into two types: encoder and non-encoder.
    A layer belongs to encoder if its name contains at least one digit.
    It uses this rule since a model's encoder in Pytorch's implementation
    is generally wrapped by nn.Sequential() or nn.ModuleList(),
    which produce digits in layer names.

    Parameters
    ----------
    names
        Model layer names.
    Returns
    -------
    encoder_names
        A list of encoder layer names.
    non_encoder_names
        A list of non-encoder layer names.
    """
    encoder_names = []
    non_encoder_names = []
    for n in names:
        is_encoder = False
        if any(p in n for p in post_encoder_patterns):
            non_encoder_names.append(n)
            continue
        for i in n.split("."):
            if i.isdigit():
                encoder_names.append(n)
                is_encoder = True
                break
        if not is_encoder:
            non_encoder_names.append(n)

    return encoder_names, non_encoder_names


def group_param_names(
    names: List[str],
    pre_encoder_patterns: Tuple[str, ...],
    post_encoder_patterns: Tuple[str, ...],
    model_prefix: Optional[str] = None,
):
    """
    Group layer names into three types: pre-encoder, encoder, and post-encoder.
    If "model_prefix" is provided, the selected layer names must start with it.
    In this case, the left names will be returned for the next-time processing.
    This function first extracts the first-level children modules' names and
    classify them into encoder and non-encoder layers. Note that an encoder may
    consist of several manually named children modules, e.g., block1 and block2.
    The non-encoder layers are further subdivided into pre-encoder and post-encoder.

    Parameters
    ----------
    names
        Model layer names
    pre_encoder_patterns
        Patterns to identify a layer as a pre-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into pre-encoder layers.
    post_encoder_patterns
        Patterns to identify a layer as a post-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into post-encoder layers.
    model_prefix
        A prefix to filter layer names. Only layer names starting with it will be selected.
    Returns
    -------
    left_names
        The layer names left for the next-time processing.
    encoder_names_grouped
        Encoder layer names.
    pre_encoder_names
        Names of layers before the encoder.
    post_encoder_names
        Names of layers after the encoder.
    """
    # two set of patterns can't have intersections
    assert all(pre_p not in post_encoder_patterns for pre_p in pre_encoder_patterns)

    left_names = []
    # in case model_prefix is provided, e.g., the clip model with image and text branches
    selected_names = []
    for n in names:
        if model_prefix is not None and not n.startswith(model_prefix):
            left_names.append(n)
        else:
            selected_names.append(n)

    # split blocks at the first level
    children_prefix = []
    for n in selected_names:
        if model_prefix is not None:
            child_name = n[len(model_prefix) + 1 :].split(".")[0]
            child_prefix = f"{model_prefix}.{child_name}"
        else:
            child_prefix = n.split(".")[0]
        if child_prefix not in children_prefix:
            children_prefix.append(child_prefix)

    encoder_names_grouped = []
    non_encoder_names = []
    for child_prefix in children_prefix:
        per_names_group = [n for n in selected_names if n.startswith(child_prefix)]
        per_encoder_names, per_non_encoder_names = split_encoder_non_encoder(per_names_group, post_encoder_patterns)
        encoder_names_grouped.append(per_encoder_names)
        non_encoder_names.extend(per_non_encoder_names)

    pre_encoder_names = []
    post_encoder_names = []
    for n in non_encoder_names:
        if any(p in n for p in pre_encoder_patterns):
            pre_encoder_names.append(n)
        elif any(p in n for p in post_encoder_patterns):
            post_encoder_names.append(n)
        else:
            raise ValueError(f"parameter name: {n} belong to neither pre or post encoder names")

    # only process left names in next iteration
    return left_names, encoder_names_grouped, pre_encoder_names, post_encoder_names


def reverse_layer_ids(
    encoder_name_to_id: dict,
    pre_enocder_name_to_id: dict,
    post_enocder_name_to_id: dict,
):
    """
    The layer ids need to increase when going from the output end to the input end.
    We need to reverse the ids which were originally assigned in a decreasing order.

    Parameters
    ----------
    encoder_name_to_id
        The layer-to-id mapping of encoder layers.
    pre_enocder_name_to_id
        The layer-to-id mapping of pre-encoder layers.
    post_enocder_name_to_id
        The layer-to-id mapping of post-encoder layers.

    Returns
    -------
    The layer-to-id mapping of all layers with layer ids reversed.
    """
    name_to_id = {**pre_enocder_name_to_id, **encoder_name_to_id, **post_enocder_name_to_id}
    if len(name_to_id) > 0:
        layer_num = max(name_to_id.values())
        # if no post encoder layers, the minimum layer id should be 1
        if len(post_enocder_name_to_id) == 0:
            layer_num += 1
    for n, layer_id in name_to_id.items():
        name_to_id[n] = layer_num - layer_id

    return name_to_id


def assign_layer_ids(
    names: List[str],
    pre_encoder_patterns: Tuple[str, ...],
    post_encoder_patterns: Tuple[str, ...],
    model_pre: Optional[str] = None,
):
    """
    Assign ids to all layers. It splits a model into three parts: pre-encoder, encoder, and post-encoder.
    Encoder is generally a stack of multiple similar layers, such as transformer layers. Since encoder is
    generally wrapped by nn.Sequential() or nn.ModuleList(), its inside layer names contain digits.
    It sets 0 as the ids of all post-encoder layers and a maximum id (layer_num) for the all the pre-encoder
    layers. The encoder layers have decreasing ids from the input to the output ends.

    Parameters
    ----------
    names
        model layer names.
    pre_encoder_patterns
        Patterns to identify a layer as a pre-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into pre-encoder layers.
    post_encoder_patterns
        Patterns to identify a layer as a post-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into post-encoder layers.
    model_pre
        The layer names' prefix. Only the layer names with this prefix will be assigned ids. The left
        layer names will be returned.

    Returns
    -------
    name_to_id
        A dictionary mapping the layer names (keys) to their ids (values).
    left_names
        The layer names not starting with the "model_pre".
    """
    try:
        left_names, encoder_names, pre_encoder_names, post_encoder_names = group_param_names(
            names=names,
            pre_encoder_patterns=pre_encoder_patterns,
            post_encoder_patterns=post_encoder_patterns,
            model_prefix=model_pre,
        )
        # add a constraint
        if len(encoder_names) == 0 and len(pre_encoder_names) != 0:
            raise ValueError(f"encoder_names is empty, but pre_encoder_names has values: {pre_encoder_names}")

        encoder_name_to_id, encoder_layer_num = assign_encoder_layer_ids(
            encoder_names=encoder_names,
        )

        pre_encoder_name_to_id = assign_non_encoder_layer_ids(non_encoder_names=pre_encoder_names, layer_id=0)

        post_encoder_name_to_id = assign_non_encoder_layer_ids(
            non_encoder_names=post_encoder_names, layer_id=encoder_layer_num + 1
        )

        name_to_id = reverse_layer_ids(
            encoder_name_to_id=encoder_name_to_id,
            pre_enocder_name_to_id=pre_encoder_name_to_id,
            post_enocder_name_to_id=post_encoder_name_to_id,
        )
    except Exception as e:
        logger.debug(
            f"When calling assign_layer_ids(), it catches exception: {e}. All the layers will use the same layer_id."
        )
        name_to_id = dict()
        left_names = names

    return name_to_id, left_names


def get_column_features(
    batch: Dict[str, torch.Tensor],
    column_name_prefix: str,
    features: torch.Tensor,
    valid_lengths: torch.Tensor,
    cls_feature: Optional[torch.Tensor] = None,
):
    """
    Index the features of one column defined by `column_name_prefix`.
    This function can be used to index both image and text features.
    The features have shape (b, n, d), where n can be the image number or
    text token number. One column corresponds to a subset of
    the n images or text tokens. One column name can only appear once in the return.

    Parameters
    ----------
    batch
        The batch input containing the feature column information, i.e., indexes.
    column_name_prefix
        The column name prefix of one modality (image or text).
    features
        The features of columns whose names starts with column_name_prefix.
    valid_lengths
        The valid image number or text token number of each sample in a batch.
    cls_feature
        The cls feature containing information from all feature columns.

    Returns
    -------
    The column features with masks. If the column has no valid features, its
    mask is 0.
    """
    column_features = {}
    feature_masks = {}

    cut_idx = len(column_name_prefix) + 1
    if cls_feature is not None:
        all_column_names = []
        # create a zero mask to do logical_or with each column's mask
        joint_mask = torch.zeros(features.shape[0]).to(features)  # (b,)
    for key in batch:
        if key.startswith(column_name_prefix):
            per_col_features = []
            per_col_masks = torch.zeros(features.shape[0]).to(features)  # (b,)
            assert batch[key].ndim == 2 and batch[key].shape[1] == 2
            for i, per_sample_col_idx in enumerate(batch[key]):
                start_idx = per_sample_col_idx[0]
                end_idx = per_sample_col_idx[1]
                if start_idx < end_idx:
                    assert end_idx <= valid_lengths[i]
                    per_col_features.append(features[i, start_idx:end_idx].mean(dim=0))
                    per_col_masks[i] = 1
                else:  # the column has no valid image/text.
                    per_col_features.append(torch.zeros_like(features[0, 0]))
                    per_col_masks[i] = 0
            column_name = key[cut_idx:]
            column_features[column_name] = torch.stack(per_col_features, dim=0)  # (b, num_features)
            feature_masks[column_name] = per_col_masks  # (b,)
            if cls_feature is not None:
                all_column_names.append(column_name)
                joint_mask = torch.logical_or(joint_mask, per_col_masks)

    # all the columns of one model's input share the model's cls feature
    if (
        cls_feature is not None and len(all_column_names) > 0
    ):  # some models', e.g, timm_image, output doesn't have the cls feature.
        # remove the individual column features since these column features not independent
        for column_name in all_column_names:
            column_features.pop(column_name)
            feature_masks.pop(column_name)

        joint_column_name = "_".join(all_column_names)
        column_features[joint_column_name] = cls_feature
        feature_masks[joint_column_name] = joint_mask.to(features)

    # print(f"column_features: {column_features}")
    return column_features, feature_masks


def create_adaptation(peft: str, layer: nn.Module, lora_r: int, lora_alpha: int, **kwargs):
    """
    Creates a model adaptation module (IA3, LoRA, IA3_LoRA) given a linear layer.

    Parameters
    ----------
    peft
        Name of the adaptation module.
    layer
       The layer the adaptation module should be applied to.
    lora_r
        The rank r of the low-rank decomposition.
    lora_alpha
        The scaling factor. Can be set to same value as r in
        most cases, as initialization is scaled already.
    filter
        Apply loRA only to linear layers filtered by name (e.g. "query.").
        If None, loRA is applied to all linear Layers in module.
    module_filter
        Apply loRA only to modules filtered by name (e.g. ".*EncDecAttention|.*DenseReluDense")
        If None, loRA is considered for all modules

    Returns
    -------
    Model with injected LoRA modules.
    """
    if "ia3_lora" in peft:
        return IA3LoRALinear(
            layer.in_features, layer.out_features, r=lora_r, lora_alpha=lora_alpha, merge_weights=False
        )
    elif "conv_lora" in peft:
        return ConvLoRALinear(
            layer.in_features,
            layer.out_features,
            r=lora_r,
            lora_alpha=lora_alpha,
            merge_weights=False,
            conv_lora_expert_num=kwargs["conv_lora_expert_num"],
        )
    elif "ia3" in peft:
        return IA3Linear(layer.in_features, layer.out_features, merge_weights=False)
    elif "lora" in peft:
        return LoRALinear(layer.in_features, layer.out_features, r=lora_r, lora_alpha=lora_alpha, merge_weights=False)
    elif peft is not None:
        raise NotImplementedError(
            f"The efficient finetuning strategy '{peft}'"
            f" is not supported. We only support"
            f" {', '.join(PEFT_ADDITIVE_STRATEGIES)}."
        )


def inject_adaptation_to_linear_layer(
    model: nn.Module,
    peft: str,
    lora_r: int = None,
    lora_alpha: int = None,
    filter: Optional[List[str]] = None,
    module_filter: Optional[List[str]] = None,
    extra_trainable_params: Optional[List[str]] = None,
    **kwargs,
) -> nn.Module:
    """
    Injects trainable adatio Low-Rank decomposition matrices (LoRA) into linear
    layers of a PyTorch model. Used for efficient fine-tuning of large
    pre-trained models.

    Parameters
    ----------
    model
        A PyTorch model.
    peft
        Efficient finetuning method that should be applied.
    lora_r
        The rank r of the low-rank decomposition.
    lora_alpha
        The scaling factor. Can be set to same value as r in
        most cases, as initialization is scaled already.
    filter
        Apply loRA only to linear layers filtered by name (e.g. "query.").
        If None, loRA is applied to all linear Layers in module.
    module_filter
        Apply loRA only to modules filtered by name (e.g. ".*EncDecAttention|.*DenseReluDense")
        If None, loRA is considered for all modules
    extra_trainable_params
        Not to apply loRA to modules filtered by name, and these modules are not frozen during training (e.g. "mask_decoder").
        If None, all the modules except for those applied loRA are frozen.
    Returns
    -------
    Model with injected LoRA modules.
    """
    # Paper alignment (Conv-LoRA, Sec. 3.3):
    # Typical usage sets module_filter to attention modules in the vision encoder (e.g., ".*vision_encoder.*attn")
    # and filter to projection names (e.g., ["q", "v"]). For "conv_lora", we create ConvLoRALinear instead of LoRA.
    for m_name, module in dict(model.named_modules()).items():
        if extra_trainable_params and any(re.match(filter_layer, m_name) for filter_layer in extra_trainable_params):
            continue
        if hasattr(model, "frozen_layers") and any(
            re.search(filter_layer, m_name) for filter_layer in model.frozen_layers
        ):
            continue
        if not module_filter or any(re.match(filter_module, m_name) for filter_module in module_filter):
            for c_name, layer in dict(module.named_children()).items():
                if not filter or any(re.match(filter_layer, c_name) for filter_layer in filter):
                    assert isinstance(
                        layer, nn.Linear
                    ), f"LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    adaptation_layer = create_adaptation(peft, layer, lora_r, lora_alpha, **kwargs)
                    adaptation_layer.weight = layer.weight
                    adaptation_layer.bias = layer.bias
                    setattr(module, c_name, adaptation_layer)

    return model  # return model to enable method chaining


def get_model_head(model: nn.Module):
    """
    Return the model's head. Different models may have different head names.

    Parameters
    ----------
    model
        A Pytorch model.

    Returns
    -------
    The model's head.
    """
    if hasattr(model, "head"):
        head = model.head  # move the head outside
    elif hasattr(model, "last_linear"):
        head = model.last_linear
    elif hasattr(model, "fc"):
        head = model.fc
    elif hasattr(model, "classifier"):
        head = model.classifier
    else:
        raise ValueError(f"Model {type(model)} doesn't have head. Need to check its implementation.")

    return head.fc if hasattr(head, "fc") else head


def get_hf_config_and_model(
    checkpoint_name: str, pretrained: Optional[bool] = True, low_cpu_mem_usage: Optional[bool] = False
):
    """
    Get a Huggingface config and model based on a checkpoint name.

    Parameters
    ----------
    checkpoint_name
        A model checkpoint name or a local path that saves a custom checkpoint.
    pretrained
         Whether using the pretrained weights. If pretrained=True, download the pretrained model.
    low_cpu_mem_usage
        Whether to turn on the optimization of reducing the peak CPU memory usage when loading the pretrained model.

    Returns
    -------
    A Huggingface config and model.
    """
    config = AutoConfig.from_pretrained(checkpoint_name)    # nosec B615

    if pretrained:
        model = AutoModel.from_pretrained(checkpoint_name, low_cpu_mem_usage=low_cpu_mem_usage) # nosec B615
    else:
        model = AutoModel.from_config(config)
    # Explicitly set the model to train mode after loading as by default it is in eval mode
    # See issue #4965
    model.train()
    return config, model


def apply_sigmoid(output: Dict):
    """
    Apply the sigmoid to logits.

    Parameters
    ----------
    output
        The model output dict.

    Returns
    -------
    The output with logits transformed by sigmoid.
    """
    for k, v in output.items():
        output[k][LOGITS] = torch.sigmoid(v[LOGITS].float())
    return output


def apply_multi_class_semantic_seg_postprocess(output: Dict):
    """
    Apply the semantic postprocessing to logits.

    Parameters
    ----------
    output
        The model output dict.

    Returns
    -------
    The output with post-proceesed semantic masks.
    """

    def semantic_inference(mask_cls, mask_pred):
        """
        Post-processing mask prediction for multi-class semantic segmentation inference based on https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py

        Args:
            mask_cls (`torch.Tensor`):
                Class logits. A tensor of shape `(num_queries, num_classes + 1)` (include the "no object" category).

            mask_pred (`torch.Tensor`):
                Mask logits. A tensor of shape `(num_queries, height, width)`.

        Returns:
            semseg (`torch.Tensor`): The processed mask prediction. A tensor of shape `(num_classes, height, width)`.

        References:
        [1] https://arxiv.org/abs/2107.06278
        [2] https://arxiv.org/abs/2112.01527
        """
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    for k, v in output.items():
        pred_classes = output[k][CLASS_LOGITS]
        pred_masks = output[k][LOGITS]
        semantic_masks = []
        for mask_cls_result, mask_pred_result in zip(
            pred_classes, pred_masks
        ):  # bs, num_q, num_class and bs, num_q, h, w
            per_sample_semantic_masks = semantic_inference(mask_cls_result, mask_pred_result)  # num_class, h, w
            semantic_masks.append(per_sample_semantic_masks)
        semantic_masks = torch.stack(semantic_masks, dim=0)
        output[k][SEMANTIC_MASK] = semantic_masks
    return output


def get_model_postprocess_fn(problem_type: str, loss_func: _Loss):
    """
    Get the postprocessing function for the model outputs.

    Parameters
    ----------
    problem_type
        The problem type, e.g., classification or regression.
    loss_func
        The loss function used in training.

    Returns
    -------
    The postprocessing function.
    """
    postprocess_func = None
    if problem_type == REGRESSION:
        if isinstance(loss_func, nn.BCEWithLogitsLoss):
            postprocess_func = apply_sigmoid
    elif problem_type == SEMANTIC_SEGMENTATION:
        if isinstance(loss_func, Mask2FormerLoss):
            postprocess_func = apply_multi_class_semantic_seg_postprocess
        else:
            postprocess_func = apply_sigmoid

    return postprocess_func


def get_mmocr_config_and_model(checkpoint_name: str):
    """
    Get an MMOCR config and model based on a checkpoint name.

    Parameters
    ----------
    checkpoint_name
        A model checkpoint name.

    Returns
    -------
    An MMOCR config and model.
    """
    from ..utils import check_if_packages_installed

    check_if_packages_installed(package_names=["mmcv"])
    try:
        import mmocr
        from mmocr.models import build_detector
    except ImportError:
        mmocr = None
    from mim.commands.download import download

    checkpoints = download(package="mmocr", configs=[checkpoint_name], dest_root=".")

    # read config files
    check_if_packages_installed(problem_type=OCR)
    config_file = checkpoint_name + ".py"
    if isinstance(config_file, str):
        config = mmcv.Config.fromfile(config_file)

    # build model and load pretrained weights
    checkpoint = checkpoints[0]
    model = build_detector(config.model, test_cfg=config.get("test_cfg"))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")
    return config, model


def lookup_mmdet_config(key, config):
    if key in config:
        return config[key]
    for subconfig in config.values():
        if isinstance(subconfig, dict):
            result = lookup_mmdet_config(key, subconfig)
            if result is not None:
                return result
        elif isinstance(subconfig, list):
            for subsubconfig in subconfig:
                if isinstance(subsubconfig, dict):
                    result = lookup_mmdet_config(key, subsubconfig)
                    if result is not None:
                        return result
    return None


def update_mmdet_config(key, value, config):
    for k, subconfig in config.items():
        if key == k:
            config[k] = value
        elif isinstance(subconfig, dict):
            update_mmdet_config(key, value, subconfig)
        elif isinstance(subconfig, list):
            for subsubconfig in subconfig:
                if isinstance(subsubconfig, dict):
                    update_mmdet_config(key, value, subsubconfig)


def run_model(model: nn.Module, batch: dict, trt_model: Optional[nn.Module] = None):
    from ..utils.onnx import OnnxModule
    from .document_transformer import DocumentTransformer
    from .fusion.fusion_mlp import MultimodalFusionMLP
    from .hf_text import HFAutoModelForTextPrediction
    from .t_few import TFewModel
    from .timm_image import TimmAutoModelForImagePrediction

    supported_models = (
        TimmAutoModelForImagePrediction,
        HFAutoModelForTextPrediction,
        MultimodalFusionMLP,
        TFewModel,
        OnnxModule,
    )
    pure_model = model
    if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
        pure_model = model._orig_mod
    if isinstance(model, nn.DataParallel):
        pure_model = model.module
    if isinstance(pure_model, OnnxModule):
        for k in batch:
            # HACK input data types in ONNX
            if batch[k].dtype == torch.int32:
                batch[k] = batch[k].to(torch.int64)
    # DocumentTransformer inherited from HFAutoModelForTextPrediction
    if (not isinstance(pure_model, DocumentTransformer)) and isinstance(pure_model, supported_models):
        input_vec = [batch[k] for k in pure_model.input_keys]
        column_names, column_values = [], []
        for k in batch.keys():
            has_image_column_prefix = isinstance(pure_model, TimmAutoModelForImagePrediction) and k.startswith(
                pure_model.image_column_prefix
            )
            has_text_column_prefix = isinstance(pure_model, HFAutoModelForTextPrediction) and k.startswith(
                pure_model.text_column_prefix
            )
            if has_image_column_prefix or has_text_column_prefix:
                column_names.append(k)
                column_values.append(batch[k])
        if column_names != [] and column_values != []:
            input_vec.append(column_names)
            input_vec.append(column_values)
        if isinstance(pure_model, OnnxModule):
            # OnnxModule doesn't support multi-gpu yet
            output_vec = pure_model(*tuple(input_vec))
        else:
            output_vec = model(*tuple(input_vec))

        output = pure_model.get_output_dict(*output_vec)
    else:
        output = model(batch)
    return output


def freeze_model_layers(model, frozen_layers):
    """
    Freeze model layers with pattern in frozen_layers.

    Parameters
    ----------
    model
        The pytorch model.
    frozen_layers
        A list of substrings of frozen layers' names.

        e.g. if frozen_layers = ["backbone", "neck"],
            all layers including "backbone" or "neck" in the name will be frozen.
    """

    if not frozen_layers:
        return

    is_frozen_layer = lambda n: any(bb in n for bb in frozen_layers)

    for n, p in model.named_parameters():
        if is_frozen_layer(n):
            p.requires_grad = False


def get_pretrained_tokenizer(
    tokenizer_name: str,
    checkpoint_name: str,
    use_fast: Optional[bool] = True,
    add_prefix_space: Optional[bool] = None,
):
    """
    Load the tokenizer for a pre-trained huggingface checkpoint.

    Parameters
    ----------
    tokenizer_name
        The tokenizer type, e.g., "bert", "clip", "electra", and "hf_auto".
    checkpoint_name
        Name of a pre-trained checkpoint.
    use_fast
        Use a fast Rust-based tokenizer if it is supported for a given model.
        If a fast tokenizer is not available for a given model, a normal Python-based tokenizer is returned instead.
        See: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer.from_pretrained.use_fast

    Returns
    -------
    A tokenizer instance.
    """
    try:
        tokenizer_class = ALL_TOKENIZERS[tokenizer_name]
        if add_prefix_space is None:
            return tokenizer_class.from_pretrained(checkpoint_name, use_fast=use_fast)
        else:
            return tokenizer_class.from_pretrained(
                checkpoint_name, use_fast=use_fast, add_prefix_space=add_prefix_space
            )
    except TypeError as e:
        try:
            tokenizer = BertTokenizer.from_pretrained(checkpoint_name)  # nosec B615
            warnings.warn(
                f"Current checkpoint {checkpoint_name} does not support AutoTokenizer. "
                "Switch to BertTokenizer instead.",
                UserWarning,
            )
            return tokenizer
        except:
            raise e


def extract_value_from_config(
    config: Dict,
    keys: Tuple[str, ...],
):
    """
    Traverse a config dictionary to get some hyper-parameter's value.

    Parameters
    ----------
    config
        A config dictionary.
    keys
        The possible names of a hyper-parameter.

    Returns
    -------
    The hyper-parameter value.
    """
    result = []
    for k, v in config.items():
        if k in keys:
            result.append(v)
        elif isinstance(v, dict):
            result += extract_value_from_config(v, keys)
        else:
            pass

    return result


def extract_image_hparams_from_config(model_name: str, config):
    """
    Extract some default hyper-parameters, e.g., image size, mean, and std,
    from a pre-trained (timm or huggingface) checkpoint.

    Parameters
    ----------
    model_name
        Name of model.
    config
        Config of a pre-trained checkpoint.

    Returns
    -------
    image_size
        Image width/height.
    mean
        Image normalization mean.
    std
        Image normalizaiton std.
    """
    if model_name.lower().startswith((TIMM_IMAGE, META_TRANSFORMER)):
        image_size = config["input_size"][-1]
        image_mean = config["mean"]
        image_std = config["std"]
    elif model_name.lower().startswith((CLIP, DOCUMENT_TRANSFORMER)):
        extracted = extract_value_from_config(
            config=config.to_diff_dict(),
            keys=("image_size",),
        )
        if len(extracted) == 0:
            image_size = None
        elif len(extracted) >= 1:
            image_size = extracted[0]
            if isinstance(image_size, tuple):
                image_size = image_size[-1]
        else:
            raise ValueError(f" more than one image_size values are detected: {extracted}")
        image_mean = None
        image_std = None
    else:
        raise ValueError(f"Unknown image processor prefix: {model_name}")
    return image_size, image_mean, image_std


def image_mean_std(norm_type: str):
    """
    Get image normalization mean and std by its name.

    Parameters
    ----------
    norm_type
        Name of image normalization.

    Returns
    -------
    Normalization mean and std.
    """
    if norm_type == "inception":
        return IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    elif norm_type == "imagenet":
        return IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    elif norm_type == "clip":
        return CLIP_IMAGE_MEAN, CLIP_IMAGE_STD
    else:
        raise ValueError(f"unknown image normalization: {norm_type}")


def get_image_size_mean_std(
    model_name: str,
    config,
    provided_size: int,
    provided_norm_type: str,
    support_variable_input_size: Optional[bool] = False,
):
    image_size, image_mean, image_std = extract_image_hparams_from_config(
        model_name=model_name,
        config=config,
    )
    if support_variable_input_size and provided_size is not None:
        # We have detected that the model supports using an image size that is
        # different from the pretrained model, e.g., ConvNets with global pooling
        if provided_size < image_size:
            logger.warning(
                f"The provided image size={provided_size} is smaller than the default size "
                f"of the pretrained backbone, which is {image_size}. "
                f"Detailed configuration of the backbone is in {config}. "
                f"You may like to double check your configuration."
            )
        image_size = provided_size
    elif provided_size is not None and provided_size != image_size:
        logger.warning(
            f"The model does not support using an image size that is different from the default size. "
            f"Provided image size={provided_size}. Default size={image_size}. "
            f"Detailed model configuration={config}. We have ignored the provided image size."
        )

    if image_size is None:
        if provided_size is not None:
            image_size = provided_size
            logger.debug(f"using provided image size: {image_size}.")
        else:
            raise ValueError("image size is missing.")
    else:
        logger.debug(f"using detected image size: {image_size}")

    if image_mean is None or image_std is None:
        if provided_norm_type is not None:
            image_mean, image_std = image_mean_std(provided_norm_type)
            logger.debug(f"using provided normalization: {provided_norm_type}.")
        else:
            raise ValueError("image normalization mean and std are missing.")
    else:
        logger.debug(f"using detected image normalization: {image_mean} and {image_std}.")

    return image_size, image_mean, image_std


def get_text_segment_num(config, provided_segment_num: int, checkpoint_name: str):
    extracted = extract_value_from_config(config=config.to_diff_dict(), keys=("type_vocab_size",))
    if len(extracted) == 0:
        default_segment_num = 1
    elif len(extracted) == 1:
        default_segment_num = extracted[0]
    else:
        raise ValueError(f" more than one type_vocab_size values are detected: {extracted}")

    if default_segment_num <= 0:
        default_segment_num = 1

    if provided_segment_num < default_segment_num:
        warnings.warn(
            f"provided text_segment_num: {provided_segment_num} "
            f"is smaller than {checkpoint_name}'s default: {default_segment_num}"
        )
    text_segment_num = min(provided_segment_num, default_segment_num)
    assert text_segment_num >= 1
    logger.debug(f"text segment num: {text_segment_num}")

    return text_segment_num


def get_text_token_max_len(provided_max_len, config, tokenizer, checkpoint_name):
    """
    Compute the allowable max length of token sequences.

    Parameters
    ----------
    provided_max_len
        The provided max length.
    config
        Model config.
    tokenizer
        Text tokenizer.
    checkpoint_name
        Name of checkpoint.

    Returns
    -------
    Token sequence max length.
    """
    if hasattr(config, "relative_attention") and config.relative_attention:
        default_max_len = tokenizer.model_max_length
    elif hasattr(config, "position_embedding_type") and "relative" in config.position_embedding_type:
        default_max_len = tokenizer.model_max_length
    elif hasattr(config, "max_position_embeddings"):
        default_max_len = config.max_position_embeddings
    else:
        default_max_len = tokenizer.model_max_length

    if provided_max_len is None or provided_max_len <= 0:
        max_len = default_max_len
    else:
        if provided_max_len < default_max_len:
            if default_max_len < 10**6:  # Larger than this value usually means infinite.
                warnings.warn(
                    f"provided max length: {provided_max_len} "
                    f"is smaller than {checkpoint_name}'s default: {default_max_len}"
                )
        max_len = min(provided_max_len, default_max_len)

    logger.debug(f"text max length: {max_len}")

    return max_len


def replace_missing_images_with_learnable(
    images: torch.Tensor,
    image_masks,
    learnable_image: nn.Parameter,
):
    b, n, c, h, w = images.shape
    assert learnable_image.shape == (c, h, w)
    for i in range(b):
        for j in range(n):
            if not image_masks[i][j]:  # False means a missing image
                images[i][j] = learnable_image

    return images


def select_model(
    config: DictConfig,
    df_preprocessor,
    strict: Optional[bool] = True,
):
    """
    Filter model config through the detected modalities in the training data.
    If MultiModalFeaturePreprocessor can't detect some modality,
    this function will remove the models that use this modality. This function is to
    maximize the user flexibility in defining the config.
    For example, if one uses the default, including hf_text and timm_image, as the model config template
    but the training data don't have images, this function will filter out timm_image.

    Parameters
    ----------
    config
        A DictConfig object. The model config should be accessible by "config.model"
    df_preprocessor
        A MultiModalFeaturePreprocessor object, which has called .fit() on the training data.
        Column names of the same modality are grouped into one list. If a modality's list is empty,
        it means the training data don't have this modality.
    strict
        If False, allow retaining one model when partial modalities are available for that model.

    Returns
    -------
    Config with some unused models removed.
    """
    data_status = {}
    for per_modality in ALL_MODALITIES:
        data_status[per_modality] = False
    if len(df_preprocessor.image_feature_names) > 0:
        data_status[IMAGE] = True
    if len(df_preprocessor.text_feature_names) > 0:
        data_status[TEXT] = True
    if len(df_preprocessor.categorical_feature_names) > 0:
        data_status[CATEGORICAL] = True
    if len(df_preprocessor.numerical_feature_names) > 0:
        data_status[NUMERICAL] = True
    if len(df_preprocessor.ner_feature_names) > 0:
        data_status[TEXT_NER] = True
    if len(df_preprocessor.document_feature_names) > 0:
        data_status[DOCUMENT] = True
    if len(df_preprocessor.semantic_segmentation_feature_names) > 0:
        data_status[SEMANTIC_SEGMENTATION_IMG] = True

    names = config.model.names
    if isinstance(names, str):
        names = [names]
    selected_model_names = []
    fusion_model_name = []
    for model_name in names:
        model_config = getattr(config.model, model_name)
        strict = getattr(model_config, "requires_all_dtypes", strict)
        if not model_config.data_types:
            fusion_model_name.append(model_name)
            continue
        model_data_status = [data_status[d_type] for d_type in model_config.data_types]
        if all(model_data_status):
            selected_model_names.append(model_name)
        else:
            if any(model_data_status) and not strict:
                selected_model_names.append(model_name)
                # update data types to be consistent with detected
                model_config.data_types = [d_type for d_type in model_config.data_types if data_status[d_type]]
            else:
                delattr(config.model, model_name)

    if len(selected_model_names) == 0:
        raise ValueError("No model is available for this dataset.")
    # only allow no more than 1 fusion model
    if len(fusion_model_name) > 1:
        raise ValueError(f"More than one fusion models `{fusion_model_name}` are detected, but only one is allowed.")

    if len(selected_model_names) > 1:
        assert len(fusion_model_name) == 1
        selected_model_names.extend(fusion_model_name)
    elif len(fusion_model_name) == 1 and hasattr(config.model, fusion_model_name[0]):
        delattr(config.model, fusion_model_name[0])

    config.model.names = selected_model_names
    logger.debug(f"selected models: {selected_model_names}")
    for model_name in selected_model_names:
        logger.debug(f"model dtypes: {getattr(config.model, model_name).data_types}")

    # clean up unused model configs
    model_keys = list(config.model.keys())
    for model_name in model_keys:
        if model_name not in selected_model_names + ["names"]:
            delattr(config.model, model_name)

    return config


def create_model(
    model_name: str,
    model_config: DictConfig,
    num_classes: Optional[int] = 0,
    classes: Optional[list] = None,
    num_numerical_columns: Optional[int] = None,
    num_categories: Optional[Dict] = None,
    numerical_fill_values: Optional[Dict] = None,
    pretrained: Optional[bool] = True,
    is_matching: Optional[bool] = False,
):
    """
    Create a single model.

    Parameters
    ----------
    model_name
        Name of the model.
    model_config
        Config of the model.
    num_classes
        The class number for a classification task. It should be 1 for a regression task.
    classes
        All classes in this dataset.
    num_numerical_columns
        The number of numerical columns in the training dataframe.
    num_categories
        The category number for each categorical column in the training dataframe.
    numerical_fill_values
        If numerical values are null, fill them with these.
    pretrained
        Whether using the pretrained timm models. If pretrained=True, download the pretrained model.
    is_matching
        Whether the model is used for semantic matching.

    Returns
    -------
    A model.
    """
    if model_name.lower().startswith(CLIP):
        from .clip import CLIPForImageText

        model = CLIPForImageText(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
            has_image=IMAGE in model_config.data_types,
            has_text=TEXT in model_config.data_types,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
            image_chan_num=model_config.image_chan_num,
            use_learnable_image=model_config.use_learnable_image,
            max_text_len=model_config.max_text_len,
            text_segment_num=model_config.text_segment_num,
            is_matching=is_matching,
        )
    elif model_name.lower().startswith(TIMM_IMAGE):
        from .timm_image import TimmAutoModelForImagePrediction

        model = TimmAutoModelForImagePrediction(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            mix_choice=model_config.mix_choice,
            pretrained=pretrained,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
            image_chan_num=model_config.image_chan_num,
            use_learnable_image=model_config.use_learnable_image,
        )
    elif model_name.lower().startswith(HF_TEXT):
        from .hf_text import HFAutoModelForTextPrediction

        model = HFAutoModelForTextPrediction(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pooling_mode=model_config.pooling_mode,
            gradient_checkpointing=model_config.gradient_checkpointing,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
            max_text_len=model_config.max_text_len,
            text_segment_num=model_config.text_segment_num,
            use_fast=model_config.use_fast,
        )
    elif model_name.lower().startswith(T_FEW):
        from .t_few import TFewModel

        model = TFewModel(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            length_norm=model_config.length_norm,  # Normalizes length to adjust for length bias in target template
            unlikely_loss=model_config.unlikely_loss,  # Adds loss term that lowers probability of incorrect outputs
            mc_loss=model_config.mc_loss,  # Adds multiple choice cross entropy loss
            num_classes=num_classes,
            gradient_checkpointing=model_config.gradient_checkpointing,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
            max_text_len=model_config.max_text_len,
            text_segment_num=model_config.text_segment_num,
        )
    elif model_name.lower().startswith(NUMERICAL_MLP):
        from .numerical_mlp import NumericalMLP

        model = NumericalMLP(
            prefix=model_name,
            in_features=num_numerical_columns,
            hidden_features=model_config.hidden_size,
            out_features=model_config.hidden_size,
            num_layers=model_config.num_layers,
            activation=model_config.activation,
            dropout=model_config.dropout,
            normalization=model_config.normalization,
            token_dim=model_config.token_dim,
            embedding_arch=model_config.embedding_arch,
            num_classes=num_classes,
            numerical_fill_values=numerical_fill_values,
        )
    elif model_name.lower().startswith(CATEGORICAL_MLP):
        from .categorical_mlp import CategoricalMLP

        model = CategoricalMLP(
            prefix=model_name,
            num_categories=num_categories,
            out_features=model_config.hidden_size,
            num_layers=model_config.num_layers,
            activation=model_config.activation,
            dropout=model_config.dropout,
            normalization=model_config.normalization,
            num_classes=num_classes,
        )
    elif model_name.lower().startswith(DOCUMENT_TRANSFORMER):
        from .document_transformer import DocumentTransformer

        model = DocumentTransformer(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pooling_mode=model_config.pooling_mode,
            gradient_checkpointing=model_config.gradient_checkpointing,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
        )
    elif model_name.lower().startswith(MMDET_IMAGE):
        from .mmdet_image import MMDetAutoModelForObjectDetection

        model = MMDetAutoModelForObjectDetection(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            config_file=model_config.config_file,
            classes=classes,
            pretrained=pretrained,
            output_bbox_format=model_config.output_bbox_format,
            frozen_layers=model_config.frozen_layers,
        )
    elif model_name.lower().startswith(MMOCR_TEXT_DET):
        from .mmocr_text_detection import MMOCRAutoModelForTextDetection

        model = MMOCRAutoModelForTextDetection(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
        )
    elif model_name.lower().startswith(MMOCR_TEXT_RECOG):
        from .mmocr_text_recognition import MMOCRAutoModelForTextRecognition

        model = MMOCRAutoModelForTextRecognition(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
        )
    elif model_name.lower().startswith(NER_TEXT):
        from .ner_text import HFAutoModelForNER

        model = HFAutoModelForNER(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            gradient_checkpointing=model_config.gradient_checkpointing,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
        )
    elif model_name.lower().startswith(FUSION_MLP):
        from .fusion import MultimodalFusionMLP

        model = functools.partial(
            MultimodalFusionMLP,
            prefix=model_name,
            hidden_features=model_config.hidden_sizes,
            num_classes=num_classes,
            adapt_in_features=model_config.adapt_in_features,
            activation=model_config.activation,
            dropout=model_config.dropout,
            normalization=model_config.normalization,
            aux_loss_weight=model_config.aux_loss_weight,
        )
    elif model_name.lower().startswith(FUSION_NER):
        from .fusion import MultimodalFusionNER

        model = functools.partial(
            MultimodalFusionNER,
            prefix=model_name,
            hidden_features=model_config.hidden_sizes,
            num_classes=num_classes,
            adapt_in_features=model_config.adapt_in_features,
            activation=model_config.activation,
            dropout_prob=model_config.drop_rate,
            normalization=model_config.normalization,
            loss_weight=model_config.weight if hasattr(model_config, "weight") else None,
        )
    elif model_name.lower().startswith(FUSION_TRANSFORMER):
        from .fusion import MultimodalFusionTransformer

        model = functools.partial(
            MultimodalFusionTransformer,
            prefix=model_name,
            hidden_features=model_config.hidden_size,
            num_classes=num_classes,
            num_blocks=model_config.num_blocks,
            attention_num_heads=model_config.attention_num_heads,
            ffn_hidden_size=model_config.ffn_hidden_size,
            attention_dropout=model_config.attention_dropout,
            residual_dropout=model_config.residual_dropout,
            ffn_dropout=model_config.ffn_dropout,
            attention_normalization=model_config.normalization,
            ffn_normalization=model_config.normalization,
            head_normalization=model_config.normalization,
            ffn_activation=model_config.ffn_activation,
            head_activation=model_config.head_activation,
            adapt_in_features=model_config.adapt_in_features,
            aux_loss_weight=model_config.aux_loss_weight,
            additive_attention=model_config.additive_attention,
            share_qv_weights=model_config.share_qv_weights,
        )
    elif model_name.lower().startswith(FT_TRANSFORMER):
        from .ft_transformer import FT_Transformer

        model = FT_Transformer(
            prefix=model_name,
            num_numerical_columns=num_numerical_columns,
            num_categories=num_categories,
            numerical_fill_values=numerical_fill_values,
            embedding_arch=model_config.embedding_arch,
            token_dim=model_config.token_dim,
            hidden_size=model_config.hidden_size,
            hidden_features=model_config.hidden_size,
            num_classes=num_classes,
            num_blocks=model_config.num_blocks,
            attention_num_heads=model_config.attention_num_heads,
            attention_dropout=model_config.attention_dropout,
            attention_normalization=model_config.normalization,
            ffn_hidden_size=model_config.ffn_hidden_size,
            ffn_dropout=model_config.ffn_dropout,
            ffn_normalization=model_config.normalization,
            ffn_activation=model_config.ffn_activation,
            residual_dropout=model_config.residual_dropout,
            head_normalization=model_config.normalization,
            head_activation=model_config.head_activation,
            additive_attention=model_config.additive_attention,
            share_qv_weights=model_config.share_qv_weights,
            pooling_mode=model_config.pooling_mode,
            checkpoint_name=model_config.checkpoint_name,
            pretrained=pretrained,
        )
    elif model_name.lower().startswith(SAM):
        from .sam import SAMForSemanticSegmentation

        model = SAMForSemanticSegmentation(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pretrained=pretrained,
            frozen_layers=model_config.frozen_layers,
            num_mask_tokens=model_config.num_mask_tokens,
            image_norm=model_config.image_norm,
        )
    elif model_name.lower().startswith(META_TRANSFORMER):
        from .meta_transformer import MetaTransformer

        model = MetaTransformer(
            prefix=model_name,
            checkpoint_path=model_config.checkpoint_path,
            num_classes=num_classes,
            model_version=model_config.model_version,
            has_image=IMAGE in model_config.data_types,
            has_text=TEXT in model_config.data_types,
            num_numerical_columns=num_numerical_columns,
            num_categories=num_categories,
            numerical_fill_values=numerical_fill_values,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
            image_chan_num=model_config.image_chan_num,
            use_learnable_image=model_config.use_learnable_image,
            max_text_len=model_config.max_text_len,
            text_segment_num=model_config.text_segment_num,
        )
    else:
        raise ValueError(f"unknown model name: {model_name}")

    return model


def create_fusion_model(
    config: DictConfig,
    num_classes: Optional[int] = None,
    classes: Optional[list] = None,
    num_numerical_columns: Optional[int] = None,
    num_categories: Optional[Dict] = None,
    numerical_fill_values: Optional[Dict] = None,
    pretrained: Optional[bool] = True,
):
    """
    Create models. It supports the auto models of huggingface text and timm image.
    Multimodal models, e.g., CLIP, should be added case-by-case since their configs and usages
    may be different. It uses MLP for the numerical features, categorical features, and late-fusion.

    Parameters
    ----------
    config
        A DictConfig object. The model config should be accessible by "config.model".
    num_classes
        The class number for a classification task. It should be 1 for a regression task.
    classes
        All classes in this dataset.
    num_numerical_columns
        The number of numerical columns in the training dataframe.
    num_categories
        The category number for each categorical column in the training dataframe.
    numerical_fill_values
        If numerical values are null, fill them with these.
    pretrained
        Whether using the pretrained timm models. If pretrained=True, download the pretrained model.

    Returns
    -------
    A Pytorch model.
    """
    names = config.model.names
    if isinstance(names, str):
        names = [names]
    # make sure no duplicate model names
    assert len(names) == len(set(names))
    logger.debug(f"output_shape: {num_classes}")
    names = sorted(names)
    config.model.names = names
    single_models = []
    fusion_model = None

    for model_name in names:
        model_config = getattr(config.model, model_name)
        model = create_model(
            model_name=model_name,
            model_config=model_config,
            num_classes=num_classes,
            classes=classes,
            num_numerical_columns=num_numerical_columns,
            num_categories=num_categories,
            numerical_fill_values=numerical_fill_values,
            pretrained=pretrained,
        )

        if isinstance(model, functools.partial):  # fusion model
            if fusion_model is None:
                fusion_model = model
            else:
                raise ValueError(
                    f"More than one fusion models are detected in {names}. Only one fusion model is allowed."
                )
        else:  # single model
            if config.optim.peft is not None:
                model = apply_peft_adaptation(model, config)
            single_models.append(model)

    if len(single_models) > 1:
        # must have one fusion model if there are multiple independent models
        model = fusion_model(models=single_models)
    elif len(single_models) == 1:
        model = single_models[0]
    else:
        raise ValueError(f"No available models for {names}")

    # build augmenter for multimodal data augmentation
    if config.optim.lemda.turn_on:
        from .fusion import MultimodalFusionMLP

        assert isinstance(model, MultimodalFusionMLP)
        from .augmenter import Augmenter

        augmenter = Augmenter(
            arch_type=config.optim.lemda.arch_type,
            input_dim=model.augmenter_in_features,
            z_dim=config.optim.lemda.z_dim,
            num_layers=config.optim.lemda.num_layers,
            adv_weight=config.optim.lemda.adv_weight,
        )
        model.augmenter = augmenter

    return model


def apply_peft_adaptation(model: nn.Module, config: DictConfig) -> nn.Module:
    """
    Apply an adaptation to the model for efficient fine-tuning.

    Parameters
    ----------
    model
        A PyTorch model.
    config:
        A DictConfig object. The optimization config should be accessible by "config.optimization".
    """
    if config.optim.peft in PEFT_ADDITIVE_STRATEGIES:
        model = inject_adaptation_to_linear_layer(
            model=model,
            peft=config.optim.peft,
            lora_r=config.optim.lora.r,
            lora_alpha=config.optim.lora.alpha,
            module_filter=config.optim.lora.module_filter,
            filter=config.optim.lora.filter,
            extra_trainable_params=config.optim.extra_trainable_params,
            conv_lora_expert_num=config.optim.lora.conv_lora_expert_num,
        )
        model.name_to_id = model.get_layer_ids()  # Need to update name to id dictionary.

    return model


def modify_duplicate_model_names(
    learner,
    postfix: str,
    blacklist: List[str],
):
    """
    Modify a learner's model names if they exist in a blacklist.

    Parameters
    ----------
    learner
        A BaseLearner object.
    postfix
        The postfix used to change the duplicate names.
    blacklist
        A list of names. The provided learner can't use model names in the list.

    Returns
    -------
    The learner guaranteed has no duplicate model names with the blacklist names.
    """
    model_names = []
    for n in learner._config.model.names:
        if n in blacklist:
            new_name = f"{n}_{postfix}"
            assert new_name not in blacklist
            assert new_name not in learner._config.model.names
            # modify model prefix
            if n == learner._model.prefix:
                learner._model.prefix = new_name
            else:
                assert isinstance(learner._model.model, nn.ModuleList)
                for per_model in learner._model.model:
                    if n == per_model.prefix:
                        per_model.prefix = new_name
                        break
            # modify data processor prefix
            for per_modality_processors in learner._data_processors.values():
                for per_processor in per_modality_processors:
                    if n == per_processor.prefix:
                        per_processor.prefix = new_name
            # modify model config keys
            setattr(learner._config.model, new_name, getattr(learner._config.model, n))
            delattr(learner._config.model, n)

            model_names.append(new_name)
        else:
            model_names.append(n)

    learner._config.model.names = model_names

    return learner


def list_timm_models(pretrained=True):
    return timm.list_models(pretrained=pretrained)


def is_lazy_weight_tensor(p: torch.Tensor) -> bool:
    from torch.nn.parameter import UninitializedParameter

    if isinstance(p, UninitializedParameter):
        warnings.warn(
            "A layer with UninitializedParameter was found. "
            "Thus, the total number of parameters detected may be inaccurate."
        )
        return True
    return False

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

I added paper-aligned annotations across the Conv-LoRA codepath and verified no lint errors. Next I’ll summarize where each step in the code ties back to the paper’s sections.

- Completed: annotated `ConvLoRALinear`, `MoEGate`, `SparseDispatcher`.
- Completed: annotated SAM integration points in attention qkv and MoE loss propagation.
- Completed: annotated `SAMForSemanticSegmentation` around MoE loss usage.
- Completed: annotated adaptation injection path in `models/utils.py`.
- Completed: brief comments in the example script to map r and M to the paper.
- Completed: lints run; no issues.

Key changes and where to look
- Conv-LoRA core with paper mapping
  - File: `multimodal/src/autogluon/multimodal/models/adaptation_layers.py`
  - What’s annotated:
    - `ConvLoRALinear` docstring and forward pass: Sec. 3 (method), Sec. 3.2 (MoE-Conv + Noisy-TopK gating), Sec. 3.3 (integration), Sec. 4 (MoE load-balancing loss).
    - Inline comments for steps: LoRA A/B factors, spatial reshape to HxW, gating, expert routing with optional up/down sampling, gated combine, projection back via B and residual add.
    - `MoEGate.forward`: Sec. 3.2 alignment, GAP -> noisy logits -> TopK -> gates + CV^2 loss.
    - `SparseDispatcher` class docstring: used for expert routing/combination in Sec. 3.2.

- SAM attention integration points
  - File: `multimodal/src/autogluon/multimodal/models/custom_hf_models/modeling_sam_for_conv_lora.py`
  - What’s annotated:
    - `SamVisionAttention.forward`: comment at qkv showing the integration hook for Conv-LoRA (Sec. 3.3) and how `moe_loss` is returned when requested; note on propagating this loss upward (Sec. 4).
    - `SamVisionLayer.forward`: notes on returning the per-layer MoE loss.
    - `SamVisionEncoder.forward`: notes on accumulating `all_moe_loss` (Sec. 4).

- SAM wrapper: enabling and exposing MoE loss
  - File: `multimodal/src/autogluon/multimodal/models/sam.py`
  - What’s annotated:
    - Class docstring: Conv-LoRA training considerations, Sec. 4.
    - `train`: toggles `output_moe_loss=True` if `ConvLoRALinear` is present (Sec. 4).
    - Forward output dict already propagates `MOE_LOSS`; comment added near merge to explain.

- Adaptation injection path (how Conv‑LoRA is applied)
  - File: `multimodal/src/autogluon/multimodal/models/utils.py`
  - What’s annotated:
    - In `inject_adaptation_to_linear_layer`: mapping to Sec. 3.3, typical filters for attention projections (`.*vision_encoder.*attn` and `q`/`v`), and switching to `ConvLoRALinear` when `optim.peft=conv_lora`.

- Example script hyperparameters mapping
  - File: `examples/automm/Conv-LoRA/run_semantic_segmentation.py`
  - What’s annotated:
    - Inline comment where hyperparameters are set: `optim.peft` (Sec. 3), `optim.lora.r` (rank r), `optim.lora.conv_lora_expert_num` (experts M, Sec. 3.2).

- No changes to prompt handling (FYI)
  - Prompt mechanism remains SAM’s default; training uses no explicit points/boxes/masks. This is unchanged; the annotations focus on Conv‑LoRA injection and training.

- No linter issues introduced.

- Where each step ties to the paper
  - Sec. 3 (Method): `ConvLoRALinear` docstring and A/B low‑rank path in `forward`.
  - Sec. 3.2 (MoE‑Conv & Gating): `ConvLoRALinear` forward (gating, experts, up/down sampling, combine), `MoEGate`, `SparseDispatcher`.
  - Sec. 3.3 (Integration): `SamVisionAttention.forward` qkv hook and note; `inject_adaptation_to_linear_layer` filter comments.
  - Sec. 4 (Training): `SamVisionAttention` returns `moe_loss`; `SamVisionLayer`/`SamVisionEncoder` accumulate it; `SAMForSemanticSegmentation.train` toggles it; wrapper merges `MOE_LOSS` into outputs.

If you want, I can also add a brief developer README snippet in `examples/automm/Conv-LoRA/README.md` summarizing these code pointers for quick navigation.

