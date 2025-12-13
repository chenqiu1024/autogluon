# ISIC2017 SAM segmentation with predicted bboxes
_Exported on 12/13/2025 at 15:29:51 PST from Cursor (2.2.20)_

---

**User**

通过 docs/cursor_bbox_predict-251211.md 恢复之前的会话，然后继续我们新的对话。我现在想知道，如果使用我们已经训练好的 预测bounding box 的模型（使用的训练命令是`nohup python3 run_bbox_predictor.py   \
--data_dir datasets/isic2017/isic2017   \
--output_dir outputs/bbox_predict-251212   \
--epochs 60   --pretrained_backbone   --export_preds   --image_size 320   \
--num_workers 4   > outputs/bbox_predict-251212.log 2>&1 &`），通过输入待分割图像来预测得到bbox，再把这个bbox输入未经任何改进的SAM（不使用任何Adapter，也包括不使用本项目所基于的Conv-LoRA方法，仅使用原始的SAM）去做分割，那么在ISIC2017 测试集上能够获得的平均分割性能是多少（DICE）？请为此写出脚本来实现它，告诉我如何使用

---

**Cursor**

已新增评估脚本 `examples/automm/Conv-LoRA/eval_sam_with_pred_bbox.py`：用已训练的 bbox 模型生成 box prompt，再用“原始 SAM（无 Adapter / 无 Conv-LoRA）”在 ISIC2017 测试集计算平均 Dice。脚本支持直接读取已有 bbox CSV，或在线调用 `BBoxPromptPredictor` 预测框。

**依赖/准备**
- 本地 SAM 权重（如 `sam_vit_h_4b8939.pth`），安装 `segment-anything`。
- 已训练好的 bbox 模型（如 `outputs/bbox_predict-251212/best_bbox.pth`），或对应的预测 CSV。
- 数据集 CSV 位于 `datasets/isic2017/isic2017/test.csv`，含 `image,label` 列。

**运行示例**
1) 训练 bbox（你已有同款命令，仅供确认）：
```
nohup python3 run_bbox_predictor.py \
  --data_dir datasets/isic2017/isic2017 \
  --output_dir outputs/bbox_predict-251212 \
  --epochs 60 --pretrained_backbone --export_preds \
  --image_size 320 --num_workers 4 \
  > outputs/bbox_predict-251212.log 2>&1 &
```
2) 用原始 SAM 评估（在线预测框，保存框 CSV 与掩码）：
```
python3 examples/automm/Conv-LoRA/eval_sam_with_pred_bbox.py \
  --data_dir datasets/isic2017/isic2017 \
  --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
  --sam_model_type vit_h \
  --bbox_model_ckpt outputs/bbox_predict-251212/best_bbox.pth \
  --bbox_image_size 320 \
  --save_pred_csv outputs/bbox_predict-251212/test_bbox_preds.csv \
  --save_mask_dir outputs/bbox_predict-251212/sam_masks
```
3) 若已有 bbox 预测 CSV（离线生成）：
```
python3 examples/automm/Conv-LoRA/eval_sam_with_pred_bbox.py \
  --data_dir datasets/isic2017/isic2017 \
  --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
  --sam_model_type vit_h \
  --bbox_preds_csv outputs/bbox_predict-251212/test_bbox_preds.csv
```
执行后会输出“测试集平均 Dice: X.XXXX（样本数 N）”，即你要的平均分割性能。当前未在本地跑实测，请按上面命令结合你的 SAM 权重与数据集执行即可获得具体 Dice。

```plaintext
nohup python3 run_bbox_predictor.py \
  --data_dir datasets/isic2017/isic2017 \
  --output_dir outputs/bbox_predict-251212 \
  --epochs 60 --pretrained_backbone --export_preds \
  --image_size 320 --num_workers 4 \
  > outputs/bbox_predict-251212.log 2>&1 &
```

```plaintext
python3 examples/automm/Conv-LoRA/eval_sam_with_pred_bbox.py \
  --data_dir datasets/isic2017/isic2017 \
  --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
  --sam_model_type vit_h \
  --bbox_model_ckpt outputs/bbox_predict-251212/best_bbox.pth \
  --bbox_image_size 320 \
  --save_pred_csv outputs/bbox_predict-251212/test_bbox_preds.csv \
  --save_mask_dir outputs/bbox_predict-251212/sam_masks
```

```plaintext
python3 examples/automm/Conv-LoRA/eval_sam_with_pred_bbox.py \
  --data_dir datasets/isic2017/isic2017 \
  --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
  --sam_model_type vit_h \
  --bbox_preds_csv outputs/bbox_predict-251212/test_bbox_preds.csv
```

---

**User**

目前我们是在推理（预测masks）时加入box prompt以提升性能，作为与TTA并存的一种推理时增强方法。那么如果我们在训练时（基于目前的包含LoRA on Attention, FC Bottle-neck Adapters, Conv-LoRA, GSPO 的方案）也对SAM加入Box prompts，是否会有什么有益作用？为什么？

---

**Cursor**

### 结论
**在训练阶段也喂 Box Prompt 往往是“有益的”，前提是训练时的 prompt 分布要尽量接近推理时（尤其是你现在用的是“预测框”而不是 GT 框）。**否则很容易出现“训练用 GT 框很强、推理用预测框变弱”的分布偏移，收益会被抵消甚至变差。

---

### 为什么训练时加 Box Prompt 可能带来收益？
- **减少任务歧义、让模型把容量用在“精细分割”而不是“找目标”**  
  ISIC2017 这类数据里，模型既要定位病灶又要分割边界。Box prompt 相当于给了强定位先验，训练时会显著降低“背景误检/多目标混淆”的梯度噪声，使 LoRA/Adapter/Conv-LoRA 更专注学习边界细节、纹理与形状细化，从而更容易提升 Dice。

- **让训练-推理条件一致（如果推理确实要用 box prompt）**  
  你现在把“box prompt + TTA”当推理增强。如果训练时完全没见过 box prompt，推理时突然加 prompt，模型内部对 prompt embedding 的利用可能是“非最优/不稳定”的；训练时加入 prompt，会让模型学会“如何正确使用 prompt 信息”，通常能提高稳定性和上限。

- **对小目标/低对比目标尤其有效**  
  病灶很小或边界弱时，box prompt 能提高召回、减少模型错过目标导致的 Dice 大幅下降；训练时加入这种先验，等价于强化“目标存在且在此区域”的学习信号。

- **对你现有的参数高效微调模块（LoRA/Adapter/Conv-LoRA/GSPO）更友好**  
  这些方法本质是在有限可训练参数下提升能力。box prompt 把“定位”外包给提示输入，**降低需要微调参数去学习定位的压力**，让有限的可训练参数更有效地提升分割质量；同时也可能降低 GSPO 这类优化中的回报方差，使训练更稳。

---

### 主要风险：GT Box 训练会“过于乐观”
- **推理时你用的是预测框（带噪声）**，而训练时若用的是从 GT mask 算出来的“完美框”，模型会形成依赖：  
  **框一旦偏移/偏小/偏大，分割就明显退化**（分布偏移）。
- **如果未来某些场景没有 box prompt**，纯 box 训练也可能降低“无 prompt 推理”的鲁棒性。

---

### 实践建议（怎么做才更可能真的提升推理 Dice）
- **混合训练（强烈推荐）**：同一批次/同一样本，随机切换 prompt 模式  
  - `p=0.3`：无 box（保持基础能力）  
  - `p=0.3`：GT box（上限/收敛更快）  
  - `p=0.4`：噪声 box / 预测 box（对齐真实推理）
- **对 box 做抖动增强**（模拟预测框误差）：随机平移、缩放、扩张/收缩、轻微长宽扰动；并设置最小框尺寸，避免框塌缩。
- **若已有 bbox predictor**：用它在训练集/验证集生成预测框，训练分割时直接喂“预测框”（或 teacher-student：预测框 + 少量 GT 框混合）。

---

### 你应该期待什么变化？
- **在“推理也用 box prompt（尤其是预测框）”的前提下**：通常会看到 **Dice 更稳、下限更高**，并且对困难样本（小/弱对比）提升更明显。  
- **若只用 GT box 训练但推理用预测框**：可能出现 **val/test Dice 不升反降**（典型的 prompt 分布偏移）。

如果你告诉我你现在推理时 box prompt 的策略是 `add` 还是 `replace`、以及预测框的平均 IoU 大概多少，我可以更具体地判断“该偏向 GT/噪声/预测框”哪一种训练配比更合适。

---

**User**

我不明白：box prompt 的两种策略 add 和 replace 的区别是什么？在代码中是如何体现的？

---

**Cursor**

我先在仓库里定位 `tta_box_prompt_mode` / `box_prompt_mode` 的具体实现位置，看看 `add` 和 `replace` 在代码路径里到底如何影响 SAM 的输入（是叠加到原有 prompt，还是直接替换）。

### `add` vs `replace` 在概念上是什么？
在“提示（prompt）”语境里，这两个词通常意味着：

- **add**：把 **box prompt 作为额外提示**加入（与已有提示并存，比如已有 points / mask prompt）。
- **replace**：用 **box prompt 替换掉原有提示**（只用 box，不用原来的 points / mask prompt）。

但这只是“语义上可能的设计意图”。

---

### 在你们当前代码里，`add` 和 `replace` 实际有没有区别？
**几乎没有区别：当前实现里 `add` 和 `replace` 走的是同一条代码分支，效果相同；唯一不同的是 `off` 会禁用 box prompt。**

关键点：在 TTA 推理的 `predict_fn` 中，只要模式是 `add` 或 `replace`，就把 box 塞进 batch；没有任何地方区分两者的后续行为。

你可以直接看这里（`SemanticSegmentationLearner` 的 TTA 推理函数）：

```385:430:multimodal/src/autogluon/multimodal/learners/semantic_segmentation.py
        def predict_fn(image: np.ndarray, box_prompt: Optional[np.ndarray] = None, box_ref_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
            ...
            box_tensor = None
            if box_prompt is not None:
                ...
                box_tensor = torch.from_numpy(box_scaled).to(device).unsqueeze(0).unsqueeze(0)  # (1,1,4)
            ...
                batch = {
                    model.prefix + '_image': img_tensor,
                    model.prefix + '_label': torch.zeros((1, model.image_size, model.image_size),
                                                         dtype=torch.long, device=device)
                }
                if box_tensor is not None and self._tta_box_prompt_mode in ["add", "replace"]:
                    batch[model.box_key] = box_tensor
                outputs = model(batch)
```

可以看到：**`["add","replace"]` 并列出现，没有进一步分支**。

---

### box prompt 最终是怎么喂给 SAM 的？
`batch[model.box_key]` 最终变成 HF SAM 的 `input_boxes` 参数：

```405:456:multimodal/src/autogluon/multimodal/models/sam.py
    @property
    def box_key(self):
        return f"{self.prefix}_box"

    def forward(self, batch):
        ...
        rets = self.model(
            batch[self.image_key],
            input_boxes=batch.get(self.box_key, None),
            multimask_output=False,
            output_moe_loss=self.output_moe_loss,
        )
```

所以在当前实现下：

- **`off`**：不会放 `batch[model.box_key]`，等价于 `input_boxes=None`（无 box prompt）。
- **`add` / `replace`**：都会放 `batch[model.box_key]`，等价于 `input_boxes=...`（有 box prompt），**两者无差别**。

---

### CLI 里为什么会出现 `add/replace`？
只是参数层面暴露了这三个选项（但实现只用到了“是否开启”这一层）。比如：

```108:111:examples/automm/Conv-LoRA/run_semantic_segmentation.py
    parser.add_argument("--tta_box_prompt_mode", type=str, default="off",
                        choices=["off", "add", "replace"],
                        help="Use GT box as SAM box prompt (off/add/replace); box not transformed with TTA")
```

---

如果你希望 **`add` 和 `replace` 真正有区别**，就需要在“模型输入提示”层面同时存在两类提示（例如 points + box），然后在 forward 前决定是“同时提供”还是“只提供 box”。当前这条管线里基本只有 box（或无 box），所以自然区分不出来。

```python
        def predict_fn(image: np.ndarray, box_prompt: Optional[np.ndarray] = None, box_ref_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
            ...
            box_tensor = None
            if box_prompt is not None:
                ...
                box_tensor = torch.from_numpy(box_scaled).to(device).unsqueeze(0).unsqueeze(0)  # (1,1,4)
            ...
                batch = {
                    model.prefix + '_image': img_tensor,
                    model.prefix + '_label': torch.zeros((1, model.image_size, model.image_size),
                                                         dtype=torch.long, device=device)
                }
                if box_tensor is not None and self._tta_box_prompt_mode in ["add", "replace"]:
                    batch[model.box_key] = box_tensor
                outputs = model(batch)
```

```python
    @property
    def box_key(self):
        return f"{self.prefix}_box"

    def forward(self, batch):
        ...
        rets = self.model(
            batch[self.image_key],
            input_boxes=batch.get(self.box_key, None),
            multimask_output=False,
            output_moe_loss=self.output_moe_loss,
        )
```

```python
    parser.add_argument("--tta_box_prompt_mode", type=str, default="off",
                        choices=["off", "add", "replace"],
                        help="Use GT box as SAM box prompt (off/add/replace); box not transformed with TTA")
```

---

**User**

所以目前的实现里不管add还是replace，最终效果都相当于是add，对吗？这应该是一个你之前实现中的疏漏，请把两种模式都实现出来

---

**Cursor**

我会先把当前 TTA 评估里 “box prompt 从哪里来、是否随增广变换” 这条链路梳理清楚，然后把 `add` 与 `replace` 的行为落到代码上：`add`=在不改变现有 prompt 的前提下额外提供 box；`replace`=强制只用 box（并禁用/清空其他 prompt）。接着补一条最小单测/自检路径，确保两种模式输出确实不同。

我已经把 `add` / `replace` 的行为在 TTA 推理里真正区分开了：`replace` 只用 box-prompt 的一次前向；`add` 会做两次前向（无 box + 有 box）并把概率图做均值融合，相当于把“box prompt 推理”当作额外视角加入到融合里。接下来我会跑一下该文件的静态检查，确保没引入语法/类型错误。

已实现，现在你说的判断是对的：**之前 `add`/`replace` 实际效果等价**，确实是疏漏；我已把两种模式区分开。

### 现在 `add` / `replace` 的真实含义
- **`off`**：不传 box prompt（完全无框提示）。
- **`replace`**：**只用 box prompt 的预测**（有框就用框；相当于“用框替换原本无框的推理”）。
- **`add`**：把 box prompt 当成“额外推理视角”：对同一张图做
  - 一次 **无 box** 前向
  - 一次 **有 box** 前向  
  然后对两次输出的概率图做 **均值融合**，再进入后续 TTA 融合。

### 代码体现位置
修改在 `SemanticSegmentationLearner._evaluate_with_tta()` 里定义的 `predict_fn`，核心逻辑现在是：

- `replace`：`prob = forward_prob(box_tensor)`
- `add`：`prob = (forward_prob(None) + forward_prob(box_tensor)) / 2`

文件：`multimodal/src/autogluon/multimodal/learners/semantic_segmentation.py`

你可以直接用原来的命令行参数验证差异，例如：

- `--tta_box_prompt_mode replace`
- `--tta_box_prompt_mode add`

两者现在会产生**不同的推理行为**（`add` 计算量约增加 1 倍，因为每个 TTA 视角要多跑一次无框前向）。

---

**User**

回到训练时添加bbox prompt的话题：推理时 box prompt 的策略是 add , 预测框的平均 DICE 大概0.6442(在ISIC2017 test集上)，请更具体地判断“该偏向 GT/噪声/预测框”哪一种训练配比更合适，并且实现出相应的脚本代码，告诉我应该执行什么命令以启动训练。

---

**Cursor**

已实现训练阶段的 box prompt 混合策略（写在 `multimodal/src/autogluon/multimodal/optim/lit_semantic_seg.py` 的 `_shared_step` 内）：  
- 仅在 **训练** 时启用，验证/测试不受影响。  
- 批级随机策略（基于你提供的预测框 Dice≈0.644，推理 `add`）：  
  - `p(no prompt)=0.2`  
  - `p(GT box)=0.3`  
  - `p(noisy box)=0.5`（噪声幅度 12% bbox 宽高，等比例平移/缩放后裁剪）  
- 同一批次采用同一种模式，避免批内不一致。  
- box 直接来自当前 batch 的 mask（已含数据增广后的尺寸），无额外依赖。

### 为什么这样配比？
- 推理用 `add`，预测框 Dice≈0.64：训练时要 **主要对齐“有噪声的框”场景**（0.5），同时保留部分 **理想框**（0.3）提升上限，并保留 **无框**（0.2）维持鲁棒性和无 prompt 场景的性能。

### 训练命令示例（沿用原脚本，无需新增参数）
```bash
python3 examples/automm/Conv-LoRA/run_semantic_segmentation.py \
  --task isic2017 \
  --output_dir outputs/seg_train_box_mix \
  --rank 3 --expert_num 8 \
  --batch_size 4 --per_gpu_batch_size 1 \
  --num_gpus 1
```
（其余超参按你现有习惯补齐即可）

### 推理/评估（保持你当前策略：预测框 + `add`）
- 若在线用 bbox 模型预测框：
```bash
python3 examples/automm/Conv-LoRA/run_semantic_segmentation.py \
  --task isic2017 --eval \
  --ckpt_path <SEG_CKPT> \
  --tta_enable \
  --tta_box_prompt_mode add \
  --bbox_prompt_source predict \
  --bbox_model_ckpt outputs/bbox_predict-251212/best_bbox.pth \
  --bbox_model_image_size 320
```
- 若已有 bbox 预测 CSV：
```bash
python3 examples/automm/Conv-LoRA/run_semantic_segmentation.py \
  --task isic2017 --eval \
  --ckpt_path <SEG_CKPT> \
  --tta_enable \
  --tta_box_prompt_mode add \
  --bbox_prompt_source predict \
  --bbox_preds_csv outputs/bbox_predict-251212/test_bbox_preds.csv
```

### 代码要点（训练侧）
- 位置：`multimodal/src/autogluon/multimodal/optim/lit_semantic_seg.py`  
- 逻辑：在训练批次中按上述概率选择 no/gt/noisy 三种模式，计算/扰动 bbox 后塞入 `batch[self.model.box_key]`，SAM 前向即可使用。验证/推理不改动。

```shellscript
python3 examples/automm/Conv-LoRA/run_semantic_segmentation.py \
  --task isic2017 \
  --output_dir outputs/seg_train_box_mix \
  --rank 3 --expert_num 8 \
  --batch_size 4 --per_gpu_batch_size 1 \
  --num_gpus 1
```

```shellscript
python3 examples/automm/Conv-LoRA/run_semantic_segmentation.py \
  --task isic2017 --eval \
  --ckpt_path <SEG_CKPT> \
  --tta_enable \
  --tta_box_prompt_mode add \
  --bbox_prompt_source predict \
  --bbox_model_ckpt outputs/bbox_predict-251212/best_bbox.pth \
  --bbox_model_image_size 320
```

```shellscript
python3 examples/automm/Conv-LoRA/run_semantic_segmentation.py \
  --task isic2017 --eval \
  --ckpt_path <SEG_CKPT> \
  --tta_enable \
  --tta_box_prompt_mode add \
  --bbox_prompt_source predict \
  --bbox_preds_csv outputs/bbox_predict-251212/test_bbox_preds.csv
```

---

**User**

训练时加入box prompt混合策略你是在multimodal/src/autogluon/multimodal/optim/lit_semantic_seg.py 里写死的吗？应该让它可以根据需要开启，并且批级随机策略的概率分布也应该有可选调节的参数

---

**Cursor**

已修复：训练时的 box prompt 不再写死，改为可配置开关与概率。

**实现要点**
- `lit_semantic_seg.py`：训练阶段读取 `train_box_prompt_cfg`，支持 `mode=off/gt/noisy/mix`，`p_no/p_gt/p_noisy`（mix 时归一化），`noise_frac`。默认 `mode=off`。
- `SemanticSegmentationLearner`：保存 `_train_box_prompt_cfg`，构建 `LitModule` 时传入。
- `run_semantic_segmentation.py` 新增 CLI：
  - `--train_box_prompt_mode {off,gt,noisy,mix}` (默认 off)
  - `--train_box_prob_no_prompt` (默认 0.2)
  - `--train_box_prob_gt` (默认 0.3)
  - `--train_box_prob_noisy` (默认 0.5)
  - `--train_box_noise_frac` (默认 0.12)

**推荐配置（与你的场景对齐：推理用 add，预测框 Dice≈0.644）**
- `--train_box_prompt_mode mix`
- `--train_box_prob_no_prompt 0.2 --train_box_prob_gt 0.3 --train_box_prob_noisy 0.5`
- `--train_box_noise_frac 0.12`

**训练命令示例**
```bash
python3 examples/automm/Conv-LoRA/run_semantic_segmentation.py \
  --task isic2017 \
  --output_dir outputs/seg_train_box_mix \
  --rank 3 --expert_num 8 \
  --batch_size 4 --per_gpu_batch_size 1 --num_gpus 1 \
  --train_box_prompt_mode mix \
  --train_box_prob_no_prompt 0.2 \
  --train_box_prob_gt 0.3 \
  --train_box_prob_noisy 0.5 \
  --train_box_noise_frac 0.12
```

**推理（保持你当前策略：预测框 + add）**
- 在线预测框：
```bash
python3 examples/automm/Conv-LoRA/run_semantic_segmentation.py \
  --task isic2017 --eval \
  --ckpt_path <SEG_CKPT> \
  --tta_enable --tta_box_prompt_mode add \
  --bbox_prompt_source predict \
  --bbox_model_ckpt outputs/bbox_predict-251212/best_bbox.pth \
  --bbox_model_image_size 320
```
- 预生成框 CSV：
```bash
python3 examples/automm/Conv-LoRA/run_semantic_segmentation.py \
  --task isic2017 --eval \
  --ckpt_path <SEG_CKPT> \
  --tta_enable --tta_box_prompt_mode add \
  --bbox_prompt_source predict \
  --bbox_preds_csv outputs/bbox_predict-251212/test_bbox_preds.csv
```

如需调节概率/噪声，只改上述 CLI 即可；默认 `off`，不会改变现有训练行为。

```shellscript
python3 examples/automm/Conv-LoRA/run_semantic_segmentation.py \
  --task isic2017 \
  --output_dir outputs/seg_train_box_mix \
  --rank 3 --expert_num 8 \
  --batch_size 4 --per_gpu_batch_size 1 --num_gpus 1 \
  --train_box_prompt_mode mix \
  --train_box_prob_no_prompt 0.2 \
  --train_box_prob_gt 0.3 \
  --train_box_prob_noisy 0.5 \
  --train_box_noise_frac 0.12
```

```shellscript
python3 examples/automm/Conv-LoRA/run_semantic_segmentation.py \
  --task isic2017 --eval \
  --ckpt_path <SEG_CKPT> \
  --tta_enable --tta_box_prompt_mode add \
  --bbox_prompt_source predict \
  --bbox_model_ckpt outputs/bbox_predict-251212/best_bbox.pth \
  --bbox_model_image_size 320
```

```shellscript
python3 examples/automm/Conv-LoRA/run_semantic_segmentation.py \
  --task isic2017 --eval \
  --ckpt_path <SEG_CKPT> \
  --tta_enable --tta_box_prompt_mode add \
  --bbox_prompt_source predict \
  --bbox_preds_csv outputs/bbox_predict-251212/test_bbox_preds.csv
```

