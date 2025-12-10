# 理解实验建议并讨论
_Exported on 12/10/2025 at 00:51:13 PST from Cursor (2.1.50)_

---

**User**

读docs/cursor_tta-251204.md以了解前文。然后，看看截图里给出的下一步实验建议，思考并讲讲你的理解

---

**Cursor**

```plaintext
去掉旋转和形态学只保留6次TTA，减少过重配置导致dice降低风险，轻量TTA结合一致性损失为实验首选...
```

我先看了 `docs/cursor_tta-251204.md`，其中记录了之前的讨论与实测：全量 18 视角（多尺度+翻转+旋转+形态学、固定阈值 0.5）在 ISIC2017 上 Dice 反而略低于基线，建议改用轻量 TTA、去掉旋转/形态学并重调阈值，再逐步加功能验证收益。示例如：去掉旋转与形态学，只保留 6 次 TTA 作为安全起点（见文档的“立刻可试的三步”第一条）。```75449:75465:docs/cursor_tta-251204.md
当前 TTA（18 视角 + 形态学 + 固定阈值 0.5 + 原图分辨率评估）把 Dice 拉到 0.8565...建议按“由轻到重、先验证再上全量”的顺序调整...
1) 去掉旋转和形态学，只保留 6 次 TTA
...
预期：Dice 接近或略高于基线。如果好，再去掉 --quick_test 跑全量。
```  

结合截图里的“下一步实验建议”，我的理解是：

- 目标：在加一致性损失（consistency loss）时，只动 TTA / multi-scale 相关的少数超参，控制搜索空间，优先验证是否能带来 0.3%~1% 的 Dice 增益或稳定性。
- TTA 向：主要调增强类型（翻转、旋转、镜像等）、每样本增强次数（2~4）、以及随机应用概率（0.5~1.0）。这延续了文档里“先用轻量、保守的几何变换”的思路，避免过度扰动边界。
- Multi-scale 向：控制输入缩放比例（如 [0.75, 1.0, 1.25] 或 [0.8, 1.0, 1.2]）、插值方式（bilinear/bicubic），以及 consistency loss 对多尺度的一致性权重（0.1~0.2），避免多尺度约束过强导致欠拟合或边界被抹平。
- 优先级与流程：  
  1) 先在单尺度上加入 TTA consistency（翻转/旋转的轻量版）做快速验证，看是否有立即提升或更稳的 Dice。  
  2) 再叠加 multi-scale consistency，观察额外收益。  
  这样符合文档里“先用低风险配置，再逐项加重”的策略，能快速定位有效增益点，也降低调参成本。

简单说：在已有 TTA 实现基础上，保持保守的几何增强与合理的尺度比例，先做“单尺度 + TTA 一致性”的轻量验证，再按需加 multi-scale 一致性，并控制权重在 0.1~0.2，避免像之前重度 TTA 那样掉点。

```plaintext

结合截图里的“下一步实验建议”，我的理解是：

- 目标：在加一致性损失（consistency loss）时，只动 TTA / multi-scale 相关的少数超参，控制搜索空间，优先验证是否能带来 0.3%~1% 的 Dice 增益或稳定性。
- TTA 向：主要调增强类型（翻转、旋转、镜像等）、每样本增强次数（2~4）、以及随机应用概率（0.5~1.0）。这延续了文档里“先用轻量、保守的几何变换”的思路，避免过度扰动边界。
- Multi-scale 向：控制输入缩放比例（如 [0.75, 1.0, 1.25] 或 [0.8, 1.0, 1.2]）、插值方式（bilinear/bicubic），以及 consistency loss 对多尺度的一致性权重（0.1~0.2），避免多尺度约束过强导致欠拟合或边界被抹平。
- 优先级与流程：  
  1) 先在单尺度上加入 TTA consistency（翻转/旋转的轻量版）做快速验证，看是否有立即提升或更稳的 Dice。  
  2) 再叠加 multi-scale consistency，观察额外收益。  
  这样符合文档里“先用低风险配置，再逐项加重”的策略，能快速定位有效增益点，也降低调参成本。

简单说：在已有 TTA 实现基础上，保持保守的几何增强与合理的尺度比例，先做“单尺度 + TTA 一致性”的轻量验证，再按需加 multi-scale 一致性，并控制权重在 0.1~0.2，避免像之前重度 TTA 那样掉点。
```

```plaintext

结合截图里的“下一步实验建议”，我的理解是：

- 目标：在加一致性损失（consistency loss）时，只动 TTA / multi-scale 相关的少数超参，控制搜索空间，优先验证是否能带来 0.3%~1% 的 Dice 增益或稳定性。
- TTA 向：主要调增强类型（翻转、旋转、镜像等）、每样本增强次数（2~4）、以及随机应用概率（0.5~1.0）。这延续了文档里“先用轻量、保守的几何变换”的思路，避免过度扰动边界。
- Multi-scale 向：控制输入缩放比例（如 [0.75, 1.0, 1.25] 或 [0.8, 1.0, 1.2]）、插值方式（bilinear/bicubic），以及 consistency loss 对多尺度的一致性权重（0.1~0.2），避免多尺度约束过强导致欠拟合或边界被抹平。
- 优先级与流程：  
  1) 先在单尺度上加入 TTA consistency（翻转/旋转的轻量版）做快速验证，看是否有立即提升或更稳的 Dice。  
  2) 再叠加 multi-scale consistency，观察额外收益。  
  这样符合文档里“先用低风险配置，再逐项加重”的策略，能快速定位有效增益点，也降低调参成本。

简单说：在已有 TTA 实现基础上，保持保守的几何增强与合理的尺度比例，先做“单尺度 + TTA 一致性”的轻量验证，再按需加 multi-scale 一致性，并控制权重在 0.1~0.2，避免像之前重度 TTA 那样掉点。
```

---

**User**

按照这张图里给出的建议超参，如果之前没有的话，在脚本中加入控制它们的命令行参数，并且实现相应的参数控制。然后给出推荐的实验设置，可以把多次实验的命令写到一个批处理脚本里，便于连续地无人值守运行

---

**Cursor**

变更说明：
- 在 `run_semantic_segmentation.py` 增加一致性实验相关的 CLI：`--tta_scale_ratios`（多尺度别名）、`--tta_resize_method`（bilinear/bicubic）、`--tta_augment_types`（horizontal_flip/vertical_flip/rotate/none）、`--tta_num_aug`、`--tta_prob`、`--tta_multi_scale_weight`、`--tta_random_seed`。若设置 multi_scale_weight，会自动切到 weighted_mean 融合并为非 1.0 尺度赋权。
- TTA 内核增强：`TTAPredictor` 支持可选 bicubic 缩放、按概率/数量裁剪增广集合（仍保证至少有 identity），并接受可复现的 random_seed。
- 新增批处理脚本 `examples/automm/Conv-LoRA/run_consistency_experiments.sh`，封装 3 组推荐实验（单尺度+轻量 TTA、一组轻量多尺度、一组稍强多尺度），便于无人值守顺序运行。

使用要点：
- 运行脚本（需先设置 CKPT，其他可用环境变量 DATASET/OUT_ROOT/SEED 调整）：
  - `CKPT=AutogluonModels/your_ckpt bash run_consistency_experiments.sh`
- 单次自定义示例（轻量多尺度一致性风格）：
  - `python3 run_semantic_segmentation.py --task isic2017 --eval --ckpt_path <CKPT> --tta_enable --tta_augment_types horizontal_flip rotate --tta_rotations -10 0 10 --tta_scale_ratios 0.8 1.0 1.2 --tta_multi_scale_weight 0.2 --tta_prob 0.8 --tta_resize_method bilinear --output_dir outputs/custom_run`
- 若只想快速验证单尺度+TTA，一条：
  - `python3 run_semantic_segmentation.py --task isic2017 --eval --ckpt_path <CKPT> --tta_enable --tta_augment_types horizontal_flip rotate --tta_rotations -10 0 10 --tta_num_aug 4 --tta_prob 0.8 --output_dir outputs/tta_single_scale`

备注：
- `--tta_scale_ratios` 覆盖 `--tta_scales`，便于直接对表中建议的缩放比例做实验。
- `--tta_prob` 和 `--tta_num_aug` 在初始化时对子集采样，运行期间保持确定性（可设 `--tta_random_seed`，默认沿用 `--seed`）。
- bicubic 可能在大尺度上更平滑，若关注边缘可用 bilinear。

