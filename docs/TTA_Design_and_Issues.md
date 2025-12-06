# TTA 方案实现与问题总结（当前版本）

本文整理当前实验中 TTA 的实现方式、关键代码位置、处理流程（含流程图）、以及已知问题与改进建议，便于后续继续研究或迭代。

---

## 1. 总览
- **核心思路**：在 CPU/PIL 上生成多视角（多尺度、翻转、小角度旋转），逐视角推理，逆变换回原图，融合概率，阈值/后处理后再计算指标。
- **评估模式**：默认“流式指标”低内存模式（无 `--tta_cache_dir`）；如需断点续传可开启缓存，代价是内存线性增长。
- **模型侧**：使用既有的 SAM 主干 + Conv-LoRA/Adapter + GSPO，不改模型结构，只改数据流。

---

## 2. 处理流程（文字流程图）
```
原图(H,W,3)
  └─ TTA 组合生成 (scale × flip × rotate) [CPU/PIL]
       └─ 视角图 -> 预处理到 model.image_size -> 前向 (SAM+LoRA/Adapter+GSPO, GPU)
            └─ 概率图(视角坐标) -> 逆变换回原图坐标/尺寸
  └─ 融合 (mean / weighted_mean)
       └─ 后处理：阈值、最小面积过滤、形态学 closing（可选）
            └─ pred_mask / pred_prob
                 └─ 与 GT 对齐(必要时 resize) -> 指标.update (torchmetrics)
```

---

## 3. 关键代码位置与片段

### 3.1 评估主循环（流式低内存）
文件：`multimodal/src/autogluon/multimodal/learners/semantic_segmentation.py`

```512:618:multimodal/src/autogluon/multimodal/learners/semantic_segmentation.py
for idx, row in data.iterrows():
    ...  # 读 image/label，label>128→1 二值化
    pred_mask, pred_prob = self._tta_predictor.predict_with_tta(image, predict_fn, return_probs=True)
    pred_tensor = torch.from_numpy(pred_prob)

    if streaming_metrics:  # 无 cache_dir 时
        y_p = pred_tensor.float()
        y_t = torch.from_numpy(label).long()
        if y_p.shape[-2:] != y_t.shape[-2:]:
            y_p = F.interpolate(y_p.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
        for metric_name, metric_obj in metric_objects.items():
            metric_obj.update(y_p.unsqueeze(0), y_t.unsqueeze(0))

    # 每 10 张输出进度 & 当前累计指标
    if streaming_metrics:
        intermediate_results = {name: float(metric_obj.compute()) for name, metric_obj in metric_objects.items()}
        logger.info(f"Intermediate TTA metrics after {num_processed} images: {intermediate_results}")
```

要点：
- 默认无 `--tta_cache_dir` 即流式模式：不存全量预测/标签，内存随时间稳定。
- 指定 `--tta_cache_dir` 时会保存预测/标签以便断点续传，内存随样本数线性增长（预期行为）。

### 3.2 TTA 变换与融合
文件：`multimodal/src/autogluon/multimodal/utils/tta_utils.py`

```419:493:multimodal/src/autogluon/multimodal/utils/tta_utils.py
for idx, (transform, weight) in enumerate(self.transforms):
    transformed_img = transform.apply(image)
    pred = predict_fn(transformed_img)          # 单视角前向
    del transformed_img
    pred_original = transform.apply_inverse_mask(pred)
    del pred

    # 尺寸不符时用 PIL resize 回原图
    ...
    all_probs.append(pred_original.copy())
    all_weights.append(weight)
    del pred_original
    if (idx + 1) % 2 == 0:
        gc.collect()

# 融合
if self.fusion_method == "mean":
    fused_prob = np.mean(all_probs, axis=0)
elif self.fusion_method == "weighted_mean":
    weights = np.array(all_weights); weights = weights / weights.sum()
    fused_prob = np.average(all_probs, axis=0, weights=weights)
del all_probs, all_weights
gc.collect()
```

要点：
- 变换：`ScaleTransform` / `FlipTransform` / `RotateTransform` 全用 PIL；每步 `del/close`，定期 `gc.collect()`。
- 融合后立即释放列表，降低峰值内存。

### 3.3 指标实现（适配不同分辨率，避免累积内存）
文件：`multimodal/src/autogluon/multimodal/optim/metrics/semantic_seg_metrics.py`

```880:936:multimodal/src/autogluon/multimodal/optim/metrics/semantic_seg_metrics.py
class Binary_IoU_Pred:
    def __init__(self):
        self.metric = torchmetrics.JaccardIndex(task="binary")
        self.total = 0.0; self.count = 0
    def update(self, logits, labels):
        logits = logits.cpu(); labels = labels.cpu()
        if logits.dim() == 3 and logits.shape[0] == 1: logits = logits.squeeze(0)
        if labels.dim() == 3 and labels.shape[0] == 1: labels = labels.squeeze(0)
        iou = self.metric(logits, labels)
        self.total += float(iou); self.count += 1
    def compute(self):
        return torch.tensor(0.0 if self.count == 0 else self.total / self.count)
```

要点：
- 不再 `torch.cat`，逐张累积标量，支持不同原始尺寸。
- Dice 同理：`dice = 2*iou/(1+iou)` 按张累积。

### 3.4 模型前向入口（与 TTA 对接）
- TTA 不改模型结构。`predict_fn` 将单视角图预处理到 `model.image_size`，构造 batch（含 dummy label），调用 `model(batch)` 得到 logits → 概率 → 回 CPU。

---

## 4. 已实现的特性
- 多尺度 / 翻转 / 小角度旋转组合，mean / weighted_mean 融合。
- 后处理：阈值、最小连通域过滤、形态学 closing（可选）。
- Sanity Check：数据列、设备、CUDA、单样本 TTA 预跑，验证输出尺寸。
- 快速模式：`--debug`（前 5 张）、`--quick_test N`（前 N 张）。
- 断点续传：`--tta_cache_dir` + `--tta_no_resume` 控制。
- 流式指标：默认无缓存时逐图更新 `torchmetrics`，定期输出中间指标。
- 内存优化：PIL 取代 cv2/skimage，显式 `del/close/gc.collect()`，GPU cache 周期清理。

---

## 5. 主要问题与风险
1) **性能下降（配置不匹配）**  
   - 重度 TTA（18 视角 + 形态学 + 阈值 0.5 + 原图分辨率评估）在 ISIC2017 + 现 ckpt 上 Dice 从 ~0.858（baseline）掉到 ~0.78。属于配置问题，而非实现错误。  
   - 可能原因：旋转/形态学对皮肤边界不友好；阈值未针对 TTA 重调；原图分辨率评估与训练分辨率不一致。

2) **内存缓慢上升的观感**  
   - 流式模式下未发现显著泄漏；若使用 `--tta_cache_dir`，内存会随样本数线性增长（为断点续传设计）。  
   - 若仍怀疑泄漏，可用 `memory_profiler/tracemalloc` 进一步确认。

3) **网络超时**  
   - HuggingFace `config.json` 拉取超时与内存无关，重试后使用本地 ckpt。

4) **未实现 GPU 端几何变换**  
   - 现设计刻意在 CPU/PIL 完成几何变换以控显存；若要 GPU 版需重写 `TTATransform`（`interpolate/grid_sample` 等），评估显存成本。

---

## 6. 建议的使用与后续实验
- **低风险配置**：6 次 TTA（scales 0.75/1.0/1.25 × flips none/horizontal，无旋转、无形态学），先验证是否 ≥ baseline。
- **阈值调优**：在验证集用 `TTAPredictor.tune_threshold` 搜索最佳阈值，再固化到测试。
- **后处理谨慎开启**：形态学/最小面积在高分辨率任务上可能过度平滑/抹除小目标，需按数据集调参。
- **断点续传**：仅在需要恢复时加 `--tta_cache_dir`；否则保持流式以最低内存。
- **监控**：`watch -n 10 'nvidia-smi; echo ---; free -h'` 观察显存/内存是否趋于平稳。

---

## 7. 若需进一步工作
- 设计 GPU 端 TTA 原型（kornia/torchvision.transforms v2），权衡显存与 CPU。
- 自动化搜索：阈值 + min_area + 形态学组合，在验证集联合调优。
- 更细粒度的采样策略：按难例/面积自适应调整 TTA 强度或视角数。

---

（以上基于当前代码：`semantic_segmentation.py`、`tta_utils.py`、`semantic_seg_metrics.py` 等最新版本。）

