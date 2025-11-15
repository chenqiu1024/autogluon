# DICE Metric Support for Semantic Segmentation

## 概述

现在已经为语义分割任务添加了 DICE 系数指标支持。DICE 系数是医学图像分割中常用的评估指标。

## DICE 与 IoU 的关系

DICE 系数和 IoU (Intersection over Union) 有以下关系：

```
DICE = 2 * IoU / (1 + IoU)
```

或反过来：

```
IoU = DICE / (2 - DICE)
```

## 使用方法

### 1. 在评估脚本中使用

修改后的 `run_semantic_segmentation.py` 现在会同时输出 IoU 和 DICE 指标：

```bash
python run_semantic_segmentation.py \
  --task isic2017 \
  --eval \
  --ckpt_path AutogluonModels/ag-20251113_165105 \
  --output_dir ~/autodl-tmp/works/outputs/convlora-eval
```

输出示例：
```
Evaluation results for test dataset isic2017: {'iou': 0.7731, 'dice': 0.8719}
```

### 2. 在代码中直接使用

```python
from autogluon.multimodal import MultiModalPredictor

# 加载模型
predictor = MultiModalPredictor.load("path/to/model")

# 评估时指定 DICE 指标
results = predictor.evaluate(test_data, metrics=["dice"])
# 或同时评估多个指标
results = predictor.evaluate(test_data, metrics=["iou", "dice"])
```

## 实现细节

### 修改的文件

1. **constants.py**: 添加 DICE 常量定义
2. **semantic_seg_metrics.py**: 
   - 添加 `Binary_DICE` 类（训练时使用）
   - 添加 `Binary_DICE_Pred` 类（预测时使用）
   - 添加 `Multiclass_DICE` 类（多类分割）
   - 添加 `Multiclass_DICE_Pred` 类（多类分割预测）
3. **semantic_segmentation.py**: 在 `get_metric_predict` 函数中添加 DICE 支持
4. **utils.py**: 在 `get_torchmetric` 函数中添加 DICE 支持
5. **run_semantic_segmentation.py**: 修改评估部分同时输出 IoU 和 DICE

### 计算方法

DICE 通过 IoU 计算：

```python
iou = JaccardIndex(logit, label)
dice = 2 * iou / (1 + iou)
```

## 示例转换

| IoU    | DICE   |
|--------|--------|
| 0.7731 | 0.8719 |
| 0.8000 | 0.8889 |
| 0.8500 | 0.9189 |

## 验证

运行测试验证 DICE 实现正确：

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA
conda run -n conv-lora python -c "
import torch
from autogluon.multimodal.optim.metrics.semantic_seg_metrics import Binary_DICE, Binary_IoU

metric = Binary_DICE()
logits = torch.tensor([[0.9, 0.8], [0.7, 0.6]]).unsqueeze(0)
labels = torch.tensor([[1.0, 1.0], [0.0, 1.0]]).unsqueeze(0)

metric.update(logits, labels)
print(f'DICE score: {metric.compute().item():.4f}')
"
```

