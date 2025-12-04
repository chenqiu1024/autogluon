# TTA Label Key 问题修复

## 问题
```
KeyError: 'sam_label'
```

## 原因
SAM 模型的 `forward()` 方法在非训练模式（`self.training == False`）时，会尝试访问 `batch[self.label_key]`，但 TTA 的 `predict_fn` 只提供了 image，没有 label。

## 解决方案
在 batch 中添加一个 dummy label（全零 tensor），满足模型接口要求：

```python
batch = {
    model.prefix + '_image': img_tensor,
    model.prefix + '_label': torch.zeros((1, model.image_size, model.image_size), 
                                         dtype=torch.long, device=model.device)
}
```

这个 dummy label 不会被用于任何计算，只是为了满足接口。

## 修改文件
- `multimodal/src/autogluon/multimodal/learners/semantic_segmentation.py`

---

**修复日期**: 2025-12-04  
**修复类型**: 添加 dummy label

