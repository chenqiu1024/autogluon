# 实现细节文档

## 1. 模块概览

### 核心模块列表

| 模块 | 文件 | 功能 | 对应公式 |
|------|------|------|---------|
| EMA Teacher | `ema_teacher.py` | EMA 更新、K次采样 | 公式 2.1 |
| 质量评估器 | `quality_estimator.py` | 一致性+置信度评估 | 公式 2.2 |
| 伪标签生成器 | `pseudo_label_gen.py` | 生成并过滤伪标签 | 公式 2.3 |
| 数据模块 | `data_module.py` | 混合数据加载 | - |
| GSPO扩展 | `gspo_semi_trainer.py` | GSPO+质量proxy | 公式 2.5 |

---

## 2. 快速开始

### 2.1 准备数据
```bash
python prepare_semi_supervised_data.py \
    --task isic2017 \
    --data_dir datasets/isic2017 \
    --labeled_ratio 0.1 \
    --box_noise_std 0.10
```

### 2.2 使用模块

```python
from semi_supervised import (
    EMATeacher,
    ConsistencyQualityEstimator,
    PseudoLabelGenerator
)

# 1. 初始化 EMA Teacher
teacher = EMATeacher(student_model, momentum=0.999)

# 2. K次采样
predictions = teacher.forward_k_times(batch, K=5)

# 3. 质量评估
estimator = ConsistencyQualityEstimator()
q, q_cons, q_conf = estimator.estimate_quality(predictions)

# 4. 生成伪标签
generator = PseudoLabelGenerator(min_quality_threshold=0.6)
pseudo_labels, valid_mask = generator.generate_from_predictions(predictions, q)

# 5. 每步更新 Teacher
teacher.update(student_model)
```

---

## 3. 关键代码片段

### 公式 2.1: EMA 更新
```python
# ema_teacher.py:42
def update(self, student_model):
    m = self.momentum
    for teacher_param, student_param in zip(...):
        teacher_param.data.mul_(m).add_(student_param.data, alpha=1-m)
```

### 公式 2.2: 质量评估
```python
# quality_estimator.py:102
def estimate_quality(self, predictions):
    q_cons = self.compute_consistency_quality(predictions)
    q_conf = self.compute_confidence_quality(predictions)
    q = self.consistency_weight * q_cons + self.confidence_weight * q_conf
    return q, q_cons, q_conf
```

### 公式 2.3: 质量加权
```python
# pseudo_label_gen.py:75
def compute_quality_weight(self, quality_scores, beta=10.0, q0=0.5):
    weights = torch.sigmoid(beta * (quality_scores - q0))
    return weights
```

---

## 4. 超参数调优

### 关键超参数

| 参数 | 默认值 | 调优范围 | 影响 |
|------|--------|---------|------|
| `ema_momentum` | 0.999 | 0.99-0.9999 | Teacher稳定性 |
| `quality_k_samples` | 5 | 3-7 | 精度vs计算成本 |
| `quality_min_threshold` | 0.6 | 0.5-0.7 | 过滤严格程度 |
| `pseudo_lambda_warmup_epochs` | 5 | 3-10 | 启用速度 |

### 调优策略
1. 先调基础参数（ema_momentum, quality_k_samples）
2. 再调质量过滤（quality_min_threshold）
3. 最后调损失权重（pseudo_lambda_warmup_epochs）

---

## 5. 常见问题排查

### Q1: 显存不足
- 减小 `quality_k_samples` 至 3
- 减小 `batch_size`

### Q2: 伪标签使用率过低
- 降低 `quality_min_threshold` 至 0.5
- 减小 `box_noise_std` 至 0.05

### Q3: 训练不稳定
- 增加 `pseudo_lambda_warmup_epochs` 至 10
- 提高 `ema_momentum` 至 0.9995

---

## 6. 公式与代码对应

| 公式 | 代码位置 | 函数 |
|------|---------|------|
| 2.1 | `ema_teacher.py:42` | `update()` |
| 2.2 | `quality_estimator.py:102` | `estimate_quality()` |
| 2.3 | `pseudo_label_gen.py:75` | `compute_quality_weight()` |
| 2.5 | `gspo_semi_trainer.py:68` | `compute_segmentation_quality()` |



