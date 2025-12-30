# LitModule 集成指南

## 概述

要让半监督训练真正生效，需要将所有组件集成到 AutoGluon 的训练循环中。本文档提供两种集成方案。

---

## ⚠️ 重要说明

**当前状态**: `run_semi_supervised_train.py` 已经完成了所有组件的初始化和注入，但训练循环仍按标准监督学习执行。

**需要做什么**: 在训练的每个 step 中调用半监督逻辑（EMA Teacher、质量评估、伪标签生成等）。

---

## 方案 A: 修改 LitModule（直接集成）

### 文件位置
`multimodal/src/autogluon/multimodal/optim/lit_semantic_seg.py`

### 需要修改的方法
`SemanticSegmentationLitModule.training_step`

### 完整实现代码

```python
def training_step(self, batch, batch_idx):
    """
    扩展的训练步骤（支持半监督）
    
    对应设计文档中的完整训练流程：
    1. 区分 labeled 和 weak 样本
    2. Teacher K 次采样 → 质量评估 → 伪标签生成
    3. Student 前向 → 计算半监督损失
    4. 更新 EMA Teacher
    """
    # 检查是否启用半监督
    if hasattr(self, 'ema_teacher') and hasattr(self, 'quality_estimator'):
        return self._semi_supervised_training_step(batch, batch_idx)
    else:
        # 原有的全监督训练步骤
        return self._standard_training_step(batch, batch_idx)

def _standard_training_step(self, batch, batch_idx):
    """标准监督学习训练步骤（原有逻辑）"""
    # 保持原有的 training_step 代码不变
    output = self._shared_step(batch)
    
    loss = output[self.model.prefix + "_loss"]
    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
    
    return loss

def _semi_supervised_training_step(self, batch, batch_idx):
    """
    半监督训练步骤（完整实现 Phase 1-4）
    
    对应设计文档中的训练流程图
    """
    device = self.model.device
    
    # 1. 区分 labeled 和 weak 样本
    is_labeled = batch.get('is_labeled', torch.ones(len(batch), dtype=torch.bool, device=device))
    has_weak = (~is_labeled).any()
    
    # 2. 对 weak 样本：Teacher 生成伪标签（Phase 1-2）
    pseudo_labels = None
    quality_scores = None
    valid_mask = None
    
    if has_weak:
        # 2.1 Teacher K 次采样（对应公式 2.1）
        with torch.no_grad():
            teacher_predictions = self.ema_teacher.forward_k_times(
                batch, 
                K=self.semi_supervised_config.get('quality_k_samples', 5)
            )
        
        # 2.2 质量评估（对应公式 2.2）
        quality_scores, q_cons, q_conf = self.quality_estimator.estimate_quality(
            teacher_predictions
        )
        
        # 2.3 生成伪标签（对应公式 2.3）
        pseudo_labels, valid_mask = self.pseudo_label_gen.generate_from_predictions(
            teacher_predictions, quality_scores
        )
        
        # 日志记录
        stats = self.pseudo_label_gen.get_statistics(quality_scores, valid_mask)
        for key, value in stats.items():
            self.log(f"semi/{key}", value, on_step=False, on_epoch=True)
        
        self.log("semi/q_cons", q_cons.mean(), on_step=False, on_epoch=True)
        self.log("semi/q_conf", q_conf.mean(), on_step=False, on_epoch=True)
    
    # 3. Student 前向传播
    output = self._shared_step(batch)
    student_pred = output[self.model.prefix + "_prediction"]
    
    # 4. 计算半监督损失（Phase 3）
    if hasattr(self, 'gspo_trainer'):
        # 使用 GSPO 半监督训练器（含质量联动，对应公式 2.5）
        
        # 准备 GT mask
        gt_mask = batch[self.model.label_key] if is_labeled.any() else None
        
        # 计算质量权重
        if has_weak and quality_scores is not None:
            quality_weight = self.quality_estimator.compute_quality_weight(
                quality_scores,
                beta=self.semi_supervised_config.get('quality_weighting_beta', 10.0),
                q0=0.5
            )
            # 应用质量过滤
            quality_weight = quality_weight * valid_mask.float()
        else:
            quality_weight = None
        
        # Box Jitter 一致性（Phase 4，可选）
        box_jitter_pred1 = None
        box_jitter_pred2 = None
        if self.semi_supervised_config.get('enable_box_jitter', False) and has_weak:
            # 这里需要额外的前向传播（不同 box jitter）
            # 为简化，当前版本跳过，后续可扩展
            pass
        
        # 计算损失（对应公式 2.4）
        loss, loss_dict = self.gspo_trainer.compute_semi_supervised_loss(
            student_pred=student_pred,
            gt_mask=gt_mask,
            pseudo_label=pseudo_labels,
            quality_weight=quality_weight,
            is_labeled=is_labeled,
            epoch=self.current_epoch,
            loss_fn=self.loss_func,
            box_jitter_pred1=box_jitter_pred1,
            box_jitter_pred2=box_jitter_pred2
        )
        
        # 记录各项损失
        for key, value in loss_dict.items():
            self.log(f"train/{key}", value, on_step=True, on_epoch=True)
    
    else:
        # 简化版（无 GSPO）
        loss = output[self.model.prefix + "_loss"]
    
    # 5. 更新 EMA Teacher（每步更新，对应公式 2.1）
    if self.ema_update_freq == 'step':
        self.ema_teacher.update(self.model)
    
    # 主损失日志
    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
    
    return loss

def on_train_epoch_end(self):
    """Epoch 结束时的处理"""
    # 如果是 epoch 级别的 EMA 更新
    if hasattr(self, 'ema_teacher') and self.ema_update_freq == 'epoch':
        self.ema_teacher.update(self.model)
    
    # 调用父类方法
    super().on_train_epoch_end()
```

---

## 方案 B: 使用 Callback（推荐，无侵入）

### 创建 Callback 文件

创建文件: `examples/automm/Conv-LoRA/semi_supervised/callbacks.py`

```python
"""
Semi-Supervised Training Callbacks

通过 PyTorch Lightning Callback 实现无侵入的半监督训练
"""

import torch
from lightning.pytorch.callbacks import Callback


class SemiSupervisedCallback(Callback):
    """
    半监督训练 Callback
    
    在训练步骤中自动插入半监督逻辑
    """
    
    def __init__(self, config):
        self.config = config
        self.step_count = 0
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """
        在训练 batch 开始前生成伪标签
        """
        # 检查组件是否已初始化
        if not hasattr(pl_module, 'ema_teacher'):
            return
        
        device = pl_module.device
        is_labeled = batch.get('is_labeled', torch.ones(len(batch), dtype=torch.bool, device=device))
        has_weak = (~is_labeled).any()
        
        if not has_weak:
            return
        
        # Teacher K 次采样
        with torch.no_grad():
            teacher_predictions = pl_module.ema_teacher.forward_k_times(
                batch, K=self.config.get('quality_k_samples', 5)
            )
        
        # 质量评估
        quality_scores, q_cons, q_conf = pl_module.quality_estimator.estimate_quality(
            teacher_predictions
        )
        
        # 生成伪标签
        pseudo_labels, valid_mask = pl_module.pseudo_label_gen.generate_from_predictions(
            teacher_predictions, quality_scores
        )
        
        # 注入到 batch
        batch['pseudo_labels'] = pseudo_labels
        batch['quality_scores'] = quality_scores
        batch['pseudo_valid_mask'] = valid_mask
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        在训练 batch 结束后更新 EMA
        """
        if hasattr(pl_module, 'ema_teacher'):
            pl_module.ema_teacher.update(pl_module.model)
            self.step_count += 1


# 使用方式
def create_semi_supervised_callback(args):
    return SemiSupervisedCallback(config=vars(args))
```

### 在训练脚本中使用

```python
# 在 run_semi_supervised_train.py 中添加

from semi_supervised.callbacks import create_semi_supervised_callback

# 创建 callback
semi_callback = create_semi_supervised_callback(args)

# 注入到 trainer（需要访问 PyTorch Lightning Trainer）
# 这需要在 predictor.fit() 之前设置
# 具体实现取决于 AutoGluon 的 API
```

---

## 方案对比

| 特性 | 方案 A（修改 LitModule） | 方案 B（Callback） |
|------|----------------------|------------------|
| 侵入性 | 高（修改核心代码） | 低（可插拔） |
| 灵活性 | 低 | 高 |
| 调试难度 | 中 | 低 |
| 维护成本 | 高 | 低 |
| **推荐度** | ⚠️ | ✅ 推荐 |

---

## 测试清单

在实际运行前，请确认：

- [ ] 所有半监督组件已正确初始化
- [ ] EMA Teacher 每步正确更新
- [ ] Teacher K 次采样返回正确的预测列表
- [ ] 质量评估返回合理的分数（0-1 范围）
- [ ] 伪标签生成正确应用质量过滤
- [ ] 损失函数包含所有三项（L_s + λ_u·L_u + λ_c·L_cons）
- [ ] 训练日志包含半监督指标（quality, pseudo_ratio等）

---

## 当前限制

由于 AutoGluon 的封装，完整集成需要：

1. **访问 PyTorch Lightning Trainer 对象**：才能添加 Callback
2. **修改 LitModule**：才能在 training_step 中插入逻辑
3. **或扩展 AutoGluon API**：添加半监督训练的官方支持

**临时方案**: 
- 使用 `run_semi_supervised_train.py` 完成组件初始化
- 手动修改 LitModule 文件实现方案 A
- 或参考 Callback 代码自行适配

---

## 完整的训练流程（伪代码）

```python
for epoch in range(max_epochs):
    for batch in dataloader:
        # 1. 区分样本类型
        is_labeled = batch['is_labeled']
        
        # 2. Teacher 生成伪标签（weak 样本）
        if (~is_labeled).any():
            predictions_k = teacher.forward_k_times(batch, K=5)
            q, _, _ = quality_estimator.estimate_quality(predictions_k)
            pseudo_labels, valid = pseudo_gen.generate(predictions_k, q)
        
        # 3. Student 前向
        student_pred = student(batch)
        
        # 4. 计算损失
        loss = gspo_trainer.compute_semi_supervised_loss(
            student_pred, gt, pseudo_labels, q, is_labeled, epoch, loss_fn
        )
        
        # 5. 反向传播
        loss.backward()
        optimizer.step()
        
        # 6. EMA 更新
        teacher.update(student)
```

---

## 下一步行动建议

1. **测试组件独立功能**：
   - 单独测试 EMA Teacher 的 K 次采样
   - 验证质量评估器的输出范围
   - 检查伪标签生成的质量

2. **选择集成方案**：
   - 方案 A：直接修改，完全控制
   - 方案 B：Callback，更灵活

3. **实施集成**：
   - 按上述代码修改相应文件
   - 逐步测试每个 Phase
   - 验证训练日志

4. **性能验证**：
   - 在小数据集上 overfit 测试
   - 验证 DICE 是否提升
   - 检查伪标签质量分布



