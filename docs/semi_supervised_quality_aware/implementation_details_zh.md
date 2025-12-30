# 实现细节文档

## 模块概览

| 模块 | 文件 | 功能 | 对应公式 |
|------|------|------|---------|
| EMA Teacher | ema_teacher.py | EMA副本、K次采样 | 公式 2.1 |
| 质量评估器 | quality_estimator.py | 一致性+置信度 | 公式 2.2 |
| 伪标签生成 | pseudo_label_gen.py | 生成并加权 | 公式 2.3 |
| 数据模块 | data_module.py | 混合数据加载 | - |
| GSPO扩展 | gspo_semi_trainer.py | 质量proxy | 公式 2.5 |

## 训练流程

1. 加载混合 batch（labeled + weak）
2. 对 weak 样本：Teacher K次采样 → 质量评估 → 生成伪标签
3. Student 前向
4. 计算损失：L_s + λ_u*L_u + λ_c*L_cons
5. 反向传播
6. EMA 更新 teacher（每步）

## 超参数调优

| 参数 | 默认值 | 调优范围 |
|------|--------|---------|
| ema_momentum | 0.999 | 0.99-0.9999 |
| quality_k_samples | 5 | 3-7 |
| quality_min_threshold | 0.6 | 0.5-0.7 |

## 常见问题

### 显存不足
- 减小 quality_k_samples 至 3
- 减小 batch_size

### 伪标签使用率低
- 降低 quality_min_threshold 至 0.5
- 减小 box_noise_std

### 训练不稳定
- 增加 warmup_epochs
- 提高 ema_momentum
