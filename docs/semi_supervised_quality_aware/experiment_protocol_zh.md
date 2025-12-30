# 实验协议文档

## 1. 数据拆分协议

### 1.1 源数据
- 数据集：ISIC2017 训练集
- 总样本数：2000 张图像
- 原始标注：完整的二值 mask

### 1.2 分层抽样策略
按照 mask 面积四分位数进行分层，确保不同难度样本均匀分布

### 1.3 拆分比例
- **10% 有标注集**（200 张）：保留 full mask
- **90% 弱标注集**（1800 张）：仅保留 noisy box
- 验证集、测试集：保持不变

### 1.4 Box 生成与噪声注入
对 90% 弱标注集：
1. 从 GT mask 计算 tight bounding box
2. 添加随机噪声：`box_noisy = box_gt * (1 + σ * randn(4))`, σ = 0.10

---

## 2. 对比实验

| 实验 | 标注形式 | 方法 | 目标 |
|------|---------|------|------|
| Exp-A | 100% mask | 全监督 | 上限 |
| Exp-B | 10% mask | 仅监督 | 下限 |
| Exp-C | 10% + 90% box | Teacher-Student | 基础半监督 |
| Exp-D | 10% + 90% box | + 质量加权 | 质量感知 |
| Exp-E | 10% + 90% box | + GSPO(q) | 策略优化 |
| Exp-F | 10% + 90% box | 完整方案 | 主方法 |

---

## 3. 关键超参数

| 参数 | 默认值 |
|------|--------|
| `ema_momentum` | 0.999 |
| `quality_k_samples` | 5 |
| `quality_min_threshold` | 0.6 |
| `pseudo_lambda_warmup_epochs` | 5 |
| `consistency_lambda` | 0.1 |



