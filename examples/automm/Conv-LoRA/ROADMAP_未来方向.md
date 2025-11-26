# RL Conv-LoRA 项目Roadmap

## 当前状态（2024-11-25）

### ✅ 已完成

1. **基础Conv-LoRA** (ICLR 2024论文实现)
   - 全32层激活
   - IoU: 0.7751, DICE: 0.8562

2. **两阶段RL训练** (已验证失败)
   - Per-image动态层选择
   - 结果: IoU 0.7646 (-1.36%) ❌
   - 根本原因: 分布偏移 + 层间依赖破坏

3. **固定配置联合训练** (当前实现)
   - 学习全局最优的固定层配置
   - 使用Gumbel-Softmax实现可微优化
   - 状态: ✅ 代码完成，待验证
   - 预期: IoU ≥ 0.775

---

## 🗺️ 未来发展路线

### Phase 1: 固定配置联合训练（当前）

**目标**: 验证联合训练的核心价值

**实现**: ✅ 已完成
- 固定mask（所有图像共享）
- Gumbel-Softmax + 同步优化
- 避免分布偏移

**验证指标**:
- [ ] IoU ≥ baseline
- [ ] 激活16-25层
- [ ] 训练稳定

**时间线**: 2024-11-25 (本周)

---

### Phase 2: Per-Image动态选择（联合训练版）⭐ 重要

**目标**: 在联合训练框架下实现per-image动态适应

**关键差异** vs 失败的两阶段RL:
```
两阶段RL (失败):
  1. 训练Conv-LoRA（假设全32层）
  2. 训练Policy（在冻结的Conv-LoRA上）
  → 分布不匹配

Per-Image联合训练 (计划):
  1. Conv-LoRA和Policy同时训练
  2. Conv-LoRA见到各种per-image masks
  3. 学会在动态环境中工作
  → 分布匹配 ✅
```

**技术要点**:
- ✅ Gumbel-Softmax（已实现）
- ✅ Policy网络（已实现，需集成）
- ⏳ 手动训练循环（需重写）
- ⏳ Patch embeddings提取（需完善）
- ⏳ 稳定性优化（需实验）

**实现工作量**: 
- 代码: 额外500-800行
- 调试: 1-2周
- 验证: 多次实验

**前置条件**:
- ✅ Phase 1证明联合训练可行
- ✅ Phase 1建立的基础设施

**时间线**: 如果Phase 1成功 → 2024年12月

---

### Phase 3: 混合方案（如果Phase 2成功）

**目标**: K个固定配置 + 动态选择

**思路**:
```
1. 学习K个固定层配置（如K=3）
   - 配置A: 低层为主（简单病变）
   - 配置B: 中层为主（中等复杂度）
   - 配置C: 高层为主（复杂病变）

2. 学习一个分类器
   classifier: image → {A, B, C}

3. 推理时
   config = classifier(image)
   mask = configs[config]
   output = model(image, mask)
```

**优势**:
- 比固定配置灵活
- 比完全per-image简单
- 平衡性能和复杂度

**时间线**: 2025年Q1

---

### Phase 4: 实用化和优化（如果有效）

**目标**: 将研究原型变成可部署的系统

**工作内容**:
1. **参数真正减少**
   - 动态加载/卸载Conv-LoRA参数
   - 或知识蒸馏到小模型

2. **推理优化**
   - 去除策略网络开销
   - 预计算和缓存
   - 量化和加速

3. **医疗场景适配**
   - 可解释性增强
   - 不确定性估计
   - FDA/NMPA合规性

4. **多数据集验证**
   - Polyp
   - Road segmentation
   - 其他医学图像

**时间线**: 2025年Q2-Q3

---

## 🔬 实验计划

### 当前实验（Phase 1）

**目标**: 验证固定配置联合训练

```bash
./运行联合训练.sh
```

**关键问题**:
- [ ] 是否收敛？
- [ ] 性能≥baseline？
- [ ] 学到有意义的配置？

**决策点**:
- ✅ 如果成功 → 进入Phase 2
- ❌ 如果失败 → 重新评估方向

---

### 未来实验（Phase 2）- Per-Image动态

**实验设计**:

#### 实验2.1: 简单per-image
```
实现: 单层MLP policy
条件: Phase 1成功
目标: 验证per-image在联合训练下是否更好
```

#### 实验2.2: 对比分析
```
对比:
  - 固定配置（Phase 1结果）
  - Per-image动态（Phase 2实现）
  - Baseline（全32层）
  - 随机选择

指标:
  - IoU/DICE
  - 计算量
  - 参数量
  - 训练稳定性
```

#### 实验2.3: 消融研究
```
变量:
  - Policy网络大小
  - 温度schedule
  - 学习率比例
  - Warmstart时长
```

---

## 📋 技术债务追踪

### 当前方案的已知限制

1. **固定配置的理论上限**
   - 无法适应图像差异
   - 性能上限可能低于per-image

2. **AutoGluon集成不完整**
   - 绕过了高层API
   - 手动训练循环

3. **真实embeddings vs 随机embeddings**
   - 当前per-image代码仍使用随机embeddings
   - Phase 2需要实现真实提取

### 计划改进

- [ ] Phase 2: 实现per-image动态（联合训练版）
- [ ] 完善patch embeddings提取
- [ ] 深度集成到AutoGluon
- [ ] 参数真正减少（动态加载）
- [ ] 推理加速优化

---

## 📊 成功标准演化

### Phase 1 (当前)
```
Minimum: IoU ≥ 0.755
Target:  IoU ≥ 0.775
Stretch: IoU ≥ 0.782
```

### Phase 2 (per-image)
```
Minimum: IoU ≥ Phase 1结果
Target:  IoU > Phase 1结果 + 0.5%
Stretch: IoU > Baseline + 2%
```

### Phase 3 (混合方案)
```
Minimum: IoU ≥ Phase 2结果
Target:  IoU > Baseline + 1%, 计算量-30%
Stretch: IoU > Baseline + 2%, 计算量-40%
```

---

## 💡 决策树

```
当前: 运行Phase 1固定配置联合训练
  ↓
  ├─ 成功 (IoU ≥ 0.775)
  │   ↓
  │   进入Phase 2: Per-Image动态联合训练
  │   ↓
  │   ├─ Per-Image > 固定配置
  │   │   ↓
  │   │   采用Per-Image，进入Phase 3优化
  │   │
  │   └─ Per-Image ≈ 固定配置
  │       ↓
  │       采用固定配置（更简单），进入Phase 4实用化
  │
  └─ 失败 (IoU < 0.765)
      ↓
      重新评估:
        - 是否层选择根本不适合ISIC 2017？
        - 或者需要其他方法（知识蒸馏、Early Exit等）
        - 或者转向其他研究方向
```

---

## 📝 记录保存

本roadmap将持续更新：
- ✅ 完成项标记为✅
- ⏳ 进行中标记为⏳
- ⬜ 未开始标记为⬜
- ❌ 放弃的方向标记为❌

**下次更新**: Phase 1完成后（约2天后）

---

## 🎯 核心承诺

**记住**: Per-image动态选择是最终目标，但要在联合训练框架下实现！

**不会重蹈覆辙**: 
- ❌ 不再做两阶段训练
- ✅ 任何per-image方案都必须是联合训练
- ✅ 渐进式验证，先简单后复杂

**当前优先级**:
1. 验证固定配置联合训练（本周）
2. 如果成功，立即规划Phase 2 per-image实现
3. 保持对最终目标的focus

---

**创建时间**: 2024-11-25  
**下次review**: Phase 1结果出来后  
**长期目标**: Per-Image动态联合训练 ⭐⭐⭐⭐⭐

