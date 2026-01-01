# 半监督质量感知 GSPO-Conv-LoRA 技术架构详解

## 📋 目录

1. [系统概览](#1-系统概览)
2. [模型架构](#2-模型架构)
3. [时序流程](#3-时序流程)
4. [损失函数详解](#4-损失函数详解)
5. [强化学习机制](#5-强化学习机制)
6. [参数配置](#6-参数配置)

---

## 1. 系统概览

### 1.1 核心问题

在医学图像分割任务中：
- **有标注数据稀缺**：获取像素级精确标注（full mask）成本极高
- **弱标注易获取**：粗略的 bounding box 标注相对容易
- **模型容量受限**：SAM-ViT-Huge 参数量巨大，全量微调不现实

### 1.2 解决方案

本系统结合了三个关键技术：

| 技术模块 | 作用 | 创新点 |
|---------|------|--------|
| **Conv-LoRA MoE** | 参数高效微调 | 多专家卷积核 + 质量感知门控 |
| **半监督学习** | 利用弱标注数据 | EMA Teacher + 质量评估 + 伪标签 |
| **GSPO 强化学习** | 策略优化 | 组级采样 + 质量反馈环路 |

---

## 2. 模型架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    半监督质量感知 GSPO-Conv-LoRA                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────┐         ┌──────────────┐                             │
│  │ 输入数据   │         │  数据类型    │                             │
│  ├────────────┤         ├──────────────┤                             │
│  │ Image      │────────▶│ 10% Labeled  │ (Full Mask)                │
│  │ Label/Box  │         │ 90% Weak     │ (Noisy Box)                │
│  └────────────┘         └──────┬───────┘                             │
│                                │                                     │
│                                ▼                                     │
│         ┌──────────────────────────────────────────┐                 │
│         │       Teacher-Student 框架               │                 │
│         ├──────────────────────────────────────────┤                 │
│         │                                          │                 │
│         │  ┌─────────────────────────────────┐    │                 │
│         │  │   EMA Teacher (冻结参数)         │    │                 │
│         │  │   ┌─────────────────────────┐   │    │                 │
│         │  │   │  SAM Encoder (冻结)      │   │    │                 │
│         │  │   │  - ViT-Huge backbone    │   │    │                 │
│         │  │   │  - FC Adapter (训练)    │   │◀──┐ │                 │
│         │  │   └─────────────────────────┘   │   │ │                 │
│         │  │   ┌─────────────────────────┐   │   │ │                 │
│         │  │   │  SAM Decoder            │   │   │ │                 │
│         │  │   │  - Conv-LoRA MoE (冻结) │   │   │EMA更新            │
│         │  │   │    * M个专家 (4-8个)     │   │   │θ_T←m*θ_T+(1-m)*θ_S│
│         │  │   │    * 质量感知门控 (冻结) │   │   │ │                 │
│         │  │   └─────────────────────────┘   │   │ │                 │
│         │  │                                  │   │ │                 │
│         │  │   输出: K次预测 (多样性采样)      │   │ │                 │
│         │  │   p_1, p_2, ..., p_K             │   │ │                 │
│         │  └──────────┬──────────────────────┘   │ │                 │
│         │             │                           │ │                 │
│         │             ▼                           │ │                 │
│         │  ┌──────────────────────────────────┐  │ │                 │
│         │  │   质量评估器 (无需GT)             │  │ │                 │
│         │  │   ┌────────────────────────────┐ │  │ │                 │
│         │  │   │ 一致性质量 q_cons          │ │  │ │                 │
│         │  │   │ = avg(IoU(p_i, p_j))       │ │  │ │                 │
│         │  │   └────────────────────────────┘ │  │ │                 │
│         │  │   ┌────────────────────────────┐ │  │ │                 │
│         │  │   │ 置信度质量 q_conf          │ │  │ │                 │
│         │  │   │ = avg(max(p, 1-p))         │ │  │ │                 │
│         │  │   └────────────────────────────┘ │  │ │                 │
│         │  │                                  │  │ │                 │
│         │  │   输出: q = α*q_cons + (1-α)*q_conf│ │                  │
│         │  └──────────┬───────────────────────┘  │ │                 │
│         │             │                           │ │                 │
│         │             ▼                           │ │                 │
│         │  ┌──────────────────────────────────┐  │ │                 │
│         │  │   伪标签生成器                    │  │ │                 │
│         │  │   - 平均K次预测: p̄ = avg(p_k)    │  │ │                 │
│         │  │   - 质量过滤: q >= q_min         │  │ │                 │
│         │  │   - 质量加权: w(q)=sigmoid(β(q-q0))│ │                  │
│         │  └──────────┬───────────────────────┘  │ │                 │
│         │             │                           │ │                 │
│         └─────────────┼───────────────────────────┘ │                 │
│                       │ 伪标签 ŷ_weak               │                 │
│                       ▼                             │                 │
│         ┌──────────────────────────────────────────┐│                 │
│         │   Student Model (可训练参数)              ││                 │
│         │   ┌─────────────────────────┐            ││                 │
│         │   │  SAM Encoder (冻结)      │            ││                 │
│         │   │  - ViT-Huge backbone    │            ││                 │
│         │   │  - FC Adapter (训练)✓   │            ││                 │
│         │   └─────────────────────────┘            ││                 │
│         │   ┌─────────────────────────┐            ││                 │
│         │   │  SAM Decoder            │            ││                 │
│         │   │  - Conv-LoRA MoE (训练)✓│            ││                 │
│         │   │    * M个专家卷积核       │            ││                 │
│         │   │    * 质量感知门控        │◀───────────┘│                 │
│         │   │    * GSPO组级优化       │  质量反馈     │                 │
│         │   └─────────────────────────┘             │                 │
│         │                                            │                 │
│         │   输入: labeled (GT y) + weak (伪标签 ŷ)   │                 │
│         │   输出: 预测 p_student                     │                 │
│         └────────────┬───────────────────────────────┘                 │
│                      │                                                 │
│                      ▼                                                 │
│         ┌──────────────────────────────────────────┐                   │
│         │   损失计算与优化                          │                   │
│         │   L = L_supervised                       │                   │
│         │     + λ_u(t) * Σw(q)*L_pseudo           │                   │
│         │     + λ_c * L_consistency                │                   │
│         │     + L_gspo                             │                   │
│         └──────────────────────────────────────────┘                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 各组件详解

#### 2.2.1 SAM 基础模型

```
SAM (Segment Anything Model)
├── Encoder (ViT-Huge)
│   ├── Image Embedding (冻结)
│   │   - 参数量: ~600M
│   │   - 不参与训练
│   └── FC Adapter (可选，训练)
│       - 位置: ViT每层后
│       - 参数量: ~0.5M/layer
│       - 作用: 质量反馈路径
│
└── Decoder (Mask Decoder)
    ├── Prompt Encoder (冻结)
    │   - 处理 box prompt
    ├── Conv-LoRA MoE (训练)
    │   - 参数量: ~2-4M
    │   - 核心可训练模块
    └── Mask Head (冻结)
        - 输出最终分割mask
```

**参数统计**：
- **总参数**: ~640M
- **可训练参数**: ~2-5M (Conv-LoRA + Adapter)
- **训练比例**: < 1%
- **冻结参数**: SAM Encoder (ViT backbone) + SAM Decoder 主干

#### 2.2.2 Conv-LoRA MoE 结构

```
Conv-LoRA Module (每个解码器层)
├── 输入特征: x [B, C, H, W]
│
├── LoRA 分支 (训练)
│   ├── 下投影: W_down [C, r]
│   │   - r: LoRA rank (默认4-16)
│   │
│   ├── MoE 专家层
│   │   ├── Expert 1: Conv(r, r, k=3, dilation=1)
│   │   ├── Expert 2: Conv(r, r, k=3, dilation=2)
│   │   ├── ...
│   │   └── Expert M: Conv(r, r, k=3, dilation=M)
│   │   │
│   │   └── 门控网络 (质量感知)
│   │       ├── 输入: Global Avg Pool(x) [B, C]
│   │       ├── 质量偏置: quality_bias [M]
│   │       │   = 0.8 * normalize(quality_history)
│   │       │     + 0.2 * exploration_bonus
│   │       │
│   │       ├── Logits: x @ W_gate + quality_bias
│   │       ├── TopK 选择: k=1 (标准) 或 k=G (GSPO组采样)
│   │       └── 输出: selected_experts [B, k]
│   │
│   └── 上投影: W_up [r, C]
│
├── 残差连接: x + α * LoRA(x)
│   - α: LoRA scaling factor
│
└── 输出: x' [B, C, H, W]
```

**关键特性**：
1. **多感受野专家**：通过不同 dilation rate 捕获多尺度信息
2. **质量感知门控**：历史质量反馈影响专家选择
3. **参数高效**：只训练 LoRA 权重，主干冻结

#### 2.2.3 EMA Teacher 模型

```
EMA Teacher
├── 初始化: deepcopy(Student Model)
│   - 所有参数 requires_grad = False
│
├── 更新策略 (每 step 或 epoch)
│   θ_teacher^(t) = m * θ_teacher^(t-1) + (1-m) * θ_student^(t)
│   - m: momentum (默认 0.999)
│   - 只更新 Conv-LoRA 和 Adapter 部分
│
├── 前向模式: K 次采样
│   ├── 启用 Dropout (产生多样性)
│   ├── 启用 MoE 噪声 (不同专家选择)
│   └── 输出: [p_1, p_2, ..., p_K]
│       - 每个 p_k shape: [B, H, W]
│
└── 作用
    ├── 生成稳定的伪标签 (平滑训练)
    └── 提供质量评估依据 (一致性)
```

#### 2.2.4 质量评估器

```
Quality Estimator (无监督)
├── 输入: K个预测 [p_1, ..., p_K]
│
├── 一致性质量 q_cons
│   ├── 计算所有配对IoU: IoU(p_i, p_j) for i<j
│   │   - 配对数: C(K,2) = K(K-1)/2
│   │   - IoU计算: (pred_i ∩ pred_j) / (pred_i ∪ pred_j)
│   │
│   └── 平均: q_cons = (2/K(K-1)) * Σ IoU(p_i, p_j)
│       - 物理意义: 预测的自洽性
│       - 范围: [0, 1]
│       - 高 → 模型确定，低 → 模型困惑
│
├── 置信度质量 q_conf
│   ├── 每个预测的确定性: max(p_k, 1-p_k)
│   │   - 接近0.5 → 不确定
│   │   - 接近0或1 → 确定
│   │
│   └── 空间+采样平均: q_conf = (1/K) * Σ mean(max(p_k, 1-p_k))
│       - 物理意义: 预测的果断性
│       - 范围: [0.5, 1.0]
│
├── 融合质量 q
│   q = α * q_cons + (1-α) * q_conf
│   - α: 一致性权重 (默认 0.7)
│   - 综合考虑自洽性和果断性
│
└── 质量权重 w(q)
    w(q) = sigmoid(β(q - q0))
    - β: 斜率 (默认 10)
    - q0: 中心点 (默认 0.5)
    - 作用: 平滑的质量过滤函数
```

**为什么无需GT？**
- **核心假设**：模型对简单样本的多次预测应该一致
- **一致性高** → 样本简单 OR 模型有把握 → 伪标签可靠
- **一致性低** → 样本困难 OR 模型困惑 → 伪标签不可靠

---

## 3. 时序流程

### 3.1 训练全流程（单个 Batch）

```
Epoch t, Batch b
│
├─ [Step 1] 数据加载
│  ├─ 读取 batch: {image, label/box, is_labeled}
│  │   - image: [B, 3, H, W]
│  │   - label: [B, H, W] (仅 labeled 样本有效)
│  │   - box: [B, 4] (仅 weak 样本有效)
│  │   - is_labeled: [B] bool
│  │
│  └─ 区分样本类型
│      ├─ labeled_mask = is_labeled == True
│      └─ weak_mask = is_labeled == False
│
├─ [Step 2] Weak 样本伪标签生成 (如果有)
│  │
│  ├─ 2.1 EMA Teacher K次采样
│  │   with torch.no_grad():
│  │       for k in range(K):  # K=5
│  │           batch['box'] = weak_boxes  # 使用 noisy box
│  │           p_k = ema_teacher(batch)
│  │           predictions.append(p_k)
│  │   
│  │   # 每次采样产生不同结果因为:
│  │   # - Dropout 状态不同
│  │   # - MoE 门控噪声不同
│  │
│  ├─ 2.2 质量评估
│  │   # 计算所有配对 IoU
│  │   iou_pairs = []
│  │   for i in range(K):
│  │       for j in range(i+1, K):
│  │           iou = compute_iou(predictions[i], predictions[j])
│  │           iou_pairs.append(iou)
│  │   q_cons = mean(iou_pairs)  # [B]
│  │   
│  │   # 计算置信度
│  │   conf_scores = []
│  │   for p_k in predictions:
│  │       conf = max(p_k, 1-p_k).mean(dim=(1,2))
│  │       conf_scores.append(conf)
│  │   q_conf = mean(conf_scores)  # [B]
│  │   
│  │   # 融合
│  │   q = 0.7 * q_cons + 0.3 * q_conf  # [B]
│  │
│  ├─ 2.3 伪标签生成
│  │   # 平均预测
│  │   p_avg = mean(predictions, dim=0)  # [B, H, W]
│  │   
│  │   # 硬标签 (默认) or 软标签
│  │   pseudo_label = (p_avg > 0.5).float()  # [B, H, W]
│  │   
│  │   # 质量过滤
│  │   valid_mask = (q >= q_min)  # q_min=0.6, [B]
│  │   
│  │   # 质量权重
│  │   w = sigmoid(10 * (q - 0.5))  # [B]
│  │   w = w * valid_mask.float()  # 低质量样本权重=0
│  │
│  └─ 输出
│      ├─ pseudo_label: [B_weak, H, W]
│      ├─ quality_weight: [B_weak]
│      └─ valid_mask: [B_weak]
│
├─ [Step 3] Student 前向传播
│  │
│  ├─ 3.1 标准前向 (如果无GSPO)
│  │   output = student_model(batch)
│  │   pred = output['logits']  # [B, 1, H, W]
│  │   pred_prob = sigmoid(pred).squeeze(1)  # [B, H, W]
│  │
│  └─ 3.2 GSPO 组级前向 (如果启用)
│      # 对每个样本生成 G 个预测
│      for g in range(G):  # G=4
│          # 每次前向的差异来自:
│          # - Dropout 随机性
│          # - MoE TopK=G (选择不同专家组合)
│          pred_g, experts_g = student_model(batch)
│          
│          # 计算质量 (labeled用GT, weak用quality_proxy)
│          if labeled:
│              quality_g = IoU(pred_g, GT)
│          else:
│              quality_g = q  # 使用一致性质量
│          
│          group_preds.append(pred_g)
│          group_qualities.append(quality_g)
│          group_experts.append(experts_g)
│
├─ [Step 4] 损失计算
│  │
│  ├─ 4.1 监督损失 (Labeled 样本)
│  │   L_s = loss_fn(pred[labeled_mask], GT[labeled_mask])
│  │   # 典型: BCE Loss 或 Dice Loss
│  │
│  ├─ 4.2 伪监督损失 (Weak 样本)
│  │   # 逐样本损失
│  │   L_u_per_sample = loss_fn(
│  │       pred[weak_mask], 
│  │       pseudo_label[weak_mask]
│  │   )  # [B_weak]
│  │   
│  │   # 质量加权
│  │   L_u = (L_u_per_sample * w[weak_mask]).sum() / w.sum()
│  │   
│  │   # Warmup (逐渐增加伪标签权重)
│  │   λ_u = min(1.0, epoch / warmup_epochs)  # 线性增长
│  │   L_u_weighted = λ_u * L_u
│  │
│  ├─ 4.3 GSPO 损失 (如果启用)
│  │   # 组内质量 baseline
│  │   Q = stack(group_qualities)  # [G, B]
│  │   Q_mean = Q.mean(dim=0)  # [B]
│  │   
│  │   # Advantage (相对质量优势)
│  │   A = Q - Q_mean  # [G, B]
│  │   # A>0: 该预测比组平均好
│  │   # A<0: 该预测比组平均差
│  │   
│  │   # 加权损失 (好的预测权重高)
│  │   weights = softmax(A / temperature)  # [G, B]
│  │   L_gspo = Σ weights[g] * loss_fn(pred_g, GT)
│  │   
│  │   # 对比损失 (拉近好的，推开差的)
│  │   features = [encoder_output_g for g in G]
│  │   L_contrast = contrastive_loss(features, Q)
│  │
│  ├─ 4.4 一致性损失 (可选，Box Jitter)
│  │   # 同一样本，不同 box → 预测应一致
│  │   box1 = jitter(GT_box, noise=0.08)
│  │   box2 = jitter(GT_box, noise=0.08)
│  │   pred1 = student_model(batch, box=box1)
│  │   pred2 = student_model(batch, box=box2)
│  │   L_cons = KL(pred1 || pred2)
│  │
│  └─ 4.5 总损失
│      L_total = L_s 
│               + λ_u(t) * L_u 
│               + λ_gspo * L_gspo
│               + λ_c * L_cons
│
├─ [Step 5] 反向传播与优化
│  │
│  ├─ 5.1 梯度计算
│  │   L_total.backward()
│  │   # 梯度流向:
│  │   # - Conv-LoRA 参数 ✓
│  │   # - FC Adapter 参数 ✓
│  │   # - SAM Encoder/Decoder 主干 ✗ (冻结)
│  │
│  ├─ 5.2 参数更新
│  │   optimizer.step()
│  │   # 只更新 requires_grad=True 的参数
│  │   # (~2-5M 参数)
│  │
│  └─ 5.3 质量反馈 (GSPO)
│      # 更新专家质量历史 (移动平均)
│      for layer in conv_lora_layers:
│          for expert_id in selected_experts:
│              old_quality = layer.expert_quality[expert_id]
│              new_quality = current_quality
│              layer.expert_quality[expert_id] = (
│                  momentum * old_quality + (1-momentum) * new_quality
│              )
│      # 这将影响下一次的门控决策
│
└─ [Step 6] EMA Teacher 更新
   │
   └─ 如果 update_freq == 'step':
       for param_t, param_s in zip(teacher.parameters(), 
                                     student.parameters()):
           param_t.data = (
               momentum * param_t.data + (1-momentum) * param_s.data
           )
       # momentum = 0.999
       # Teacher 平滑跟踪 Student，提供稳定伪标签
```

### 3.2 训练阶段划分

```
Training Timeline
│
├── Epoch 0-5: Warmup 阶段
│   ├── λ_u = 0.0 → 1.0 (线性增长)
│   │   - 初期：只用 labeled 数据训练
│   │   - 后期：逐渐引入伪标签
│   │
│   ├── GSPO 预热 (如果启用)
│   │   - 收集专家质量统计
│   │   - 质量偏置权重较小
│   │
│   └── EMA Teacher 稳定
│       - Teacher 逐渐偏离初始化状态
│       - 伪标签质量提升
│
├── Epoch 6-15: 稳定训练阶段
│   ├── λ_u = 1.0 (全权重)
│   ├── GSPO 全激活
│   │   - 质量反馈环路稳定
│   │   - 专家选择优化
│   │
│   └── 伪标签质量监控
│       - pseudo_label_ratio: 通过质量过滤的比例
│       - mean_quality: 平均质量分数
│       - 正常范围: 60-80% 样本通过
│
└── Epoch 16-30: 精调阶段
    ├── 降低学习率 (如 1e-5 → 1e-6)
    ├── 可选: 提高 q_min 阈值 (0.6 → 0.7)
    └── Early stopping 监控 val_loss
```

---

## 4. 损失函数详解

### 4.1 监督损失 L_s

```python
# 对有标注样本
L_s = (1/N_labeled) * Σ BCE(pred_i, GT_i)

# BCE (Binary Cross Entropy)
BCE(p, y) = -[y*log(p) + (1-y)*log(1-p)]

# 或使用 Dice Loss (对不平衡数据更鲁棒)
Dice(p, y) = 1 - (2*Σ(p*y) + smooth) / (Σp + Σy + smooth)
```

**特点**：
- 标准监督学习损失
- 只作用于 10% labeled 样本
- 提供可靠的训练信号

### 4.2 伪监督损失 L_u

```python
# 完整公式
L_u = (1/Σw_i) * Σ w_i * BCE(pred_i, pseudo_i)

其中:
- w_i = sigmoid(β(q_i - q0)) * I(q_i >= q_min)
- q_i: 样本i的质量分数
- q_min: 质量过滤阈值 (0.6)
- β: sigmoid斜率 (10.0)
- q0: sigmoid中心 (0.5)
- I(.): 指示函数
```

**质量权重曲线**：

```
w(q)
1.0 ┤        ╭──────
    │      ╭─╯
0.5 ┤    ╭─╯  ← q0
    │  ╭─╯
0.0 ┤──╯
    └─┬──┬──┬──┬──┬─→ q
     0.0 0.5 1.0
         ↑
        q_min
```

**物理意义**：
- **q < q_min**: w=0，样本被过滤
- **q ≈ q0**: w≈0.5，中等权重
- **q → 1.0**: w→1.0，高权重

### 4.3 GSPO 损失 L_gspo

```python
# 组级训练
G = group_size  # 4
for g in range(G):
    pred_g, quality_g = forward_and_evaluate()
    
# Advantage 函数 (核心)
A_g = quality_g - mean(quality_1, ..., quality_G)

# 策略梯度权重
weight_g = exp(A_g / τ) / Σ exp(A_i / τ)
# τ: temperature (5.0)

# 加权损失
L_gspo = Σ weight_g * BCE(pred_g, GT)

# 对比损失 (可选)
features_g = encoder_output_g  # [B, D]
L_contrast = Σ_i,j [
    -log(exp(sim(f_i, f_j)/t) / Σ_k exp(sim(f_i, f_k)/t))
] * I(quality_i和quality_j同类)
# 高质量拉近，低质量推开
```

**核心思想**：
1. **组内竞争**：G个预测相互比较
2. **相对优势**：不看绝对质量，看相对质量
3. **权重梯度**：好的预测贡献更多梯度
4. **探索利用**：通过温度参数平衡

### 4.4 一致性损失 L_cons

```python
# Box Jitter 一致性
box1 = GT_box + noise1  # 噪声标准差 ~0.08
box2 = GT_box + noise2

pred1 = student(image, box=box1)
pred2 = student(image, box=box2)

# KL 散度 (概率分布间距离)
L_cons = KL(pred1 || pred2)
       = Σ pred1 * log(pred1 / pred2)

# 或 MSE
L_cons = MSE(pred1, pred2)
       = mean((pred1 - pred2)²)
```

**物理意义**：
- 不同 box prompt → 预测应该相似
- 增强对 prompt 扰动的鲁棒性
- 正则化作用

### 4.5 完整损失

```python
L_total = L_s                      # 监督损失
        + λ_u(t) * L_u            # 伪监督损失 (warmup)
        + λ_gspo * L_gspo         # GSPO策略损失
        + λ_c * L_cons            # 一致性损失

# 典型权重
λ_u(t) = min(1.0, t/5)  # epoch 0→5 线性增长
λ_gspo = 1.0
λ_c = 0.1
```

---

## 5. 强化学习机制

### 5.1 RL 映射关系

| RL 概念 | GSPO-Conv-LoRA 对应 |
|---------|---------------------|
| **Agent** | MoE 门控网络 |
| **State** | 图像特征 (encoder output) |
| **Action** | 选择哪个专家 (或专家组合) |
| **Action Space** | {1, 2, ..., M}^k (M个专家选k个) |
| **Policy** | 门控 softmax(logits + quality_bias) |
| **Reward** | 分割质量 (IoU/Dice) |
| **Value Function** | Expert Quality History |
| **Advantage** | A = Q_current - Q_group_mean |

### 5.2 反馈环路详解

```
┌─────────────────────────────────────────────────────────────┐
│                   GSPO 强化学习反馈环路                      │
└─────────────────────────────────────────────────────────────┘

Step t-1: 历史状态
│
├─ Expert Quality History: Q_history[expert_id]
│   Expert 1: [0.72, 0.71, 0.73, ...] → μ=0.72, σ=0.01
│   Expert 2: [0.68, 0.69, 0.67, ...] → μ=0.68, σ=0.01
│   ...
│   Expert M: [0.75, 0.74, 0.76, ...] → μ=0.75, σ=0.01
│
└─ Usage Count: N_usage[expert_id]
    [120, 95, 110, ..., 105]  # 负载平衡

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step t: 当前决策
│
├─ [前向传播] Image → Encoder → features [B, C]
│
├─ [门控决策]
│   # 1. 基础 logits
│   logits = features @ W_gate  # [B, M]
│   
│   # 2. 质量偏置 (exploit)
│   exploit = normalize(Q_history)  # [M]
│   # 高质量专家 → 正偏置
│   # 低质量专家 → 负偏置
│   
│   # 3. 探索奖励 (explore)
│   explore = 1 / sqrt(N_usage + 1)  # [M]
│   # 少用的专家 → 高探索奖励
│   
│   # 4. 组合
│   quality_bias = 0.8 * exploit + 0.2 * explore
│   
│   # 5. 最终 logits
│   logits_final = logits + quality_bias
│   probs = softmax(logits_final)
│   
│   # 6. TopK 采样 (GSPO: k=G)
│   selected_experts = topk(probs, k=G)  # [B, G]
│
└─ [执行专家]
    pred = expert_forward(features, selected_experts)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step t: 质量评估
│
├─ [计算 Reward]
│   # Labeled 样本: 真实 IoU
│   R_labeled = IoU(pred, GT)
│   
│   # Weak 样本: 一致性质量 (无GT)
│   R_weak = q_cons  # 来自 Teacher K次采样
│   
│   R = [R_labeled, R_weak]  # [B]
│
├─ [组内 Advantage]
│   # GSPO 组级评估
│   R_group = [R_1, R_2, ..., R_G]  # [G, B]
│   R_mean = mean(R_group, dim=0)    # [B]
│   
│   A = R - R_mean  # [G, B]
│   # A[g] > 0: 第g个预测好于组平均 → 增强此策略
│   # A[g] < 0: 第g个预测差于组平均 → 抑制此策略
│
└─ [质量反馈]
    # 更新专家质量历史 (EMA)
    for expert_id in selected_experts:
        Q_new = R[expert_id]
        Q_history[expert_id] = (
            momentum * Q_history[expert_id] + (1-momentum) * Q_new
        )
        N_usage[expert_id] += 1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step t: 策略更新
│
├─ [GSPO 策略梯度]
│   # 基于 Advantage 加权损失
│   weight = softmax(A / temperature)  # [G, B]
│   loss = Σ weight[g] * BCE(pred_g, target)
│   
│   # 反向传播
│   loss.backward()
│   # 梯度流向:
│   # - Conv LoRA 参数 (专家参数)
│   # - Gate 参数 (门控网络)
│   
│   optimizer.step()
│   # 更新后，gate会更倾向选择高质量专家
│
└─ [效果]
    # 下一次前向传播时
    # quality_bias[高质量expert] 增大
    # → 选择概率增大
    # → 正反馈循环

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

长期效应
│
├─ 专家分工 (Specialization)
│   Expert 1 → 擅长大目标 (质量高 on large lesions)
│   Expert 2 → 擅长小目标 (质量高 on small lesions)
│   Expert 3 → 擅长复杂边界
│   ...
│
├─ 自适应选择 (Adaptation)
│   图像 A (大目标) → 自动选 Expert 1
│   图像 B (小目标) → 自动选 Expert 2
│
└─ 性能提升
    原始 Conv-LoRA (TopK=1, 无反馈): IoU ~0.75
    + GSPO (TopK=G, 质量反馈):      IoU ~0.78-0.80
```

### 5.3 动作空间分析

**标准 Conv-LoRA (无GSPO)**：
```
Action Space = {1, 2, ..., M}  # M=4或8
每次选择 1 个专家 (TopK=1)
动作数 = M
```

**GSPO-Conv-LoRA**：
```
Action Space = Combinations(M, G)  # 例如 M=8, G=4
每次选择 G 个专家 (TopK=G)
动作数 = C(M, G) = M! / (G! * (M-G)!)
例如: C(8, 4) = 70 种组合

实际实现: TopK softmax采样
不是枚举所有组合，而是：
1. Gate 输出 [M] 概率
2. TopK 选出前 G 个
3. G个专家并行执行并平均
```

**关键差异**：
- **标准**: 确定性选择1个专家
- **GSPO**: 随机性选择G个专家（dropout+噪声产生多样性）

### 5.4 Weak样本中的强化学习

**核心挑战**: Weak 样本没有 GT，如何计算 Reward？

**解决方案**: 使用一致性质量作为 Reward 代理

```
┌──────────────────────────────────────────────────┐
│     Weak 样本的质量代理 (Quality Proxy)           │
├──────────────────────────────────────────────────┤
│                                                    │
│  真实 Reward (有GT):                               │
│    R_true = IoU(pred, GT)                         │
│                                                    │
│  代理 Reward (无GT):                               │
│    R_proxy = q_cons = avg(IoU(teacher_pred_i,     │
│                                 teacher_pred_j))   │
│                                                    │
│  假设:                                             │
│    - 高质量样本 → Teacher 预测一致 → q_cons高      │
│    - 低质量样本 → Teacher 预测不一致 → q_cons低    │
│                                                    │
│  实验验证:                                         │
│    Correlation(R_true, R_proxy) ~ 0.7-0.8        │
│    证明代理的有效性                                │
│                                                    │
└──────────────────────────────────────────────────┘
```

**在 GSPO 中的应用**：

```python
def compute_advantage(pred, gt_mask, is_labeled, quality_proxy):
    """统一处理 labeled 和 weak 样本"""
    
    # 计算质量分数
    quality = []
    for i in range(len(pred)):
        if is_labeled[i]:
            # Labeled: 真实 IoU
            q = IoU(pred[i], gt_mask[i])
        else:
            # Weak: 一致性代理
            q = quality_proxy[i]
        quality.append(q)
    
    # 组内 baseline
    quality_mean = mean(quality)
    
    # Advantage
    advantage = quality - quality_mean
    
    return advantage
```

**效果**：
- Labeled 和 Weak 样本**统一处理**
- 都参与 GSPO 的策略优化
- Weak 样本虽然没有精确标签，但仍能提供**质量反馈信号**

---

## 6. 参数配置

### 6.1 关键超参数

#### 6.1.1 数据相关

```yaml
labeled_ratio: 0.1          # 有标注数据比例
weak_ratio: 0.9             # 弱标注数据比例

box_jitter_mode: expand_only  # Box 扰动模式
box_expand_ratio: 0.15      # 外扩比例 (相对bbox尺寸)
```

#### 6.1.2 Conv-LoRA MoE

```yaml
lora:
  r: 4                      # LoRA rank (典型: 4-16)
  alpha: 32                 # LoRA scaling (典型: 2*r 或 4*r)
  conv_lora_expert_num: 4   # 专家数量 M
  
gspo:
  enabled: True
  group_size: 4             # 组大小 G (建议 = expert_num)
  warmup_epochs: 5          # GSPO 预热轮数
  quality_momentum: 0.9     # 质量历史 EMA 动量
  advantage_temperature: 5.0  # Advantage softmax 温度
  contrastive_weight: 0.1   # 对比损失权重
```

#### 6.1.3 半监督

```yaml
semi_supervised:
  # EMA Teacher
  ema_momentum: 0.999       # Teacher 更新动量 (接近1更平滑)
  ema_update_freq: step     # 更新频率: step 或 epoch
  
  # 质量评估
  quality_k_samples: 5      # Teacher 采样次数 K
  quality_consistency_weight: 0.7  # 一致性质量权重 α
  quality_min_threshold: 0.6  # 质量过滤阈值 q_min
  quality_weighting_beta: 10.0  # 质量权重 sigmoid 斜率
  
  # 伪标签
  use_soft_label: False     # 是否使用软标签 (False=硬标签)
  pseudo_label_threshold: 0.5  # 硬标签二值化阈值
  
  # 损失权重
  pseudo_lambda_init: 0.0   # 初始伪监督权重
  pseudo_lambda_final: 1.0  # 最终伪监督权重
  pseudo_lambda_warmup_epochs: 5  # Warmup 轮数
  consistency_lambda: 0.1   # 一致性损失权重
```

#### 6.1.4 训练

```yaml
training:
  max_epochs: 30
  batch_size: 4
  per_gpu_batch_size: 1
  lr: 1e-4
  patience: 30              # Early stopping 耐心值
  val_check_interval: 1.0   # 验证频率
```

### 6.2 冻结 vs 训练参数

```
模型参数分布 (SAM-ViT-Huge + Conv-LoRA)
│
├── 冻结参数 (~635M, 99.3%)
│   ├── SAM Image Encoder (ViT-Huge)
│   │   - Patch Embedding: ~200K
│   │   - Transformer Blocks: ~630M
│   │   - Output Projection: ~5M
│   │
│   ├── SAM Prompt Encoder
│   │   - Point/Box Embedding: ~100K
│   │   - Mask Embedding: ~50K
│   │
│   └── SAM Mask Decoder (部分)
│       - Attention layers: ~10M
│       - Output Head: ~500K
│
└── 训练参数 (~4.5M, 0.7%)
    ├── Conv-LoRA (每层)
    │   ├── Down projection: C × r
    │   ├── Expert Convs: M × (r × r × k²)
    │   ├── Gate network: C × M
    │   ├── Up projection: r × C
    │   └── 单层约: ~100K (r=4, M=4, C=256)
    │       × 10层 = ~1M
    │
    ├── FC Adapter (可选, 每层)
    │   ├── Down: C × d
    │   ├── Up: d × C
    │   └── 单层约: ~64K (d=64, C=256)
    │       × 24层 = ~1.5M
    │
    └── Quality Tracking (不计入参数)
        ├── expert_quality_history: M × 1
        └── expert_usage_count: M × 1

总计: ~640M, 其中可训练 ~4.5M (0.7%)
```

### 6.3 推荐配置组合

#### 6.3.1 基础配置 (入门)

```bash
python run_semi_supervised_train_v3_fixed.py \
  --labeled_ratio 0.1 \
  --rank 4 \
  --expert_num 4 \
  --quality_k_samples 5 \
  --quality_min_threshold 0.6 \
  --ema_momentum 0.999 \
  --max_epochs 30 \
  --batch_size 4 \
  --lr 1e-4
```

#### 6.3.2 高性能配置 (推荐)

```bash
python run_semi_supervised_train_v3_fixed.py \
  --labeled_ratio 0.1 \
  --rank 8 \
  --expert_num 8 \
  --quality_k_samples 7 \
  --quality_min_threshold 0.65 \
  --ema_momentum 0.995 \
  --gspo_enable \
  --gspo_group_size 8 \
  --adapter_enable \
  --adapter_dim 64 \
  --max_epochs 50 \
  --batch_size 8 \
  --lr 5e-5
```

#### 6.3.3 快速验证配置

```bash
python run_semi_supervised_train_v3_fixed.py \
  --labeled_ratio 0.2 \     # 更多 labeled (更快收敛)
  --rank 4 \
  --expert_num 4 \
  --quality_k_samples 3 \   # 更少采样 (更快)
  --max_epochs 10 \
  --batch_size 8
```

---

## 7. 常见问题解答

### Q1: 强化学习的反馈环路具体作用在哪里？

**A**: 反馈环路作用在 **MoE 门控网络**：

```
[质量评估] → [专家质量历史] → [质量偏置] → [门控决策] → [专家选择]
      ↑                                                      ↓
      └──────────────────── [分割质量] ←──────────────────┘
```

- **位置**: Conv-LoRA 每一层的 MoE Gate
- **机制**: 根据历史质量调整专家选择概率
- **效果**: 高质量专家被更频繁选择

### Q2: GSPO 的动作空间是什么？

**A**: 动作 = **选择哪些专家组合**

- **标准 MoE**: 动作空间 = {1, 2, ..., M}，选1个
- **GSPO**: 动作空间 = C(M, G)，选G个组合
  - 例如 M=8, G=4 → 70种可能组合
  - 实际通过 TopK 软采样实现

### Q3: 损失函数包含哪些项？

**A**: 4 个主要损失项：

```python
L = L_supervised        # BCE/Dice (labeled样本)
  + λ_u * L_pseudo     # 加权BCE (weak样本 + 伪标签)
  + λ_gspo * L_gspo    # Advantage加权 + 对比损失
  + λ_c * L_consistency  # Box jitter一致性
```

### Q4: 哪些部分参与训练，哪些被冻结？

**A**:

| 模块 | 参数量 | 状态 | requires_grad |
|------|--------|------|---------------|
| SAM Encoder (ViT) | ~630M | 冻结 | False |
| SAM Decoder (主干) | ~10M | 冻结 | False |
| Conv-LoRA (Down/Up/Expert/Gate) | ~2M | **训练** | True |
| FC Adapter (可选) | ~1.5M | **训练** | True |
| EMA Teacher (所有) | ~640M | 冻结 | False |

**梯度流向**:
```
Loss → Student Pred → Conv-LoRA ✓
                   → FC Adapter ✓
                   → SAM Decoder ✗
                   → SAM Encoder ✗
     
     → Teacher Pred → (无梯度)
```

### Q5: Teacher 为什么需要 K 次采样？

**A**: 产生预测多样性，用于**无监督质量评估**

- **单次预测**: 确定性，无法判断质量
- **K次预测**: 通过一致性推断质量
  - 一致 → 模型有把握 → 高质量
  - 不一致 → 模型困惑 → 低质量

### Q6: Weak 样本如何参与 GSPO？

**A**: 使用**一致性质量作为 Reward 代理**

```python
# Labeled: 真实 IoU
if is_labeled:
    reward = IoU(pred, GT)

# Weak: 一致性质量
else:
    reward = quality_proxy  # 来自 Teacher K次采样

# 统一计算 Advantage
advantage = reward - mean(group_rewards)
```

---

## 8. 总结

### 8.1 系统核心创新

1. **参数高效**: 只训练 0.7% 参数即可微调 SAM
2. **半监督学习**: 利用 90% 弱标注数据
3. **无监督质量评估**: 无需 GT 即可评估伪标签质量
4. **质量感知策略**: GSPO 根据历史质量优化专家选择
5. **统一框架**: Labeled 和 Weak 样本统一处理

### 8.2 技术栈

```
半监督学习 + 参数高效微调 + 强化学习
    │              │                │
    ▼              ▼                ▼
Teacher-Student  Conv-LoRA MoE    GSPO
    │              │                │
    ├─ EMA 更新    ├─ 多专家卷积   ├─ 组级采样
    ├─ K次采样     ├─ 质量门控     ├─ Advantage
    ├─ 质量评估    └─ LoRA微调     └─ 策略优化
    └─ 伪标签生成
```

### 8.3 预期性能

| 配置 | mIoU (验证集) |
|------|--------------|
| 只用 10% labeled | ~0.72 |
| + 90% weak (简单伪标签) | ~0.75 |
| + 质量感知 | ~0.77 |
| + GSPO | ~0.78-0.80 |

---

**文档版本**: v1.0  
**创建日期**: 2025-12-31  
**作者**: AutoML System  
**适用版本**: AutoGluon MultiModal + Conv-LoRA + GSPO
