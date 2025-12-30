# 方案设计文档

## 核心公式

### 公式 2.1: EMA 更新
\`\`\`
θ_teacher ← m * θ_teacher + (1-m) * θ_student
m = 0.999
\`\`\`

### 公式 2.2: 质量评估
\`\`\`
q_cons = mean(IoU(p_i, p_j)) for i<j
q_conf = mean(max(p, 1-p))
q = α * q_cons + (1-α) * q_conf, α=0.7
\`\`\`

### 公式 2.3: 质量加权
\`\`\`
w(q) = sigmoid(β(q - q0))
β = 10, q0 = 0.5
\`\`\`

### 公式 2.4: 总损失
\`\`\`
L = L_supervised + λ_u * Σ w(q) * L_pseudo + λ_c * L_cons
\`\`\`

### 公式 2.5: GSPO 扩展
\`\`\`
q_quality = IoU(pred, gt) if labeled else q_cons
A = q_quality - mean(q_g)
\`\`\`

## 创新点

1. **方法论**：首次将策略优化与半监督自训练结合
2. **技术**：多次采样一致性作为无标注质量 proxy
3. **应用**：医学弱监督场景显著降低标注成本
