# 公式速查表

## EMA 更新
```
θ_T^(t) = m * θ_T^(t-1) + (1-m) * θ_S^(t)
m = 0.999
```

## 质量评估
```
q_cons = (2/K(K-1)) * Σ IoU(p_i, p_j) for i<j
q_conf = (1/K) * Σ mean(max(p_k, 1-p_k))
q = α * q_cons + (1-α) * q_conf, α=0.7
```

## 质量加权
```
w(q) = sigmoid(β(q - q_0))
β = 10, q_0 = 0.5
```

## 损失函数
```
L = L_s + λ_u(t) * L_u + λ_c * L_cons + L_GSPO
λ_u(t) = min(1.0, t/5)
λ_c = 0.1
```

## GSPO 扩展
```
q_quality = IoU(pred, gt) if labeled else q_cons
A = q_quality - mean(q_g)
```




