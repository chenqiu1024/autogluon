# 质量感知半监督分割方案 - 完整索引

## 📖 文档导航（推荐阅读顺序）

### 第一步：快速了解
1. **[README.md](README.md)** - 3步快速开始，5分钟上手

### 第二步：理解原理
2. **[design_overview_zh.md](design_overview_zh.md)** - 方案原理、核心公式
3. **[formulas_reference.md](formulas_reference.md)** - 公式速查表

### 第三步：实现细节
4. **[implementation_details_zh.md](implementation_details_zh.md)** - 代码结构、超参调优

### 第四步：实验设计
5. **[experiment_protocol_zh.md](experiment_protocol_zh.md)** - 数据拆分、实验设计

### 第五步：完整总结
6. **[FINAL_SUMMARY.md](FINAL_SUMMARY.md)** - 最终总结、完整状态

---

## 🗂️ 代码模块索引

### 核心模块 (`examples/automm/Conv-LoRA/semi_supervised/`)

```
semi_supervised/
├── __init__.py                 # 模块导出
├── ema_teacher.py              # EMA Teacher（公式 2.1）
├── quality_estimator.py        # 质量评估器（公式 2.2）
├── pseudo_label_gen.py         # 伪标签生成器（公式 2.3）
├── data_module.py              # 数据加载模块
└── gspo_semi_trainer.py        # GSPO 扩展（公式 2.5）
```

### 工具脚本 (`examples/automm/Conv-LoRA/`)

```
├── prepare_semi_supervised_data.py   # 数据拆分脚本
└── run_semi_supervised_train.sh      # 一键运行示例
```

---

## 🎯 快速查找

### 我想...

- **快速开始训练** → [README.md](README.md)
- **理解核心原理** → [design_overview_zh.md](design_overview_zh.md)
- **查看公式定义** → [formulas_reference.md](formulas_reference.md)
- **了解实现细节** → [implementation_details_zh.md](implementation_details_zh.md)
- **设计实验对比** → [experiment_protocol_zh.md](experiment_protocol_zh.md)
- **查看完整状态** → [FINAL_SUMMARY.md](FINAL_SUMMARY.md)

### 我遇到了...

- **显存不足** → [implementation_details_zh.md](implementation_details_zh.md) Q1
- **伪标签率低** → [implementation_details_zh.md](implementation_details_zh.md) Q2
- **训练不稳定** → [implementation_details_zh.md](implementation_details_zh.md) Q3

---

## 📋 所有文件清单

### 文档文件（6个）

| # | 文件名 | 说明 |
|---|--------|------|
| 1 | README.md | 快速开始 |
| 2 | design_overview_zh.md | 方案原理 |
| 3 | formulas_reference.md | 公式速查 |
| 4 | experiment_protocol_zh.md | 实验协议 |
| 5 | implementation_details_zh.md | 实现细节 |
| 6 | FINAL_SUMMARY.md | 最终总结 |

### 代码文件（6个）

| # | 文件名 | 代码行数 |
|---|--------|---------|
| 1 | \_\_init\_\_.py | ~20 |
| 2 | ema_teacher.py | ~150 |
| 3 | quality_estimator.py | ~180 |
| 4 | pseudo_label_gen.py | ~120 |
| 5 | data_module.py | ~100 |
| 6 | gspo_semi_trainer.py | ~200 |

### 工具脚本（2个）

| # | 文件名 | 代码行数 |
|---|--------|---------|
| 1 | prepare_semi_supervised_data.py | ~250 |
| 2 | run_semi_supervised_train.sh | ~30 |

**总计**: 约 1050 行核心代码

---

## ✅ 完成状态

- [x] 所有文档编写完成
- [x] 所有代码实现完成
- [x] 所有公式对应完成
- [x] 使用示例编写完成
- [x] 问题排查文档完成

**状态**: Production Ready ✅  
**日期**: 2025-12-30  
**版本**: v1.0

---

## 🚀 从这里开始

1. 阅读 [README.md](README.md) 快速开始
2. 准备 ISIC2017 数据集
3. 运行数据拆分脚本
4. 执行训练并验证结果

**祝训练顺利！🎉**
