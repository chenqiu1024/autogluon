# 半/弱监督在 SAM + Conv‑LoRA + GSPO 框架下的实践手册

本页汇总了可直接落地于当前代码库（SAM 主干冻结，训练 Conv‑LoRA(MoE)+Adapter，GSPO 在 MoE 上做质量驱动优化，支持 box prompt 注入）的半监督与弱监督方法：原理、框架图、流程与必要公式，便于工程快速实现。

结构：
- `overview.md`：方法概述、流程图与公式（推荐先读）
- `implementation.md`：可运行的实现建议、配置开关和伪代码

代码关联（参考实现位置）：
- Trainer / group sampling: [examples/automm/Conv-LoRA/gspo_trainer.py](examples/automm/Conv-LoRA/gspo_trainer.py)
- Conv‑LoRA / MoE / Adapter 层: [multimodal/src/autogluon/multimodal/models/adaptation_layers.py](multimodal/src/autogluon/multimodal/models/adaptation_layers.py)
- 训练入口与数据/loader: [examples/automm/Conv-LoRA/run_semantic_segmentation.py](examples/automm/Conv-LoRA/run_semantic_segmentation.py)
- Learner: [multimodal/src/autogluon/multimodal/learners/semantic_segmentation.py](multimodal/src/autogluon/multimodal/learners/semantic_segmentation.py)

阅读顺序：`overview.md` → `implementation.md`。
