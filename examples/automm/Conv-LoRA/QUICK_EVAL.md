# 快速评估 RLOO 模型

## 🚀 一键评估

运行以下命令即可评估所有 RLOO checkpoint 并生成对比报告：

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA

# 方式 1: 使用便捷脚本
bash run_evaluation.sh

# 方式 2: 直接运行 Python 脚本
conda activate conv-lora
python compare_rloo_models.py \
  --rloo_output_dir outputs/rloo/isic2017/20251127 \
  --base_checkpoint_path AutogluonModels/ag-20251126_062717 \
  --task isic2017
```

## 📊 预期输出

评估完成后会显示对比表格：

```
================================================================================
评估结果对比
================================================================================
模型                                      IoU          Dice         样本数
--------------------------------------------------------------------------------
Baseline (Before RLOO)                   0.XXXX±0.XXXX  0.XXXX±0.XXXX    XXXX
RLOO - epoch=00 (reward=1.8524)          0.XXXX±0.XXXX  0.XXXX±0.XXXX    XXXX
RLOO - epoch=01 (reward=1.8479)          0.XXXX±0.XXXX  0.XXXX±0.XXXX    XXXX
RLOO - epoch=02 (reward=1.8514)          0.XXXX±0.XXXX  0.XXXX±0.XXXX    XXXX
================================================================================

RLOO 训练改进:
--------------------------------------------------------------------------------
RLOO - epoch=00:
  IoU:  +X.XXXX (+X.XX%)
  Dice: +X.XXXX (+X.XX%)
```

## 📁 结果文件

评估结果保存在：

```
outputs/rloo/isic2017/20251127/evaluation_results/
├── eval_baseline_model.json          # 基线模型结果
├── eval_rloo_rloo-epoch=00-....json  # RLOO Epoch 0 结果
├── eval_rloo_rloo-epoch=01-....json  # RLOO Epoch 1 结果
├── eval_rloo_rloo-epoch=02-....json  # RLOO Epoch 2 结果
└── comparison_report_YYYYMMDD_HHMMSS.json  # 完整对比报告
```

## 🎯 单独评估特定 Checkpoint

如果只想评估某个特定的 checkpoint：

```bash
conda activate conv-lora
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA

# 评估最佳 RLOO 模型
python evaluate_rloo_lightning.py \
  --checkpoint_path outputs/rloo/isic2017/20251127/checkpoints/rloo-epoch=00-rloo_mean_reward=1.8524.ckpt \
  --base_predictor_path AutogluonModels/ag-20251126_062717 \
  --task isic2017 \
  --output_file results_best.json

# 查看结果
cat results_best.json | jq
```

## 📚 详细文档

- **完整评估指南**: `EVALUATION_GUIDE.md`
- **RLOO 训练指南**: `RLOO_REAL_TRAINING_GUIDE_zh.md`
- **Bug 修复记录**: `BUGFIX_RLOO_TRAINING.md`

## ⚡ 常见问题

### Q: 如何选择最佳 checkpoint？

A: 查看对比报告中 IoU/Dice 最高的模型。通常 `epoch=00` 或 `epoch=02` 表现较好。

### Q: 评估需要多长时间？

A: 取决于测试集大小和 GPU。对于 ISIC2017（约 600 张图片），大约 5-10 分钟。

### Q: 如何在其他数据集上评估？

A: 修改 `--task` 参数，如：
```bash
python compare_rloo_models.py \
  --task leaf_disease_segmentation \
  ...
```

---

**快速评估完成后，查看结果并分析 RLOO 训练的效果！** 🎊

