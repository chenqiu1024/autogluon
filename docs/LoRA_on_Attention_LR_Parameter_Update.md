# LoRA on Attention - 学习率参数更新

## 更新日期
2024-12-08

## 新功能
添加了 `--lr` 命令行参数，允许用户自定义学习率，无需修改代码。

## 修改内容

### 1. 命令行参数
在 `run_semantic_segmentation.py` 中添加：
```python
parser.add_argument("--lr", type=float, default=None,
                    help="Learning rate (default: auto based on task, typically 1e-4 or 3e-4)")
```

### 2. 逻辑处理
```python
# 获取默认学习率
validation_metric, loss, max_epoch, lr = get_default_training_setting(dataset_name)

# 如果用户通过 CLI 指定了学习率，则覆盖默认值
if args.lr is not None:
    lr = args.lr
    print(f"Using custom learning rate: {lr}")

# 应用到 hyperparameters
hyperparameters.update({
    "optim.lr": lr,
    # ...
})
```

## 使用方法

### 基础用法（使用默认学习率）
```bash
python3 run_semantic_segmentation.py \
  --task isic2017 \
  --decoder_attn_lora_enable \
  --decoder_attn_lora_r 8 \
  --output_dir outputs/default_lr
```
输出：使用默认 lr=1e-4（isic2017）

### 自定义学习率
```bash
python3 run_semantic_segmentation.py \
  --task isic2017 \
  --decoder_attn_lora_enable \
  --decoder_attn_lora_r 8 \
  --lr 3e-4 \
  --output_dir outputs/custom_lr_3e4
```
输出：`Using custom learning rate: 0.0003`

## 默认学习率（按任务）

| 任务 | 默认学习率 | 说明 |
|------|-----------|------|
| isic2017 | 1e-4 | 医学图像分割 |
| polyp | 1e-4 | 息肉分割 |
| camo_sem_seg | 1e-4 | 伪装对象分割 |
| SBU-shadow | 1e-4 | 阴影检测 |
| road_segmentation | 3e-4 | 道路分割（较高） |
| leaf_disease_segmentation | 3e-4 | 叶病分割（较高） |

## 推荐学习率

### LoRA on Attention
- **轻量级（r=4）**: 2e-4 ~ 5e-4
- **标准配置（r=8）**: 2e-4 ~ 3e-4
- **高表达力（r=16）**: 1e-4 ~ 2e-4

### 组合配置
- **Conv-LoRA + LoRA Attention**: 1e-4 ~ 3e-4
- **全部 PEFT**: 1e-4 ~ 2e-4（保守）

## 学习率搜索建议

使用 `--quick_test` 快速测试不同学习率：

```bash
# 测试 1: 默认
python3 run_semantic_segmentation.py \
  --task isic2017 \
  --decoder_attn_lora_enable \
  --decoder_attn_lora_r 8 \
  --quick_test 50 \
  --output_dir outputs/lr_test_1e4

# 测试 2: 2倍
python3 run_semantic_segmentation.py \
  --task isic2017 \
  --decoder_attn_lora_enable \
  --decoder_attn_lora_r 8 \
  --lr 2e-4 \
  --quick_test 50 \
  --output_dir outputs/lr_test_2e4

# 测试 3: 3倍
python3 run_semantic_segmentation.py \
  --task isic2017 \
  --decoder_attn_lora_enable \
  --decoder_attn_lora_r 8 \
  --lr 3e-4 \
  --quick_test 50 \
  --output_dir outputs/lr_test_3e4
```

然后对比验证集指标，选择最佳学习率进行全量训练。

## 向后兼容性
- ✅ 完全向后兼容
- ✅ 不指定 `--lr` 时使用任务默认值
- ✅ 旧脚本无需修改

## 相关文档
- `LoRA_on_Attention_Design.md` - 完整设计文档
- `LoRA_on_Attention_QuickStart.md` - 快速开始指南
- `example_train_with_custom_lr.sh` - 使用示例脚本

## 文件修改清单
- ✅ `examples/automm/Conv-LoRA/run_semantic_segmentation.py` - 添加 CLI 参数
- ✅ `docs/LoRA_on_Attention_Design.md` - 更新参数说明
- ✅ `docs/LoRA_on_Attention_QuickStart.md` - 添加使用示例
- ✅ `examples/automm/Conv-LoRA/example_train_with_custom_lr.sh` - 新增示例脚本
- ✅ `docs/LoRA_on_Attention_LR_Parameter_Update.md` - 本文档

---

**功能状态**: ✅ 已实现并测试  
**维护者**: AutoGluon Team  
**版本**: v1.1

