# RLOO 训练 Bug 修复记录

## 问题描述

运行以下命令时出错：

```bash
nohup python run_semantic_segmentation_rloo_real.py \
  --task isic2017 \
  --ckpt_path AutogluonModels/ag-20251126_062717 \
  --output_dir outputs/rloo/isic2017/20251127 \
  --num_generations 4 \
  --beta 0.05 \
  --reward_type combo \
  --learning_rate 1e-5 \
  --epochs 3 \
  --batch_size 2 > train_rloo-251127.log 2>&1 &
```

错误信息：

```
AttributeError: 'SemanticSegmentationLearner' object has no attribute '_data_module'
```

## 根本原因

`SemanticSegmentationLearner` 不存储 `_data_module` 作为实例属性。相反，它通过 `get_datamodule_per_run()` 方法动态创建 DataModule。

## 修复内容

### 1. 修复 DataModule 创建

**原代码** (`run_semantic_segmentation_rloo_real.py` 第 104 行):

```python
def setup_model_and_datamodule(self, predictor: MultiModalPredictor):
    learner = predictor._learner
    model = learner._model
    datamodule = learner._data_module  # ❌ 不存在
    return model, datamodule
```

**修复后**:

```python
def setup_model_and_datamodule(self, predictor: MultiModalPredictor):
    learner = predictor._learner
    model = learner._model
    
    # 创建 DataModule（调用 learner 的方法）
    datamodule = learner.get_datamodule_per_run(
        df_preprocessor=learner._df_preprocessor,
        data_processors=learner._data_processors,
        per_gpu_batch_size=learner._config.env.per_gpu_batch_size,
        num_workers=learner._config.env.num_workers,
        is_train=True,
    )
    
    return model, datamodule
```

### 2. 修复优化器配置

**原代码** (第 201-209 行):

```python
# 获取优化器配置
learner = predictor._learner
optim_kwargs = learner._config.optim  # DictConfig 对象

# 更新学习率
optim_kwargs['lr'] = self.learning_rate  # ❌ 不能直接赋值
```

**修复后**:

```python
# 获取优化器配置
learner = predictor._learner

# 获取验证指标
validation_metric, custom_metric_func = learner.get_validation_metric_per_run(
    output_shape=learner._output_shape
)

# 获取 loss 函数
loss_func, aug_loss_func = learner.get_loss_func_per_run(learner._config)

# 构建优化器配置（转换为普通字典）
optim_config = learner._config.optim
optim_kwargs = dict(
    optim_type=optim_config.optim_type,
    lr_choice=optim_config.lr_choice,
    lr_schedule=optim_config.lr_schedule,
    lr=self.learning_rate,  # ✅ 使用我们指定的学习率
    lr_decay=optim_config.lr_decay,
    end_lr=optim_config.end_lr,
    lr_mult=optim_config.lr_mult,
    weight_decay=optim_config.weight_decay,
    warmup_steps=optim_config.warmup_steps,
    loss_func=loss_func,
    validation_metric=validation_metric,
    validation_metric_name=learner._validation_metric_name,
    custom_metric_func=custom_metric_func,
)
```

### 3. 添加 Checkpoint Callback

**原代码** (第 212-221 行):

```python
trainer = pl.Trainer(
    max_epochs=self.epochs,
    accelerator='auto',
    devices=1,
    default_root_dir=self.output_dir,
    enable_progress_bar=True,
    log_every_n_steps=10,
    enable_checkpointing=True,  # ❌ 但没有配置 callback
    logger=True,
)
```

**修复后**:

```python
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(self.output_dir, 'checkpoints'),
    filename='rloo-{epoch:02d}-{rloo_mean_reward:.4f}',
    monitor='rloo_mean_reward',
    mode='max',
    save_top_k=3,
    save_last=True,
)

trainer = pl.Trainer(
    max_epochs=self.epochs,
    accelerator='auto',
    devices=1,
    default_root_dir=self.output_dir,
    enable_progress_bar=True,
    log_every_n_steps=10,
    callbacks=[checkpoint_callback],  # ✅ 添加 callback
    logger=True,
)
```

### 4. 修复 Checkpoint 路径打印

**原代码** (第 233 行):

```python
print(f"  最终 checkpoint: {trainer.checkpoint_callback.best_model_path}")
```

**修复后**:

```python
if checkpoint_callback.best_model_path:
    print(f"  最佳 checkpoint: {checkpoint_callback.best_model_path}")
if checkpoint_callback.last_model_path:
    print(f"  最后 checkpoint: {checkpoint_callback.last_model_path}")
```

## 验证

修复后，所有导入测试通过：

```bash
✓ RLOOTrainer 导入成功
✓ PyTorch 2.7.1+cu126
✓ PyTorch Lightning 2.5.6
✓ AutoGluon MultiModal 导入成功
```

## 如何使用修复后的版本

### 1. 清理之前的日志

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA
rm train_rloo-251127.log  # 清理旧日志
```

### 2. 激活环境并重新运行

```bash
# 激活环境
conda activate conv-lora

# 重新运行训练
nohup python run_semantic_segmentation_rloo_real.py \
  --task isic2017 \
  --ckpt_path AutogluonModels/ag-20251126_062717 \
  --output_dir outputs/rloo/isic2017/20251127 \
  --num_generations 4 \
  --beta 0.05 \
  --reward_type combo \
  --learning_rate 1e-5 \
  --epochs 3 \
  --batch_size 2 > train_rloo-251127.log 2>&1 &
```

### 3. 监控训练进度

```bash
# 查看日志
tail -f train_rloo-251127.log

# 或者使用 watch
watch -n 5 tail -30 train_rloo-251127.log
```

## 预期输出

训练应该正常启动，输出如下：

```
================================================================================
开始真实的 RLOO 训练
================================================================================
配置:
  Checkpoint: AutogluonModels/ag-20251126_062717
  输出目录: outputs/rloo/isic2017/20251127
  候选数量 (G): 4
  KL 系数 (β): 0.05
  Reward 类型: combo
  学习率: 1e-05
  Epochs: 3
  Batch size: 2
================================================================================

参数统计:
  可训练参数: 514,560
  冻结参数: 637,951,488
  总参数: 638,466,048

开始训练循环...

Epoch 1/3
  Batch 1: rloo_mean_reward=0.xxxx, rloo_mean_iou=0.xxxx, ...
  ...
```

## 修复 2：get_validation_metric_per_run() 参数错误

### 错误信息

```
TypeError: BaseLearner.get_validation_metric_per_run() got an unexpected keyword argument 'output_shape'
```

### 原因

`BaseLearner.get_validation_metric_per_run()` 方法**不接受任何参数**，它直接使用 `self._output_shape`。

### 修复

```python
# 修复前
validation_metric, custom_metric_func = learner.get_validation_metric_per_run(
    output_shape=learner._output_shape  # ❌ 不需要这个参数
)

# 修复后
validation_metric, custom_metric_func = learner.get_validation_metric_per_run()  # ✅ 无参数
```

## 修复 3：Lightning 包命名空间不一致

### 错误信息

```
TypeError: `model` must be a `LightningModule` or `torch._dynamo.OptimizedModule`, got `RLOOSemanticSegmentationLitModule`
```

### 原因

- AutoGluon 使用 `lightning.pytorch` (新包名)
- 我们的脚本使用 `pytorch_lightning` (旧包名)
- 虽然是同一个包，但命名空间不同，导致 `isinstance` 和 `issubclass` 检查失败

### 验证问题

```python
import pytorch_lightning as pl
from lit_semantic_seg_rloo import RLOOSemanticSegmentationLitModule

# ❌ 返回 False
print(issubclass(RLOOSemanticSegmentationLitModule, pl.LightningModule))

import lightning.pytorch as L
# ✅ 返回 True  
print(issubclass(RLOOSemanticSegmentationLitModule, L.LightningModule))
```

### 修复

**修复前**:
```python
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

pl.seed_everything(seed)
trainer = pl.Trainer(...)
```

**修复后**:
```python
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

seed_everything(seed)
trainer = Trainer(...)
```

## 修复 4：训练数据为 None

### 错误信息

```
TypeError: 'NoneType' object is not subscriptable
File "/root/autodl-tmp/works/autogluon/multimodal/src/autogluon/multimodal/data/preprocess_dataframe.py", line 562, in transform_semantic_segmentation_img
    col_value = df[col_name]
```

### 原因

- 加载保存的模型后，`learner._train_data` 和 `learner._tuning_data` 为 None
- `get_datamodule_per_run()` 依赖这些属性来创建 DataModule
- DataModule 初始化时找不到训练数据

### 修复

添加方法重新加载训练数据：

```python
def load_train_data(self, task: str):
    """重新加载训练数据集"""
    dataset_name = task
    dataset_dir = os.path.join(f"datasets/{dataset_name}", dataset_name)
    train_csv = os.path.join(dataset_dir, "train.csv")
    
    train_df = pd.read_csv(train_csv)
    
    # 展开路径
    for col in ["image", "label"]:
        if col in train_df.columns:
            train_df[col] = train_df[col].apply(
                lambda ele: os.path.join(dataset_dir, ele)
            )
    
    return train_df

def setup_model_and_datamodule(self, predictor, task):
    learner = predictor._learner
    model = learner._model
    
    # ✅ 重新加载训练数据
    train_data = self.load_train_data(task)
    
    # ✅ 手动设置到 learner
    learner._train_data = train_data
    learner._tuning_data = None
    
    # ✅ 现在创建 DataModule
    datamodule = learner.get_datamodule_per_run(...)
    
    return model, datamodule
```

### 验证

```bash
✓ 成功加载训练数据: 2000 行
  列: ['Unnamed: 0', 'image', 'label']
```

## 修复 4b：验证数据也为 None

### 问题

修复了训练数据后，在 setup 验证集时又出现同样的错误：

```
File "/root/autodl-tmp/works/autogluon/multimodal/src/autogluon/multimodal/data/datamodule.py", line 126, in setup
    self.set_dataset(VALIDATE)
TypeError: 'NoneType' object is not subscriptable
```

### 修复

同时加载训练和验证数据：

```python
def load_train_data(self, task: str):
    """重新加载训练和验证数据集"""
    dataset_dir = os.path.join(f"datasets/{task}", task)
    
    # 加载训练集
    train_df = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
    # 展开路径...
    
    # 加载验证集
    val_df = None
    val_csv = os.path.join(dataset_dir, "val.csv")
    if os.path.exists(val_csv):
        val_df = pd.read_csv(val_csv)
        # 展开路径...
    
    return train_df, val_df

def setup_model_and_datamodule(self, predictor, task):
    learner = predictor._learner
    model = learner._model
    
    # ✅ 同时加载训练和验证数据
    train_data, val_data = self.load_train_data(task)
    
    # ✅ 都设置到 learner
    learner._train_data = train_data
    learner._tuning_data = val_data  # 必须设置，否则 DataModule setup 会失败
    
    datamodule = learner.get_datamodule_per_run(...)
    return model, datamodule
```

### 验证

```bash
✓ 训练数据: 2000 行
✓ 验证数据: 150 行
```

## 修复完成时间

2025-11-27（第五次修复）

## 相关文件

- `run_semantic_segmentation_rloo_real.py` - 主要修复文件
- `lit_semantic_seg_rloo.py` - 无需修改
- `RLOO_REAL_TRAINING_GUIDE_zh.md` - 使用指南

## 注意事项

1. **必须激活 conda 环境**：`conda activate conv-lora`
2. **检查 checkpoint 路径**：确保 `AutogluonModels/ag-20251126_062717` 存在
3. **监控内存使用**：如果 OOM，降低 `--batch_size` 或 `--num_generations`
4. **网络连接**：可能需要从 Hugging Face 下载模型配置

## 后续步骤

1. 重新运行训练命令
2. 监控训练进度和指标
3. 检查 checkpoint 保存
4. 评估训练后的模型性能

---

**修复完成！可以重新开始训练。** ✅

