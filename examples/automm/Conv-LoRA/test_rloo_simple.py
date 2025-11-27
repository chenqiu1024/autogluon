"""
简单的 RLOO 测试脚本
用于验证基本功能
"""
import torch
from autogluon.multimodal import MultiModalPredictor

print("="*80)
print("RLOO 实现测试")
print("="*80)

# 加载模型
ckpt_path = "AutogluonModels/ag-20251126_062717"
print(f"\n1. 加载模型: {ckpt_path}")
predictor = MultiModalPredictor.load(ckpt_path)
print("✓ 模型加载成功")

# 获取 learner 和内部组件
learner = predictor._learner
model = learner._model
datamodule = learner._data_module

print(f"\n2. 模型信息:")
print(f"  模型类型: {type(model).__name__}")
print(f"  DataModule 类型: {type(datamodule).__name__}")

# 检查参数
print(f"\n3. 参数统计:")
total_params = 0
trainable_params = 0
conv_lora_params = 0

for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
    if 'lora_A' in name or 'lora_B' in name:
        conv_lora_params += param.numel()
        print(f"  Conv-LoRA 参数: {name} - {param.shape}")

print(f"\n  总参数: {total_params:,}")
print(f"  可训练参数: {trainable_params:,}")
print(f"  Conv-LoRA 参数: {conv_lora_params:,}")

# 测试导入自定义模块
print(f"\n4. 测试自定义模块导入:")
try:
    from lit_semantic_seg_rloo import RLOOSemanticSegmentationLitModule
    print("✓ RLOOSemanticSegmentationLitModule 导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    exit(1)

try:
    from rloo_utils import rloo_loss, compute_segmentation_reward
    print("✓ rloo_utils 导入成功")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    exit(1)

# 测试 DataModule
print(f"\n5. 测试 DataModule:")
try:
    # 设置 DataModule
    datamodule.prepare_data()
    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()
    print(f"✓ DataModule 设置成功")
    print(f"  训练集大小: {len(datamodule.train_dataset)}")
    print(f"  Batch 数量: {len(train_dataloader)}")
    
    # 获取一个 batch
    batch = next(iter(train_dataloader))
    print(f"\n  Batch 内容:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"    {key}: {value.shape}")
        else:
            print(f"    {key}: {type(value)}")
    
except Exception as e:
    print(f"✗ DataModule 测试失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试前向传播
print(f"\n6. 测试前向传播:")
try:
    from autogluon.multimodal.models.utils import run_model
    from autogluon.multimodal.constants import LOGITS
    
    model.eval()
    with torch.no_grad():
        output = run_model(model, batch)
        logits = output[model.prefix][LOGITS]
        print(f"✓ 前向传播成功")
        print(f"  输出 logits 形状: {logits.shape}")
        
except Exception as e:
    print(f"✗ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*80)
print("✓ 所有测试通过！可以开始 RLOO 训练。")
print("="*80)

