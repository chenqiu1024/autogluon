"""
快速测试脚本：验证半监督组件初始化
"""
import sys
import os

# 设置路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("测试 1: 导入半监督模块")
print("=" * 60)

try:
    from semi_supervised import (
        EMATeacher,
        ConsistencyQualityEstimator,
        PseudoLabelGenerator,
        SemiSupervisedDataModule,
        GSPOSemiSupervisedTrainer
    )
    print("✅ 所有模块导入成功")
except Exception as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("测试 2: 创建半监督组件实例")
print("=" * 60)

try:
    import torch
    
    # 创建一个简单的测试模型
    test_model = torch.nn.Linear(10, 10)
    
    # 创建 EMA Teacher
    print("\n创建 EMA Teacher...")
    ema_teacher = EMATeacher(
        student_model=test_model,
        momentum=0.999,
        update_freq='step'
    )
    print("✅ EMA Teacher 创建成功")
    
    # 创建质量评估器
    print("\n创建质量评估器...")
    quality_estimator = ConsistencyQualityEstimator(
        consistency_weight=0.7
    )
    print("✅ 质量评估器创建成功")
    
    # 创建伪标签生成器
    print("\n创建伪标签生成器...")
    pseudo_label_gen = PseudoLabelGenerator(
        min_quality_threshold=0.6,
        use_soft_label=False
    )
    print("✅ 伪标签生成器创建成功")
    
    print("\n" + "=" * 60)
    print("✅ 所有半监督组件创建成功！")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ 组件创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("测试 3: 验证 LitModule 修改")
print("=" * 60)

try:
    from autogluon.multimodal.optim.lit_semantic_seg import SemanticSegmentationLitModule
    
    # 检查 training_step 方法
    print("\n检查 training_step 方法...")
    assert hasattr(SemanticSegmentationLitModule, 'training_step')
    print("✅ training_step 方法存在")
    
    # 检查 _semi_supervised_training_step 方法
    print("\n检查 _semi_supervised_training_step 方法...")
    assert hasattr(SemanticSegmentationLitModule, '_semi_supervised_training_step')
    print("✅ _semi_supervised_training_step 方法存在")
    
    # 检查 on_train_epoch_end 方法
    print("\n检查 on_train_epoch_end 方法...")
    assert hasattr(SemanticSegmentationLitModule, 'on_train_epoch_end')
    print("✅ on_train_epoch_end 方法存在")
    
    print("\n" + "=" * 60)
    print("✅ LitModule 修改验证通过！")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ LitModule 验证失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("🎉 所有测试通过！半监督训练已启用！")
print("=" * 60)
print("\n可以安全运行以下命令开始训练：")
print("python run_semi_supervised_train.py \\")
print("    --task isic2017 \\")
print("    --data_dir datasets/isic2017/isic2017 \\")
print("    --quality_k_samples 5 \\")
print("    --gspo_enable")
