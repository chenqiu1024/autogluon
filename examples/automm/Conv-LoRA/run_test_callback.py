"""
简化测试脚本：验证 Callback 机制
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semi_supervised import (
    EMATeacher,
    ConsistencyQualityEstimator,
    PseudoLabelGenerator
)
from semi_supervised_callback import SemiSupervisedCallback

# 创建半监督组件
print("初始化半监督组件...")
ema_teacher = EMATeacher(student_model=None, momentum=0.999, update_freq='step')
quality_estimator = ConsistencyQualityEstimator(consistency_weight=0.7)
pseudo_label_gen = PseudoLabelGenerator(min_quality_threshold=0.6, use_soft_label=False)

# 创建 Callback
print("创建 Callback...")
callback = SemiSupervisedCallback(
    ema_teacher=ema_teacher,
    quality_estimator=quality_estimator,
    pseudo_label_gen=pseudo_label_gen,
    semi_supervised_config={'quality_k_samples': 5}
)

print("✅ 所有组件初始化成功！")
print("   - EMA Teacher (延迟初始化)")
print("   - 质量评估器")
print("   - 伪标签生成器")
print("   - 半监督 Callback")

# 创建一个简单的模拟 LitModule 来测试 callback
print("\n测试 Callback 的 on_fit_start...")
import torch.nn as nn

class MockLitModule:
    def __init__(self):
        self.model = nn.Linear(10, 10)

class MockTrainer:
    current_epoch = 0

mock_module = MockLitModule()
mock_trainer = MockTrainer()

callback.on_fit_start(mock_trainer, mock_module)

print("\n✅ Callback 测试通过！")
print("   - EMA Teacher 模型已初始化")
print("   - 半监督组件已注入到 LitModule")

# 验证组件已注入
assert hasattr(mock_module, 'ema_teacher')
assert hasattr(mock_module, 'quality_estimator')
assert hasattr(mock_module, 'pseudo_label_gen')
assert mock_module.ema_teacher.teacher_model is not None

print("\n🎉 所有测试通过！Callback 机制工作正常！")
