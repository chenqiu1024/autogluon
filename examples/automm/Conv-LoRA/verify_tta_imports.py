#!/usr/bin/env python3
"""
快速验证 TTA 功能的导入和基本功能
"""

print("=" * 60)
print("TTA 功能验证脚本")
print("=" * 60)
print()

# 测试 1: 导入 TTA 模块
print("测试 1: 导入 TTA 工具模块...")
try:
    from autogluon.multimodal.utils.tta_utils import (
        TTAPredictor, 
        TTATransform,
        ScaleTransform,
        FlipTransform,
        RotateTransform,
        dice_coefficient,
        iou_score
    )
    print("  ✅ TTA 工具模块导入成功")
except Exception as e:
    print(f"  ❌ 导入失败: {e}")
    exit(1)

# 测试 2: 导入 learner
print("\n测试 2: 导入 SemanticSegmentationLearner...")
try:
    from autogluon.multimodal.learners import SemanticSegmentationLearner
    print("  ✅ SemanticSegmentationLearner 导入成功")
except Exception as e:
    print(f"  ❌ 导入失败: {e}")
    exit(1)

# 测试 3: 创建 TTA Predictor
print("\n测试 3: 创建 TTA Predictor...")
try:
    tta_predictor = TTAPredictor(
        scales=[0.75, 1.0, 1.25],
        flips=["none", "horizontal"],
        rotations=[0],
    )
    print(f"  ✅ TTA Predictor 创建成功")
    print(f"     - 变换数量: {len(tta_predictor.transforms)}")
except Exception as e:
    print(f"  ❌ 创建失败: {e}")
    exit(1)

# 测试 4: 测试变换功能
print("\n测试 4: 测试变换功能...")
try:
    import numpy as np
    
    # 创建测试图像
    test_image = np.random.rand(256, 256, 3).astype(np.float32)
    
    # 测试 ScaleTransform
    scale_transform = ScaleTransform(0.75)
    scaled = scale_transform.apply(test_image)
    print(f"  ✅ ScaleTransform 正常工作")
    print(f"     - 原始形状: {test_image.shape}")
    print(f"     - 缩放后: {scaled.shape}")
    
    # 测试 FlipTransform
    flip_transform = FlipTransform("horizontal")
    flipped = flip_transform.apply(test_image)
    print(f"  ✅ FlipTransform 正常工作")
    print(f"     - 翻转后形状: {flipped.shape}")
    
    # 测试 RotateTransform
    rotate_transform = RotateTransform(10)
    rotated = rotate_transform.apply(test_image)
    print(f"  ✅ RotateTransform 正常工作")
    print(f"     - 旋转后形状: {rotated.shape}")
    
except Exception as e:
    print(f"  ❌ 变换测试失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 测试 5: 测试度量函数
print("\n测试 5: 测试度量函数...")
try:
    # 创建测试 mask
    pred = np.random.rand(256, 256) > 0.5
    target = np.random.rand(256, 256) > 0.5
    
    dice = dice_coefficient(pred.astype(np.uint8), target.astype(np.uint8))
    iou = iou_score(pred.astype(np.uint8), target.astype(np.uint8))
    
    print(f"  ✅ 度量函数正常工作")
    print(f"     - Dice: {dice:.4f}")
    print(f"     - IoU: {iou:.4f}")
    
except Exception as e:
    print(f"  ❌ 度量测试失败: {e}")
    exit(1)

# 测试 6: 检查依赖库
print("\n测试 6: 检查依赖库...")
try:
    import numpy
    import scipy
    from PIL import Image
    from skimage.transform import resize
    
    print(f"  ✅ numpy 版本: {numpy.__version__}")
    print(f"  ✅ scipy 版本: {scipy.__version__}")
    print(f"  ✅ Pillow (PIL) 已安装")
    print(f"  ✅ scikit-image 已安装")
    
except Exception as e:
    print(f"  ❌ 依赖检查失败: {e}")
    exit(1)

print("\n" + "=" * 60)
print("🎉 所有测试通过！TTA 功能已就绪！")
print("=" * 60)
print()
print("下一步：运行完整测试")
print("  cd examples/automm/Conv-LoRA")
print("  ./test_tta.sh")
print()

