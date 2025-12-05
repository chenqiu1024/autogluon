#!/usr/bin/env python3
"""
快速验证 TTA 新功能是否正常
- 方案 1: 快速验证模式
- 方案 2: 断点续传机制
- 方案 4: Sanity Check

使用方法:
    python verify_tta_features.py
"""

import sys
import inspect

def check_import():
    """检查导入是否正常"""
    print("🔍 检查 1: 验证模块导入...")
    try:
        from autogluon.multimodal import MultiModalPredictor
        from autogluon.multimodal.learners import SemanticSegmentationLearner
        print("  ✅ 模块导入成功")
        return True
    except Exception as e:
        print(f"  ❌ 模块导入失败: {e}")
        return False

def check_enable_tta_signature():
    """检查 enable_tta 方法签名"""
    print("\n🔍 检查 2: 验证 enable_tta 方法签名...")
    try:
        from autogluon.multimodal import MultiModalPredictor
        
        # 获取 enable_tta 方法签名
        sig = inspect.signature(MultiModalPredictor.enable_tta)
        params = list(sig.parameters.keys())
        
        print(f"  方法参数: {params}")
        
        # 检查是否包含新参数
        required_params = ['cache_dir', 'resume_from_cache']
        missing_params = [p for p in required_params if p not in params]
        
        if missing_params:
            print(f"  ❌ 缺少参数: {missing_params}")
            return False
        
        print("  ✅ enable_tta 签名正确（包含 cache_dir 和 resume_from_cache）")
        return True
    except Exception as e:
        print(f"  ❌ 检查失败: {e}")
        return False

def check_learner_methods():
    """检查 SemanticSegmentationLearner 的方法"""
    print("\n🔍 检查 3: 验证 SemanticSegmentationLearner 方法...")
    try:
        from autogluon.multimodal.learners import SemanticSegmentationLearner
        
        # 检查是否有 enable_tta 方法
        if not hasattr(SemanticSegmentationLearner, 'enable_tta'):
            print("  ❌ 缺少 enable_tta 方法")
            return False
        
        # 检查 enable_tta 方法签名
        sig = inspect.signature(SemanticSegmentationLearner.enable_tta)
        params = list(sig.parameters.keys())
        
        required_params = ['cache_dir', 'resume_from_cache']
        missing_params = [p for p in required_params if p not in params]
        
        if missing_params:
            print(f"  ❌ enable_tta 缺少参数: {missing_params}")
            return False
        
        # 检查是否有 _evaluate_with_tta 方法
        if not hasattr(SemanticSegmentationLearner, '_evaluate_with_tta'):
            print("  ❌ 缺少 _evaluate_with_tta 方法")
            return False
        
        # 检查 _evaluate_with_tta 方法签名
        sig = inspect.signature(SemanticSegmentationLearner._evaluate_with_tta)
        params = list(sig.parameters.keys())
        
        required_params = ['cache_dir', 'resume_from_cache']
        missing_params = [p for p in required_params if p not in params]
        
        if missing_params:
            print(f"  ❌ _evaluate_with_tta 缺少参数: {missing_params}")
            return False
        
        print("  ✅ SemanticSegmentationLearner 方法签名正确")
        return True
    except Exception as e:
        print(f"  ❌ 检查失败: {e}")
        return False

def check_tta_utils():
    """检查 TTA utils"""
    print("\n🔍 检查 4: 验证 TTA utils...")
    try:
        from autogluon.multimodal.utils.tta_utils import TTAPredictor
        
        print("  ✅ TTAPredictor 导入成功")
        return True
    except Exception as e:
        print(f"  ❌ TTAPredictor 导入失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("TTA 新功能验证脚本")
    print("=" * 60)
    
    checks = [
        check_import,
        check_enable_tta_signature,
        check_learner_methods,
        check_tta_utils,
    ]
    
    results = []
    for check in checks:
        results.append(check())
    
    print("\n" + "=" * 60)
    print("验证结果总结")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print(f"✅ 所有检查通过 ({passed}/{total})")
        print("\n🎉 TTA 新功能已正确安装！")
        print("\n下一步：")
        print("  1. 运行快速测试: python run_semantic_segmentation.py ... --debug")
        print("  2. 或运行渐进式测试: ./test_tta_progressive.sh")
        return 0
    else:
        print(f"❌ 部分检查失败 ({passed}/{total})")
        print("\n请检查上述错误信息并修复")
        return 1

if __name__ == "__main__":
    sys.exit(main())

