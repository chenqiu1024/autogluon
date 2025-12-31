import pandas as pd
import os
from autogluon.multimodal import MultiModalPredictor

# 数据路径
data_dir = "datasets/isic2017/isic2017"
model_path = "outputs/semi_supervised"

# 加载模型
print("加载训练好的模型...")
predictor = MultiModalPredictor.load(model_path)

# 读取测试集
test_csv = os.path.join(data_dir, "test.csv")
if os.path.exists(test_csv):
    test_df = pd.read_csv(test_csv)
    # 扩展路径
    for col in ["image", "label"]:
        if col in test_df.columns:
            test_df[col] = test_df[col].apply(lambda ele: os.path.join(data_dir, ele))
    
    # 评估
    print(f"\n开始评估（测试集: {len(test_df)} 样本）...")
    score = predictor.evaluate(test_df, metrics=["iou"])
    print(f"\n✅ 测试集评估结果:")
    print(f"   IoU: {score['iou']:.4f}")
else:
    print(f"❌ 测试集不存在: {test_csv}")
