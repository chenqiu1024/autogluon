"""
Semi-Supervised Training Script v3 Fixed - 正确的 Conv-LoRA 配置
"""
import argparse
import os
import sys
import torch
import pandas as pd
from autogluon.multimodal import MultiModalPredictor

# 导入半监督模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from semi_supervised import (
    EMATeacher,
    ConsistencyQualityEstimator,
    PseudoLabelGenerator,
    SemiSupervisedDataModule,
    GSPOSemiSupervisedTrainer
)


def get_default_training_setting(dataset_name):
    """获取数据集默认配置"""
    validation_metric = "iou"
    loss = "structure_loss"
    max_epoch = 30
    lr = 1e-4
    
    if dataset_name == "isic2017":
        validation_metric = "iou"
        max_epoch = 30
        lr = 1e-4
    
    return validation_metric, loss, max_epoch, lr


def expand_path(df, dataset_dir):
    """扩展 CSV 中的相对路径为绝对路径"""
    for col in ["image", "label"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


def main():
    parser = argparse.ArgumentParser(description="质量感知半监督训练脚本 v3")
    
    # 基础参数
    parser.add_argument("--task", type=str, default="isic2017")
    parser.add_argument("--data_dir", type=str, default="datasets/isic2017")
    parser.add_argument("--output_dir", type=str, default="outputs/semi_supervised")
    parser.add_argument("--seed", type=int, default=42686693)
    
    # 数据参数
    parser.add_argument("--labeled_ratio", type=float, default=0.1, help="有标注比例")
    
    # 训练参数
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--per_gpu_batch_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    # EMA Teacher 参数
    parser.add_argument("--ema_momentum", type=float, default=0.999, help="EMA 动量")
    parser.add_argument("--ema_update_freq", type=str, default="step", choices=["step", "epoch"])
    
    # 质量评估参数
    parser.add_argument("--quality_k_samples", type=int, default=5, help="质量评估采样次数 K")
    parser.add_argument("--quality_min_threshold", type=float, default=0.6, help="质量过滤阈值 q_min")
    parser.add_argument("--quality_consistency_weight", type=float, default=0.7, help="一致性权重 α")
    
    # 半监督损失参数
    parser.add_argument("--pseudo_lambda_warmup_epochs", type=int, default=5, help="λ_u warmup epochs")
    parser.add_argument("--consistency_lambda", type=float, default=0.1, help="一致性损失权重 λ_c")
    
    # GSPO 参数
    parser.add_argument("--gspo_enable", action="store_true", help="启用 GSPO")
    parser.add_argument("--gspo_group_size", type=int, default=4)
    parser.add_argument("--gspo_warmup_epochs", type=int, default=5)
    parser.add_argument("--gspo_contrastive_weight", type=float, default=0.1)
    parser.add_argument("--gspo_quality_momentum", type=float, default=0.9)
    
    # Conv-LoRA 参数
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--expert_num", type=int, default=4)
    
    # Adapter 参数
    parser.add_argument("--adapter_enable", action="store_true")
    parser.add_argument("--adapter_dim", type=int, default=64)
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    # ========== 步骤 1: 准备数据 ==========
    print("\n" + "="*60)
    print("步骤 1: 数据准备")
    print("="*60)
    
    labeled_csv = os.path.join(
        args.data_dir, f"train_labeled_{int(args.labeled_ratio*100)}pct.csv"
    )
    weak_csv = os.path.join(
        args.data_dir, f"train_weak_{int((1-args.labeled_ratio)*100)}pct.csv"
    )
    
    if not os.path.exists(labeled_csv) or not os.path.exists(weak_csv):
        print(f"\n❌ 错误: 数据文件不存在")
        print(f"   缺少: {labeled_csv}")
        print(f"   或: {weak_csv}")
        return
    
    # 读取并合并数据
    print(f"读取数据...")
    data_module = SemiSupervisedDataModule(
        labeled_csv=labeled_csv,
        weak_csv=weak_csv,
        batch_size=args.batch_size,
        labeled_ratio_in_batch=0.5
    )
    
    train_df = data_module.merge_dataframes_for_autogluon()
    train_df = expand_path(train_df, args.data_dir)
    
    # 验证集
    val_csv = os.path.join(args.data_dir, "val.csv")
    val_df = None
    if os.path.exists(val_csv):
        val_df = pd.read_csv(val_csv)
        val_df = expand_path(val_df, args.data_dir)
        print(f"验证集: {len(val_df)} 样本")
    
    print(f"训练集: {len(train_df)} 样本")
    
    # ========== 步骤 2: 初始化半监督组件 ==========
    print("\n" + "="*60)
    print("步骤 2: 半监督组件初始化")
    print("="*60)
    
    # EMA Teacher（延迟初始化）
    print(f"\n初始化 EMA Teacher (momentum={args.ema_momentum}, freq={args.ema_update_freq})...")
    ema_teacher = EMATeacher(
        student_model=None,
        momentum=args.ema_momentum,
        update_freq=args.ema_update_freq
    )
    
    # 质量评估器
    print(f"\n初始化质量评估器 (K={args.quality_k_samples}, α={args.quality_consistency_weight})...")
    quality_estimator = ConsistencyQualityEstimator(
        consistency_weight=args.quality_consistency_weight
    )
    
    # 伪标签生成器
    print(f"\n初始化伪标签生成器 (q_min={args.quality_min_threshold})...")
    pseudo_label_gen = PseudoLabelGenerator(
        min_quality_threshold=args.quality_min_threshold,
        use_soft_label=False
    )
    
    # ========== 步骤 3: 配置模型 ==========
    print("\n" + "="*60)
    print("步骤 3: 模型配置")
    print("="*60)
    
    validation_metric, loss, max_epoch, base_lr = get_default_training_setting(args.task)
    
    # 正确的 Conv-LoRA 配置
    hyperparameters = {
        "optim.peft": "conv_lora",  # 使用 Conv-LoRA
        "optim.lora.conv_lora_expert_num": args.expert_num,  # MoE expert 数量
        "optim.lora.r": args.rank,  # LoRA rank
        "optim.lr": args.lr,
        "optim.max_epochs": args.max_epochs,
        "optim.patience": 10,
        "optim.val_check_interval": 1.0,
        "optim.loss_func": loss,
        "env.per_gpu_batch_size": args.per_gpu_batch_size,
        "env.batch_size": args.batch_size,
        "env.num_gpus": args.num_gpus,
    }
    
    # GSPO 配置
    if args.gspo_enable:
        hyperparameters.update({
            "optim.lora.gspo_enabled": True,
            "optim.lora.gspo_group_size": args.gspo_group_size,
            "optim.lora.gspo_quality_momentum": args.gspo_quality_momentum,
        })
    
    # Adapter 配置
    if args.adapter_enable:
        print(f"配置 Adapter: dim={args.adapter_dim}")
        hyperparameters.update({
            "model.sam.adapter_enabled": True,
            "model.sam.adapter_dim": args.adapter_dim,
        })
    
    print("\n✅ 半监督组件初始化完成！")
    
    # 创建 predictor
    print(f"\n创建 MultiModalPredictor...")
    predictor = MultiModalPredictor(
        problem_type="semantic_segmentation",
        validation_metric=validation_metric,
        eval_metric=validation_metric,
        hyperparameters=hyperparameters,
        label="label",
        path=args.output_dir
    )
    
    # 注入半监督组件到 predictor._learner
    print(f"\n注入半监督组件到 learner...")
    predictor._learner._semi_supervised_components = {
        'ema_teacher': ema_teacher,
        'quality_estimator': quality_estimator,
        'pseudo_label_gen': pseudo_label_gen,
        'labeled_count': data_module.labeled_count,
        'weak_start_idx': data_module.weak_start_idx,
        'config': vars(args)
    }
    
    # 现在初始化 GSPO（如果启用）
    if args.gspo_enable:
        print(f"初始化 GSPO 半监督训练器...")
        gspo_trainer = GSPOSemiSupervisedTrainer(
            predictor=predictor,
            group_size=args.gspo_group_size,
            warmup_epochs=args.gspo_warmup_epochs,
            contrastive_weight=args.gspo_contrastive_weight,
            pseudo_lambda_warmup_epochs=args.pseudo_lambda_warmup_epochs,
            consistency_lambda=args.consistency_lambda
        )
        predictor._learner._semi_supervised_components['gspo_trainer'] = gspo_trainer
        print("✅ GSPO 半监督训练器已启用")
    
    print("✅ 半监督组件已注入")
    
    # ========== 步骤 4: 训练 ==========
    print("\n" + "="*60)
    print("步骤 4: 开始训练")
    print("="*60)
    
    print("\n✅ 半监督训练已启用！")
    print(f"   - {len(data_module.labeled_df)} 个有标注样本")
    print(f"   - {len(data_module.weak_df)} 个弱标注样本")
    print(f"\n开始训练...\n")
    
    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        time_limit=None,
    )
    
    print(f"\n✅ 训练完成！模型保存在: {args.output_dir}")
    
    # ========== 步骤 5: 评估 ==========
    print("\n" + "="*60)
    print("步骤 5: 评估")
    print("="*60)
    
    test_csv = os.path.join(args.data_dir, "test.csv")
    if os.path.exists(test_csv):
        test_df = pd.read_csv(test_csv)
        test_df = expand_path(test_df, args.data_dir)
        score = predictor.evaluate(test_df, metrics=[validation_metric])
        print(f"测试集评估: {score}")
    else:
        print(f"测试集不存在: {test_csv}")


if __name__ == "__main__":
    main()
