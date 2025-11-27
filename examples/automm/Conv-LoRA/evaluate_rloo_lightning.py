"""
直接从 Lightning checkpoint 评估 RLOO 模型
在测试集上评估模型性能（IoU 和 Dice）
"""
import argparse
import os
import pandas as pd
import torch
from tqdm import tqdm

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.models.utils import run_model
from autogluon.multimodal.constants import LOGITS, LABEL

from lit_semantic_seg_rloo import RLOOSemanticSegmentationLitModule
from rloo_utils import compute_binary_iou, compute_binary_dice


def expand_path(df: pd.DataFrame, dataset_dir: str) -> pd.DataFrame:
    """展开数据路径"""
    for col in ["image", "label"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


def evaluate_autogluon_predictor(
    predictor_path: str,
    task: str,
    test_csv: str = "test.csv",
    batch_size: int = 1,
    num_workers: int = 4,
):
    """
    评估 AutoGluon 原生 Predictor（基线模型）
    
    这个函数处理非 Lightning 格式的 checkpoint，即原始的监督训练模型
    """
    print("使用 AutoGluon Predictor 评估模式\n")
    
    # 加载测试数据
    dataset_dir = os.path.join(f"datasets/{task}", task)
    test_csv_path = os.path.join(dataset_dir, test_csv)
    
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test data not found: {test_csv_path}")
    
    test_df = pd.read_csv(test_csv_path)
    test_df = expand_path(test_df, dataset_dir)
    
    print(f"✓ 测试集大小: {len(test_df)} 样本\n")
    
    # 加载 predictor
    print(f"加载 AutoGluon Predictor: {predictor_path}")
    predictor = MultiModalPredictor.load(predictor_path)
    learner = predictor._learner
    model = learner._model
    
    # 设置数据
    learner._train_data = test_df[:10] if len(test_df) > 10 else test_df
    learner._tuning_data = test_df
    
    # 创建 DataModule
    print("创建 DataModule...")
    datamodule = learner.get_datamodule_per_run(
        df_preprocessor=learner._df_preprocessor,
        data_processors=learner._data_processors,
        per_gpu_batch_size=batch_size,
        num_workers=num_workers,
        is_train=True,
    )
    
    datamodule.prepare_data()
    datamodule.setup("fit")
    test_dataloader = datamodule.val_dataloader()
    
    print(f"✓ DataLoader 准备完成 (batches: {len(test_dataloader)})\n")
    
    # 设置模型为评估模式
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"✓ 模型加载完成 (device: {device})\n")
    
    # 评估
    print("开始评估...")
    all_iou = []
    all_dice = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="评估进度")):
            try:
                # 移动到设备
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # 前向传播
                output = run_model(model, batch)
                pred_logits = output[model.prefix][LOGITS]
                
                # 获取 GT masks（使用 model 的 label_key）
                gt_masks = batch[model.label_key]
                
                # 二值化预测
                pred_masks = (torch.sigmoid(pred_logits) > 0.5).float()
                
                # 计算指标
                batch_size_actual = pred_masks.shape[0]
                for i in range(batch_size_actual):
                    iou = compute_binary_iou(
                        pred_masks[i:i+1], 
                        gt_masks[i:i+1]
                    ).item()
                    dice = compute_binary_dice(
                        pred_masks[i:i+1], 
                        gt_masks[i:i+1]
                    ).item()
                    
                    all_iou.append(iou)
                    all_dice.append(dice)
                    
            except Exception as e:
                print(f"\n警告: Batch {batch_idx} 评估失败: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 汇总结果
    if len(all_iou) == 0:
        print("\n错误: 没有成功评估的样本！")
        return None
    
    mean_iou = sum(all_iou) / len(all_iou)
    mean_dice = sum(all_dice) / len(all_dice)
    std_iou = torch.tensor(all_iou).std().item()
    std_dice = torch.tensor(all_dice).std().item()
    
    print("\n" + "="*80)
    print("评估结果")
    print("="*80)
    print(f"测试样本数: {len(all_iou)}")
    print(f"平均 IoU:  {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"平均 Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    print("="*80)
    
    return {
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
        'std_iou': std_iou,
        'std_dice': std_dice,
        'num_samples': len(all_iou),
        'all_iou': all_iou,
        'all_dice': all_dice,
    }


def is_lightning_checkpoint(checkpoint_path: str) -> bool:
    """
    检查 checkpoint 是否是 PyTorch Lightning 格式
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # Lightning checkpoint 有特定的键
        return 'pytorch-lightning_version' in checkpoint or 'epoch' in checkpoint
    except:
        return False


def evaluate_lightning_checkpoint(
    checkpoint_path: str,
    base_predictor_path: str,
    task: str,
    test_csv: str = "test.csv",
    batch_size: int = 1,
    num_workers: int = 4,
):
    """
    从 checkpoint 加载模型并在测试集上评估
    
    支持两种 checkpoint 格式：
    1. PyTorch Lightning checkpoint（RLOO 训练后的模型）
    2. AutoGluon 原生 checkpoint（基线模型）
    
    Parameters
    ----------
    checkpoint_path : str
        Checkpoint 路径 (.ckpt 文件)
    base_predictor_path : str
        基础 AutoGluon predictor 路径（用于加载 data processors）
    task : str
        任务名称
    test_csv : str
        测试数据文件名
    batch_size : int
        评估时的 batch size
    num_workers : int
        DataLoader workers 数量
    
    Returns
    -------
    dict
        评估结果字典，包含 mean_iou、mean_dice 等
    """
    print("="*80)
    print("模型评估 - 测试集")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"基础模型: {base_predictor_path}")
    print(f"任务: {task}")
    print(f"Batch size: {batch_size}")
    print("="*80 + "\n")
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if not os.path.exists(base_predictor_path):
        raise FileNotFoundError(f"Base predictor not found: {base_predictor_path}")
    
    # 检查是否是 Lightning checkpoint
    is_lightning = is_lightning_checkpoint(checkpoint_path)
    print(f"Checkpoint 类型: {'PyTorch Lightning' if is_lightning else 'AutoGluon 原生'}\n")
    
    # 如果是 AutoGluon 原生 checkpoint（基线模型），使用不同的加载方式
    if not is_lightning:
        # 基线模型：checkpoint 路径就是 predictor 路径的一部分
        # 例如 AutogluonModels/ag-xxx/model.ckpt -> AutogluonModels/ag-xxx
        predictor_dir = os.path.dirname(checkpoint_path)
        return evaluate_autogluon_predictor(
            predictor_path=predictor_dir,
            task=task,
            test_csv=test_csv,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    
    # 加载基础 predictor（获取 data processors 和配置）
    print("加载基础模型配置...")
    base_predictor = MultiModalPredictor.load(base_predictor_path)
    learner = base_predictor._learner
    
    # 加载测试数据
    dataset_dir = os.path.join(f"datasets/{task}", task)
    test_csv_path = os.path.join(dataset_dir, test_csv)
    
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test data not found: {test_csv_path}")
    
    test_df = pd.read_csv(test_csv_path)
    test_df = expand_path(test_df, dataset_dir)
    
    print(f"✓ 测试集大小: {len(test_df)} 样本\n")
    
    # 设置数据到 learner（用于创建 DataModule）
    # 注意：即使是评估模式，DataModule 的 setup("fit") 也会尝试设置训练集
    # 所以我们需要提供训练数据（虽然实际不会用到）
    learner._train_data = test_df[:10] if len(test_df) > 10 else test_df  # 使用少量数据作为虚拟训练集
    learner._tuning_data = test_df  # 使用测试数据作为验证集
    
    # 创建 DataModule
    print("创建 DataModule...")
    datamodule = learner.get_datamodule_per_run(
        df_preprocessor=learner._df_preprocessor,
        data_processors=learner._data_processors,
        per_gpu_batch_size=batch_size,
        num_workers=num_workers,
        is_train=True,  # 需要设置为 True 才能正确初始化
    )
    
    # 准备数据
    datamodule.prepare_data()
    datamodule.setup("fit")
    test_dataloader = datamodule.val_dataloader()  # 使用验证集加载器加载测试数据
    
    print(f"✓ DataLoader 准备完成 (batches: {len(test_dataloader)})\n")
    
    # 加载 Lightning checkpoint
    print(f"加载 checkpoint: {checkpoint_path}")
    
    # 获取优化器配置（评估时不需要，但 checkpoint 加载时可能需要）
    validation_metric, custom_metric_func = learner.get_validation_metric_per_run()
    loss_func, aug_loss_func = learner.get_loss_func_per_run(learner._config)
    
    optim_config = learner._config.optim
    optim_kwargs = dict(
        optim_type=optim_config.optim_type,
        lr_choice=optim_config.lr_choice,
        lr_schedule=optim_config.lr_schedule,
        lr=optim_config.lr,
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
    
    lit_module = RLOOSemanticSegmentationLitModule.load_from_checkpoint(
        checkpoint_path,
        model=learner._model,
        enable_rloo=False,  # 评估时不需要 RLOO
        **optim_kwargs,
    )
    lit_module.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_module = lit_module.to(device)
    
    print(f"✓ 模型加载完成 (device: {device})\n")
    
    # 评估
    print("开始评估...")
    all_iou = []
    all_dice = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="评估进度")):
            try:
                # 移动到设备
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # 前向传播
                output = run_model(lit_module.model, batch)
                pred_logits = output[lit_module.model.prefix][LOGITS]
                
                # 获取 GT masks（使用 model 的 label_key）
                gt_masks = batch[lit_module.model.label_key]
                
                # 二值化预测
                pred_masks = (torch.sigmoid(pred_logits) > 0.5).float()
                
                # 计算指标（每个样本单独计算）
                batch_size_actual = pred_masks.shape[0]
                for i in range(batch_size_actual):
                    iou = compute_binary_iou(
                        pred_masks[i:i+1], 
                        gt_masks[i:i+1]
                    ).item()
                    dice = compute_binary_dice(
                        pred_masks[i:i+1], 
                        gt_masks[i:i+1]
                    ).item()
                    
                    all_iou.append(iou)
                    all_dice.append(dice)
                    
            except Exception as e:
                print(f"\n警告: Batch {batch_idx} 评估失败: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 汇总结果
    if len(all_iou) == 0:
        print("\n错误: 没有成功评估的样本！")
        return None
    
    mean_iou = sum(all_iou) / len(all_iou)
    mean_dice = sum(all_dice) / len(all_dice)
    std_iou = torch.tensor(all_iou).std().item()
    std_dice = torch.tensor(all_dice).std().item()
    
    print("\n" + "="*80)
    print("评估结果")
    print("="*80)
    print(f"测试样本数: {len(all_iou)}")
    print(f"平均 IoU:  {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"平均 Dice: {mean_dice:.4f} ± {std_dice:.4f}")
    print("="*80)
    
    return {
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
        'std_iou': std_iou,
        'std_dice': std_dice,
        'num_samples': len(all_iou),
        'all_iou': all_iou,
        'all_dice': all_dice,
    }


def main():
    parser = argparse.ArgumentParser(
        description="从 Lightning checkpoint 评估 RLOO 模型"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Lightning checkpoint 文件路径 (.ckpt)"
    )
    parser.add_argument(
        "--base_predictor_path",
        type=str,
        required=True,
        help="基础 AutoGluon predictor 路径"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="isic2017",
        choices=["polyp", "leaf_disease_segmentation", "camo_sem_seg", 
                 "isic2017", "road_segmentation", "SBU-shadow"],
        help="任务名称"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="test.csv",
        help="测试数据文件名"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="评估时的 batch size"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="结果保存路径（JSON 格式，可选）"
    )
    
    args = parser.parse_args()
    
    results = evaluate_lightning_checkpoint(
        checkpoint_path=args.checkpoint_path,
        base_predictor_path=args.base_predictor_path,
        task=args.task,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # 保存结果
    if results and args.output_file:
        import json
        # 只保存可序列化的内容
        save_results = {
            'checkpoint_path': args.checkpoint_path,
            'task': args.task,
            'mean_iou': results['mean_iou'],
            'mean_dice': results['mean_dice'],
            'std_iou': results['std_iou'],
            'std_dice': results['std_dice'],
            'num_samples': results['num_samples'],
        }
        with open(args.output_file, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\n✓ 结果已保存到: {args.output_file}")


if __name__ == "__main__":
    main()

