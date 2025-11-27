"""
真实的 RLOO 训练脚本 - 带梯度更新
基于 AutoGluon 内部组件构建，直接使用 PyTorch Lightning Trainer
"""
import argparse
import os
from typing import Dict, List, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import LABEL

from lit_semantic_seg_rloo import RLOOSemanticSegmentationLitModule


def expand_path(df: pd.DataFrame, dataset_dir: str) -> pd.DataFrame:
    """复用原脚本中的路径展开逻辑"""
    for col in ["image", "label"]:
        df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


class RLOOTrainer:
    """
    RLOO 训练器，管理真实的 RL 训练循环
    """
    
    def __init__(
        self,
        ckpt_path: str,
        output_dir: str,
        num_generations: int = 4,
        beta: float = 0.05,
        reward_type: str = "combo",
        learning_rate: float = 1e-5,
        epochs: int = 3,
        batch_size: int = 2,
        num_workers: int = 4,
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        ckpt_path
            阶段一 Conv-LoRA SAM checkpoint 路径
        output_dir
            输出目录
        num_generations
            每个样本生成的候选 mask 数量
        beta
            KL 正则系数
        reward_type
            reward 类型："iou", "dice", 或 "combo"
        learning_rate
            学习率
        epochs
            训练 epoch 数
        batch_size
            Batch size
        num_workers
            DataLoader workers 数量
        seed
            随机种子
        """
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        self.num_generations = num_generations
        self.beta = beta
        self.reward_type = reward_type
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        # 处理路径
        if self.ckpt_path.endswith('.ckpt'):
            self.ckpt_path = os.path.dirname(self.ckpt_path)
        
        if not os.path.isdir(self.ckpt_path):
            raise ValueError(
                f"Checkpoint path '{self.ckpt_path}' does not exist or is not a directory."
            )
        
        os.makedirs(self.output_dir, exist_ok=True)
        torch.manual_seed(self.seed)
        pl.seed_everything(self.seed)
        
    def setup_model_and_datamodule(self, predictor: MultiModalPredictor):
        """
        从 AutoGluon Predictor 中提取模型和 DataModule
        """
        learner = predictor._learner
        
        # 获取模型
        model = learner._model
        
        # 获取 DataModule
        datamodule = learner._data_module
        
        return model, datamodule
    
    def freeze_non_conv_lora_params(self, model: nn.Module):
        """
        冻结非 Conv-LoRA 参数，只训练 Conv-LoRA
        """
        trainable_count = 0
        frozen_count = 0
        
        for name, param in model.named_parameters():
            # 只保持 Conv-LoRA 参数可训练
            if 'lora_A' in name or 'lora_B' in name:
                param.requires_grad = True
                trainable_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()
        
        print(f"\n参数统计:")
        print(f"  可训练参数: {trainable_count:,}")
        print(f"  冻结参数: {frozen_count:,}")
        print(f"  总参数: {trainable_count + frozen_count:,}\n")
        
        return trainable_count
    
    def get_trainable_param_names(self, model: nn.Module) -> List[str]:
        """
        获取 Conv-LoRA 参数名称列表
        """
        trainable_names = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_names.append(name)
        return trainable_names
    
    def create_lit_module(
        self,
        model: nn.Module,
        optim_kwargs: Dict,
    ) -> RLOOSemanticSegmentationLitModule:
        """
        创建 RLOO LitModule
        """
        # 获取可训练参数名称
        trainable_param_names = self.get_trainable_param_names(model)
        
        # 创建 LitModule
        lit_module = RLOOSemanticSegmentationLitModule(
            model=model,
            enable_rloo=True,
            num_generations=self.num_generations,
            beta=self.beta,
            reward_type=self.reward_type,
            trainable_param_names=trainable_param_names,
            **optim_kwargs,
        )
        
        return lit_module
    
    def train(self, predictor: MultiModalPredictor):
        """
        执行 RLOO 训练
        """
        print("="*80)
        print("开始真实的 RLOO 训练")
        print("="*80)
        print(f"配置:")
        print(f"  Checkpoint: {self.ckpt_path}")
        print(f"  输出目录: {self.output_dir}")
        print(f"  候选数量 (G): {self.num_generations}")
        print(f"  KL 系数 (β): {self.beta}")
        print(f"  Reward 类型: {self.reward_type}")
        print(f"  学习率: {self.learning_rate}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print("="*80 + "\n")
        
        # 设置模型和数据
        model, datamodule = self.setup_model_and_datamodule(predictor)
        
        # 冻结非 Conv-LoRA 参数
        trainable_params = self.freeze_non_conv_lora_params(model)
        
        if trainable_params == 0:
            raise RuntimeError(
                "没有找到可训练参数！请检查模型是否使用 Conv-LoRA 训练。"
            )
        
        # 获取优化器配置
        learner = predictor._learner
        optim_kwargs = learner._config.optim
        
        # 更新学习率
        optim_kwargs['lr'] = self.learning_rate
        
        # 创建 LitModule
        lit_module = self.create_lit_module(model, optim_kwargs)
        
        # 创建 PyTorch Lightning Trainer
        trainer = pl.Trainer(
            max_epochs=self.epochs,
            accelerator='auto',
            devices=1,
            default_root_dir=self.output_dir,
            enable_progress_bar=True,
            log_every_n_steps=10,
            enable_checkpointing=True,
            logger=True,
        )
        
        # 开始训练
        print("\n开始训练循环...\n")
        trainer.fit(lit_module, datamodule=datamodule)
        
        print("\n" + "="*80)
        print("RLOO 训练完成！")
        print("="*80)
        print(f"\n模型已保存到: {self.output_dir}")
        print("\n训练统计:")
        print(f"  总 epochs: {self.epochs}")
        print(f"  最终 checkpoint: {trainer.checkpoint_callback.best_model_path}")
        print("="*80 + "\n")
        
        return trainer, lit_module


def main():
    parser = argparse.ArgumentParser(
        description="真实的 RLOO 微调 - 带梯度更新的完整实现"
    )
    
    # 数据和模型
    parser.add_argument(
        "--task",
        type=str,
        default="leaf_disease_segmentation",
        choices=[
            "polyp",
            "leaf_disease_segmentation",
            "camo_sem_seg",
            "isic2017",
            "road_segmentation",
            "SBU-shadow"
        ],
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="阶段一 Conv-LoRA SAM checkpoint 目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_rloo_real",
        help="输出目录"
    )
    
    # RLOO 超参数
    parser.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="每个样本生成的候选 mask 数量 G"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="KL 正则系数"
    )
    parser.add_argument(
        "--reward_type",
        type=str,
        default="combo",
        choices=["iou", "dice", "combo"],
        help="reward 类型"
    )
    
    # 训练参数
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="学习率"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="训练 epoch 数"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    args = parser.parse_args()
    
    # 加载 AutoGluon Predictor
    print(f"\n加载模型: {args.ckpt_path}")
    predictor = MultiModalPredictor.load(args.ckpt_path)
    
    # 创建 RLOO Trainer
    rloo_trainer = RLOOTrainer(
        ckpt_path=args.ckpt_path,
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        beta=args.beta,
        reward_type=args.reward_type,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    
    # 训练
    trainer, lit_module = rloo_trainer.train(predictor)
    
    print("\n✓ 训练完成！")
    print(f"✓ 输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()

