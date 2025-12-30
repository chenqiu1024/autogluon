"""
Semi-Supervised Data Module

管理有标注和弱标注数据的混合加载

核心功能：
1. 同时加载 labeled 和 weak 数据
2. 在每个 batch 中混合两类样本
3. 支持 box prompt 的加载和传递
"""

import pandas as pd
import torch
from typing import Dict, Tuple, Optional


class SemiSupervisedDataModule:
    """
    半监督数据模块
    
    管理有标注集（10% full mask）和弱标注集（90% noisy box）的混合加载。
    
    Args:
        labeled_csv: 有标注集 CSV 路径
        weak_csv: 弱标注集 CSV 路径（包含 box 列）
        batch_size: 总 batch size
        labeled_ratio_in_batch: batch 中有标注样本的比例（默认 0.5）
    """
    
    def __init__(
        self,
        labeled_csv: str,
        weak_csv: str,
        batch_size: int = 4,
        labeled_ratio_in_batch: float = 0.5
    ):
        self.labeled_csv = labeled_csv
        self.weak_csv = weak_csv
        self.batch_size = batch_size
        self.labeled_ratio = labeled_ratio_in_batch
        
        # 读取数据
        self.labeled_df = pd.read_csv(labeled_csv)
        self.weak_df = pd.read_csv(weak_csv)
        
        # 计算每个 batch 中的样本数
        self.n_labeled_per_batch = max(1, int(batch_size * labeled_ratio_in_batch))
        self.n_weak_per_batch = batch_size - self.n_labeled_per_batch
        
        print(f"[Data Module] Labeled: {len(self.labeled_df)}, Weak: {len(self.weak_df)}")
        print(f"[Data Module] Batch: {self.n_labeled_per_batch} labeled + "
              f"{self.n_weak_per_batch} weak = {batch_size} total")
    
    def parse_box(self, box_str: str) -> torch.Tensor:
        """
        解析 box 字符串为 tensor
        
        Args:
            box_str: "x_min,y_min,x_max,y_max"
        
        Returns:
            box: [4] tensor
        """
        coords = [int(x) for x in box_str.split(',')]
        return torch.tensor(coords, dtype=torch.float32)
    
    def merge_dataframes_for_autogluon(self) -> pd.DataFrame:
        """
        合并 labeled 和 weak 数据为单个 dataframe（用于 AutoGluon）
        
        策略：
        - Labeled 样本保留 image 和 label
        - Weak 样本保留 image 和 box（label 设为 None 或占位符）
        
        Returns:
            merged_df: 合并后的 dataframe
        """
        # 复制 labeled 数据
        labeled = self.labeled_df.copy()
        labeled['is_labeled'] = True
        labeled['box'] = None  # 无 box
        
        # 复制 weak 数据
        weak = self.weak_df.copy()
        weak['is_labeled'] = False
        weak['label'] = labeled['label'].iloc[0]  # 占位符（训练时不使用）
        
        # 合并
        merged = pd.concat([labeled, weak], ignore_index=True)
        
        return merged
