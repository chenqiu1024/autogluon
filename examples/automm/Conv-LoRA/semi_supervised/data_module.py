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
        
        注意：AutoGluon 会过滤掉额外的列，所以不能添加 is_labeled 等元数据。
        
        策略：
        - Labeled 数据在前（索引 0 到 len(labeled)-1）
        - Weak 数据在后（索引 len(labeled) 到 end）
        - LitModule 需要根据索引或其他方式识别数据类型
        
        Returns:
            merged_df: 合并后的 dataframe，只包含 image 和 label 列
        """
        # 复制 labeled 数据（只保留 image 和 label）
        labeled = self.labeled_df[['image', 'label']].copy()
        
        # 复制 weak 数据
        weak = self.weak_df[['image']].copy()
        # 为 weak 数据添加占位符 label（训练时会被忽略）
        if len(labeled) > 0:
            weak['label'] = labeled['label'].iloc[0]
        else:
            # 如果没有 labeled 数据，使用一个虚拟路径
            weak['label'] = weak['image'].iloc[0]  # 占位符
        
        # 合并（labeled 在前，weak 在后）
        merged = pd.concat([labeled, weak], ignore_index=True)
        
        # 记录分界点（供 Callback 使用）
        self.labeled_count = len(labeled)
        self.weak_start_idx = len(labeled)
        
        print(f"[Data Module] Merged: {len(labeled)} labeled + {len(weak)} weak = {len(merged)} total")
        print(f"[Data Module] Labeled indices: 0-{self.labeled_count-1}, Weak indices: {self.weak_start_idx}-{len(merged)-1}")
        
        return merged
