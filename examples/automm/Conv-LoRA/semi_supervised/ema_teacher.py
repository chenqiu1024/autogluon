"""
EMA Teacher Module for Semi-Supervised Learning

实现指数移动平均（EMA）teacher 模型，提供稳定的伪标签生成。

核心功能：
1. 每步更新 teacher 参数（对应公式：θ_T ← m*θ_T + (1-m)*θ_S）
2. K 次采样前向传播（不同 dropout/噪声状态）
3. 梯度冻结（teacher 不参与反传）
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
from copy import deepcopy


class EMATeacher(nn.Module):
    """
    Exponential Moving Average Teacher for semi-supervised learning.
    
    该模块维护 student 模型的 EMA 副本，用于生成稳定的伪标签。
    
    Args:
        student_model: Student 模型（将被复制）
        momentum: EMA 动量系数 m（默认 0.999）
        update_freq: 更新频率（'step' 或 'epoch'）
    
    对应设计文档中的公式 2.1：
        θ_teacher^(t) ← m * θ_teacher^(t-1) + (1-m) * θ_student^(t)
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        momentum: float = 0.999,
        update_freq: str = 'step'
    ):
        super().__init__()
        
        self.momentum = momentum
        self.update_freq = update_freq
        
        # 深拷贝 student 模型参数到 teacher（如果提供）
        if student_model is not None:
            self.teacher_model = deepcopy(student_model)
            
            # 冻结 teacher 参数（不计算梯度）
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            
            # 设置 teacher 为 eval 模式（但仍可通过 dropout 产生多样性）
            self.teacher_model.eval()           
            print(f"[EMA Teacher] 初始化完成，momentum={momentum}, update_freq={update_freq}")
        else:
            self.teacher_model = None
            print(f"[EMA Teacher] 延迟初始化模式，momentum={momentum}, update_freq={update_freq}")
    
    @torch.no_grad()
    def update(self, student_model: nn.Module):
        """
        更新 teacher 参数（EMA 滑动平均）
        
        公式：θ_T ← m * θ_T + (1-m) * θ_S
        
        Args:
            student_model: 当前的 student 模型
        """
        m = self.momentum
        
        # 遍历所有参数进行 EMA 更新
        for teacher_param, student_param in zip(
            self.teacher_model.parameters(),
            student_model.parameters()
        ):
            teacher_param.data.mul_(m).add_(student_param.data, alpha=1 - m)
    
    @torch.no_grad()
    def forward_k_times(
        self,
        batch: Dict,
        K: int = 5,
        enable_dropout: bool = True
    ) -> List[torch.Tensor]:
        """
        对同一 batch 进行 K 次前向传播，产生多个预测
        
        通过不同的 dropout 和 MoE 噪声状态，生成预测多样性，用于质量评估。
        
        Args:
            batch: 输入 batch（包含 image, box_prompt 等）
            K: 采样次数（默认 5）
            enable_dropout: 是否启用 dropout（产生多样性）
        
        Returns:
            predictions: K 个预测 mask 的列表，每个形状 [B, H, W]
        """
        predictions = []
        
        # 临时启用 dropout（在 eval 模式下也生效）
        if enable_dropout:
            self.teacher_model.train()  # dropout 需要 train 模式
        
        for k in range(K):
            # 每次前向传播都会因 dropout/MoE 噪声产生不同结果
            output = self.teacher_model(batch)
            
            # 提取预测 mask（根据模型输出格式调整）
            if isinstance(output, dict):
                pred_mask = output.get('pred_mask', output.get('logits'))
            else:
                pred_mask = output
            
            # 转为概率（如果是 logits）
            if pred_mask.dim() == 4 and pred_mask.shape[1] > 1:
                # Multi-class: 取 softmax
                pred_mask = torch.softmax(pred_mask, dim=1)[:, 1]  # 取前景类
            elif pred_mask.dim() == 4 and pred_mask.shape[1] == 1:
                # Binary: 取 sigmoid
                pred_mask = torch.sigmoid(pred_mask).squeeze(1)
            else:
                # 已经是概率
                pred_mask = torch.sigmoid(pred_mask) if pred_mask.max() > 1 else pred_mask
            
            predictions.append(pred_mask)
        
        # 恢复 eval 模式
        if enable_dropout:
            self.teacher_model.eval()
        
        return predictions
    
    def forward(self, batch: Dict) -> torch.Tensor:
        """
        单次前向传播（用于验证/测试）
        
        Args:
            batch: 输入 batch
        
        Returns:
            pred_mask: 预测 mask [B, H, W]
        """
        with torch.no_grad():
            output = self.teacher_model(batch)
            
            if isinstance(output, dict):
                pred_mask = output.get('pred_mask', output.get('logits'))
            else:
                pred_mask = output
            
            # 转为概率
            if pred_mask.dim() == 4 and pred_mask.shape[1] > 1:
                pred_mask = torch.softmax(pred_mask, dim=1)[:, 1]
            elif pred_mask.dim() == 4 and pred_mask.shape[1] == 1:
                pred_mask = torch.sigmoid(pred_mask).squeeze(1)
            else:
                pred_mask = torch.sigmoid(pred_mask) if pred_mask.max() > 1 else pred_mask
        
        return pred_mask
    
    def load_state_dict(self, state_dict, strict: bool = True):
        """加载 teacher 模型权重"""
        return self.teacher_model.load_state_dict(state_dict, strict=strict)
    
    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):
        """
        返回 teacher 模型权重
        兼容 PyTorch / Lightning 的 state_dict 接口（允许传入 destination/prefix/keep_vars）
        """
        return self.teacher_model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
