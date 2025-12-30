"""GSPO 半监督扩展 - 公式 2.5"""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gspo_trainer import GSPOConvLoRATrainer

class GSPOSemiSupervisedTrainer(GSPOConvLoRATrainer):
    def __init__(self, predictor, group_size=4, warmup_epochs=5, **kwargs):
        super().__init__(predictor, group_size, warmup_epochs, **kwargs)
        self.pseudo_warmup = kwargs.get('pseudo_lambda_warmup_epochs', 5)
    
    def compute_segmentation_quality(self, pred, gt=None, metric='iou', quality_proxy=None):
        if gt is not None:
            return super().compute_segmentation_quality(pred, gt, metric)
        elif quality_proxy is not None:
            return quality_proxy
        raise ValueError("需要 gt 或 quality_proxy")
    
    def get_pseudo_lambda(self, epoch):
        return min(1.0, epoch / self.pseudo_warmup) if epoch < self.pseudo_warmup else 1.0
