"""伪标签生成器 - 公式 2.3"""
import torch

class PseudoLabelGenerator:
    def __init__(self, min_quality_threshold=0.6, threshold=0.5, use_soft=False):
        self.min_quality = min_quality_threshold
        self.threshold = threshold
        self.use_soft = use_soft
    
    def generate_from_predictions(self, predictions, quality_scores):
        avg = torch.stack(predictions, dim=0).mean(dim=0)
        pseudo = avg if self.use_soft else (avg > self.threshold).float()
        valid = quality_scores >= self.min_quality
        return pseudo, valid
    
    def compute_quality_weight(self, quality_scores, beta=10.0, q0=0.5):
        return torch.sigmoid(beta * (quality_scores - q0))
