"""质量评估器 - 公式 2.2"""
import torch

class ConsistencyQualityEstimator:
    def __init__(self, consistency_weight=0.7, threshold=0.5):
        self.cons_w = consistency_weight
        self.conf_w = 1.0 - consistency_weight
        self.threshold = threshold
    
    def compute_iou(self, p1, p2):
        b1 = (p1 > self.threshold).float().view(p1.shape[0], -1)
        b2 = (p2 > self.threshold).float().view(p2.shape[0], -1)
        inter = (b1 * b2).sum(dim=1)
        union = b1.sum(dim=1) + b2.sum(dim=1) - inter
        return (inter + 1e-6) / (union + 1e-6)
    
    def estimate_quality(self, predictions):
        K, B = len(predictions), predictions[0].shape[0]
        total_iou = torch.zeros(B, device=predictions[0].device)
        count = 0
        for i in range(K):
            for j in range(i+1, K):
                total_iou += self.compute_iou(predictions[i], predictions[j])
                count += 1
        q_cons = total_iou / max(count, 1)
        total_conf = sum([torch.maximum(p, 1-p).view(B, -1).mean(dim=1) for p in predictions])
        q_conf = total_conf / K
        q = self.cons_w * q_cons + self.conf_w * q_conf
        return q, q_cons, q_conf
