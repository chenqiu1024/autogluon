"""EMA Teacher - 公式 2.1"""
import torch
import torch.nn as nn
from copy import deepcopy

class EMATeacher(nn.Module):
    def __init__(self, student_model, momentum=0.999, update_freq='step'):
        super().__init__()
        self.momentum = momentum
        self.teacher_model = deepcopy(student_model)
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        self.teacher_model.eval()
    
    @torch.no_grad()
    def update(self, student_model):
        m = self.momentum
        for tp, sp in zip(self.teacher_model.parameters(), student_model.parameters()):
            tp.data.mul_(m).add_(sp.data, alpha=1-m)
    
    @torch.no_grad()
    def forward_k_times(self, batch, K=5, enable_dropout=True):
        preds = []
        if enable_dropout:
            self.teacher_model.train()
        for _ in range(K):
            out = self.teacher_model(batch)
            pred = out.get('pred_mask', out) if isinstance(out, dict) else out
            if pred.dim() == 4 and pred.shape[1] > 1:
                pred = torch.softmax(pred, dim=1)[:, 1]
            elif pred.dim() == 4:
                pred = torch.sigmoid(pred).squeeze(1)
            preds.append(pred)
        if enable_dropout:
            self.teacher_model.eval()
        return preds
