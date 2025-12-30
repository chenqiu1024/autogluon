"""
Semi-Supervised Learning Module for Quality-Aware Segmentation

This module implements a Teacher-Student framework with:
1. EMA Teacher for stable pseudo-label generation
2. Consistency-based quality estimation
3. GSPO integration for policy optimization
4. Box jitter consistency regularization
"""

from .ema_teacher import EMATeacher
from .quality_estimator import ConsistencyQualityEstimator
from .pseudo_label_gen import PseudoLabelGenerator
from .data_module import SemiSupervisedDataModule
from .gspo_semi_trainer import GSPOSemiSupervisedTrainer

__all__ = [
    'EMATeacher',
    'ConsistencyQualityEstimator',
    'PseudoLabelGenerator',
    'SemiSupervisedDataModule',
    'GSPOSemiSupervisedTrainer',
]




