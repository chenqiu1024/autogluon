"""
Semi-Supervised Training Callback

使用 PyTorch Lightning Callback 机制注入半监督组件，
避免直接修改 AutoGluon 核心代码。
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class SemiSupervisedCallback(Callback):
    """
    半监督训练回调
    
    在训练开始时将半监督组件注入到 LitModule
    """
    
    def __init__(
        self,
        ema_teacher,
        quality_estimator,
        pseudo_label_gen,
        gspo_trainer=None,
        semi_supervised_config=None
    ):
        """
        Parameters
        ----------
        ema_teacher : EMATeacher
            EMA Teacher 组件
        quality_estimator : ConsistencyQualityEstimator
            质量评估器
        pseudo_label_gen : PseudoLabelGenerator
            伪标签生成器
        gspo_trainer : GSPOSemiSupervisedTrainer, optional
            GSPO 半监督训练器
        semi_supervised_config : dict, optional
            半监督配置
        """
        self.ema_teacher = ema_teacher
        self.quality_estimator = quality_estimator
        self.pseudo_label_gen = pseudo_label_gen
        self.gspo_trainer = gspo_trainer
        self.semi_supervised_config = semi_supervised_config or {}
        
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        训练开始时注入半监督组件
        """
        print("\n" + "="*60)
        print("🔧 注入半监督组件到 LitModule")
        print("="*60)
        
        # 现在模型已经初始化，初始化 EMA Teacher
        if hasattr(pl_module, 'model'):
            from copy import deepcopy
            import torch
            
            student_model = pl_module.model._model if hasattr(pl_module.model, '_model') else pl_module.model
            
            # 初始化 teacher 模型（深拷贝）
            if self.ema_teacher.teacher_model is None:
                self.ema_teacher.teacher_model = deepcopy(student_model)
                
                # 冻结 teacher 参数
                for param in self.ema_teacher.teacher_model.parameters():
                    param.requires_grad = False
                
                # 设置 teacher 为 eval 模式
                self.ema_teacher.teacher_model.eval()
                print(f"✓ EMA Teacher 模型已初始化")
            
            # 更新 student 指针
            self.ema_teacher.student_model = student_model
            print(f"✓ EMA Teacher student 模型已设置")
        
        # 注入组件到 LitModule
        pl_module.ema_teacher = self.ema_teacher
        pl_module.quality_estimator = self.quality_estimator
        pl_module.pseudo_label_gen = self.pseudo_label_gen
        pl_module.semi_supervised_config = self.semi_supervised_config
        
        if self.gspo_trainer is not None:
            pl_module.gspo_trainer = self.gspo_trainer
            print(f"✓ GSPO 半监督训练器已注入")
        
        print(f"✓ EMA Teacher 已注入")
        print(f"✓ 质量评估器已注入")
        print(f"✓ 伪标签生成器已注入")
        print("="*60 + "\n")
    
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        每个 epoch 开始时的处理
        """
        epoch = trainer.current_epoch
        if epoch == 0:
            print(f"\n🚀 开始半监督训练 Epoch {epoch}")
            print(f"   - 配置: K={self.semi_supervised_config.get('quality_k_samples', 5)}")
            print(f"   - EMA momentum: {self.semi_supervised_config.get('ema_momentum', 0.999)}")
            print(f"   - 质量阈值: {self.semi_supervised_config.get('quality_min_threshold', 0.6)}\n")
