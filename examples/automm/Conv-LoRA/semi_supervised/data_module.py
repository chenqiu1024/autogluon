"""数据模块"""
import pandas as pd

class SemiSupervisedDataModule:
    def __init__(self, labeled_csv, weak_csv, batch_size=4, labeled_ratio=0.5):
        self.labeled_df = pd.read_csv(labeled_csv)
        self.weak_df = pd.read_csv(weak_csv)
        self.batch_size = batch_size
        print(f"[Data] Labeled: {len(self.labeled_df)}, Weak: {len(self.weak_df)}")
    
    def merge_dataframes_for_autogluon(self):
        l = self.labeled_df.copy()
        l['is_labeled'] = True
        w = self.weak_df.copy()
        w['is_labeled'] = False
        return pd.concat([l, w], ignore_index=True)
