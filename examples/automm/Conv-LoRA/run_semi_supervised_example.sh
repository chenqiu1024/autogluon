#!/bin/bash
echo "=== 数据准备 ==="
python prepare_semi_supervised_data.py --task isic2017 --data_dir datasets/isic2017

echo "=== 开始训练 ==="
python run_semi_supervised_train.py --task isic2017 --data_dir datasets/isic2017 --gspo_enable --adapter_enable --max_epochs 30
