# nnU-Net 在 ISIC2017（2D 皮肤病灶分割）端到端指引

本指引基于 nnU-Net 官方文档（readme 与 docs），面向已有数据路径 `examples/automm/Conv-LoRA/datasets/isic2017/isic2017`。包含数据准备 → 训练 → 推理的操作说明，并配套一键脚本。

## 环境与安装
- 建议使用 Python ≥ 3.9，并先安装匹配硬件的 PyTorch（CUDA/MPS/CPU），官方提醒：**Torch 2.9 在 3D AMP 上有性能回退，2D 使用 2.8 及以下更稳**。
- 在本仓库根目录执行：
  ```bash
  cd /root/autodl-tmp/works/autogluon/external/nnUNet
  pip install -e .  # 开发模式，便于调试
  pip install pillow  # 数据转换脚本依赖
  ```
- 设定 nnU-Net 路径（脚本已内置，默认放在仓库根目录下）：
  - `nnUNet_raw`：原始/转换后数据
  - `nnUNet_preprocessed`：预处理缓存
  - `nnUNet_results`：训练权重与推理输出

## 数据集准备（ISIC2017 → nnU-Net 格式）
数据原始结构：
`train/ISIC-2017_Train/*.jpg` + `train/ISIC-2017_Training_Part1_GroundTruth/*_segmentation.png`，`val/` 同结构，`test/ISIC-2017_Test/*.jpg`。

nnU-Net 要求：`DatasetXXX_NAME/{imagesTr, labelsTr, imagesTs, dataset.json}`，图像与标签需同一无损格式且命名 `{ID}_0000.png` / `{ID}.png`。

一键转换脚本（合并 train+val 作为训练，test 作为测试）：
```bash
cd /root/autodl-tmp/works/autogluon/external/nnUNet/scripts
chmod +x isic2017_env.sh isic2017_prepare.sh
./isic2017_prepare.sh
```
脚本会：
1) 写入环境变量并创建 `nnUNet_raw/nnUNet_preprocessed/nnUNet_results` 目录；
2) 将 JPG 图像和 PNG 掩码统一转为无损 PNG，命名为 `{case}_0000.png` / `{case}.png`；
3) 生成 `Dataset701_ISIC2017/dataset.json`（单通道 RGB，自定义 label：0 背景 / 1 病灶）；
4) 运行 `nnUNetv2_plan_and_preprocess -d 701 --verify_dataset_integrity -c 2d`。

若需只用 train、不合并 val，可修改 `isic2017_prepare.py` 的 `--include-val` 参数。

## 训练
默认使用 2D 配置、5 折交叉验证并保存 softmax (`--npz` 便于后续自动挑选与集成)。
```bash
cd /root/autodl-tmp/works/autogluon/external/nnUNet/scripts
chmod +x isic2017_train.sh
# 可选：CUDA_VISIBLE_DEVICES=0 ./isic2017_train.sh 2d "0 1 2 3 4"
./isic2017_train.sh        # 默认 2d 配置、5 折、使用当前可见 GPU
```
注意：首次启动某配置需等待预处理解压完成，再并行开多折。CPU 核数少时，可通过 `export nnUNet_n_proc_DA=8` 等降低数据增强并行度。

如需自动选择最佳后处理/集成：
```bash
chmod +x isic2017_find_best.sh
./isic2017_find_best.sh    # 仅 2d 配置时仍会给出后处理与推理指令
```

## 推理
输入必须与训练时一致的文件后缀和命名（`{ID}_0000.png`）。脚本默认读取 `nnUNet_raw/Dataset701_ISIC2017/imagesTs`。
```bash
cd /root/autodl-tmp/works/autogluon/external/nnUNet/scripts
chmod +x isic2017_infer.sh
./isic2017_infer.sh                          # 使用 2d、五折集成
# 或指定输入输出
# ./isic2017_infer.sh /path/to/imagesTs /path/to/save 2d "0 1 2 3 4"
```
若训练目录存在 `postprocessing.pkl`，脚本会自动执行 `nnUNetv2_apply_postprocessing` 生成 `*-postproc` 结果。

## 常见检查项
- 环境变量：`echo $nnUNet_raw` 等确认已生效（或直接执行 `./isic2017_env.sh`）。
- 磁盘：预处理与软概率 (`--npz`) 需要额外空间，请确保 `nnUNet_preprocessed`、`nnUNet_results` 所在磁盘有余量。
- GPU/CPU：若显存不足可在训练时设置 `CUDA_VISIBLE_DEVICES` 更换 GPU，或改用 `-device cpu` 但速度会极慢。
- 数据一致性：`--verify_dataset_integrity` 会检查最常见格式问题；如修改数据需清理对应 `nnUNet_preprocessed/Dataset701_ISIC2017` 后再预处理。

