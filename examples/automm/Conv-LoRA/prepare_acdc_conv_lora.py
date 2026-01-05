#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 ABD 提供的 ACDC 预处理 H5 数据集，转换为 Conv-LoRA / AutoGluon
语义分割脚本所需的图片 + CSV 形式。

输出目录结构（相对于 --output_root，默认 datasets/acdc_conv_lora/acdc_conv_lora）:
  ├─ train/
  │   ├─ images/*.png
  │   └─ masks/*.png
  ├─ val/
  │   ├─ images/*.png
  │   └─ masks/*.png
  ├─ test/
  │   ├─ images/*.png
  │   └─ masks/*.png
  ├─ train.csv
  ├─ val.csv
  └─ test.csv

CSV 列：index, image, label （路径为相对于 dataset 根目录的相对路径）
"""
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_h5(path):
    with h5py.File(path, "r") as f:
        img = f["image"][:]
        lbl = f["label"][:]
    return img, lbl


def save_png(img, lbl, img_path: Path, lbl_path: Path):
    img_path.parent.mkdir(parents=True, exist_ok=True)
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    # 图像：0-1 -> 0-255 uint8
    img = np.clip(img, 0, 1)
    img_uint8 = (img * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(img_path)
    # 标签保持整型
    lbl_uint8 = lbl.astype(np.uint8)
    Image.fromarray(lbl_uint8).save(lbl_path)


def process_train_slices(abd_root: Path, out_root: Path):
    slices_dir = abd_root / "data" / "slices"
    train_list = (abd_root / "train_slices.list").read_text().strip().splitlines()
    records = []
    for name in tqdm(train_list, desc="Train slices"):
        h5_path = slices_dir / f"{name}.h5"
        img, lbl = load_h5(h5_path)
        img_path = out_root / "train" / "images" / f"{name}.png"
        lbl_path = out_root / "train" / "masks" / f"{name}.png"
        save_png(img, lbl, img_path, lbl_path)
        records.append((f"train/images/{name}.png", f"train/masks/{name}.png"))
    return records


def process_volume_slices(abd_root: Path, split: str, out_root: Path):
    """split in {val,test}"""
    list_path = abd_root / f"{split}.list"
    cases = list_path.read_text().strip().splitlines()
    records = []
    for case in tqdm(cases, desc=f"{split} volumes"):
        h5_path = abd_root / "data" / f"{case}.h5"
        img_vol, lbl_vol = load_h5(h5_path)
        # img_vol shape: (D, H, W)
        for idx in range(img_vol.shape[0]):
            name = f"{case}_slice_{idx+1}"
            img = img_vol[idx]
            lbl = lbl_vol[idx]
            img_path = out_root / split / "images" / f"{name}.png"
            lbl_path = out_root / split / "masks" / f"{name}.png"
            save_png(img, lbl, img_path, lbl_path)
            records.append((f"{split}/images/{name}.png", f"{split}/masks/{name}.png"))
    return records


def write_csv(records, csv_path: Path):
    lines = [",image,label"]
    for idx, (img, lbl) in enumerate(records):
        lines.append(f"{idx},{img},{lbl}")
    csv_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Convert ABD ACDC H5 to Conv-LoRA friendly PNG+CSV dataset.")
    parser.add_argument("--abd_root", type=str, default="/root/autodl-tmp/works/ABD/data/ACDC",
                        help="ABD ACDC 根目录（包含 train_slices.list / val.list / test.list 和 data/）")
    parser.add_argument("--output_root", type=str, default="datasets/acdc_conv_lora",
                        help="输出根目录，会在其中创建 acdc_conv_lora 子目录")
    args = parser.parse_args()

    abd_root = Path(args.abd_root).resolve()
    dataset_root = Path(args.output_root).resolve() / "acdc_conv_lora"
    dataset_root.mkdir(parents=True, exist_ok=True)

    print(f"ABD root: {abd_root}")
    print(f"Output dataset root: {dataset_root}")

    train_rec = process_train_slices(abd_root, dataset_root)
    val_rec = process_volume_slices(abd_root, "val", dataset_root)
    test_rec = process_volume_slices(abd_root, "test", dataset_root)

    write_csv(train_rec, dataset_root / "train.csv")
    write_csv(val_rec, dataset_root / "val.csv")
    write_csv(test_rec, dataset_root / "test.csv")

    print("\nDone.")
    print(f"Train samples: {len(train_rec)}")
    print(f"Val samples  : {len(val_rec)}")
    print(f"Test samples : {len(test_rec)}")
    print(f"CSV written to: {dataset_root}")


if __name__ == "__main__":
    main()

