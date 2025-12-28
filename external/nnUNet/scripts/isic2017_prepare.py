#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "Pillow 未安装，请先运行 `pip install pillow` 后重试。"
    ) from exc


def binarize_mask(mask: Image.Image) -> Image.Image:
    """将掩码转为 0/1 的灰度图。"""
    mask = mask.convert("L")
    return mask.point(lambda p: 1 if p > 0 else 0, mode="L")


def save_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img.convert("RGB").save(dst)


def save_mask(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as mask:
        binarize_mask(mask).save(dst)


def collect_cases(img_dir: Path, mask_dir: Path | None, is_training: bool) -> list[str]:
    cases: list[str] = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        case_id = img_path.stem
        if is_training:
            mask_path = (mask_dir / f"{case_id}_segmentation.png") if mask_dir else None
            if not mask_path or not mask_path.is_file():
                raise FileNotFoundError(f"缺少标注: {mask_path}")
        cases.append(case_id)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="将 ISIC2017 转为 nnU-Net 数据集格式")
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="原始 ISIC2017 根目录（含 train/val/test 子目录）",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="nnUNet_raw 路径",
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=701,
        help="三位数字数据集 ID",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="ISIC2017",
        help="数据集名称",
    )
    parser.add_argument(
        "--include-val",
        action="store_true",
        default=True,
        help="是否将 val 合并进训练（默认 True）",
    )
    args = parser.parse_args()

    dataset_folder = args.raw_root / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    images_tr = dataset_folder / "imagesTr"
    labels_tr = dataset_folder / "labelsTr"
    images_ts = dataset_folder / "imagesTs"
    dataset_folder.mkdir(parents=True, exist_ok=True)

    train_img_dir = args.src / "train" / "ISIC-2017_Train"
    train_mask_dir = args.src / "train" / "ISIC-2017_Training_Part1_GroundTruth"
    val_img_dir = args.src / "val" / "ISIC-2017_Val"
    val_mask_dir = args.src / "val" / "ISIC-2017_Validation_Part1_GroundTruth"
    test_img_dir = args.src / "test" / "ISIC-2017_Test"

    if not train_img_dir.is_dir() or not train_mask_dir.is_dir():
        raise SystemExit("未找到 train 图像或标注目录，请检查 --src 路径")

    train_cases = collect_cases(train_img_dir, train_mask_dir, is_training=True)
    all_train_cases = list(train_cases)
    if args.include_val:
        val_cases = collect_cases(val_img_dir, val_mask_dir, is_training=True)
        all_train_cases.extend(val_cases)

    for case_id in all_train_cases:
        if case_id in train_cases:
            img_dir, mask_dir = train_img_dir, train_mask_dir
        else:
            img_dir, mask_dir = val_img_dir, val_mask_dir
        save_image(img_dir / f"{case_id}.jpg", images_tr / f"{case_id}_0000.png")
        save_mask(mask_dir / f"{case_id}_segmentation.png", labels_tr / f"{case_id}.png")

    if test_img_dir.is_dir():
        for img_path in sorted(test_img_dir.glob("*.jpg")):
            case_id = img_path.stem
            save_image(img_path, images_ts / f"{case_id}_0000.png")

    dataset_json = {
        "channel_names": {"0": "RGB"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(all_train_cases),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    with open(dataset_folder / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"已生成 {dataset_folder}")
    print(f"训练样本数: {len(all_train_cases)}")
    if images_ts.is_dir():
        test_count = len(list(images_ts.glob("*.png")))
        print(f"测试样本数: {test_count}")


if __name__ == "__main__":
    sys.exit(main())

