# NnUNet ISIC2017 图像分割指南
_Exported on 12/27/2025 at 18:02:32 PST from Cursor (2.2.44)_

---

**User**

在external/nnUNet下有nnUNet的源码。现在我想用nnUNet对examples/automm/Conv-LoRA/datasets/isic2017/isic2017指向的ISIC2017数据集作图像分割。请从nnUNet的README.md开始，完整阅读其各项相关说明，给出详细的 数据集准备->训练->推理 的指导，并生成指导文档和便于一键开始 数据集准备/训练/推理 的各种脚本

---

**Cursor**

```markdown
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

```

```shellscript
#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/works/autogluon"
export nnUNet_raw="${nnUNet_raw:-$ROOT/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$ROOT/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$ROOT/nnUNet_results}"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"

```

```python
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

```

```shellscript
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/isic2017_env.sh"

python "${SCRIPT_DIR}/isic2017_prepare.py" \
  --src "/root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA/datasets/isic2017/isic2017" \
  --raw-root "${nnUNet_raw}" \
  --dataset-id 701 \
  --dataset-name "ISIC2017" \
  --include-val

nnUNetv2_plan_and_preprocess -d 701 --verify_dataset_integrity -c 2d

```

```shellscript
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/isic2017_env.sh"

DATASET="Dataset701_ISIC2017"
CONFIG="${1:-2d}"
FOLDS="${2:-0 1 2 3 4}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

for FOLD in ${FOLDS}; do
  echo "开始训练 ${DATASET} 配置=${CONFIG} 折=${FOLD}（GPU=${CUDA_VISIBLE_DEVICES}）"
  nnUNetv2_train "${DATASET}" "${CONFIG}" "${FOLD}" --npz
done

```

```shellscript
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/isic2017_env.sh"

DATASET="Dataset701_ISIC2017"
CONFIGS="${1:-2d}"

nnUNetv2_find_best_configuration "${DATASET}" -c ${CONFIGS}

```

```shellscript
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/isic2017_env.sh"

INPUT="${1:-${nnUNet_raw}/Dataset701_ISIC2017/imagesTs}"
OUTPUT="${2:-${nnUNet_results}/Dataset701_ISIC2017/predictions_2d}"
CONFIG="${3:-2d}"
FOLDS="${4:-0 1 2 3 4}"

echo "输入: ${INPUT}"
echo "输出: ${OUTPUT}"
echo "配置: ${CONFIG}, 折: ${FOLDS}"

nnUNetv2_predict \
  -i "${INPUT}" \
  -o "${OUTPUT}" \
  -d Dataset701_ISIC2017 \
  -c "${CONFIG}" \
  -f ${FOLDS}

POSTPROC_DIR="${nnUNet_results}/Dataset701_ISIC2017/nnUNetTrainer__nnUNetPlans__${CONFIG}"
PP_FILE="${POSTPROC_DIR}/postprocessing.pkl"
PLANS_JSON="${POSTPROC_DIR}/plans.json"
DATASET_JSON="${POSTPROC_DIR}/dataset.json"

if [ -f "${PP_FILE}" ]; then
  echo "检测到 postprocessing.pkl，正在应用后处理..."
  nnUNetv2_apply_postprocessing \
    -i "${OUTPUT}" \
    -o "${OUTPUT}-postproc" \
    --pp_pkl_file "${PP_FILE}" \
    -plans_json "${PLANS_JSON}" \
    -dataset_json "${DATASET_JSON}"
  echo "后处理完成，结果输出至 ${OUTPUT}-postproc"
else
  echo "未找到 postprocessing.pkl，跳过后处理。"
fi

```

已补充 nnU-Net 在 ISIC2017 上的端到端指南与一键脚本（环境/安装、数据转换、训练、推理），可直接在仓库内使用。核心内容和脚本位置如下所示：

```5:64:external/nnUNet/ISIC2017_nnUNet_guide.md
## 环境与安装
- 建议使用 Python ≥ 3.9，并先安装匹配硬件的 PyTorch（CUDA/MPS/CPU），官方提醒：**Torch 2.9 在 3D AMP 上有性能回退，2D 使用 2.8 及以下更稳**。
- 在本仓库根目录执行：
  cd /root/autodl-tmp/works/autogluon/external/nnUNet
  pip install -e .
  pip install pillow
…
一键转换脚本（合并 train+val 作为训练，test 作为测试）：
cd /root/autodl-tmp/works/autogluon/external/nnUNet/scripts
chmod +x isic2017_env.sh isic2017_prepare.sh
./isic2017_prepare.sh
…
默认使用 2D 配置、5 折交叉验证并保存 softmax (--npz)… ./isic2017_train.sh
…
推理：./isic2017_infer.sh，自动检测 postprocessing.pkl
```

```21:127:external/nnUNet/scripts/isic2017_prepare.py
def save_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img.convert("RGB").save(dst)
…
train_cases = collect_cases(train_img_dir, train_mask_dir, is_training=True)
…
save_image(img_dir / f"{case_id}.jpg", images_tr / f"{case_id}_0000.png")
save_mask(mask_dir / f"{case_id}_segmentation.png", labels_tr / f"{case_id}.png")
…
json.dump({"channel_names": {"0": "RGB"},
           "labels": {"background": 0, "lesion": 1},
           "numTraining": len(all_train_cases),
           "file_ending": ".png",
           "overwrite_image_reader_writer": "NaturalImage2DIO"}, ...)
```

```7:39:external/nnUNet/scripts/isic2017_infer.sh
INPUT="${1:-${nnUNet_raw}/Dataset701_ISIC2017/imagesTs}"
OUTPUT="${2:-${nnUNet_results}/Dataset701_ISIC2017/predictions_2d}"
CONFIG="${3:-2d}"
FOLDS="${4:-0 1 2 3 4}"
nnUNetv2_predict -i "${INPUT}" -o "${OUTPUT}" -d Dataset701_ISIC2017 -c "${CONFIG}" -f ${FOLDS}
…
if [ -f "${PP_FILE}" ]; then
  nnUNetv2_apply_postprocessing -i "${OUTPUT}" -o "${OUTPUT}-postproc" …
fi
```

使用步骤（默认路径已写在脚本里，可按需修改）：
- 安装与环境变量：`pip install -e . && pip install pillow`，执行 `scripts/isic2017_env.sh` 生成/打印 `nnUNet_raw` 等目录。
- 数据准备：`scripts/isic2017_prepare.sh`（合并 train+val，转换为 PNG，写 dataset.json，并运行 `nnUNetv2_plan_and_preprocess -d 701 --verify_dataset_integrity -c 2d`）。若只想用 train，可在脚本内把 `--include-val` 去掉或设为 `False`。
- 训练：`scripts/isic2017_train.sh`（默认 2d、折 0-4，可用 `CUDA_VISIBLE_DEVICES` 指定 GPU；已加 `--npz` 便于后处理/集成）。
- 自动挑选配置/后处理：`scripts/isic2017_find_best.sh`（当前仅 2d 也会输出后处理指令）。
- 推理：`scripts/isic2017_infer.sh [可选输入] [可选输出] [配置] [折列表]`，若存在 `postprocessing.pkl` 会自动生成 `*-postproc` 结果。

注意事项：
- 需先安装 Pillow；Torch 建议 ≤2.8 以避开 3D AMP 回退（2D 也安全）。
- 预处理/软概率文件占用较大，请确保 `nnUNet_preprocessed` 与 `nnUNet_results` 磁盘空间。
- 输入命名必须与训练一致（`{ID}_0000.png`），并保持无损格式。

测试：未实际运行训练/推理命令（仅编写脚本与流程说明）；如需验证请按上述脚本执行。

```plaintext
## 环境与安装
- 建议使用 Python ≥ 3.9，并先安装匹配硬件的 PyTorch（CUDA/MPS/CPU），官方提醒：**Torch 2.9 在 3D AMP 上有性能回退，2D 使用 2.8 及以下更稳**。
- 在本仓库根目录执行：
  cd /root/autodl-tmp/works/autogluon/external/nnUNet
  pip install -e .
  pip install pillow
…
一键转换脚本（合并 train+val 作为训练，test 作为测试）：
cd /root/autodl-tmp/works/autogluon/external/nnUNet/scripts
chmod +x isic2017_env.sh isic2017_prepare.sh
./isic2017_prepare.sh
…
默认使用 2D 配置、5 折交叉验证并保存 softmax (--npz)… ./isic2017_train.sh
…
推理：./isic2017_infer.sh，自动检测 postprocessing.pkl
```

```python
def save_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img.convert("RGB").save(dst)
…
train_cases = collect_cases(train_img_dir, train_mask_dir, is_training=True)
…
save_image(img_dir / f"{case_id}.jpg", images_tr / f"{case_id}_0000.png")
save_mask(mask_dir / f"{case_id}_segmentation.png", labels_tr / f"{case_id}.png")
…
json.dump({"channel_names": {"0": "RGB"},
           "labels": {"background": 0, "lesion": 1},
           "numTraining": len(all_train_cases),
           "file_ending": ".png",
           "overwrite_image_reader_writer": "NaturalImage2DIO"}, ...)
```

```shellscript
INPUT="${1:-${nnUNet_raw}/Dataset701_ISIC2017/imagesTs}"
OUTPUT="${2:-${nnUNet_results}/Dataset701_ISIC2017/predictions_2d}"
CONFIG="${3:-2d}"
FOLDS="${4:-0 1 2 3 4}"
nnUNetv2_predict -i "${INPUT}" -o "${OUTPUT}" -d Dataset701_ISIC2017 -c "${CONFIG}" -f ${FOLDS}
…
if [ -f "${PP_FILE}" ]; then
  nnUNetv2_apply_postprocessing -i "${OUTPUT}" -o "${OUTPUT}-postproc" …
fi
```

