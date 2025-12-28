# NnUNet ISIC2017 图像分割指南
_Exported on 12/27/2025 at 21:22:29 PST from Cursor (2.2.44)_

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

---

**User**

我想先开始规划下一步实验，但是在使用nnUNet测试ISIC2017基线分割性能前，先只做规划设计而不要动代码：下一步我想要把本项目examples/automm/Conv-LoRA中的Conv-LoRA与nnUNet相结合。Conv-LoRA原本是用于SAM的PEFT微调的，你要仔细考虑如何把相同的方法应用在nnUNet的什么地方。

---

**Cursor**

你这个方向很对：先在“设计层面”想清楚 **Conv‑LoRA 应该插到 nnU‑Net 的哪里、怎么训练、怎么评估**，再动代码会省很多返工。接下来我会先把 `examples/automm/Conv-LoRA` 里 Conv‑LoRA 的实现/配置读透，再对照 nnU‑Net v2 的网络构建与训练器入口，给出几套可行的融合方案（从最保守到最激进），并指出各自的风险点与实验顺序。

我先把 `Conv-LoRA` 示例的实际训练入口和它在 AutoGluon 里“PEFT=conv_lora”的实现位置找出来；弄清楚它到底对哪些层做了什么，再把同样的“插入点”映射到 nnU‑Net 的网络定义（encoder/decoder 的 conv blocks、跳连、stem 等）。接着我会给你一份**不改代码的融合设计方案**（包含优先级、消融实验矩阵、风险点）。

好消息：在 AutoGluon 这边，`Conv-LoRA` 并不是“只有论文概念”，而是已经落成在 `autogluon.multimodal.models.adaptation_layers.py` / `modeling_sam_for_conv_lora.py` 里了；下一步我会把它的**插入位置（哪些层）、参数形状、专家路由方式**读出来，然后再把同一套思路映射到 nnU‑Net 的 `build_network_architecture/get_network_from_plans` 这条网络构建链路上，给出清晰的融合点选择与实验矩阵。

### 我对“Conv‑LoRA 应用到 nnU‑Net 的哪里”的初步结论（先不动代码）

从你项目里的实现看，AutoGluon 版 **Conv‑LoRA 的关键不是“给卷积加 LoRA”**，而是：**在 LoRA 的低秩中间表示上再加一个 MoE‑Conv（多专家卷积）分支**，用全局池化做 gating（top‑1），让低秩更新具备空间/多尺度建模能力（见 `ConvLoRALinear`：先做 \(xA\)，reshape 成特征图，再 MoE‑Conv，再投回 \(B\)）。

nnU‑Net v2 的默认网络是 **纯卷积 UNet/ResidualEncoderUNet**（网络由 `nnUNetTrainer.build_network_architecture -> get_network_from_plans` 构建；推理也走同一入口），因此“把相同方法应用在 nnUNet 的什么地方”有两条路线：

---

### 路线 A（最贴合原方法）：把 Conv‑LoRA 的“MoE‑Conv on low‑rank features”迁移到 **nnU‑Net 的 Conv2d**
把 nnU‑Net 里某些 `Conv2d` 替换/包裹为“Conv‑LoRA‑Conv2d”形式（概念上）：

- **主干卷积（冻结）**：\(y = \text{Conv}(x; W)\)
- **低秩分支（可训练）**：
  - \(u = \text{Conv1x1}_A(x)\) 把通道压到 rank \(r\)
  - \(u' = u + \text{MoEConv}(u)\)（多个 3×3 专家；gating 用 GAP+线性；top‑1；可用不同 dilation/不同 upsample ratio 模拟多尺度）
  - \(\Delta y = \text{Conv1x1}_B(u') \cdot \alpha/r\)
- **输出**：\(y \leftarrow y + \Delta y\)

这就是把你们现在在 `ConvLoRALinear` 上做的那套“**低秩 + MoE‑Conv**”搬到 nnU‑Net 的卷积层上，语义完全一致，只是把 “Linear 的 token 网格” 换成了 “Conv 的特征图”。

**nnU‑Net 里最合理的插入点（优先级从高到低）**  
- **P0：encoder 的后半段（深层）卷积块**：语义强、参数量大，PEFT 最划算；且 MoE‑Conv 的多尺度更容易贡献增益。  
- **P1：bottleneck（最底部）卷积块**：最省参数/最稳定的第一版验证点。  
- **P2：decoder 的卷积块**：更偏边界与细节恢复，常见现象是提升细粒度 IoU，但也可能带来噪声。  
- **P3：skip 连接融合前（concat 前后）的 1×1/3×3**：对跨尺度信息融合做“轻量可控的域适配”，但实现上需要更了解网络结构细节（不建议第一版就上）。

**Norm 层怎么处理（策略建议）**  
- nnU‑Net 常用 `InstanceNorm(affine=True)`：建议**保留 affine 参数可训练**（和 LoRA 常见做法类似：Norm/Head 不完全冻结），其余主干 conv 权重冻结，训练更稳。

---

### 路线 B（复用现成实现）：只在 nnU‑Net 的“带 Linear 的架构”上用原版 `ConvLoRALinear`
nnU‑Net v2 里确实存在 transformer/混合架构 trainer（例如 `Primus`），这类网络有大量 `nn.Linear`，可以更“原汁原味”地直接复用你们现在的 `ConvLoRALinear` 注入逻辑。

但这条路线的代价是：你实验对象就不再是 nnU‑Net 的经典 UNet（而是 Primus 这类变体），对“nnU‑Net baseline + Conv‑LoRA”这个叙事可能不够直观；更适合后续拓展，而不是第一阶段。

---

### 我建议的实验规划顺序（只做设计，不改代码）
**先把“增益来自哪里”拆清楚**，否则容易陷入参数/位置/训练策略的组合爆炸：

1. **nnU‑Net full finetune 基线**（ISIC2017）  
2. **freeze‑backbone 只训 head/最后一两层**（下界参考）  
3. **LoRA‑Conv2d（无 MoE）**：验证“低秩更新”本身的收益  
4. **Conv‑LoRA‑Conv2d（带 MoE‑Conv）**：验证“MoE‑Conv”额外收益  
5. **插入点消融**：bottleneck → encoder 后半段 → encoder+decoder  
6. **超参网格（小规模）**：`rank r ∈ {2,4,8}`，`expert_num M ∈ {4,8}`（top‑1 固定）

每一步都记录：**IoU/Dice、可训练参数量、吞吐/显存**，这样你的“PEFT 是否值得”会非常清晰。

---

### 关键约束提醒（决定你最终该怎么融合）
因为 nnU‑Net 的网络构建入口 `build_network_architecture` 同时用于训练与推理，所以最终实现时要么：
- 用一个 **新的 Trainer 类**（例如 `nnUNetTrainer_ConvLoRA`）在 build 时注入/包裹模块；保证推理能加载同构网络；
- 要么保证注入逻辑在推理时也能复现（同一 trainer 名称/同一路径）。

---

### 我需要你确认 2 个问题，才能把“融合到哪里”收敛成唯一方案
- **目标优先级**：你更看重 **(A) 性能提升** 还是 **(B) 训练参数/显存/速度下降**（PEFT 的本意）？两者会影响插入范围与是否训练 Norm/Head。  
- **工程形态**：你希望最终 **继续用 nnU‑Net CLI 流程**（plan/train/predict 一条龙），还是希望 **集成进 AutoGluon/Conv‑LoRA 的训练脚手架**（统一日志、超参搜索、数据管线）？

你回这两点后，我就能把上面路线 A 的“插入点 + 消融矩阵 + 预期风险”收敛成一份可执行的实验设计文档（仍不改代码）。

---

**User**

看重性能提升； 集成进 AutoGluon/Conv‑LoRA 的训练脚手架。

---

**Cursor**

收到：你更看重 **性能提升**，并且希望 **把 nnU‑Net + Conv‑LoRA 融合进 AutoGluon/Conv‑LoRA 的训练脚手架**。接下来我会先把 AutoGluon 里“语义分割训练管线/模型注册/PEFT 注入点/日志与评估”几处关键入口摸清，再给出一份**实现不改代码的融合设计**：nnU‑Net 作为新模型后端如何接入，Conv‑LoRA 应该改造成 Conv2d 版本插到哪些 block，训练与推理如何对齐 nnU‑Net 规范，以及逐步实验矩阵。

接下来我会把 AutoGluon 语义分割这条链路里“模型是怎么被 select_model 构建出来、PEFT 什么时候 apply、训练时 loss/输出格式如何约定”的细节读出来。这样我们才能把 nnU‑Net 当成一个新的 `model.*` 选项接进去，并把 Conv‑LoRA 从 Linear 扩展到 Conv2d 的注入机制复用掉（而不是重写一套训练脚手架）。

### 你给的目标约束下，最合理的融合路线（结论先说）
你要 **性能优先** 且 **接入 AutoGluon/Conv‑LoRA 脚手架**，最佳策略是：

- **把 nnU‑Net 当作 AutoGluon 的一个新 `model` 后端**（像现有 `sam` 一样走 `create_model -> Lightning LitModule` 的训练/评估/保存流程）。
- **把 Conv‑LoRA 从 “Linear 版” 扩展成 “Conv2d 版（ConvLoRAConv2d）”**，并复用 AutoGluon 已经打通的 **MoE loss 端到端链路**（模型 forward 把 `MOE_LOSS` 放进输出 dict，`SemanticSegmentationLitModule` 会自动加到总 loss 里）。

这样你既能用 AutoGluon 的统一日志/超参/数据表接口，也能把 Conv‑LoRA 的核心思想落到 nnU‑Net 最关键的算子（卷积）上。

---

### 现有代码里“你要接入的接口点”已经非常清晰
#### 1) AutoGluon 语义分割管线的模型接口契约
- 模型 `forward(batch)` 返回 `Dict[prefix, Dict]`，至少要有 `LOGITS`；训练时不要求 `LABEL`，评估时会带上（参考 `SAMForSemanticSegmentation.forward`）：

```406:438:multimodal/src/autogluon/multimodal/models/sam.py
if self.training:
    rets_dict = {self.prefix: {LOGITS: pred_masks}}
else:
    rets_dict = {self.prefix: {LOGITS: pred_masks, LABEL: batch[self.label_key]}}
...
if self.output_moe_loss:
    rets_dict[self.prefix].update({MOE_LOSS: rets.vision_moe_loss})
```

- loss 侧：`SemanticSegmentationLitModule` 会对每个 `per_output[LOGITS]` 调 `loss_func(input=..., target=label)`，并自动把 `MOE_LOSS` 加进去：

```23:48:multimodal/src/autogluon/multimodal/optim/lit_semantic_seg.py
loss += self.loss_func(input=per_output[LOGITS], target=label) * weight
if MOE_LOSS in per_output:
    loss += per_output[MOE_LOSS]
```

这意味着：**只要你的 nnU‑Net 模型输出 dict 里塞进 `MOE_LOSS`，就天然接入了训练目标**，不需要改训练循环。

#### 2) 模型选择/创建入口
AutoGluon 的模型创建是集中式的：`create_model` 里有 `elif model_name.lower().startswith(SAM): ...`。你加 nnU‑Net 后端也应该对齐这个入口（新增一个分支，比如 `NNUNET`）：

```1538:1550:multimodal/src/autogluon/multimodal/models/utils.py
elif model_name.lower().startswith(SAM):
    from .sam import SAMForSemanticSegmentation
    model = SAMForSemanticSegmentation(...)
```

#### 3) 现有 PEFT 的注入模式（你要复用而不是另起炉灶）
目前 `apply_peft_adaptation` 只对 `nn.Linear` 做替换（`inject_adaptation_to_linear_layer`），而 `conv_lora` 实现也是 **Linear 版**（`ConvLoRALinear`），所以对 nnU‑Net（几乎全是 Conv2d）“什么都不会发生”。

但这个模式本身非常可复用：**新增一个 `inject_adaptation_to_conv2d_layer`**，让 `conv_lora` 同时支持 `Linear + Conv2d`（由配置决定 target）。

---

### Conv‑LoRA 应该插到 nnU‑Net 的哪里？（性能优先的建议）
nnU‑Net 的默认网络（`PlainConvUNet`/`ResidualEncoderUNet`）全是 conv blocks。你要的 Conv‑LoRA 本质是“冻结主干 + 低秩分支 + 在低秩分支上做 MoE‑Conv 扩大空间建模能力”。因此 **插入点就应该是 Conv2d 本身**。

**推荐插入顺序（从最稳到最强）**  
- **P0（最稳验证）**：只在 **bottleneck / 最深一层** 的 3×3 Conv2d 上插 Conv‑LoRA‑Conv2d  
- **P1（通常性能最划算）**：扩展到 **encoder 的后 2 个 stage 的 3×3 conv**  
- **P2（追上限）**：encoder 后半 + decoder 全部 stage 的 3×3 conv（通常对边界细节更敏感）  
- **P3（极限）**：几乎所有 Conv2d 都插（但会更难调、也更吃显存/吞吐）

**为什么不建议第一版就“全网络插满”？**  
因为你接下来还要解决：MoE loss 权重、rank、专家数、数据增强强度、输入分辨率等问题；全插会让这些因素高度耦合，排错很慢。

---

### Conv‑LoRA→Conv2d 的“等价迁移”怎么做（保持方法一致）
你们现在的 Conv‑LoRA（Linear 版）关键是：  
1) 做低秩 \(xA\) 得到 rank 特征  
2) 把 rank 特征 reshape 成特征图（或本来就是特征图）  
3) 在 rank 特征上跑 **MoE‑Conv**（带 gating + 多尺度）  
4) 再投回 \(B\)，形成增量并加回主干输出  
5) 输出一个 **load balancing 的 moe_loss**（目前实现是 `cv_squared(importance)+cv_squared(load)`）

对应到 Conv2d 的自然形式就是：
- `A`: 1×1 Conv（in→r）
- `MoEConv`: 多个 3×3 Conv 专家（可用 dilation 或 “上采样再卷积再下采样” 来做多尺度，与你们现实现一致）
- `B`: 1×1 Conv（r→out）
- 主干 Conv 权重冻结；只训练 `lora_*` 参数与（可选）norm/seg head

**关键工程细节（为了无缝接入 AutoGluon）**
- 参数命名保持 `lora_*` 前缀：AutoGluon 的 `get_peft_param_names` 会用正则 `.*lora_*.` 自动抓 trainable 参数（见 `optim/utils.py`）。
- 模型 forward 里，把所有 ConvLoRAConv2d 产生的 moe_loss 累加，放到输出 dict 的 `MOE_LOSS` 字段（AutoGluon 训练 loop 自动加到总 loss）。

---

### 作为 AutoGluon 后端的 nnU‑Net：你需要实现/对齐哪些能力
为了能像 `sam` 一样跑起来，nnU‑Net 后端模型类需要具备：

- **必要属性**（给数据处理器用）：`prefix`, `image_size`, `image_mean`, `image_std`, `num_classes`
  - `SemanticSegImageProcessor` 会读取 `model.image_size/mean/std/num_classes` 并做 resize/normalize（见 `process_semantic_seg_img.py`）。
- **必要 key 命名**：`image_key`, `label_key` 等（同 SAM 的约定：`f"{prefix}_{IMAGE}"` / `f"{prefix}_{LABEL}"`）。
- **forward 输出契约**：返回 `{prefix: {LOGITS: ...}}`（多类则 `LOGITS` 直接是 per-pixel logits，或者走 Mask2Former 风格另说）。
- **layer id 分配**：实现 `get_layer_ids()`，否则 layerwise lr decay 之类策略不好用（SAM 已实现）。

---

### “性能优先”的额外建议：别被 nnU‑Net 之名限制住
你最终要的是 **ISIC2017 上性能**，而不是“严格复刻 nnU‑Net CLI 的所有预处理与 patch 采样”。在 AutoGluon 框架里，性能上最可能的瓶颈反而是：
- 输入分辨率（`resize_to_square` 到多少）
- 数据增强（目前 semantic seg 默认增强很轻）
- loss（nnU‑Net 的 Dice+CE 往往优于纯 CE/BCE）

因此我建议把实验拆两条线并行推进：
- **线 1（快验证融合正确性）**：nnU‑Net(architecture) + AutoGluon dataloader/loss/augment，先把 Conv‑LoRA‑Conv2d 的收益验证出来。
- **线 2（冲上限）**：逐步把 nnU‑Net 的强项迁入 AutoGluon：Dice+CE、deep supervision、多尺度增强/强增强策略。这样每一步都能量化带来多少提升。

---

### 下一步实验设计（按你“性能优先”排序）
建议的阶段/消融（每步都用同一套 AutoGluon 记录 iou/dice、吞吐、显存）：

- **Stage A：架构基线**
  - A0：现有 SAM baseline（你们 Conv‑LoRA 已跑过）
  - A1：nnU‑Net(architecture) full finetune（不加 Conv‑LoRA）
- **Stage B：Conv‑LoRA 插入点消融（先小后大）**
  - B0：只 bottleneck conv
  - B1：encoder 后 2 个 stage
  - B2：encoder 后半 + decoder 全部
- **Stage C：超参网格（小规模）**
  - `rank r ∈ {2,4,8}`
  - `expert_num M ∈ {4,8}`（top‑1 先固定）
  - `moe_loss_weight ∈ {1e-3, 1e-2, 5e-2}`（你们当前实现默认 1e-2）
- **Stage D：训练策略**
  - D0：只训练 lora 参数
  - D1：lora + norm affine（推荐）
  - D2：lora + norm + seg head（通常更稳）

---

### 我建议你现在确认的 3 个“设计决策”（决定后续实现最省事）
1) **nnU‑Net 后端选型**：先用 `ResidualEncoderUNet` 还是 `PlainConvUNet`？（ISIC2017 2D 一般 ResEnc 更强但更重）  
2) **Conv‑LoRA 的多尺度实现**：沿用你们现有的 “upsample_ratio 专家”（更贴论文/已有实现），还是改成 dilation 专家（更省算力/更 CNN 原生）？  
3) **多类/二类接口**：ISIC2017 是二类（背景/病灶），我们是否统一用 `num_classes=1` 的二值 logits 形式（最省改动），还是走 `num_classes=2` 的多类 softmax（更通用）？

你回复这 3 点后，我可以把上面方案收敛成一份“最终融合蓝图”：包括需要新增的模块/配置项/超参键名清单、与现有 `run_semantic_segmentation.py` 的参数对齐方式、以及第一轮实验的具体命令模板（仍然先不动代码，先把路线定死）。

```python
if self.training:
    rets_dict = {self.prefix: {LOGITS: pred_masks}}
else:
    rets_dict = {self.prefix: {LOGITS: pred_masks, LABEL: batch[self.label_key]}}
...
if self.output_moe_loss:
    rets_dict[self.prefix].update({MOE_LOSS: rets.vision_moe_loss})
```

```python
loss += self.loss_func(input=per_output[LOGITS], target=label) * weight
if MOE_LOSS in per_output:
    loss += per_output[MOE_LOSS]
```

```python
elif model_name.lower().startswith(SAM):
    from .sam import SAMForSemanticSegmentation
    model = SAMForSemanticSegmentation(...)
```

---

**User**

1. ResEnc 2. “upsample_ratio 专家” 3. 二值 logits 形式 

---

**Cursor**

下一步我会把三件事落地：新增一个 AutoGluon 的 `nnunet` 语义分割模型后端（ResEnc UNet），扩展现有 `conv_lora` 让它能注入到 `Conv2d`（保留你选的 upsample_ratio 专家），并把这些都接到现有 `run_semantic_segmentation.py` 的超参入口上，这样你可以一条命令跑 nnUNet+Conv‑LoRA 消融。现在先把 AutoGluon 的 batch 形状/`run_model` 约定读清楚，避免接入时踩张量维度坑。

计划：先把 `ConvLoRAConv2d + 注入函数` 加到 AutoGluon 的 PEFT 体系里（保证 `optim.peft=conv_lora` 对 Conv2d 生效并能回传 `MOE_LOSS`），再新增一个 `nnunet` 语义分割模型类（ResEnc UNet、二值 logits、upsample_ratio 专家），最后把它注册进 `create_model` + 默认 YAML 配置，并在 `run_semantic_segmentation.py` 里加一个开关让你能一键切换 `sam` vs `nnunet`。接下来我会直接开始改代码实现这些接口。

```python
# Modified based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def identity(x):
    return x


class LoRALayer:
    """
    Abstract LoRA Layer.

    Parameters
    ----------
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = identity
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class IA3LoRALinear(nn.Linear, LoRALayer):
    """
    LoRA (low-rank adaptation) followed by (IA)^3 (weight rescaling) incorporated in a Linear Layer. Weights of Linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r=8,
        lora_alpha=8,
        lora_dropout: float = 0.0,
        fan_in_fan_out=False,
        merge_weights=False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        # In essence the $b$ parameter of LoRA.
        self.lora_b = nn.Parameter(torch.ones(out_features, 1))
        self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
        self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
        self.fan_in_fan_out = fan_in_fan_out
        self.weight.requires_grad = False

        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.T(self.weight), bias=self.bias)
        if self.r > 0:
            result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

        hidden = result * self.lora_b.flatten()
        return hidden

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data /= self.lora_b.flatten()
                self.weight.data -= self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        # def T(w):
        #     return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += self.T(self.lora_B @ self.lora_A) * self.scaling
                self.weight.data *= self.lora_b.flatten()
            self.merged = True
        return hidden

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class IA3Linear(nn.Linear, LoRALayer):
    """
    (IA)^3 incorporated in a Linear Layer. Weights of Linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    scaling_rank
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Liu, Haokun and Tam, Derek and Muqeeth, Mohammed and Mohta, Jay and Huang, Tenghao and Bansal, Mohit and Raffel, Colin,
    "Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning", 2022
    https://arxiv.org/pdf/2205.05638.pdf
    2. Code: https://github.com/r-three/t-few
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        merge_weights: False,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self, r=4, lora_alpha=4, lora_dropout=0.0, merge_weights=merge_weights
        )  # Default arguments, only
        # In essence the $b$ parameter of LoRA.
        self.lora_b = nn.Parameter(torch.ones(out_features, 1))
        self.weight.requires_grad = False

    def forward(self, x: torch.Tensor):
        hidden = F.linear(x, self.weight, self.bias)
        hidden = hidden * self.lora_b.flatten()
        return hidden

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data /= self.lora_b.flatten()
            self.merged = False

    def eval(self):
        # def T(w):
        #     return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data *= self.lora_b.flatten()
            self.merged = True
        return hidden

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class LoRALinear(nn.Linear, LoRALayer):
    """
    LoRA incorporated in Linear Layer. Weights of linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        # def T(w):
        #     return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += self.T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, self.T(self.weight), bias=self.bias)


class LoRAEmbedding(nn.Embedding, LoRALayer):
    """
    LoRA incorporated in Embedding Layer. Weights of embedding layer are set to be frozen per default.

    Parameters
    ----------
    num_embeddings
        size of the dictionary of embeddings.
    embedding_dim
         the size of each embedding vector.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0, merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(
                    x,
                    self.lora_A.T,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )
                result += (after_A @ self.lora_B.T) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class LoRAMergedLinear(nn.Linear, LoRALayer):
    """
    LoRA where single nn.Linear represents more than one layer (used in some implementations of attention query/key/value projections). Weights of linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing
    r
        rank r of the low-rank decomposition
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout rate for LoRA
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out)
    merge_weights
        Merging weights during inference to reduce latency

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, "The length of enable_lora must divide out_features"
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(-1, self.out_features // len(self.enable_lora) * sum(self.enable_lora))
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data -= self.zero_pad(T(delta_w * self.scaling))
            self.merged = False

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0 and any(self.enable_lora):
                delta_w = F.conv1d(
                    self.lora_A.data.unsqueeze(0), self.lora_B.data.unsqueeze(-1), groups=sum(self.enable_lora)
                ).squeeze(0)
                self.weight.data += self.zero_pad(T(delta_w * self.scaling))
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_A = F.linear(self.lora_dropout(x), self.lora_A)
                after_B = F.conv1d(
                    after_A.transpose(-2, -1), self.lora_B.unsqueeze(-1), groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                result += self.zero_pad(after_B) * self.scaling
            return result


class LoRAConv2d(nn.Conv2d, LoRALayer):
    """
    LoRA incorporated in 2d-Convolutional Layer. Weights of convolutional layer are set to be frozen per default.

    Parameters
    ----------
    in_channels
         Number of channels in the input image.
    out_channels
        Number of channels produced by the convolution.
    kernel_size
        Size of the convolving kernel.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Adding dropout to LoRA.
    merge_weights
        Merging weights during inference to reduce latency.

    References
    ----------
    1. Edward J. Hu*, Yelong Shen*, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
    "LoRA: Low-Rank Adaptation of Large Language Models", 2021
    https://arxiv.org/abs/2106.09685
    2. Code: https://github.com/microsoft/LoRA
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
        **kwargs,
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert type(kernel_size) is int
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r * kernel_size, in_channels * kernel_size)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_channels * kernel_size, r * kernel_size)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = False

    def eval(self):
        nn.Conv2d.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.conv2d(
                x,
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        return nn.Conv2d.forward(self, x)


class ConvLoRAConv2d(nn.Conv2d, LoRALayer):
    """
    Conv-LoRA for Conv2d.

    设计目标：把现有 `ConvLoRALinear` 的 “低秩 + MoE-Conv(upsample_ratio 专家) + moe_loss” 思路迁移到 Conv2d，
    用于卷积网络（如 nnU-Net/ResEnc UNet）的 PEFT。

    - 冻结原始 Conv2d 权重 (self.weight)。
    - 低秩分支：x -> (1x1 conv, stride=原stride) -> MoEConv(r通道) -> (1x1 conv) -> add to base output。
    - MoE gating：GAP + 线性门控，top-1，带 load balancing loss（与 ConvLoRALinear 相同形式）。

    备注：当前仅支持 groups==1（nnU-Net 默认即如此）。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        merge_weights: bool = False,
        conv_lora_expert_num: Optional[int] = None,
    ):
        nn.Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        if groups != 1:
            raise NotImplementedError("ConvLoRAConv2d 暂不支持 groups!=1 的卷积层。")

        self.scaling = self.lora_alpha / self.r if self.r > 0 else 1.0
        self.moe_loss = None

        if self.r > 0:
            if conv_lora_expert_num is None or conv_lora_expert_num <= 0:
                raise ValueError("conv_lora_expert_num 必须为正整数")

            # 低秩 1x1 “A”：用参数矩阵实现（等价于 1x1 conv），stride 使用原 stride 以对齐输出分辨率
            self.lora_A = nn.Parameter(torch.zeros((self.r, in_channels)))
            # 低秩 1x1 “B”
            self.lora_B = nn.Parameter(torch.zeros((out_channels, self.r)))

            # 冻结原始权重
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

            # MoE-Conv（与 ConvLoRALinear 保持一致：不同 upsample_ratio 的专家）
            topk = 1
            self.lora_moe_gating = MoEGate(M=conv_lora_expert_num, d=self.r, K=topk)
            self.lora_moe_experts = nn.ModuleList([])
            self.upsample_ratios = list(range(1, conv_lora_expert_num + 1))
            for upsample_ratio in self.upsample_ratios:
                expert = nn.Conv2d(in_channels=self.r, out_channels=self.r, kernel_size=3, stride=1, padding=1, bias=True)
                expert.bias.data.zero_()
                self.lora_moe_experts.append(nn.Sequential(expert, nn.GELU()))
            self.num_experts = conv_lora_expert_num
            self.multiply_by_gates = False

            self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def _conv1x1_from_A(self) -> torch.Tensor:
        # [r, in] -> [r, in, 1, 1]
        return self.lora_A.view(self.r, self.in_channels, 1, 1)

    def _conv1x1_from_B(self) -> torch.Tensor:
        # [out, r] -> [out, r, 1, 1]
        return self.lora_B.view(self.out_channels, self.r, 1, 1)

    def forward(self, x: torch.Tensor):
        # base conv
        result = nn.Conv2d.forward(self, x)
        if self.r <= 0:
            self.moe_loss = None
            return result

        # low-rank branch: down-proj (stride aligns output resolution)
        lora_in = self.lora_dropout(x)
        lora_res = F.conv2d(
            lora_in,
            self._conv1x1_from_A(),
            bias=None,
            stride=self.stride,
            padding=0,
            dilation=1,
            groups=1,
        )

        # MoEConv on low-rank features
        gates, moe_loss = self.lora_moe_gating(lora_res)
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(lora_res)
        expert_outputs = []
        H, W = lora_res.shape[-2:]
        for i in range(self.num_experts):
            if len(expert_inputs[i]) == 0:
                continue
            upsample_ratio = self.upsample_ratios[i]
            cur = expert_inputs[i]
            if upsample_ratio != 1:
                cur = F.interpolate(cur, scale_factor=upsample_ratio, mode="bicubic")
            cur = self.lora_moe_experts[i](cur)
            if upsample_ratio != 1:
                cur = F.interpolate(cur, size=(int(H), int(W)), mode="bicubic")
            expert_outputs.append(cur)
        temp = dispatcher.combine(expert_outputs, multiply_by_gates=self.multiply_by_gates)
        lora_res = lora_res + temp

        # up-proj
        delta = F.conv2d(
            lora_res,
            self._conv1x1_from_B(),
            bias=None,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
        )

        self.moe_loss = moe_loss
        return result + delta * self.scaling


class ConvLoRALinear(nn.Linear, LoRALayer):
    """
    Conv-LoRA incorporated in Linear Layer. Weights of linear layer are set to be frozen per default.

    Parameters
    ----------
    in_features
        input dimension, set to the original linear layer input dimension LoRA is replacing.
    out_features
        output dimension, set to the original linear layer output dimension LoRA is replacing.
    r
        rank r of the low-rank decomposition.
    lora_alpha
        Scaling factor. Can be simply set to same value as r as initialization is scaled already.
    lora_dropout
        Dropout probability.
    fan_in_fan_out
        Set this to True if the layer to replace stores weight like (fan_in, fan_out).
    merge_weights
        Merging weights during inference to reduce latency.
    conv_lora_expert_num
        The number of experts in MoE-Conv.

    References
    ----------
    1. Zihan Zhong, Zhiqiang Tang, Tong He, Haoyang Fang, Chun Yuan,
    "Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model", 2024
    https://arxiv.org/abs/2401.17868
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        conv_lora_expert_num: Optional[int] = None,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            # MoE-Conv
            topk = 1
            self.lora_moe_gating = MoEGate(M=conv_lora_expert_num, d=self.r, K=topk)
            self.lora_moe_experts = nn.ModuleList([])
            self.upsample_ratios = list(range(1, conv_lora_expert_num + 1))
            for upsample_ratio in self.upsample_ratios:
                expert = nn.Conv2d(in_channels=r, out_channels=r, kernel_size=3, stride=1, padding=1, bias=True)
                expert.bias.data.zero_()
                self.lora_moe_experts.append(nn.Sequential(expert, nn.GELU()))
            self.num_experts = conv_lora_expert_num
            self.multiply_by_gates = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def T(self, w):
        return w.T if self.fan_in_fan_out else w

    def forward(self, x: torch.Tensor):
        result = F.linear(x, self.T(self.weight), bias=self.bias)
        if self.r > 0:
            lora_res = self.lora_dropout(x) @ self.lora_A.T
            dim = lora_res.dim()
            if dim == 3:
                B, L, C = lora_res.size()
                H = W = int(math.sqrt(L))
                lora_res = lora_res.reshape(B, H, W, C)
            else:
                H, W = lora_res.size()[1:3]

            # Calculate the gating values.
            lora_res = lora_res.permute(0, 3, 1, 2).contiguous()
            gates, moe_loss = self.lora_moe_gating(lora_res)

            # Distribute data samples to experts.
            dispatcher = SparseDispatcher(self.num_experts, gates)
            expert_inputs = dispatcher.dispatch(lora_res)
            expert_outputs = []
            for i in range(self.num_experts):
                if len(expert_inputs[i]) == 0:
                    continue
                upsample_ratio = self.upsample_ratios[i]
                cur_res = expert_inputs[i]
                if upsample_ratio != 1:
                    cur_res = F.interpolate(cur_res, scale_factor=upsample_ratio, mode="bicubic")
                cur_res = self.lora_moe_experts[i](cur_res)
                if upsample_ratio != 1:
                    cur_res = F.interpolate(cur_res, size=(int(H), int(W)), mode="bicubic")
                expert_outputs.append(cur_res)

            # Combine data samples after processing by each expert.
            temp_lora_res = dispatcher.combine(expert_outputs, multiply_by_gates=self.multiply_by_gates)
            lora_res = lora_res + temp_lora_res

            lora_res = lora_res.permute(0, 2, 3, 1).contiguous()
            if dim == 3:
                lora_res = lora_res.reshape(B, L, C)
            result += (lora_res @ self.lora_B.T) * self.scaling

        return result, moe_loss


class MoEGate(nn.Module):
    def __init__(self, d, M=4, K=1, noisy_gating=True):
        """Constructor
        Args:
            d: input channel dimensionality.
            M: the number of experts.
            K: the number of chosen experts for each forward pass.
        """
        super(MoEGate, self).__init__()
        self.M = M
        self.k = K
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # global average pooling

        self.noisy_gating = noisy_gating

        self.w_gate = nn.Parameter(torch.zeros(d, M), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(d, M), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.M

    def forward(self, feats, loss_coef=1e-2, noise_epsilon=1e-2):
        batch_size = feats.shape[0]

        feats_S = self.gap(feats).view(batch_size, -1)

        clean_logits = feats_S @ self.w_gate
        if self.noisy_gating and self.training:
            raw_noise_stddev = feats_S @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.M), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True).float()
        gates = zeros.scatter(1, top_k_indices, top_k_gates).to(logits.dtype)

        if self.noisy_gating and self.k < self.M and self.training:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)

        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        return gates, loss

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.

    References
    ----------
    1. Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean,
    "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", 2017
    https://arxiv.org/abs/1701.06538
    2. Code: https://github.com/davidmrau/mixture-of-experts/blob/master/moe.py
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)  # [bs * num_of chosen experts, dim]
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()

        if multiply_by_gates:
            # stitched = stitched.mul(self._nonzero_gates)
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(-1).unsqueeze(-1))
        # zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size()[1],
            expert_out[-1].size()[2],
            expert_out[-1].size()[3],
            requires_grad=True,
            device=stitched.device,
        )
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

```

```python
import functools
import logging
import re
import warnings
from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch._dynamo
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from torch import nn
from torch.nn.modules.loss import _Loss
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertTokenizer, CLIPTokenizer, ElectraTokenizer
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss

from ..constants import (
    ALL_MODALITIES,
    CATEGORICAL,
    CATEGORICAL_MLP,
    CLASS_LOGITS,
    CLIP,
    CLIP_IMAGE_MEAN,
    CLIP_IMAGE_STD,
    DOCUMENT,
    DOCUMENT_TRANSFORMER,
    FT_TRANSFORMER,
    FUSION_MLP,
    FUSION_NER,
    FUSION_TRANSFORMER,
    HF_TEXT,
    IMAGE,
    LOGITS,
    META_TRANSFORMER,
    MMDET_IMAGE,
    MMOCR_TEXT_DET,
    MMOCR_TEXT_RECOG,
    NER_TEXT,
    NUMERICAL,
    NUMERICAL_MLP,
    OCR,
    PEFT_ADDITIVE_STRATEGIES,
    REGRESSION,
    SAM,
    SEMANTIC_MASK,
    SEMANTIC_SEGMENTATION,
    SEMANTIC_SEGMENTATION_IMG,
    T_FEW,
    TEXT,
    TEXT_NER,
    TIMM_IMAGE,
)
from .adaptation_layers import ConvLoRAConv2d, ConvLoRALinear, IA3Linear, IA3LoRALinear, LoRALinear

logger = logging.getLogger(__name__)


ALL_TOKENIZERS = {
    "bert": BertTokenizer,
    "clip": CLIPTokenizer,
    "electra": ElectraTokenizer,
    "hf_auto": AutoTokenizer,
}


class DummyLayer(nn.Module):
    """
    DummyLayer to ensure that the gradient checkpointing will assign output layer as require_grad=True.
    Reference: https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
    """

    def __init__(self):
        super().__init__()
        self.dummy_bias = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def forward(self, x):
        return x + self.dummy_bias.to(x) - self.dummy_bias.to(x)


def init_weights(module: nn.Module):
    """
    Initialize one module. It uses xavier_norm to initialize nn.Embedding
    and xavier_uniform to initialize nn.Linear's weight.

    Parameters
    ----------
    module
        A Pytorch nn.Module.
    """
    if isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def assign_encoder_layer_ids(
    encoder_names: List[List[str]],
):
    """
    Assign ids to encoder layers. The encoder may contain several blocks e.g., block1 and block2.
    This function iterates through all the layers of each block from the input end towards the output end.
    It increases 1 on the layer id when the detected digit in a layer name changes.

    Parameters
    ----------
    encoder_names
        Encoder layer names.

    Returns
    -------
    name_to_id
        The encoder layer-to-id mapping.
    encoder_layer_num
        The encoder layer number.
    """
    name_to_id = {}
    cur_id = 0
    for i, group_names in enumerate(encoder_names):
        last_inferred_id = -1
        for n in group_names:
            detect_id = False
            n_splits = n.split(".")

            for split in n_splits:
                # the first digit encountered is used to infer layer id
                if split.isdigit():
                    inferred_id = int(split)
                    # increase at most 1 one time
                    if inferred_id != last_inferred_id:
                        cur_id += 1  # layer.0 -> layer_id 1
                        last_inferred_id = inferred_id

                    name_to_id[n] = cur_id
                    detect_id = True
                    break

            if detect_id is False:
                raise ValueError(f"parameter name: {n} not has no id inside")

    if len(name_to_id) > 0:
        encoder_layer_num = max(name_to_id.values())
    else:
        encoder_layer_num = 0
    return name_to_id, encoder_layer_num


def assign_non_encoder_layer_ids(
    non_encoder_names: List[str],
    layer_id: int,
):
    """
    Assign the provided id to non-encoder layers.

    Parameters
    ----------
    non_encoder_names
        Names layers not belonging to an encoder.
    layer_id
        provided id.

    Returns
    -------
    A dictionary mapping the layer names (keys) to their ids (values).
    """
    name_to_id = {}
    for n in non_encoder_names:
        name_to_id[n] = layer_id
    return name_to_id


def split_encoder_non_encoder(names: List[str], post_encoder_patterns: Tuple[str, ...]):
    """
    Group layer names into two types: encoder and non-encoder.
    A layer belongs to encoder if its name contains at least one digit.
    It uses this rule since a model's encoder in Pytorch's implementation
    is generally wrapped by nn.Sequential() or nn.ModuleList(),
    which produce digits in layer names.

    Parameters
    ----------
    names
        Model layer names.
    Returns
    -------
    encoder_names
        A list of encoder layer names.
    non_encoder_names
        A list of non-encoder layer names.
    """
    encoder_names = []
    non_encoder_names = []
    for n in names:
        is_encoder = False
        if any(p in n for p in post_encoder_patterns):
            non_encoder_names.append(n)
            continue
        for i in n.split("."):
            if i.isdigit():
                encoder_names.append(n)
                is_encoder = True
                break
        if not is_encoder:
            non_encoder_names.append(n)

    return encoder_names, non_encoder_names


def group_param_names(
    names: List[str],
    pre_encoder_patterns: Tuple[str, ...],
    post_encoder_patterns: Tuple[str, ...],
    model_prefix: Optional[str] = None,
):
    """
    Group layer names into three types: pre-encoder, encoder, and post-encoder.
    If "model_prefix" is provided, the selected layer names must start with it.
    In this case, the left names will be returned for the next-time processing.
    This function first extracts the first-level children modules' names and
    classify them into encoder and non-encoder layers. Note that an encoder may
    consist of several manually named children modules, e.g., block1 and block2.
    The non-encoder layers are further subdivided into pre-encoder and post-encoder.

    Parameters
    ----------
    names
        Model layer names
    pre_encoder_patterns
        Patterns to identify a layer as a pre-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into pre-encoder layers.
    post_encoder_patterns
        Patterns to identify a layer as a post-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into post-encoder layers.
    model_prefix
        A prefix to filter layer names. Only layer names starting with it will be selected.
    Returns
    -------
    left_names
        The layer names left for the next-time processing.
    encoder_names_grouped
        Encoder layer names.
    pre_encoder_names
        Names of layers before the encoder.
    post_encoder_names
        Names of layers after the encoder.
    """
    # two set of patterns can't have intersections
    assert all(pre_p not in post_encoder_patterns for pre_p in pre_encoder_patterns)

    left_names = []
    # in case model_prefix is provided, e.g., the clip model with image and text branches
    selected_names = []
    for n in names:
        if model_prefix is not None and not n.startswith(model_prefix):
            left_names.append(n)
        else:
            selected_names.append(n)

    # split blocks at the first level
    children_prefix = []
    for n in selected_names:
        if model_prefix is not None:
            child_name = n[len(model_prefix) + 1 :].split(".")[0]
            child_prefix = f"{model_prefix}.{child_name}"
        else:
            child_prefix = n.split(".")[0]
        if child_prefix not in children_prefix:
            children_prefix.append(child_prefix)

    encoder_names_grouped = []
    non_encoder_names = []
    for child_prefix in children_prefix:
        per_names_group = [n for n in selected_names if n.startswith(child_prefix)]
        per_encoder_names, per_non_encoder_names = split_encoder_non_encoder(per_names_group, post_encoder_patterns)
        encoder_names_grouped.append(per_encoder_names)
        non_encoder_names.extend(per_non_encoder_names)

    pre_encoder_names = []
    post_encoder_names = []
    for n in non_encoder_names:
        if any(p in n for p in pre_encoder_patterns):
            pre_encoder_names.append(n)
        elif any(p in n for p in post_encoder_patterns):
            post_encoder_names.append(n)
        else:
            raise ValueError(f"parameter name: {n} belong to neither pre or post encoder names")

    # only process left names in next iteration
    return left_names, encoder_names_grouped, pre_encoder_names, post_encoder_names


def reverse_layer_ids(
    encoder_name_to_id: dict,
    pre_enocder_name_to_id: dict,
    post_enocder_name_to_id: dict,
):
    """
    The layer ids need to increase when going from the output end to the input end.
    We need to reverse the ids which were originally assigned in a decreasing order.

    Parameters
    ----------
    encoder_name_to_id
        The layer-to-id mapping of encoder layers.
    pre_enocder_name_to_id
        The layer-to-id mapping of pre-encoder layers.
    post_enocder_name_to_id
        The layer-to-id mapping of post-encoder layers.

    Returns
    -------
    The layer-to-id mapping of all layers with layer ids reversed.
    """
    name_to_id = {**pre_enocder_name_to_id, **encoder_name_to_id, **post_enocder_name_to_id}
    if len(name_to_id) > 0:
        layer_num = max(name_to_id.values())
        # if no post encoder layers, the minimum layer id should be 1
        if len(post_enocder_name_to_id) == 0:
            layer_num += 1
    for n, layer_id in name_to_id.items():
        name_to_id[n] = layer_num - layer_id

    return name_to_id


def assign_layer_ids(
    names: List[str],
    pre_encoder_patterns: Tuple[str, ...],
    post_encoder_patterns: Tuple[str, ...],
    model_pre: Optional[str] = None,
):
    """
    Assign ids to all layers. It splits a model into three parts: pre-encoder, encoder, and post-encoder.
    Encoder is generally a stack of multiple similar layers, such as transformer layers. Since encoder is
    generally wrapped by nn.Sequential() or nn.ModuleList(), its inside layer names contain digits.
    It sets 0 as the ids of all post-encoder layers and a maximum id (layer_num) for the all the pre-encoder
    layers. The encoder layers have decreasing ids from the input to the output ends.

    Parameters
    ----------
    names
        model layer names.
    pre_encoder_patterns
        Patterns to identify a layer as a pre-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into pre-encoder layers.
    post_encoder_patterns
        Patterns to identify a layer as a post-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into post-encoder layers.
    model_pre
        The layer names' prefix. Only the layer names with this prefix will be assigned ids. The left
        layer names will be returned.

    Returns
    -------
    name_to_id
        A dictionary mapping the layer names (keys) to their ids (values).
    left_names
        The layer names not starting with the "model_pre".
    """
    try:
        left_names, encoder_names, pre_encoder_names, post_encoder_names = group_param_names(
            names=names,
            pre_encoder_patterns=pre_encoder_patterns,
            post_encoder_patterns=post_encoder_patterns,
            model_prefix=model_pre,
        )
        # add a constraint
        if len(encoder_names) == 0 and len(pre_encoder_names) != 0:
            raise ValueError(f"encoder_names is empty, but pre_encoder_names has values: {pre_encoder_names}")

        encoder_name_to_id, encoder_layer_num = assign_encoder_layer_ids(
            encoder_names=encoder_names,
        )

        pre_encoder_name_to_id = assign_non_encoder_layer_ids(non_encoder_names=pre_encoder_names, layer_id=0)

        post_encoder_name_to_id = assign_non_encoder_layer_ids(
            non_encoder_names=post_encoder_names, layer_id=encoder_layer_num + 1
        )

        name_to_id = reverse_layer_ids(
            encoder_name_to_id=encoder_name_to_id,
            pre_enocder_name_to_id=pre_encoder_name_to_id,
            post_enocder_name_to_id=post_encoder_name_to_id,
        )
    except Exception as e:
        logger.debug(
            f"When calling assign_layer_ids(), it catches exception: {e}. All the layers will use the same layer_id."
        )
        name_to_id = dict()
        left_names = names

    return name_to_id, left_names


def get_column_features(
    batch: Dict[str, torch.Tensor],
    column_name_prefix: str,
    features: torch.Tensor,
    valid_lengths: torch.Tensor,
    cls_feature: Optional[torch.Tensor] = None,
):
    """
    Index the features of one column defined by `column_name_prefix`.
    This function can be used to index both image and text features.
    The features have shape (b, n, d), where n can be the image number or
    text token number. One column corresponds to a subset of
    the n images or text tokens. One column name can only appear once in the return.

    Parameters
    ----------
    batch
        The batch input containing the feature column information, i.e., indexes.
    column_name_prefix
        The column name prefix of one modality (image or text).
    features
        The features of columns whose names starts with column_name_prefix.
    valid_lengths
        The valid image number or text token number of each sample in a batch.
    cls_feature
        The cls feature containing information from all feature columns.

    Returns
    -------
    The column features with masks. If the column has no valid features, its
    mask is 0.
    """
    column_features = {}
    feature_masks = {}

    cut_idx = len(column_name_prefix) + 1
    if cls_feature is not None:
        all_column_names = []
        # create a zero mask to do logical_or with each column's mask
        joint_mask = torch.zeros(features.shape[0]).to(features)  # (b,)
    for key in batch:
        if key.startswith(column_name_prefix):
            per_col_features = []
            per_col_masks = torch.zeros(features.shape[0]).to(features)  # (b,)
            assert batch[key].ndim == 2 and batch[key].shape[1] == 2
            for i, per_sample_col_idx in enumerate(batch[key]):
                start_idx = per_sample_col_idx[0]
                end_idx = per_sample_col_idx[1]
                if start_idx < end_idx:
                    assert end_idx <= valid_lengths[i]
                    per_col_features.append(features[i, start_idx:end_idx].mean(dim=0))
                    per_col_masks[i] = 1
                else:  # the column has no valid image/text.
                    per_col_features.append(torch.zeros_like(features[0, 0]))
                    per_col_masks[i] = 0
            column_name = key[cut_idx:]
            column_features[column_name] = torch.stack(per_col_features, dim=0)  # (b, num_features)
            feature_masks[column_name] = per_col_masks  # (b,)
            if cls_feature is not None:
                all_column_names.append(column_name)
                joint_mask = torch.logical_or(joint_mask, per_col_masks)

    # all the columns of one model's input share the model's cls feature
    if (
        cls_feature is not None and len(all_column_names) > 0
    ):  # some models', e.g, timm_image, output doesn't have the cls feature.
        # remove the individual column features since these column features not independent
        for column_name in all_column_names:
            column_features.pop(column_name)
            feature_masks.pop(column_name)

        joint_column_name = "_".join(all_column_names)
        column_features[joint_column_name] = cls_feature
        feature_masks[joint_column_name] = joint_mask.to(features)

    # print(f"column_features: {column_features}")
    return column_features, feature_masks


def create_adaptation(peft: str, layer: nn.Module, lora_r: int, lora_alpha: int, **kwargs):
    """
    Creates a model adaptation module (IA3, LoRA, IA3_LoRA) given a linear layer.

    Parameters
    ----------
    peft
        Name of the adaptation module.
    layer
       The layer the adaptation module should be applied to.
    lora_r
        The rank r of the low-rank decomposition.
    lora_alpha
        The scaling factor. Can be set to same value as r in
        most cases, as initialization is scaled already.
    filter
        Apply loRA only to linear layers filtered by name (e.g. "query.").
        If None, loRA is applied to all linear Layers in module.
    module_filter
        Apply loRA only to modules filtered by name (e.g. ".*EncDecAttention|.*DenseReluDense")
        If None, loRA is considered for all modules

    Returns
    -------
    Model with injected LoRA modules.
    """
    if "ia3_lora" in peft:
        return IA3LoRALinear(
            layer.in_features, layer.out_features, r=lora_r, lora_alpha=lora_alpha, merge_weights=False
        )
    elif "conv_lora" in peft:
        return ConvLoRALinear(
            layer.in_features,
            layer.out_features,
            r=lora_r,
            lora_alpha=lora_alpha,
            merge_weights=False,
            conv_lora_expert_num=kwargs["conv_lora_expert_num"],
        )
    elif "ia3" in peft:
        return IA3Linear(layer.in_features, layer.out_features, merge_weights=False)
    elif "lora" in peft:
        return LoRALinear(layer.in_features, layer.out_features, r=lora_r, lora_alpha=lora_alpha, merge_weights=False)
    elif peft is not None:
        raise NotImplementedError(
            f"The efficient finetuning strategy '{peft}'"
            f" is not supported. We only support"
            f" {', '.join(PEFT_ADDITIVE_STRATEGIES)}."
        )


def inject_adaptation_to_linear_layer(
    model: nn.Module,
    peft: str,
    lora_r: int = None,
    lora_alpha: int = None,
    filter: Optional[List[str]] = None,
    module_filter: Optional[List[str]] = None,
    extra_trainable_params: Optional[List[str]] = None,
    **kwargs,
) -> nn.Module:
    """
    Injects trainable adatio Low-Rank decomposition matrices (LoRA) into linear
    layers of a PyTorch model. Used for efficient fine-tuning of large
    pre-trained models.

    Parameters
    ----------
    model
        A PyTorch model.
    peft
        Efficient finetuning method that should be applied.
    lora_r
        The rank r of the low-rank decomposition.
    lora_alpha
        The scaling factor. Can be set to same value as r in
        most cases, as initialization is scaled already.
    filter
        Apply loRA only to linear layers filtered by name (e.g. "query.").
        If None, loRA is applied to all linear Layers in module.
    module_filter
        Apply loRA only to modules filtered by name (e.g. ".*EncDecAttention|.*DenseReluDense")
        If None, loRA is considered for all modules
    extra_trainable_params
        Not to apply loRA to modules filtered by name, and these modules are not frozen during training (e.g. "mask_decoder").
        If None, all the modules except for those applied loRA are frozen.
    Returns
    -------
    Model with injected LoRA modules.
    """
    for m_name, module in dict(model.named_modules()).items():
        if extra_trainable_params and any(re.match(filter_layer, m_name) for filter_layer in extra_trainable_params):
            continue
        if hasattr(model, "frozen_layers") and any(
            re.search(filter_layer, m_name) for filter_layer in model.frozen_layers
        ):
            continue
        if not module_filter or any(re.match(filter_module, m_name) for filter_module in module_filter):
            for c_name, layer in dict(module.named_children()).items():
                if not filter or any(re.match(filter_layer, c_name) for filter_layer in filter):
                    assert isinstance(
                        layer, nn.Linear
                    ), f"LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    adaptation_layer = create_adaptation(peft, layer, lora_r, lora_alpha, **kwargs)
                    adaptation_layer.weight = layer.weight
                    adaptation_layer.bias = layer.bias
                    setattr(module, c_name, adaptation_layer)

    return model  # return model to enable method chaining


def inject_adaptation_to_conv2d_layer(
    model: nn.Module,
    peft: str,
    lora_r: int = None,
    lora_alpha: int = None,
    filter: Optional[List[str]] = None,
    module_filter: Optional[List[str]] = None,
    extra_trainable_params: Optional[List[str]] = None,
    **kwargs,
) -> nn.Module:
    """
    将 LoRA/Conv-LoRA 注入到 Conv2d（用于 nnU-Net 等卷积网络）。

    约定：
    - 仅对 `torch.nn.Conv2d` 生效
    - 目前 `peft` 仅支持 `conv_lora`（其它策略未来可扩展）
    - 注入后原卷积权重被冻结，新增参数以 `lora_*` 命名，便于 `get_peft_param_names` 选择可训练参数
    """
    if peft is None:
        return model
    if "conv_lora" not in peft:
        # 当前只扩展 conv_lora 到 Conv2d，其它 PEFT 保持原逻辑（linear-only）
        return model

    for m_name, module in dict(model.named_modules()).items():
        if extra_trainable_params and any(re.match(filter_layer, m_name) for filter_layer in extra_trainable_params):
            continue
        if hasattr(model, "frozen_layers") and any(re.search(filter_layer, m_name) for filter_layer in model.frozen_layers):
            continue
        if not module_filter or any(re.match(filter_module, m_name) for filter_module in module_filter):
            for c_name, layer in dict(module.named_children()).items():
                if not filter or any(re.match(filter_layer, c_name) for filter_layer in filter):
                    if not isinstance(layer, nn.Conv2d):
                        continue
                    adaptation_layer = ConvLoRAConv2d(
                        in_channels=layer.in_channels,
                        out_channels=layer.out_channels,
                        kernel_size=layer.kernel_size,
                        stride=layer.stride,
                        padding=layer.padding,
                        dilation=layer.dilation,
                        groups=layer.groups,
                        bias=layer.bias is not None,
                        padding_mode=layer.padding_mode,
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        merge_weights=False,
                        conv_lora_expert_num=kwargs["conv_lora_expert_num"],
                    )
                    # 复用原参数对象（weight/bias），确保 state_dict 兼容且冻结生效
                    adaptation_layer.weight = layer.weight
                    adaptation_layer.bias = layer.bias
                    setattr(module, c_name, adaptation_layer)

    return model


def get_model_head(model: nn.Module):
    """
    Return the model's head. Different models may have different head names.

    Parameters
    ----------
    model
        A Pytorch model.

    Returns
    -------
    The model's head.
    """
    if hasattr(model, "head"):
        head = model.head  # move the head outside
    elif hasattr(model, "last_linear"):
        head = model.last_linear
    elif hasattr(model, "fc"):
        head = model.fc
    elif hasattr(model, "classifier"):
        head = model.classifier
    else:
        raise ValueError(f"Model {type(model)} doesn't have head. Need to check its implementation.")

    return head.fc if hasattr(head, "fc") else head


def get_hf_config_and_model(
    checkpoint_name: str, pretrained: Optional[bool] = True, low_cpu_mem_usage: Optional[bool] = False
):
    """
    Get a Huggingface config and model based on a checkpoint name.

    Parameters
    ----------
    checkpoint_name
        A model checkpoint name or a local path that saves a custom checkpoint.
    pretrained
         Whether using the pretrained weights. If pretrained=True, download the pretrained model.
    low_cpu_mem_usage
        Whether to turn on the optimization of reducing the peak CPU memory usage when loading the pretrained model.

    Returns
    -------
    A Huggingface config and model.
    """
    config = AutoConfig.from_pretrained(checkpoint_name)    # nosec B615

    if pretrained:
        model = AutoModel.from_pretrained(checkpoint_name, low_cpu_mem_usage=low_cpu_mem_usage) # nosec B615
    else:
        model = AutoModel.from_config(config)
    # Explicitly set the model to train mode after loading as by default it is in eval mode
    # See issue #4965
    model.train()
    return config, model


def apply_sigmoid(output: Dict):
    """
    Apply the sigmoid to logits.

    Parameters
    ----------
    output
        The model output dict.

    Returns
    -------
    The output with logits transformed by sigmoid.
    """
    for k, v in output.items():
        output[k][LOGITS] = torch.sigmoid(v[LOGITS].float())
    return output


def apply_multi_class_semantic_seg_postprocess(output: Dict):
    """
    Apply the semantic postprocessing to logits.

    Parameters
    ----------
    output
        The model output dict.

    Returns
    -------
    The output with post-proceesed semantic masks.
    """

    def semantic_inference(mask_cls, mask_pred):
        """
        Post-processing mask prediction for multi-class semantic segmentation inference based on https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py

        Args:
            mask_cls (`torch.Tensor`):
                Class logits. A tensor of shape `(num_queries, num_classes + 1)` (include the "no object" category).

            mask_pred (`torch.Tensor`):
                Mask logits. A tensor of shape `(num_queries, height, width)`.

        Returns:
            semseg (`torch.Tensor`): The processed mask prediction. A tensor of shape `(num_classes, height, width)`.

        References:
        [1] https://arxiv.org/abs/2107.06278
        [2] https://arxiv.org/abs/2112.01527
        """
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    for k, v in output.items():
        pred_classes = output[k][CLASS_LOGITS]
        pred_masks = output[k][LOGITS]
        semantic_masks = []
        for mask_cls_result, mask_pred_result in zip(
            pred_classes, pred_masks
        ):  # bs, num_q, num_class and bs, num_q, h, w
            per_sample_semantic_masks = semantic_inference(mask_cls_result, mask_pred_result)  # num_class, h, w
            semantic_masks.append(per_sample_semantic_masks)
        semantic_masks = torch.stack(semantic_masks, dim=0)
        output[k][SEMANTIC_MASK] = semantic_masks
    return output


def get_model_postprocess_fn(problem_type: str, loss_func: _Loss):
    """
    Get the postprocessing function for the model outputs.

    Parameters
    ----------
    problem_type
        The problem type, e.g., classification or regression.
    loss_func
        The loss function used in training.

    Returns
    -------
    The postprocessing function.
    """
    postprocess_func = None
    if problem_type == REGRESSION:
        if isinstance(loss_func, nn.BCEWithLogitsLoss):
            postprocess_func = apply_sigmoid
    elif problem_type == SEMANTIC_SEGMENTATION:
        if isinstance(loss_func, Mask2FormerLoss):
            postprocess_func = apply_multi_class_semantic_seg_postprocess
        else:
            postprocess_func = apply_sigmoid

    return postprocess_func


def get_mmocr_config_and_model(checkpoint_name: str):
    """
    Get an MMOCR config and model based on a checkpoint name.

    Parameters
    ----------
    checkpoint_name
        A model checkpoint name.

    Returns
    -------
    An MMOCR config and model.
    """
    from ..utils import check_if_packages_installed

    check_if_packages_installed(package_names=["mmcv"])
    try:
        import mmocr
        from mmocr.models import build_detector
    except ImportError:
        mmocr = None
    from mim.commands.download import download

    checkpoints = download(package="mmocr", configs=[checkpoint_name], dest_root=".")

    # read config files
    check_if_packages_installed(problem_type=OCR)
    config_file = checkpoint_name + ".py"
    if isinstance(config_file, str):
        config = mmcv.Config.fromfile(config_file)

    # build model and load pretrained weights
    checkpoint = checkpoints[0]
    model = build_detector(config.model, test_cfg=config.get("test_cfg"))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")
    return config, model


def lookup_mmdet_config(key, config):
    if key in config:
        return config[key]
    for subconfig in config.values():
        if isinstance(subconfig, dict):
            result = lookup_mmdet_config(key, subconfig)
            if result is not None:
                return result
        elif isinstance(subconfig, list):
            for subsubconfig in subconfig:
                if isinstance(subsubconfig, dict):
                    result = lookup_mmdet_config(key, subsubconfig)
                    if result is not None:
                        return result
    return None


def update_mmdet_config(key, value, config):
    for k, subconfig in config.items():
        if key == k:
            config[k] = value
        elif isinstance(subconfig, dict):
            update_mmdet_config(key, value, subconfig)
        elif isinstance(subconfig, list):
            for subsubconfig in subconfig:
                if isinstance(subsubconfig, dict):
                    update_mmdet_config(key, value, subsubconfig)


def run_model(model: nn.Module, batch: dict, trt_model: Optional[nn.Module] = None):
    from ..utils.onnx import OnnxModule
    from .document_transformer import DocumentTransformer
    from .fusion.fusion_mlp import MultimodalFusionMLP
    from .hf_text import HFAutoModelForTextPrediction
    from .t_few import TFewModel
    from .timm_image import TimmAutoModelForImagePrediction

    supported_models = (
        TimmAutoModelForImagePrediction,
        HFAutoModelForTextPrediction,
        MultimodalFusionMLP,
        TFewModel,
        OnnxModule,
    )
    pure_model = model
    if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
        pure_model = model._orig_mod
    if isinstance(model, nn.DataParallel):
        pure_model = model.module
    if isinstance(pure_model, OnnxModule):
        for k in batch:
            # HACK input data types in ONNX
            if batch[k].dtype == torch.int32:
                batch[k] = batch[k].to(torch.int64)
    # DocumentTransformer inherited from HFAutoModelForTextPrediction
    if (not isinstance(pure_model, DocumentTransformer)) and isinstance(pure_model, supported_models):
        input_vec = [batch[k] for k in pure_model.input_keys]
        column_names, column_values = [], []
        for k in batch.keys():
            has_image_column_prefix = isinstance(pure_model, TimmAutoModelForImagePrediction) and k.startswith(
                pure_model.image_column_prefix
            )
            has_text_column_prefix = isinstance(pure_model, HFAutoModelForTextPrediction) and k.startswith(
                pure_model.text_column_prefix
            )
            if has_image_column_prefix or has_text_column_prefix:
                column_names.append(k)
                column_values.append(batch[k])
        if column_names != [] and column_values != []:
            input_vec.append(column_names)
            input_vec.append(column_values)
        if isinstance(pure_model, OnnxModule):
            # OnnxModule doesn't support multi-gpu yet
            output_vec = pure_model(*tuple(input_vec))
        else:
            output_vec = model(*tuple(input_vec))

        output = pure_model.get_output_dict(*output_vec)
    else:
        output = model(batch)
    return output


def freeze_model_layers(model, frozen_layers):
    """
    Freeze model layers with pattern in frozen_layers.

    Parameters
    ----------
    model
        The pytorch model.
    frozen_layers
        A list of substrings of frozen layers' names.

        e.g. if frozen_layers = ["backbone", "neck"],
            all layers including "backbone" or "neck" in the name will be frozen.
    """

    if not frozen_layers:
        return

    is_frozen_layer = lambda n: any(bb in n for bb in frozen_layers)

    for n, p in model.named_parameters():
        if is_frozen_layer(n):
            p.requires_grad = False


def get_pretrained_tokenizer(
    tokenizer_name: str,
    checkpoint_name: str,
    use_fast: Optional[bool] = True,
    add_prefix_space: Optional[bool] = None,
):
    """
    Load the tokenizer for a pre-trained huggingface checkpoint.

    Parameters
    ----------
    tokenizer_name
        The tokenizer type, e.g., "bert", "clip", "electra", and "hf_auto".
    checkpoint_name
        Name of a pre-trained checkpoint.
    use_fast
        Use a fast Rust-based tokenizer if it is supported for a given model.
        If a fast tokenizer is not available for a given model, a normal Python-based tokenizer is returned instead.
        See: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer.from_pretrained.use_fast

    Returns
    -------
    A tokenizer instance.
    """
    try:
        tokenizer_class = ALL_TOKENIZERS[tokenizer_name]
        if add_prefix_space is None:
            return tokenizer_class.from_pretrained(checkpoint_name, use_fast=use_fast)
        else:
            return tokenizer_class.from_pretrained(
                checkpoint_name, use_fast=use_fast, add_prefix_space=add_prefix_space
            )
    except TypeError as e:
        try:
            tokenizer = BertTokenizer.from_pretrained(checkpoint_name)  # nosec B615
            warnings.warn(
                f"Current checkpoint {checkpoint_name} does not support AutoTokenizer. "
                "Switch to BertTokenizer instead.",
                UserWarning,
            )
            return tokenizer
        except:
            raise e


def extract_value_from_config(
    config: Dict,
    keys: Tuple[str, ...],
):
    """
    Traverse a config dictionary to get some hyper-parameter's value.

    Parameters
    ----------
    config
        A config dictionary.
    keys
        The possible names of a hyper-parameter.

    Returns
    -------
    The hyper-parameter value.
    """
    result = []
    for k, v in config.items():
        if k in keys:
            result.append(v)
        elif isinstance(v, dict):
            result += extract_value_from_config(v, keys)
        else:
            pass

    return result


def extract_image_hparams_from_config(model_name: str, config):
    """
    Extract some default hyper-parameters, e.g., image size, mean, and std,
    from a pre-trained (timm or huggingface) checkpoint.

    Parameters
    ----------
    model_name
        Name of model.
    config
        Config of a pre-trained checkpoint.

    Returns
    -------
    image_size
        Image width/height.
    mean
        Image normalization mean.
    std
        Image normalizaiton std.
    """
    if model_name.lower().startswith((TIMM_IMAGE, META_TRANSFORMER)):
        image_size = config["input_size"][-1]
        image_mean = config["mean"]
        image_std = config["std"]
    elif model_name.lower().startswith((CLIP, DOCUMENT_TRANSFORMER)):
        extracted = extract_value_from_config(
            config=config.to_diff_dict(),
            keys=("image_size",),
        )
        if len(extracted) == 0:
            image_size = None
        elif len(extracted) >= 1:
            image_size = extracted[0]
            if isinstance(image_size, tuple):
                image_size = image_size[-1]
        else:
            raise ValueError(f" more than one image_size values are detected: {extracted}")
        image_mean = None
        image_std = None
    else:
        raise ValueError(f"Unknown image processor prefix: {model_name}")
    return image_size, image_mean, image_std


def image_mean_std(norm_type: str):
    """
    Get image normalization mean and std by its name.

    Parameters
    ----------
    norm_type
        Name of image normalization.

    Returns
    -------
    Normalization mean and std.
    """
    if norm_type == "inception":
        return IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    elif norm_type == "imagenet":
        return IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    elif norm_type == "clip":
        return CLIP_IMAGE_MEAN, CLIP_IMAGE_STD
    else:
        raise ValueError(f"unknown image normalization: {norm_type}")


def get_image_size_mean_std(
    model_name: str,
    config,
    provided_size: int,
    provided_norm_type: str,
    support_variable_input_size: Optional[bool] = False,
):
    image_size, image_mean, image_std = extract_image_hparams_from_config(
        model_name=model_name,
        config=config,
    )
    if support_variable_input_size and provided_size is not None:
        # We have detected that the model supports using an image size that is
        # different from the pretrained model, e.g., ConvNets with global pooling
        if provided_size < image_size:
            logger.warning(
                f"The provided image size={provided_size} is smaller than the default size "
                f"of the pretrained backbone, which is {image_size}. "
                f"Detailed configuration of the backbone is in {config}. "
                f"You may like to double check your configuration."
            )
        image_size = provided_size
    elif provided_size is not None and provided_size != image_size:
        logger.warning(
            f"The model does not support using an image size that is different from the default size. "
            f"Provided image size={provided_size}. Default size={image_size}. "
            f"Detailed model configuration={config}. We have ignored the provided image size."
        )

    if image_size is None:
        if provided_size is not None:
            image_size = provided_size
            logger.debug(f"using provided image size: {image_size}.")
        else:
            raise ValueError("image size is missing.")
    else:
        logger.debug(f"using detected image size: {image_size}")

    if image_mean is None or image_std is None:
        if provided_norm_type is not None:
            image_mean, image_std = image_mean_std(provided_norm_type)
            logger.debug(f"using provided normalization: {provided_norm_type}.")
        else:
            raise ValueError("image normalization mean and std are missing.")
    else:
        logger.debug(f"using detected image normalization: {image_mean} and {image_std}.")

    return image_size, image_mean, image_std


def get_text_segment_num(config, provided_segment_num: int, checkpoint_name: str):
    extracted = extract_value_from_config(config=config.to_diff_dict(), keys=("type_vocab_size",))
    if len(extracted) == 0:
        default_segment_num = 1
    elif len(extracted) == 1:
        default_segment_num = extracted[0]
    else:
        raise ValueError(f" more than one type_vocab_size values are detected: {extracted}")

    if default_segment_num <= 0:
        default_segment_num = 1

    if provided_segment_num < default_segment_num:
        warnings.warn(
            f"provided text_segment_num: {provided_segment_num} "
            f"is smaller than {checkpoint_name}'s default: {default_segment_num}"
        )
    text_segment_num = min(provided_segment_num, default_segment_num)
    assert text_segment_num >= 1
    logger.debug(f"text segment num: {text_segment_num}")

    return text_segment_num


def get_text_token_max_len(provided_max_len, config, tokenizer, checkpoint_name):
    """
    Compute the allowable max length of token sequences.

    Parameters
    ----------
    provided_max_len
        The provided max length.
    config
        Model config.
    tokenizer
        Text tokenizer.
    checkpoint_name
        Name of checkpoint.

    Returns
    -------
    Token sequence max length.
    """
    if hasattr(config, "relative_attention") and config.relative_attention:
        default_max_len = tokenizer.model_max_length
    elif hasattr(config, "position_embedding_type") and "relative" in config.position_embedding_type:
        default_max_len = tokenizer.model_max_length
    elif hasattr(config, "max_position_embeddings"):
        default_max_len = config.max_position_embeddings
    else:
        default_max_len = tokenizer.model_max_length

    if provided_max_len is None or provided_max_len <= 0:
        max_len = default_max_len
    else:
        if provided_max_len < default_max_len:
            if default_max_len < 10**6:  # Larger than this value usually means infinite.
                warnings.warn(
                    f"provided max length: {provided_max_len} "
                    f"is smaller than {checkpoint_name}'s default: {default_max_len}"
                )
        max_len = min(provided_max_len, default_max_len)

    logger.debug(f"text max length: {max_len}")

    return max_len


def replace_missing_images_with_learnable(
    images: torch.Tensor,
    image_masks,
    learnable_image: nn.Parameter,
):
    b, n, c, h, w = images.shape
    assert learnable_image.shape == (c, h, w)
    for i in range(b):
        for j in range(n):
            if not image_masks[i][j]:  # False means a missing image
                images[i][j] = learnable_image

    return images


def select_model(
    config: DictConfig,
    df_preprocessor,
    strict: Optional[bool] = True,
):
    """
    Filter model config through the detected modalities in the training data.
    If MultiModalFeaturePreprocessor can't detect some modality,
    this function will remove the models that use this modality. This function is to
    maximize the user flexibility in defining the config.
    For example, if one uses the default, including hf_text and timm_image, as the model config template
    but the training data don't have images, this function will filter out timm_image.

    Parameters
    ----------
    config
        A DictConfig object. The model config should be accessible by "config.model"
    df_preprocessor
        A MultiModalFeaturePreprocessor object, which has called .fit() on the training data.
        Column names of the same modality are grouped into one list. If a modality's list is empty,
        it means the training data don't have this modality.
    strict
        If False, allow retaining one model when partial modalities are available for that model.

    Returns
    -------
    Config with some unused models removed.
    """
    data_status = {}
    for per_modality in ALL_MODALITIES:
        data_status[per_modality] = False
    if len(df_preprocessor.image_feature_names) > 0:
        data_status[IMAGE] = True
    if len(df_preprocessor.text_feature_names) > 0:
        data_status[TEXT] = True
    if len(df_preprocessor.categorical_feature_names) > 0:
        data_status[CATEGORICAL] = True
    if len(df_preprocessor.numerical_feature_names) > 0:
        data_status[NUMERICAL] = True
    if len(df_preprocessor.ner_feature_names) > 0:
        data_status[TEXT_NER] = True
    if len(df_preprocessor.document_feature_names) > 0:
        data_status[DOCUMENT] = True
    if len(df_preprocessor.semantic_segmentation_feature_names) > 0:
        data_status[SEMANTIC_SEGMENTATION_IMG] = True

    names = config.model.names
    if isinstance(names, str):
        names = [names]
    selected_model_names = []
    fusion_model_name = []
    for model_name in names:
        model_config = getattr(config.model, model_name)
        strict = getattr(model_config, "requires_all_dtypes", strict)
        if not model_config.data_types:
            fusion_model_name.append(model_name)
            continue
        model_data_status = [data_status[d_type] for d_type in model_config.data_types]
        if all(model_data_status):
            selected_model_names.append(model_name)
        else:
            if any(model_data_status) and not strict:
                selected_model_names.append(model_name)
                # update data types to be consistent with detected
                model_config.data_types = [d_type for d_type in model_config.data_types if data_status[d_type]]
            else:
                delattr(config.model, model_name)

    if len(selected_model_names) == 0:
        raise ValueError("No model is available for this dataset.")
    # only allow no more than 1 fusion model
    if len(fusion_model_name) > 1:
        raise ValueError(f"More than one fusion models `{fusion_model_name}` are detected, but only one is allowed.")

    if len(selected_model_names) > 1:
        assert len(fusion_model_name) == 1
        selected_model_names.extend(fusion_model_name)
    elif len(fusion_model_name) == 1 and hasattr(config.model, fusion_model_name[0]):
        delattr(config.model, fusion_model_name[0])

    config.model.names = selected_model_names
    logger.debug(f"selected models: {selected_model_names}")
    for model_name in selected_model_names:
        logger.debug(f"model dtypes: {getattr(config.model, model_name).data_types}")

    # clean up unused model configs
    model_keys = list(config.model.keys())
    for model_name in model_keys:
        if model_name not in selected_model_names + ["names"]:
            delattr(config.model, model_name)

    return config


def create_model(
    model_name: str,
    model_config: DictConfig,
    num_classes: Optional[int] = 0,
    classes: Optional[list] = None,
    num_numerical_columns: Optional[int] = None,
    num_categories: Optional[Dict] = None,
    numerical_fill_values: Optional[Dict] = None,
    pretrained: Optional[bool] = True,
    is_matching: Optional[bool] = False,
):
    """
    Create a single model.

    Parameters
    ----------
    model_name
        Name of the model.
    model_config
        Config of the model.
    num_classes
        The class number for a classification task. It should be 1 for a regression task.
    classes
        All classes in this dataset.
    num_numerical_columns
        The number of numerical columns in the training dataframe.
    num_categories
        The category number for each categorical column in the training dataframe.
    numerical_fill_values
        If numerical values are null, fill them with these.
    pretrained
        Whether using the pretrained timm models. If pretrained=True, download the pretrained model.
    is_matching
        Whether the model is used for semantic matching.

    Returns
    -------
    A model.
    """
    if model_name.lower().startswith(CLIP):
        from .clip import CLIPForImageText

        model = CLIPForImageText(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
            has_image=IMAGE in model_config.data_types,
            has_text=TEXT in model_config.data_types,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
            image_chan_num=model_config.image_chan_num,
            use_learnable_image=model_config.use_learnable_image,
            max_text_len=model_config.max_text_len,
            text_segment_num=model_config.text_segment_num,
            is_matching=is_matching,
        )
    elif model_name.lower().startswith(TIMM_IMAGE):
        from .timm_image import TimmAutoModelForImagePrediction

        model = TimmAutoModelForImagePrediction(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            mix_choice=model_config.mix_choice,
            pretrained=pretrained,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
            image_chan_num=model_config.image_chan_num,
            use_learnable_image=model_config.use_learnable_image,
        )
    elif model_name.lower().startswith(HF_TEXT):
        from .hf_text import HFAutoModelForTextPrediction

        model = HFAutoModelForTextPrediction(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pooling_mode=model_config.pooling_mode,
            gradient_checkpointing=model_config.gradient_checkpointing,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
            max_text_len=model_config.max_text_len,
            text_segment_num=model_config.text_segment_num,
            use_fast=model_config.use_fast,
        )
    elif model_name.lower().startswith(T_FEW):
        from .t_few import TFewModel

        model = TFewModel(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            length_norm=model_config.length_norm,  # Normalizes length to adjust for length bias in target template
            unlikely_loss=model_config.unlikely_loss,  # Adds loss term that lowers probability of incorrect outputs
            mc_loss=model_config.mc_loss,  # Adds multiple choice cross entropy loss
            num_classes=num_classes,
            gradient_checkpointing=model_config.gradient_checkpointing,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
            max_text_len=model_config.max_text_len,
            text_segment_num=model_config.text_segment_num,
        )
    elif model_name.lower().startswith(NUMERICAL_MLP):
        from .numerical_mlp import NumericalMLP

        model = NumericalMLP(
            prefix=model_name,
            in_features=num_numerical_columns,
            hidden_features=model_config.hidden_size,
            out_features=model_config.hidden_size,
            num_layers=model_config.num_layers,
            activation=model_config.activation,
            dropout=model_config.dropout,
            normalization=model_config.normalization,
            token_dim=model_config.token_dim,
            embedding_arch=model_config.embedding_arch,
            num_classes=num_classes,
            numerical_fill_values=numerical_fill_values,
        )
    elif model_name.lower().startswith(CATEGORICAL_MLP):
        from .categorical_mlp import CategoricalMLP

        model = CategoricalMLP(
            prefix=model_name,
            num_categories=num_categories,
            out_features=model_config.hidden_size,
            num_layers=model_config.num_layers,
            activation=model_config.activation,
            dropout=model_config.dropout,
            normalization=model_config.normalization,
            num_classes=num_classes,
        )
    elif model_name.lower().startswith(DOCUMENT_TRANSFORMER):
        from .document_transformer import DocumentTransformer

        model = DocumentTransformer(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pooling_mode=model_config.pooling_mode,
            gradient_checkpointing=model_config.gradient_checkpointing,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
        )
    elif model_name.lower().startswith(MMDET_IMAGE):
        from .mmdet_image import MMDetAutoModelForObjectDetection

        model = MMDetAutoModelForObjectDetection(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            config_file=model_config.config_file,
            classes=classes,
            pretrained=pretrained,
            output_bbox_format=model_config.output_bbox_format,
            frozen_layers=model_config.frozen_layers,
        )
    elif model_name.lower().startswith(MMOCR_TEXT_DET):
        from .mmocr_text_detection import MMOCRAutoModelForTextDetection

        model = MMOCRAutoModelForTextDetection(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
        )
    elif model_name.lower().startswith(MMOCR_TEXT_RECOG):
        from .mmocr_text_recognition import MMOCRAutoModelForTextRecognition

        model = MMOCRAutoModelForTextRecognition(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
        )
    elif model_name.lower().startswith(NER_TEXT):
        from .ner_text import HFAutoModelForNER

        model = HFAutoModelForNER(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            gradient_checkpointing=model_config.gradient_checkpointing,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
        )
    elif model_name.lower().startswith(FUSION_MLP):
        from .fusion import MultimodalFusionMLP

        model = functools.partial(
            MultimodalFusionMLP,
            prefix=model_name,
            hidden_features=model_config.hidden_sizes,
            num_classes=num_classes,
            adapt_in_features=model_config.adapt_in_features,
            activation=model_config.activation,
            dropout=model_config.dropout,
            normalization=model_config.normalization,
            aux_loss_weight=model_config.aux_loss_weight,
        )
    elif model_name.lower().startswith(FUSION_NER):
        from .fusion import MultimodalFusionNER

        model = functools.partial(
            MultimodalFusionNER,
            prefix=model_name,
            hidden_features=model_config.hidden_sizes,
            num_classes=num_classes,
            adapt_in_features=model_config.adapt_in_features,
            activation=model_config.activation,
            dropout_prob=model_config.drop_rate,
            normalization=model_config.normalization,
            loss_weight=model_config.weight if hasattr(model_config, "weight") else None,
        )
    elif model_name.lower().startswith(FUSION_TRANSFORMER):
        from .fusion import MultimodalFusionTransformer

        model = functools.partial(
            MultimodalFusionTransformer,
            prefix=model_name,
            hidden_features=model_config.hidden_size,
            num_classes=num_classes,
            num_blocks=model_config.num_blocks,
            attention_num_heads=model_config.attention_num_heads,
            ffn_hidden_size=model_config.ffn_hidden_size,
            attention_dropout=model_config.attention_dropout,
            residual_dropout=model_config.residual_dropout,
            ffn_dropout=model_config.ffn_dropout,
            attention_normalization=model_config.normalization,
            ffn_normalization=model_config.normalization,
            head_normalization=model_config.normalization,
            ffn_activation=model_config.ffn_activation,
            head_activation=model_config.head_activation,
            adapt_in_features=model_config.adapt_in_features,
            aux_loss_weight=model_config.aux_loss_weight,
            additive_attention=model_config.additive_attention,
            share_qv_weights=model_config.share_qv_weights,
        )
    elif model_name.lower().startswith(FT_TRANSFORMER):
        from .ft_transformer import FT_Transformer

        model = FT_Transformer(
            prefix=model_name,
            num_numerical_columns=num_numerical_columns,
            num_categories=num_categories,
            numerical_fill_values=numerical_fill_values,
            embedding_arch=model_config.embedding_arch,
            token_dim=model_config.token_dim,
            hidden_size=model_config.hidden_size,
            hidden_features=model_config.hidden_size,
            num_classes=num_classes,
            num_blocks=model_config.num_blocks,
            attention_num_heads=model_config.attention_num_heads,
            attention_dropout=model_config.attention_dropout,
            attention_normalization=model_config.normalization,
            ffn_hidden_size=model_config.ffn_hidden_size,
            ffn_dropout=model_config.ffn_dropout,
            ffn_normalization=model_config.normalization,
            ffn_activation=model_config.ffn_activation,
            residual_dropout=model_config.residual_dropout,
            head_normalization=model_config.normalization,
            head_activation=model_config.head_activation,
            additive_attention=model_config.additive_attention,
            share_qv_weights=model_config.share_qv_weights,
            pooling_mode=model_config.pooling_mode,
            checkpoint_name=model_config.checkpoint_name,
            pretrained=pretrained,
        )
    elif model_name.lower().startswith(SAM):
        from .sam import SAMForSemanticSegmentation

        model = SAMForSemanticSegmentation(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pretrained=pretrained,
            frozen_layers=model_config.frozen_layers,
            num_mask_tokens=model_config.num_mask_tokens,
            image_norm=model_config.image_norm,
        )
    elif model_name.lower().startswith(META_TRANSFORMER):
        from .meta_transformer import MetaTransformer

        model = MetaTransformer(
            prefix=model_name,
            checkpoint_path=model_config.checkpoint_path,
            num_classes=num_classes,
            model_version=model_config.model_version,
            has_image=IMAGE in model_config.data_types,
            has_text=TEXT in model_config.data_types,
            num_numerical_columns=num_numerical_columns,
            num_categories=num_categories,
            numerical_fill_values=numerical_fill_values,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
            image_chan_num=model_config.image_chan_num,
            use_learnable_image=model_config.use_learnable_image,
            max_text_len=model_config.max_text_len,
            text_segment_num=model_config.text_segment_num,
        )
    else:
        raise ValueError(f"unknown model name: {model_name}")

    return model


def create_fusion_model(
    config: DictConfig,
    num_classes: Optional[int] = None,
    classes: Optional[list] = None,
    num_numerical_columns: Optional[int] = None,
    num_categories: Optional[Dict] = None,
    numerical_fill_values: Optional[Dict] = None,
    pretrained: Optional[bool] = True,
):
    """
    Create models. It supports the auto models of huggingface text and timm image.
    Multimodal models, e.g., CLIP, should be added case-by-case since their configs and usages
    may be different. It uses MLP for the numerical features, categorical features, and late-fusion.

    Parameters
    ----------
    config
        A DictConfig object. The model config should be accessible by "config.model".
    num_classes
        The class number for a classification task. It should be 1 for a regression task.
    classes
        All classes in this dataset.
    num_numerical_columns
        The number of numerical columns in the training dataframe.
    num_categories
        The category number for each categorical column in the training dataframe.
    numerical_fill_values
        If numerical values are null, fill them with these.
    pretrained
        Whether using the pretrained timm models. If pretrained=True, download the pretrained model.

    Returns
    -------
    A Pytorch model.
    """
    names = config.model.names
    if isinstance(names, str):
        names = [names]
    # make sure no duplicate model names
    assert len(names) == len(set(names))
    logger.debug(f"output_shape: {num_classes}")
    names = sorted(names)
    config.model.names = names
    single_models = []
    fusion_model = None

    for model_name in names:
        model_config = getattr(config.model, model_name)
        model = create_model(
            model_name=model_name,
            model_config=model_config,
            num_classes=num_classes,
            classes=classes,
            num_numerical_columns=num_numerical_columns,
            num_categories=num_categories,
            numerical_fill_values=numerical_fill_values,
            pretrained=pretrained,
        )

        if isinstance(model, functools.partial):  # fusion model
            if fusion_model is None:
                fusion_model = model
            else:
                raise ValueError(
                    f"More than one fusion models are detected in {names}. Only one fusion model is allowed."
                )
        else:  # single model
            if config.optim.peft is not None:
                model = apply_peft_adaptation(model, config)
            single_models.append(model)

    if len(single_models) > 1:
        # must have one fusion model if there are multiple independent models
        model = fusion_model(models=single_models)
    elif len(single_models) == 1:
        model = single_models[0]
    else:
        raise ValueError(f"No available models for {names}")

    # build augmenter for multimodal data augmentation
    if config.optim.lemda.turn_on:
        from .fusion import MultimodalFusionMLP

        assert isinstance(model, MultimodalFusionMLP)
        from .augmenter import Augmenter

        augmenter = Augmenter(
            arch_type=config.optim.lemda.arch_type,
            input_dim=model.augmenter_in_features,
            z_dim=config.optim.lemda.z_dim,
            num_layers=config.optim.lemda.num_layers,
            adv_weight=config.optim.lemda.adv_weight,
        )
        model.augmenter = augmenter

    return model


def apply_peft_adaptation(model: nn.Module, config: DictConfig) -> nn.Module:
    """
    Apply an adaptation to the model for efficient fine-tuning.

    Parameters
    ----------
    model
        A PyTorch model.
    config:
        A DictConfig object. The optimization config should be accessible by "config.optimization".
    """
    if config.optim.peft in PEFT_ADDITIVE_STRATEGIES:
        # 先注入 Linear（现有逻辑，SAM/Transformer 用）
        model = inject_adaptation_to_linear_layer(
            model=model,
            peft=config.optim.peft,
            lora_r=config.optim.lora.r,
            lora_alpha=config.optim.lora.alpha,
            module_filter=config.optim.lora.module_filter,
            filter=config.optim.lora.filter,
            extra_trainable_params=config.optim.extra_trainable_params,
            conv_lora_expert_num=config.optim.lora.conv_lora_expert_num,
        )
        # 再注入 Conv2d（新增逻辑，nnU-Net 等卷积网络用）
        model = inject_adaptation_to_conv2d_layer(
            model=model,
            peft=config.optim.peft,
            lora_r=config.optim.lora.r,
            lora_alpha=config.optim.lora.alpha,
            module_filter=config.optim.lora.module_filter,
            filter=config.optim.lora.filter,
            extra_trainable_params=config.optim.extra_trainable_params,
            conv_lora_expert_num=config.optim.lora.conv_lora_expert_num,
        )
        model.name_to_id = model.get_layer_ids()  # Need to update name to id dictionary.

    return model


def modify_duplicate_model_names(
    learner,
    postfix: str,
    blacklist: List[str],
):
    """
    Modify a learner's model names if they exist in a blacklist.

    Parameters
    ----------
    learner
        A BaseLearner object.
    postfix
        The postfix used to change the duplicate names.
    blacklist
        A list of names. The provided learner can't use model names in the list.

    Returns
    -------
    The learner guaranteed has no duplicate model names with the blacklist names.
    """
    model_names = []
    for n in learner._config.model.names:
        if n in blacklist:
            new_name = f"{n}_{postfix}"
            assert new_name not in blacklist
            assert new_name not in learner._config.model.names
            # modify model prefix
            if n == learner._model.prefix:
                learner._model.prefix = new_name
            else:
                assert isinstance(learner._model.model, nn.ModuleList)
                for per_model in learner._model.model:
                    if n == per_model.prefix:
                        per_model.prefix = new_name
                        break
            # modify data processor prefix
            for per_modality_processors in learner._data_processors.values():
                for per_processor in per_modality_processors:
                    if n == per_processor.prefix:
                        per_processor.prefix = new_name
            # modify model config keys
            setattr(learner._config.model, new_name, getattr(learner._config.model, n))
            delattr(learner._config.model, n)

            model_names.append(new_name)
        else:
            model_names.append(n)

    learner._config.model.names = model_names

    return learner


def list_timm_models(pretrained=True):
    return timm.list_models(pretrained=pretrained)


def is_lazy_weight_tensor(p: torch.Tensor) -> bool:
    from torch.nn.parameter import UninitializedParameter

    if isinstance(p, UninitializedParameter):
        warnings.warn(
            "A layer with UninitializedParameter was found. "
            "Thus, the total number of parameters detected may be inaccurate."
        )
        return True
    return False

```

```python
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..constants import COLUMN, IMAGE, LABEL, LOGITS, MOE_LOSS
from .adaptation_layers import ConvLoRAConv2d
from .utils import image_mean_std

logger = logging.getLogger(__name__)


class ResEncUNetForSemanticSegmentation(nn.Module):
    """
    ResEnc U-Net（dynamic-network-architectures）包装成 AutoGluon semantic_segmentation 的统一接口。

    目标：
    - 走 AutoGluon 的 data processor（resize/normalize）与 Lightning 训练循环
    - 允许通过 optim.peft=conv_lora 注入 ConvLoRAConv2d，并把 MOE_LOSS 透传到训练 loss
    - 输出二值 logits（shape: [B, 1, H, W]）
    """

    def __init__(
        self,
        prefix: str,
        num_classes: int = 1,
        image_size: int = 512,
        image_norm: Optional[str] = "imagenet",
        pretrained: bool = False,
        # ResEnc UNet hyperparams（先给出合理默认，后续可用 config 覆盖）
        n_stages: int = 7,
        features_per_stage=(32, 64, 128, 256, 512, 512, 512),
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6, 6),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1, 1),
        deep_supervision: bool = False,
    ):
        super().__init__()
        if pretrained:
            logger.warning("ResEncUNetForSemanticSegmentation 当前不支持加载预训练权重，pretrained=True 将被忽略。")

        assert num_classes == 1, "当前选择的是二值 logits 形式，请将 num_classes 设为 1。"

        self.prefix = prefix
        self.num_classes = num_classes
        self.image_size = image_size
        self.image_mean, self.image_std = image_mean_std(image_norm)

        from dynamic_network_architectures.architectures.residual_unet import ResidualEncoderUNet

        self.model = ResidualEncoderUNet(
            input_channels=3,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=nn.Conv2d,
            kernel_sizes=3,
            strides=(1, 2, 2, 2, 2, 2, 2),
            n_blocks_per_stage=n_blocks_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=True,
            norm_op=nn.InstanceNorm2d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=deep_supervision,
        )

    @property
    def image_key(self):
        return f"{self.prefix}_{IMAGE}"

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    @property
    def image_column_prefix(self):
        return f"{self.image_key}_{COLUMN}"

    def _collect_moe_loss(self) -> Optional[torch.Tensor]:
        moe = None
        for m in self.modules():
            if isinstance(m, ConvLoRAConv2d) and getattr(m, "moe_loss", None) is not None:
                moe = m.moe_loss if moe is None else (moe + m.moe_loss)
        return moe

    def forward(self, batch):
        x = batch[self.image_key]
        # semantic seg processor 会产出 [B, N, C, H, W]，本任务默认 N=1
        if x.dim() == 5:
            x = x[:, 0]
        # 网络输出期望 [B, 1, H, W]
        logits = self.model(x)
        # deep_supervision=False 时 logits 是 Tensor；True 时为 list/tuple（暂不支持）
        if isinstance(logits, (list, tuple)):
            raise NotImplementedError("deep_supervision=True 暂未接入 AutoGluon loss，请先使用 deep_supervision=False。")

        # 对齐 label 的空间分辨率（保险起见）
        if logits.shape[-2:] != batch[self.label_key].shape[-2:]:
            logits = F.interpolate(logits, batch[self.label_key].shape[-2:], mode="bilinear", align_corners=False)

        if self.training:
            out = {self.prefix: {LOGITS: logits}}
        else:
            out = {self.prefix: {LOGITS: logits, LABEL: batch[self.label_key]}}

        moe_loss = self._collect_moe_loss()
        if moe_loss is not None:
            out[self.prefix][MOE_LOSS] = moe_loss
        return out

    def get_layer_ids(self):
        # 先给一个最简单的实现，确保 layerwise lr decay 的调用不报错
        return {n: 0 for n, _ in self.named_parameters()}


```

```python
from .augmenter import Augmenter
from .categorical_mlp import CategoricalMLP
from .clip import CLIPForImageText
from .document_transformer import DocumentTransformer
from .ft_transformer import FT_Transformer
from .fusion import (
    AbstractMultimodalFusionModel,
    MultimodalFusionMLP,
    MultimodalFusionNER,
    MultimodalFusionTransformer,
)
from .hf_text import HFAutoModelForTextPrediction
from .meta_transformer import MetaTransformer
from .mmdet_image import MMDetAutoModelForObjectDetection
from .mmocr_text_detection import MMOCRAutoModelForTextDetection
from .mmocr_text_recognition import MMOCRAutoModelForTextRecognition
from .ner_text import HFAutoModelForNER
from .numerical_mlp import NumericalMLP
from .sam import SAMForSemanticSegmentation
from .nnunet import ResEncUNetForSemanticSegmentation
from .t_few import TFewModel
from .timm_image import TimmAutoModelForImagePrediction
from .utils import (
    create_fusion_model,
    create_model,
    get_model_postprocess_fn,
    is_lazy_weight_tensor,
    list_timm_models,
    modify_duplicate_model_names,
    select_model,
)

```

```python
"""Storing the constants"""

from autogluon.core.metrics import METRICS

# Column/Label Types
NULL = "null"
CATEGORICAL = "categorical"
TEXT = "text"
TEXT_NER = "text_ner"  # Added for NER text column
NUMERICAL = "numerical"
IMAGE_PATH = "image_path"
IMAGE_BYTEARRAY = "image_bytearray"
IMAGE_BASE64_STR = "image_base64_str"
IDENTIFIER = "identifier"
DOCUMENT = "document"
DOCUMENT_IMAGE = "document_image"
DOCUMENT_PDF = "document_pdf"

# Scarcity modes
FEW_SHOT = "few_shot"
DEFAULT_SHOT = "default_shot"
ZERO_SHOT = "zero_shot"

# Problem types
CLASSIFICATION = "classification"
BINARY = "binary"
MULTICLASS = "multiclass"
REGRESSION = "regression"
NER = "ner"
NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
FEATURE_EXTRACTION = "feature_extraction"
ZERO_SHOT_IMAGE_CLASSIFICATION = "zero_shot_image_classification"
OBJECT_DETECTION = "object_detection"
OCR = "ocr"
OCR_TEXT_DETECTION = f"{OCR}_text_detection"
OCR_TEXT_RECOGNITION = f"{OCR}_text_recognition"
IMAGE_SIMILARITY = "image_similarity"
TEXT_SIMILARITY = "text_similarity"
IMAGE_TEXT_SIMILARITY = "image_text_similarity"
FEW_SHOT_CLASSIFICATION = "few_shot_classification"
SEMANTIC_SEGMENTATION = "semantic_segmentation"

# Input keys
IMAGE = "image"
IMAGE_META = "image_meta"
IMAGE_VALID_NUM = "image_valid_num"
LABEL = "label"
TEXT_TOKEN_IDS = "text_token_ids"
CHOICES_IDS = "choices_ids"
TEXT_VALID_LENGTH = "text_valid_length"
TEXT_SEGMENT_IDS = "text_segment_ids"
COLUMN = "column"
ATTENTION_MASK = "attention_mask"
TOKEN_TYPE_IDS = "token_type_ids"
PIXEL_VALUES = "pixel_values"
INPUT_IDS = "input_ids"
SEMANTIC_SEGMENTATION_IMG = "semantic_segmentation_img"
SEMANTIC_SEGMENTATION_GT = "semantic_segmentation_gt"

# Output keys
LOGITS = "logits"
ORI_LOGITS = "ori_logits"
AUG_LOGITS = "aug_logits"
TEMPLATE_LOGITS = "template_logits"
LM_TARGET = "lm_target"
LOSS = "loss"
OUTPUT = "output"
WEIGHT = "weight"
FEATURES = "features"
MULTIMODAL_FEATURES = "multimodal_features"  # used for the adapted multimodal features before the fusion module
MULTIMODAL_FEATURES_PRE_AUG = "multimodal_features_pre_aug"
MULTIMODAL_FEATURES_POST_AUG = "multimodal_features_post_aug"
RAW_FEATURES = "raw_features"
MASKS = "masks"
PROBABILITY = "probability"
COLUMN_FEATURES = "column_features"
BBOX = "bbox"
ROIS = "rois"
SCORE = "score"
LOGIT_SCALE = "logit_scale"
VAE_MEAN = "vae_mean"
VAE_VAR = "vae_var"

# Loss
MOE_LOSS = "moe_loss"

# Metric for Object Detection
MAP = "map"
MEAN_AVERAGE_PRECISION = "mean_average_precision"
MAP_50 = "map_50"
MAP_75 = "map_75"
MAP_SMALL = "map_small"
MAP_MEDIUM = "map_medium"
MAP_LARGE = "map_large"
MAR_1 = "mar_1"
MAR_10 = "mar_10"
MAR_100 = "mar_100"
MAR_SMALL = "mar_small"
MAR_MEDIUM = "mar_medium"
MAR_LARGE = "mar_large"
DETECTION_METRICS = [
    MAP,
    MEAN_AVERAGE_PRECISION,
    MAP_50,
    MAP_75,
    MAP_SMALL,
    MAP_MEDIUM,
    MAP_LARGE,
    MAR_1,
    MAR_10,
    MAR_100,
    MAR_SMALL,
    MAR_MEDIUM,
    MAR_LARGE,
]

# Metric
MAX = "max"
MIN = "min"
ACCURACY = "accuracy"
ACC = "acc"
OVERALL_ACCURACY = "overall_accuracy"
RMSE = "rmse"
ROOT_MEAN_SQUARED_ERROR = "root_mean_squared_error"
R2 = "r2"
PEARSONR = "pearsonr"
SPEARMANR = "spearmanr"
QUADRATIC_KAPPA = "quadratic_kappa"
ROC_AUC = "roc_auc"
AVERAGE_PRECISION = "average_precision"
LOG_LOSS = "log_loss"
CROSS_ENTROPY = "cross_entropy"
COSINE_EMBEDDING_LOSS = "cosine_embedding_loss"
F1 = "f1"
OVERALL_F1 = "overall_f1"
F1_MACRO = "f1_macro"
F1_MICRO = "f1_micro"
F1_WEIGHTED = "f1_weighted"
NER_TOKEN_F1 = "ner_token_f1"
DIRECT_LOSS = "direct_loss"
HIT_RATE = "hit_rate"
NDCG = "ndcg"
PRECISION = "precision"
RECALL = "recall"
MRR = "mrr"
SM = "sm"
EM = "em"
FM = "fm"
MAE = "mae"
BER = "ber"
IOU = "iou"
COVERAGE = "coverage"
RETRIEVAL_METRICS = [NDCG, PRECISION, RECALL, MRR]
METRIC_MODE_MAP = {
    ACC: MAX,
    ACCURACY: MAX,
    DIRECT_LOSS: MIN,
    RMSE: MIN,
    ROOT_MEAN_SQUARED_ERROR: MIN,
    R2: MAX,
    QUADRATIC_KAPPA: MAX,
    ROC_AUC: MAX,
    LOG_LOSS: MIN,
    CROSS_ENTROPY: MIN,
    PEARSONR: MAX,
    SPEARMANR: MAX,
    F1: MAX,
    F1_MACRO: MAX,
    F1_MICRO: MAX,
    F1_WEIGHTED: MAX,
    MAP: MAX,
    MEAN_AVERAGE_PRECISION: MAX,
    NER_TOKEN_F1: MAX,
    OVERALL_F1: MAX,
    RECALL: MAX,
    SM: MAX,
    IOU: MAX,
    BER: MIN,
    COVERAGE: MAX,
}

MATCHING_METRICS = {
    BINARY: [ROC_AUC, ROC_AUC],
    MULTICLASS: [SPEARMANR, SPEARMANR],
    REGRESSION: [SPEARMANR, SPEARMANR],
}
MATCHING_METRICS_WITHOUT_PROBLEM_TYPE = [RECALL, NDCG]

EVALUATION_METRICS = {
    # Use evaluation metrics from METRICS for these types
    BINARY: list(METRICS[BINARY].keys()) + [COVERAGE],
    MULTICLASS: METRICS[MULTICLASS].keys(),
    REGRESSION: METRICS[REGRESSION].keys(),
    OBJECT_DETECTION: DETECTION_METRICS,
    SEMANTIC_SEGMENTATION: [IOU, BER, SM],
    NER: [OVERALL_F1, NER_TOKEN_F1],
    NAMED_ENTITY_RECOGNITION: [OVERALL_F1, NER_TOKEN_F1],
    FEW_SHOT_CLASSIFICATION: METRICS[MULTICLASS].keys(),
}

VALIDATION_METRICS = {
    problem_type: [metric for metric in metrics if metric in METRIC_MODE_MAP] + [DIRECT_LOSS]
    for problem_type, metrics in EVALUATION_METRICS.items()
}

# Training status
TRAIN = "train"
VALIDATE = "validate"
VAL = "val"
TEST = "test"
PREDICT = "predict"

# Model sources
HUGGINGFACE = "huggingface"
TIMM = "timm"
MMDET = "mmdet"
MMOCR = "mmocr"

# Modality keys. may need to update here if new modality keys are added in above.
ALL_MODALITIES = [IMAGE, TEXT, CATEGORICAL, NUMERICAL, TEXT_NER, DOCUMENT, SEMANTIC_SEGMENTATION_IMG]

# Keys to compute metrics
Y_PRED = "y_pred"
Y_PRED_PROB = "y_pred_prob"
Y_TRUE = "y_true"

# Configuration keys
MODEL = "model"
DATA = "data"
OPTIM = "optim"
ENV = "env"
DISTILLER = "distiller"
MATCHER = "matcher"
VALID_CONFIG_KEYS = [MODEL, DATA, OPTIM, ENV, DISTILLER, MATCHER]

# Image normalization mean and std. This is only to normalize images for the CLIP model.
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

# Logger name
AUTOMM = "automm"

# environment variables
AUTOMM_TUTORIAL_MODE = "AUTOMM_TUTORIAL_MODE"

# error try
GET_ITEM_ERROR_RETRY = 50

# top-k checkpoint average methods
UNIFORM_SOUP = "uniform_soup"
GREEDY_SOUP = "greedy_soup"
BEST = "best"

# efficient finetuning strategies
NORM_FIT = "norm_fit"
BIT_FIT = "bit_fit"
LORA = "lora"
LORA_BIAS = "lora_bias"
LORA_NORM = "lora_norm"
IA3 = "ia3"
IA3_BIAS = "ia3_bias"
IA3_NORM = "ia3_norm"
IA3_LORA = "ia3_lora"
IA3_LORA_BIAS = "ia3_lora_bias"
IA3_LORA_NORM = "Ia3_lora_norm"
CONV_LORA = "conv_lora"
PEFT_ADDITIVE_STRATEGIES = [
    LORA,
    LORA_BIAS,
    LORA_NORM,
    IA3,
    IA3_BIAS,
    IA3_NORM,
    IA3_LORA,
    IA3_LORA_BIAS,
    IA3_LORA_NORM,
    CONV_LORA,
]
PEFT_NON_ADDITIVE_STRATEGIES = [
    BIT_FIT,
    NORM_FIT,
]
PEFT_STRATEGIES = list(set(PEFT_ADDITIVE_STRATEGIES) | set(PEFT_NON_ADDITIVE_STRATEGIES))

# DeepSpeed constants
DEEPSPEED_OFFLOADING = "deepspeed_stage_3_offload"
DEEPSPEED_STRATEGY = "deepspeed"
DEEPSPEED_MODULE = "autogluon.multimodal.optim.deepspeed"
DEEPSPEED_MIN_PL_VERSION = "1.7.1"

# registered model keys. TODO: document how to add new models.
CLIP = "clip"
TIMM_IMAGE = "timm_image"
HF_TEXT = "hf_text"
T_FEW = "t_few"
NUMERICAL_MLP = "numerical_mlp"
CATEGORICAL_MLP = "categorical_mlp"
FUSION = "fusion"
FUSION_MLP = f"{FUSION}_mlp"
FUSION_TRANSFORMER = f"{FUSION}_transformer"
FT_TRANSFORMER = "ft_transformer"
FUSION_NER = f"{FUSION}_{NER}"
MMDET_IMAGE = "mmdet_image"
MMOCR_TEXT_DET = "mmocr_text_detection"
MMOCR_TEXT_RECOG = "mmocr_text_recognition"
NER_TEXT = "ner_text"
DOCUMENT_TRANSFORMER = "document_transformer"
HF_MODELS = (HF_TEXT, T_FEW, CLIP, NER_TEXT, DOCUMENT_TRANSFORMER)
MMLAB_MODELS = (MMDET_IMAGE, MMOCR_TEXT_DET, MMOCR_TEXT_RECOG)
SAM = "sam"
NNUNET = "nnunet"
META_TRANSFORMER = "meta_transformer"

# matcher loss type
CONTRASTIVE_LOSS = "contrastive_loss"
MULTI_NEGATIVES_SOFTMAX_LOSS = "multi_negatives_softmax_loss"

# matcher distance type
COSINE_SIMILARITY = "cosine_similarity"

# matcher miner type
PAIR_MARGIN_MINER = "pair_margin_miner"

# checkpoints
RAY_TUNE_CHECKPOINT = "ray_tune_checkpoint.ckpt"
BEST_K_MODELS_FILE = "best_k_models.yaml"
LAST_CHECKPOINT = "last.ckpt"
MODEL_CHECKPOINT = "model.ckpt"

# url
S3_PREFIX = "s3://"
SOURCEPROMPT_URL = "https://automl-mm-bench.s3.amazonaws.com/few_shot/templates.zip"
SOURCEPROMPT_SHA1 = "c25cdf3730ff96ab4859b72e18d46ff117b62bd6"

# ner
ENTITY_GROUP = "entity_group"
START_OFFSET = "start"
END_OFFSET = "end"
TOKEN_WORD_MAPPING = "token_word_mapping"
WORD_OFFSETS = "word_offsets"
NER_RET = "ner_ret"
NER_ANNOTATION = "ner_annotation"

# matcher
QUERY = "query"
RESPONSE = "response"
QUERY_RESPONSE = f"{QUERY}_{RESPONSE}"
PAIR = "pair"
TRIPLET = "triplet"

# mmdet
XYWH = "xywh"
XYXY = "xyxy"
BBOX_FORMATS = [XYWH, XYXY]

# sam (multi-class)
CLASS_LOGITS = "class_logits"
MASK_LABEL = "mask_label"
CLASS_LABEL = "class_label"
SEMANTIC_MASK = "semantic_mask"

# presets
DEFAULT = "default"
HIGH_QUALITY = "high_quality"
MEDIUM_QUALITY = "medium_quality"
BEST_QUALITY = "best_quality"
ALL_MODEL_QUALITIES = [HIGH_QUALITY, MEDIUM_QUALITY, BEST_QUALITY, DEFAULT]

# datasets
DEFAULT_DATASET = "default_dataset"
MULTI_IMAGE_MIX_DATASET = "multi_image_mix_dataset"

# strategies
DDP = "ddp"
DDP_FIND_UNUSED_PARAMETERS_FALSE = "ddp_find_unused_parameters_false"
DDP_FIND_UNUSED_PARAMETERS_TRUE = "ddp_find_unused_parameters_true"
DDP_STRATEGIES = [DDP, DDP_FIND_UNUSED_PARAMETERS_FALSE, DDP_FIND_UNUSED_PARAMETERS_TRUE]

# torch constants
TORCH_COMPILE_MIN_VERSION = "2.2.0.dev20230908"

```

```python
import functools
import logging
import re
import warnings
from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch._dynamo
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from torch import nn
from torch.nn.modules.loss import _Loss
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertTokenizer, CLIPTokenizer, ElectraTokenizer
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss

from ..constants import (
    ALL_MODALITIES,
    CATEGORICAL,
    CATEGORICAL_MLP,
    CLASS_LOGITS,
    CLIP,
    CLIP_IMAGE_MEAN,
    CLIP_IMAGE_STD,
    DOCUMENT,
    DOCUMENT_TRANSFORMER,
    FT_TRANSFORMER,
    FUSION_MLP,
    FUSION_NER,
    FUSION_TRANSFORMER,
    HF_TEXT,
    IMAGE,
    LOGITS,
    META_TRANSFORMER,
    MMDET_IMAGE,
    MMOCR_TEXT_DET,
    MMOCR_TEXT_RECOG,
    NER_TEXT,
    NUMERICAL,
    NUMERICAL_MLP,
    OCR,
    PEFT_ADDITIVE_STRATEGIES,
    REGRESSION,
    SAM,
    NNUNET,
    SEMANTIC_MASK,
    SEMANTIC_SEGMENTATION,
    SEMANTIC_SEGMENTATION_IMG,
    T_FEW,
    TEXT,
    TEXT_NER,
    TIMM_IMAGE,
)
from .adaptation_layers import ConvLoRAConv2d, ConvLoRALinear, IA3Linear, IA3LoRALinear, LoRALinear

logger = logging.getLogger(__name__)


ALL_TOKENIZERS = {
    "bert": BertTokenizer,
    "clip": CLIPTokenizer,
    "electra": ElectraTokenizer,
    "hf_auto": AutoTokenizer,
}


class DummyLayer(nn.Module):
    """
    DummyLayer to ensure that the gradient checkpointing will assign output layer as require_grad=True.
    Reference: https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
    """

    def __init__(self):
        super().__init__()
        self.dummy_bias = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def forward(self, x):
        return x + self.dummy_bias.to(x) - self.dummy_bias.to(x)


def init_weights(module: nn.Module):
    """
    Initialize one module. It uses xavier_norm to initialize nn.Embedding
    and xavier_uniform to initialize nn.Linear's weight.

    Parameters
    ----------
    module
        A Pytorch nn.Module.
    """
    if isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def assign_encoder_layer_ids(
    encoder_names: List[List[str]],
):
    """
    Assign ids to encoder layers. The encoder may contain several blocks e.g., block1 and block2.
    This function iterates through all the layers of each block from the input end towards the output end.
    It increases 1 on the layer id when the detected digit in a layer name changes.

    Parameters
    ----------
    encoder_names
        Encoder layer names.

    Returns
    -------
    name_to_id
        The encoder layer-to-id mapping.
    encoder_layer_num
        The encoder layer number.
    """
    name_to_id = {}
    cur_id = 0
    for i, group_names in enumerate(encoder_names):
        last_inferred_id = -1
        for n in group_names:
            detect_id = False
            n_splits = n.split(".")

            for split in n_splits:
                # the first digit encountered is used to infer layer id
                if split.isdigit():
                    inferred_id = int(split)
                    # increase at most 1 one time
                    if inferred_id != last_inferred_id:
                        cur_id += 1  # layer.0 -> layer_id 1
                        last_inferred_id = inferred_id

                    name_to_id[n] = cur_id
                    detect_id = True
                    break

            if detect_id is False:
                raise ValueError(f"parameter name: {n} not has no id inside")

    if len(name_to_id) > 0:
        encoder_layer_num = max(name_to_id.values())
    else:
        encoder_layer_num = 0
    return name_to_id, encoder_layer_num


def assign_non_encoder_layer_ids(
    non_encoder_names: List[str],
    layer_id: int,
):
    """
    Assign the provided id to non-encoder layers.

    Parameters
    ----------
    non_encoder_names
        Names layers not belonging to an encoder.
    layer_id
        provided id.

    Returns
    -------
    A dictionary mapping the layer names (keys) to their ids (values).
    """
    name_to_id = {}
    for n in non_encoder_names:
        name_to_id[n] = layer_id
    return name_to_id


def split_encoder_non_encoder(names: List[str], post_encoder_patterns: Tuple[str, ...]):
    """
    Group layer names into two types: encoder and non-encoder.
    A layer belongs to encoder if its name contains at least one digit.
    It uses this rule since a model's encoder in Pytorch's implementation
    is generally wrapped by nn.Sequential() or nn.ModuleList(),
    which produce digits in layer names.

    Parameters
    ----------
    names
        Model layer names.
    Returns
    -------
    encoder_names
        A list of encoder layer names.
    non_encoder_names
        A list of non-encoder layer names.
    """
    encoder_names = []
    non_encoder_names = []
    for n in names:
        is_encoder = False
        if any(p in n for p in post_encoder_patterns):
            non_encoder_names.append(n)
            continue
        for i in n.split("."):
            if i.isdigit():
                encoder_names.append(n)
                is_encoder = True
                break
        if not is_encoder:
            non_encoder_names.append(n)

    return encoder_names, non_encoder_names


def group_param_names(
    names: List[str],
    pre_encoder_patterns: Tuple[str, ...],
    post_encoder_patterns: Tuple[str, ...],
    model_prefix: Optional[str] = None,
):
    """
    Group layer names into three types: pre-encoder, encoder, and post-encoder.
    If "model_prefix" is provided, the selected layer names must start with it.
    In this case, the left names will be returned for the next-time processing.
    This function first extracts the first-level children modules' names and
    classify them into encoder and non-encoder layers. Note that an encoder may
    consist of several manually named children modules, e.g., block1 and block2.
    The non-encoder layers are further subdivided into pre-encoder and post-encoder.

    Parameters
    ----------
    names
        Model layer names
    pre_encoder_patterns
        Patterns to identify a layer as a pre-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into pre-encoder layers.
    post_encoder_patterns
        Patterns to identify a layer as a post-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into post-encoder layers.
    model_prefix
        A prefix to filter layer names. Only layer names starting with it will be selected.
    Returns
    -------
    left_names
        The layer names left for the next-time processing.
    encoder_names_grouped
        Encoder layer names.
    pre_encoder_names
        Names of layers before the encoder.
    post_encoder_names
        Names of layers after the encoder.
    """
    # two set of patterns can't have intersections
    assert all(pre_p not in post_encoder_patterns for pre_p in pre_encoder_patterns)

    left_names = []
    # in case model_prefix is provided, e.g., the clip model with image and text branches
    selected_names = []
    for n in names:
        if model_prefix is not None and not n.startswith(model_prefix):
            left_names.append(n)
        else:
            selected_names.append(n)

    # split blocks at the first level
    children_prefix = []
    for n in selected_names:
        if model_prefix is not None:
            child_name = n[len(model_prefix) + 1 :].split(".")[0]
            child_prefix = f"{model_prefix}.{child_name}"
        else:
            child_prefix = n.split(".")[0]
        if child_prefix not in children_prefix:
            children_prefix.append(child_prefix)

    encoder_names_grouped = []
    non_encoder_names = []
    for child_prefix in children_prefix:
        per_names_group = [n for n in selected_names if n.startswith(child_prefix)]
        per_encoder_names, per_non_encoder_names = split_encoder_non_encoder(per_names_group, post_encoder_patterns)
        encoder_names_grouped.append(per_encoder_names)
        non_encoder_names.extend(per_non_encoder_names)

    pre_encoder_names = []
    post_encoder_names = []
    for n in non_encoder_names:
        if any(p in n for p in pre_encoder_patterns):
            pre_encoder_names.append(n)
        elif any(p in n for p in post_encoder_patterns):
            post_encoder_names.append(n)
        else:
            raise ValueError(f"parameter name: {n} belong to neither pre or post encoder names")

    # only process left names in next iteration
    return left_names, encoder_names_grouped, pre_encoder_names, post_encoder_names


def reverse_layer_ids(
    encoder_name_to_id: dict,
    pre_enocder_name_to_id: dict,
    post_enocder_name_to_id: dict,
):
    """
    The layer ids need to increase when going from the output end to the input end.
    We need to reverse the ids which were originally assigned in a decreasing order.

    Parameters
    ----------
    encoder_name_to_id
        The layer-to-id mapping of encoder layers.
    pre_enocder_name_to_id
        The layer-to-id mapping of pre-encoder layers.
    post_enocder_name_to_id
        The layer-to-id mapping of post-encoder layers.

    Returns
    -------
    The layer-to-id mapping of all layers with layer ids reversed.
    """
    name_to_id = {**pre_enocder_name_to_id, **encoder_name_to_id, **post_enocder_name_to_id}
    if len(name_to_id) > 0:
        layer_num = max(name_to_id.values())
        # if no post encoder layers, the minimum layer id should be 1
        if len(post_enocder_name_to_id) == 0:
            layer_num += 1
    for n, layer_id in name_to_id.items():
        name_to_id[n] = layer_num - layer_id

    return name_to_id


def assign_layer_ids(
    names: List[str],
    pre_encoder_patterns: Tuple[str, ...],
    post_encoder_patterns: Tuple[str, ...],
    model_pre: Optional[str] = None,
):
    """
    Assign ids to all layers. It splits a model into three parts: pre-encoder, encoder, and post-encoder.
    Encoder is generally a stack of multiple similar layers, such as transformer layers. Since encoder is
    generally wrapped by nn.Sequential() or nn.ModuleList(), its inside layer names contain digits.
    It sets 0 as the ids of all post-encoder layers and a maximum id (layer_num) for the all the pre-encoder
    layers. The encoder layers have decreasing ids from the input to the output ends.

    Parameters
    ----------
    names
        model layer names.
    pre_encoder_patterns
        Patterns to identify a layer as a pre-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into pre-encoder layers.
    post_encoder_patterns
        Patterns to identify a layer as a post-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into post-encoder layers.
    model_pre
        The layer names' prefix. Only the layer names with this prefix will be assigned ids. The left
        layer names will be returned.

    Returns
    -------
    name_to_id
        A dictionary mapping the layer names (keys) to their ids (values).
    left_names
        The layer names not starting with the "model_pre".
    """
    try:
        left_names, encoder_names, pre_encoder_names, post_encoder_names = group_param_names(
            names=names,
            pre_encoder_patterns=pre_encoder_patterns,
            post_encoder_patterns=post_encoder_patterns,
            model_prefix=model_pre,
        )
        # add a constraint
        if len(encoder_names) == 0 and len(pre_encoder_names) != 0:
            raise ValueError(f"encoder_names is empty, but pre_encoder_names has values: {pre_encoder_names}")

        encoder_name_to_id, encoder_layer_num = assign_encoder_layer_ids(
            encoder_names=encoder_names,
        )

        pre_encoder_name_to_id = assign_non_encoder_layer_ids(non_encoder_names=pre_encoder_names, layer_id=0)

        post_encoder_name_to_id = assign_non_encoder_layer_ids(
            non_encoder_names=post_encoder_names, layer_id=encoder_layer_num + 1
        )

        name_to_id = reverse_layer_ids(
            encoder_name_to_id=encoder_name_to_id,
            pre_enocder_name_to_id=pre_encoder_name_to_id,
            post_enocder_name_to_id=post_encoder_name_to_id,
        )
    except Exception as e:
        logger.debug(
            f"When calling assign_layer_ids(), it catches exception: {e}. All the layers will use the same layer_id."
        )
        name_to_id = dict()
        left_names = names

    return name_to_id, left_names


def get_column_features(
    batch: Dict[str, torch.Tensor],
    column_name_prefix: str,
    features: torch.Tensor,
    valid_lengths: torch.Tensor,
    cls_feature: Optional[torch.Tensor] = None,
):
    """
    Index the features of one column defined by `column_name_prefix`.
    This function can be used to index both image and text features.
    The features have shape (b, n, d), where n can be the image number or
    text token number. One column corresponds to a subset of
    the n images or text tokens. One column name can only appear once in the return.

    Parameters
    ----------
    batch
        The batch input containing the feature column information, i.e., indexes.
    column_name_prefix
        The column name prefix of one modality (image or text).
    features
        The features of columns whose names starts with column_name_prefix.
    valid_lengths
        The valid image number or text token number of each sample in a batch.
    cls_feature
        The cls feature containing information from all feature columns.

    Returns
    -------
    The column features with masks. If the column has no valid features, its
    mask is 0.
    """
    column_features = {}
    feature_masks = {}

    cut_idx = len(column_name_prefix) + 1
    if cls_feature is not None:
        all_column_names = []
        # create a zero mask to do logical_or with each column's mask
        joint_mask = torch.zeros(features.shape[0]).to(features)  # (b,)
    for key in batch:
        if key.startswith(column_name_prefix):
            per_col_features = []
            per_col_masks = torch.zeros(features.shape[0]).to(features)  # (b,)
            assert batch[key].ndim == 2 and batch[key].shape[1] == 2
            for i, per_sample_col_idx in enumerate(batch[key]):
                start_idx = per_sample_col_idx[0]
                end_idx = per_sample_col_idx[1]
                if start_idx < end_idx:
                    assert end_idx <= valid_lengths[i]
                    per_col_features.append(features[i, start_idx:end_idx].mean(dim=0))
                    per_col_masks[i] = 1
                else:  # the column has no valid image/text.
                    per_col_features.append(torch.zeros_like(features[0, 0]))
                    per_col_masks[i] = 0
            column_name = key[cut_idx:]
            column_features[column_name] = torch.stack(per_col_features, dim=0)  # (b, num_features)
            feature_masks[column_name] = per_col_masks  # (b,)
            if cls_feature is not None:
                all_column_names.append(column_name)
                joint_mask = torch.logical_or(joint_mask, per_col_masks)

    # all the columns of one model's input share the model's cls feature
    if (
        cls_feature is not None and len(all_column_names) > 0
    ):  # some models', e.g, timm_image, output doesn't have the cls feature.
        # remove the individual column features since these column features not independent
        for column_name in all_column_names:
            column_features.pop(column_name)
            feature_masks.pop(column_name)

        joint_column_name = "_".join(all_column_names)
        column_features[joint_column_name] = cls_feature
        feature_masks[joint_column_name] = joint_mask.to(features)

    # print(f"column_features: {column_features}")
    return column_features, feature_masks


def create_adaptation(peft: str, layer: nn.Module, lora_r: int, lora_alpha: int, **kwargs):
    """
    Creates a model adaptation module (IA3, LoRA, IA3_LoRA) given a linear layer.

    Parameters
    ----------
    peft
        Name of the adaptation module.
    layer
       The layer the adaptation module should be applied to.
    lora_r
        The rank r of the low-rank decomposition.
    lora_alpha
        The scaling factor. Can be set to same value as r in
        most cases, as initialization is scaled already.
    filter
        Apply loRA only to linear layers filtered by name (e.g. "query.").
        If None, loRA is applied to all linear Layers in module.
    module_filter
        Apply loRA only to modules filtered by name (e.g. ".*EncDecAttention|.*DenseReluDense")
        If None, loRA is considered for all modules

    Returns
    -------
    Model with injected LoRA modules.
    """
    if "ia3_lora" in peft:
        return IA3LoRALinear(
            layer.in_features, layer.out_features, r=lora_r, lora_alpha=lora_alpha, merge_weights=False
        )
    elif "conv_lora" in peft:
        return ConvLoRALinear(
            layer.in_features,
            layer.out_features,
            r=lora_r,
            lora_alpha=lora_alpha,
            merge_weights=False,
            conv_lora_expert_num=kwargs["conv_lora_expert_num"],
        )
    elif "ia3" in peft:
        return IA3Linear(layer.in_features, layer.out_features, merge_weights=False)
    elif "lora" in peft:
        return LoRALinear(layer.in_features, layer.out_features, r=lora_r, lora_alpha=lora_alpha, merge_weights=False)
    elif peft is not None:
        raise NotImplementedError(
            f"The efficient finetuning strategy '{peft}'"
            f" is not supported. We only support"
            f" {', '.join(PEFT_ADDITIVE_STRATEGIES)}."
        )


def inject_adaptation_to_linear_layer(
    model: nn.Module,
    peft: str,
    lora_r: int = None,
    lora_alpha: int = None,
    filter: Optional[List[str]] = None,
    module_filter: Optional[List[str]] = None,
    extra_trainable_params: Optional[List[str]] = None,
    **kwargs,
) -> nn.Module:
    """
    Injects trainable adatio Low-Rank decomposition matrices (LoRA) into linear
    layers of a PyTorch model. Used for efficient fine-tuning of large
    pre-trained models.

    Parameters
    ----------
    model
        A PyTorch model.
    peft
        Efficient finetuning method that should be applied.
    lora_r
        The rank r of the low-rank decomposition.
    lora_alpha
        The scaling factor. Can be set to same value as r in
        most cases, as initialization is scaled already.
    filter
        Apply loRA only to linear layers filtered by name (e.g. "query.").
        If None, loRA is applied to all linear Layers in module.
    module_filter
        Apply loRA only to modules filtered by name (e.g. ".*EncDecAttention|.*DenseReluDense")
        If None, loRA is considered for all modules
    extra_trainable_params
        Not to apply loRA to modules filtered by name, and these modules are not frozen during training (e.g. "mask_decoder").
        If None, all the modules except for those applied loRA are frozen.
    Returns
    -------
    Model with injected LoRA modules.
    """
    for m_name, module in dict(model.named_modules()).items():
        if extra_trainable_params and any(re.match(filter_layer, m_name) for filter_layer in extra_trainable_params):
            continue
        if hasattr(model, "frozen_layers") and any(
            re.search(filter_layer, m_name) for filter_layer in model.frozen_layers
        ):
            continue
        if not module_filter or any(re.match(filter_module, m_name) for filter_module in module_filter):
            for c_name, layer in dict(module.named_children()).items():
                if not filter or any(re.match(filter_layer, c_name) for filter_layer in filter):
                    assert isinstance(
                        layer, nn.Linear
                    ), f"LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}."
                    adaptation_layer = create_adaptation(peft, layer, lora_r, lora_alpha, **kwargs)
                    adaptation_layer.weight = layer.weight
                    adaptation_layer.bias = layer.bias
                    setattr(module, c_name, adaptation_layer)

    return model  # return model to enable method chaining


def inject_adaptation_to_conv2d_layer(
    model: nn.Module,
    peft: str,
    lora_r: int = None,
    lora_alpha: int = None,
    filter: Optional[List[str]] = None,
    module_filter: Optional[List[str]] = None,
    extra_trainable_params: Optional[List[str]] = None,
    **kwargs,
) -> nn.Module:
    """
    将 LoRA/Conv-LoRA 注入到 Conv2d（用于 nnU-Net 等卷积网络）。

    约定：
    - 仅对 `torch.nn.Conv2d` 生效
    - 目前 `peft` 仅支持 `conv_lora`（其它策略未来可扩展）
    - 注入后原卷积权重被冻结，新增参数以 `lora_*` 命名，便于 `get_peft_param_names` 选择可训练参数
    """
    if peft is None:
        return model
    if "conv_lora" not in peft:
        # 当前只扩展 conv_lora 到 Conv2d，其它 PEFT 保持原逻辑（linear-only）
        return model

    for m_name, module in dict(model.named_modules()).items():
        if extra_trainable_params and any(re.match(filter_layer, m_name) for filter_layer in extra_trainable_params):
            continue
        if hasattr(model, "frozen_layers") and any(re.search(filter_layer, m_name) for filter_layer in model.frozen_layers):
            continue
        if not module_filter or any(re.match(filter_module, m_name) for filter_module in module_filter):
            for c_name, layer in dict(module.named_children()).items():
                if not filter or any(re.match(filter_layer, c_name) for filter_layer in filter):
                    if not isinstance(layer, nn.Conv2d):
                        continue
                    adaptation_layer = ConvLoRAConv2d(
                        in_channels=layer.in_channels,
                        out_channels=layer.out_channels,
                        kernel_size=layer.kernel_size,
                        stride=layer.stride,
                        padding=layer.padding,
                        dilation=layer.dilation,
                        groups=layer.groups,
                        bias=layer.bias is not None,
                        padding_mode=layer.padding_mode,
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        merge_weights=False,
                        conv_lora_expert_num=kwargs["conv_lora_expert_num"],
                    )
                    # 复用原参数对象（weight/bias），确保 state_dict 兼容且冻结生效
                    adaptation_layer.weight = layer.weight
                    adaptation_layer.bias = layer.bias
                    setattr(module, c_name, adaptation_layer)

    return model


def get_model_head(model: nn.Module):
    """
    Return the model's head. Different models may have different head names.

    Parameters
    ----------
    model
        A Pytorch model.

    Returns
    -------
    The model's head.
    """
    if hasattr(model, "head"):
        head = model.head  # move the head outside
    elif hasattr(model, "last_linear"):
        head = model.last_linear
    elif hasattr(model, "fc"):
        head = model.fc
    elif hasattr(model, "classifier"):
        head = model.classifier
    else:
        raise ValueError(f"Model {type(model)} doesn't have head. Need to check its implementation.")

    return head.fc if hasattr(head, "fc") else head


def get_hf_config_and_model(
    checkpoint_name: str, pretrained: Optional[bool] = True, low_cpu_mem_usage: Optional[bool] = False
):
    """
    Get a Huggingface config and model based on a checkpoint name.

    Parameters
    ----------
    checkpoint_name
        A model checkpoint name or a local path that saves a custom checkpoint.
    pretrained
         Whether using the pretrained weights. If pretrained=True, download the pretrained model.
    low_cpu_mem_usage
        Whether to turn on the optimization of reducing the peak CPU memory usage when loading the pretrained model.

    Returns
    -------
    A Huggingface config and model.
    """
    config = AutoConfig.from_pretrained(checkpoint_name)    # nosec B615

    if pretrained:
        model = AutoModel.from_pretrained(checkpoint_name, low_cpu_mem_usage=low_cpu_mem_usage) # nosec B615
    else:
        model = AutoModel.from_config(config)
    # Explicitly set the model to train mode after loading as by default it is in eval mode
    # See issue #4965
    model.train()
    return config, model


def apply_sigmoid(output: Dict):
    """
    Apply the sigmoid to logits.

    Parameters
    ----------
    output
        The model output dict.

    Returns
    -------
    The output with logits transformed by sigmoid.
    """
    for k, v in output.items():
        output[k][LOGITS] = torch.sigmoid(v[LOGITS].float())
    return output


def apply_multi_class_semantic_seg_postprocess(output: Dict):
    """
    Apply the semantic postprocessing to logits.

    Parameters
    ----------
    output
        The model output dict.

    Returns
    -------
    The output with post-proceesed semantic masks.
    """

    def semantic_inference(mask_cls, mask_pred):
        """
        Post-processing mask prediction for multi-class semantic segmentation inference based on https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py

        Args:
            mask_cls (`torch.Tensor`):
                Class logits. A tensor of shape `(num_queries, num_classes + 1)` (include the "no object" category).

            mask_pred (`torch.Tensor`):
                Mask logits. A tensor of shape `(num_queries, height, width)`.

        Returns:
            semseg (`torch.Tensor`): The processed mask prediction. A tensor of shape `(num_classes, height, width)`.

        References:
        [1] https://arxiv.org/abs/2107.06278
        [2] https://arxiv.org/abs/2112.01527
        """
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    for k, v in output.items():
        pred_classes = output[k][CLASS_LOGITS]
        pred_masks = output[k][LOGITS]
        semantic_masks = []
        for mask_cls_result, mask_pred_result in zip(
            pred_classes, pred_masks
        ):  # bs, num_q, num_class and bs, num_q, h, w
            per_sample_semantic_masks = semantic_inference(mask_cls_result, mask_pred_result)  # num_class, h, w
            semantic_masks.append(per_sample_semantic_masks)
        semantic_masks = torch.stack(semantic_masks, dim=0)
        output[k][SEMANTIC_MASK] = semantic_masks
    return output


def get_model_postprocess_fn(problem_type: str, loss_func: _Loss):
    """
    Get the postprocessing function for the model outputs.

    Parameters
    ----------
    problem_type
        The problem type, e.g., classification or regression.
    loss_func
        The loss function used in training.

    Returns
    -------
    The postprocessing function.
    """
    postprocess_func = None
    if problem_type == REGRESSION:
        if isinstance(loss_func, nn.BCEWithLogitsLoss):
            postprocess_func = apply_sigmoid
    elif problem_type == SEMANTIC_SEGMENTATION:
        if isinstance(loss_func, Mask2FormerLoss):
            postprocess_func = apply_multi_class_semantic_seg_postprocess
        else:
            postprocess_func = apply_sigmoid

    return postprocess_func


def get_mmocr_config_and_model(checkpoint_name: str):
    """
    Get an MMOCR config and model based on a checkpoint name.

    Parameters
    ----------
    checkpoint_name
        A model checkpoint name.

    Returns
    -------
    An MMOCR config and model.
    """
    from ..utils import check_if_packages_installed

    check_if_packages_installed(package_names=["mmcv"])
    try:
        import mmocr
        from mmocr.models import build_detector
    except ImportError:
        mmocr = None
    from mim.commands.download import download

    checkpoints = download(package="mmocr", configs=[checkpoint_name], dest_root=".")

    # read config files
    check_if_packages_installed(problem_type=OCR)
    config_file = checkpoint_name + ".py"
    if isinstance(config_file, str):
        config = mmcv.Config.fromfile(config_file)

    # build model and load pretrained weights
    checkpoint = checkpoints[0]
    model = build_detector(config.model, test_cfg=config.get("test_cfg"))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")
    return config, model


def lookup_mmdet_config(key, config):
    if key in config:
        return config[key]
    for subconfig in config.values():
        if isinstance(subconfig, dict):
            result = lookup_mmdet_config(key, subconfig)
            if result is not None:
                return result
        elif isinstance(subconfig, list):
            for subsubconfig in subconfig:
                if isinstance(subsubconfig, dict):
                    result = lookup_mmdet_config(key, subsubconfig)
                    if result is not None:
                        return result
    return None


def update_mmdet_config(key, value, config):
    for k, subconfig in config.items():
        if key == k:
            config[k] = value
        elif isinstance(subconfig, dict):
            update_mmdet_config(key, value, subconfig)
        elif isinstance(subconfig, list):
            for subsubconfig in subconfig:
                if isinstance(subsubconfig, dict):
                    update_mmdet_config(key, value, subsubconfig)


def run_model(model: nn.Module, batch: dict, trt_model: Optional[nn.Module] = None):
    from ..utils.onnx import OnnxModule
    from .document_transformer import DocumentTransformer
    from .fusion.fusion_mlp import MultimodalFusionMLP
    from .hf_text import HFAutoModelForTextPrediction
    from .t_few import TFewModel
    from .timm_image import TimmAutoModelForImagePrediction

    supported_models = (
        TimmAutoModelForImagePrediction,
        HFAutoModelForTextPrediction,
        MultimodalFusionMLP,
        TFewModel,
        OnnxModule,
    )
    pure_model = model
    if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
        pure_model = model._orig_mod
    if isinstance(model, nn.DataParallel):
        pure_model = model.module
    if isinstance(pure_model, OnnxModule):
        for k in batch:
            # HACK input data types in ONNX
            if batch[k].dtype == torch.int32:
                batch[k] = batch[k].to(torch.int64)
    # DocumentTransformer inherited from HFAutoModelForTextPrediction
    if (not isinstance(pure_model, DocumentTransformer)) and isinstance(pure_model, supported_models):
        input_vec = [batch[k] for k in pure_model.input_keys]
        column_names, column_values = [], []
        for k in batch.keys():
            has_image_column_prefix = isinstance(pure_model, TimmAutoModelForImagePrediction) and k.startswith(
                pure_model.image_column_prefix
            )
            has_text_column_prefix = isinstance(pure_model, HFAutoModelForTextPrediction) and k.startswith(
                pure_model.text_column_prefix
            )
            if has_image_column_prefix or has_text_column_prefix:
                column_names.append(k)
                column_values.append(batch[k])
        if column_names != [] and column_values != []:
            input_vec.append(column_names)
            input_vec.append(column_values)
        if isinstance(pure_model, OnnxModule):
            # OnnxModule doesn't support multi-gpu yet
            output_vec = pure_model(*tuple(input_vec))
        else:
            output_vec = model(*tuple(input_vec))

        output = pure_model.get_output_dict(*output_vec)
    else:
        output = model(batch)
    return output


def freeze_model_layers(model, frozen_layers):
    """
    Freeze model layers with pattern in frozen_layers.

    Parameters
    ----------
    model
        The pytorch model.
    frozen_layers
        A list of substrings of frozen layers' names.

        e.g. if frozen_layers = ["backbone", "neck"],
            all layers including "backbone" or "neck" in the name will be frozen.
    """

    if not frozen_layers:
        return

    is_frozen_layer = lambda n: any(bb in n for bb in frozen_layers)

    for n, p in model.named_parameters():
        if is_frozen_layer(n):
            p.requires_grad = False


def get_pretrained_tokenizer(
    tokenizer_name: str,
    checkpoint_name: str,
    use_fast: Optional[bool] = True,
    add_prefix_space: Optional[bool] = None,
):
    """
    Load the tokenizer for a pre-trained huggingface checkpoint.

    Parameters
    ----------
    tokenizer_name
        The tokenizer type, e.g., "bert", "clip", "electra", and "hf_auto".
    checkpoint_name
        Name of a pre-trained checkpoint.
    use_fast
        Use a fast Rust-based tokenizer if it is supported for a given model.
        If a fast tokenizer is not available for a given model, a normal Python-based tokenizer is returned instead.
        See: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer.from_pretrained.use_fast

    Returns
    -------
    A tokenizer instance.
    """
    try:
        tokenizer_class = ALL_TOKENIZERS[tokenizer_name]
        if add_prefix_space is None:
            return tokenizer_class.from_pretrained(checkpoint_name, use_fast=use_fast)
        else:
            return tokenizer_class.from_pretrained(
                checkpoint_name, use_fast=use_fast, add_prefix_space=add_prefix_space
            )
    except TypeError as e:
        try:
            tokenizer = BertTokenizer.from_pretrained(checkpoint_name)  # nosec B615
            warnings.warn(
                f"Current checkpoint {checkpoint_name} does not support AutoTokenizer. "
                "Switch to BertTokenizer instead.",
                UserWarning,
            )
            return tokenizer
        except:
            raise e


def extract_value_from_config(
    config: Dict,
    keys: Tuple[str, ...],
):
    """
    Traverse a config dictionary to get some hyper-parameter's value.

    Parameters
    ----------
    config
        A config dictionary.
    keys
        The possible names of a hyper-parameter.

    Returns
    -------
    The hyper-parameter value.
    """
    result = []
    for k, v in config.items():
        if k in keys:
            result.append(v)
        elif isinstance(v, dict):
            result += extract_value_from_config(v, keys)
        else:
            pass

    return result


def extract_image_hparams_from_config(model_name: str, config):
    """
    Extract some default hyper-parameters, e.g., image size, mean, and std,
    from a pre-trained (timm or huggingface) checkpoint.

    Parameters
    ----------
    model_name
        Name of model.
    config
        Config of a pre-trained checkpoint.

    Returns
    -------
    image_size
        Image width/height.
    mean
        Image normalization mean.
    std
        Image normalizaiton std.
    """
    if model_name.lower().startswith((TIMM_IMAGE, META_TRANSFORMER)):
        image_size = config["input_size"][-1]
        image_mean = config["mean"]
        image_std = config["std"]
    elif model_name.lower().startswith((CLIP, DOCUMENT_TRANSFORMER)):
        extracted = extract_value_from_config(
            config=config.to_diff_dict(),
            keys=("image_size",),
        )
        if len(extracted) == 0:
            image_size = None
        elif len(extracted) >= 1:
            image_size = extracted[0]
            if isinstance(image_size, tuple):
                image_size = image_size[-1]
        else:
            raise ValueError(f" more than one image_size values are detected: {extracted}")
        image_mean = None
        image_std = None
    else:
        raise ValueError(f"Unknown image processor prefix: {model_name}")
    return image_size, image_mean, image_std


def image_mean_std(norm_type: str):
    """
    Get image normalization mean and std by its name.

    Parameters
    ----------
    norm_type
        Name of image normalization.

    Returns
    -------
    Normalization mean and std.
    """
    if norm_type == "inception":
        return IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    elif norm_type == "imagenet":
        return IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    elif norm_type == "clip":
        return CLIP_IMAGE_MEAN, CLIP_IMAGE_STD
    else:
        raise ValueError(f"unknown image normalization: {norm_type}")


def get_image_size_mean_std(
    model_name: str,
    config,
    provided_size: int,
    provided_norm_type: str,
    support_variable_input_size: Optional[bool] = False,
):
    image_size, image_mean, image_std = extract_image_hparams_from_config(
        model_name=model_name,
        config=config,
    )
    if support_variable_input_size and provided_size is not None:
        # We have detected that the model supports using an image size that is
        # different from the pretrained model, e.g., ConvNets with global pooling
        if provided_size < image_size:
            logger.warning(
                f"The provided image size={provided_size} is smaller than the default size "
                f"of the pretrained backbone, which is {image_size}. "
                f"Detailed configuration of the backbone is in {config}. "
                f"You may like to double check your configuration."
            )
        image_size = provided_size
    elif provided_size is not None and provided_size != image_size:
        logger.warning(
            f"The model does not support using an image size that is different from the default size. "
            f"Provided image size={provided_size}. Default size={image_size}. "
            f"Detailed model configuration={config}. We have ignored the provided image size."
        )

    if image_size is None:
        if provided_size is not None:
            image_size = provided_size
            logger.debug(f"using provided image size: {image_size}.")
        else:
            raise ValueError("image size is missing.")
    else:
        logger.debug(f"using detected image size: {image_size}")

    if image_mean is None or image_std is None:
        if provided_norm_type is not None:
            image_mean, image_std = image_mean_std(provided_norm_type)
            logger.debug(f"using provided normalization: {provided_norm_type}.")
        else:
            raise ValueError("image normalization mean and std are missing.")
    else:
        logger.debug(f"using detected image normalization: {image_mean} and {image_std}.")

    return image_size, image_mean, image_std


def get_text_segment_num(config, provided_segment_num: int, checkpoint_name: str):
    extracted = extract_value_from_config(config=config.to_diff_dict(), keys=("type_vocab_size",))
    if len(extracted) == 0:
        default_segment_num = 1
    elif len(extracted) == 1:
        default_segment_num = extracted[0]
    else:
        raise ValueError(f" more than one type_vocab_size values are detected: {extracted}")

    if default_segment_num <= 0:
        default_segment_num = 1

    if provided_segment_num < default_segment_num:
        warnings.warn(
            f"provided text_segment_num: {provided_segment_num} "
            f"is smaller than {checkpoint_name}'s default: {default_segment_num}"
        )
    text_segment_num = min(provided_segment_num, default_segment_num)
    assert text_segment_num >= 1
    logger.debug(f"text segment num: {text_segment_num}")

    return text_segment_num


def get_text_token_max_len(provided_max_len, config, tokenizer, checkpoint_name):
    """
    Compute the allowable max length of token sequences.

    Parameters
    ----------
    provided_max_len
        The provided max length.
    config
        Model config.
    tokenizer
        Text tokenizer.
    checkpoint_name
        Name of checkpoint.

    Returns
    -------
    Token sequence max length.
    """
    if hasattr(config, "relative_attention") and config.relative_attention:
        default_max_len = tokenizer.model_max_length
    elif hasattr(config, "position_embedding_type") and "relative" in config.position_embedding_type:
        default_max_len = tokenizer.model_max_length
    elif hasattr(config, "max_position_embeddings"):
        default_max_len = config.max_position_embeddings
    else:
        default_max_len = tokenizer.model_max_length

    if provided_max_len is None or provided_max_len <= 0:
        max_len = default_max_len
    else:
        if provided_max_len < default_max_len:
            if default_max_len < 10**6:  # Larger than this value usually means infinite.
                warnings.warn(
                    f"provided max length: {provided_max_len} "
                    f"is smaller than {checkpoint_name}'s default: {default_max_len}"
                )
        max_len = min(provided_max_len, default_max_len)

    logger.debug(f"text max length: {max_len}")

    return max_len


def replace_missing_images_with_learnable(
    images: torch.Tensor,
    image_masks,
    learnable_image: nn.Parameter,
):
    b, n, c, h, w = images.shape
    assert learnable_image.shape == (c, h, w)
    for i in range(b):
        for j in range(n):
            if not image_masks[i][j]:  # False means a missing image
                images[i][j] = learnable_image

    return images


def select_model(
    config: DictConfig,
    df_preprocessor,
    strict: Optional[bool] = True,
):
    """
    Filter model config through the detected modalities in the training data.
    If MultiModalFeaturePreprocessor can't detect some modality,
    this function will remove the models that use this modality. This function is to
    maximize the user flexibility in defining the config.
    For example, if one uses the default, including hf_text and timm_image, as the model config template
    but the training data don't have images, this function will filter out timm_image.

    Parameters
    ----------
    config
        A DictConfig object. The model config should be accessible by "config.model"
    df_preprocessor
        A MultiModalFeaturePreprocessor object, which has called .fit() on the training data.
        Column names of the same modality are grouped into one list. If a modality's list is empty,
        it means the training data don't have this modality.
    strict
        If False, allow retaining one model when partial modalities are available for that model.

    Returns
    -------
    Config with some unused models removed.
    """
    data_status = {}
    for per_modality in ALL_MODALITIES:
        data_status[per_modality] = False
    if len(df_preprocessor.image_feature_names) > 0:
        data_status[IMAGE] = True
    if len(df_preprocessor.text_feature_names) > 0:
        data_status[TEXT] = True
    if len(df_preprocessor.categorical_feature_names) > 0:
        data_status[CATEGORICAL] = True
    if len(df_preprocessor.numerical_feature_names) > 0:
        data_status[NUMERICAL] = True
    if len(df_preprocessor.ner_feature_names) > 0:
        data_status[TEXT_NER] = True
    if len(df_preprocessor.document_feature_names) > 0:
        data_status[DOCUMENT] = True
    if len(df_preprocessor.semantic_segmentation_feature_names) > 0:
        data_status[SEMANTIC_SEGMENTATION_IMG] = True

    names = config.model.names
    if isinstance(names, str):
        names = [names]
    selected_model_names = []
    fusion_model_name = []
    for model_name in names:
        model_config = getattr(config.model, model_name)
        strict = getattr(model_config, "requires_all_dtypes", strict)
        if not model_config.data_types:
            fusion_model_name.append(model_name)
            continue
        model_data_status = [data_status[d_type] for d_type in model_config.data_types]
        if all(model_data_status):
            selected_model_names.append(model_name)
        else:
            if any(model_data_status) and not strict:
                selected_model_names.append(model_name)
                # update data types to be consistent with detected
                model_config.data_types = [d_type for d_type in model_config.data_types if data_status[d_type]]
            else:
                delattr(config.model, model_name)

    if len(selected_model_names) == 0:
        raise ValueError("No model is available for this dataset.")
    # only allow no more than 1 fusion model
    if len(fusion_model_name) > 1:
        raise ValueError(f"More than one fusion models `{fusion_model_name}` are detected, but only one is allowed.")

    if len(selected_model_names) > 1:
        assert len(fusion_model_name) == 1
        selected_model_names.extend(fusion_model_name)
    elif len(fusion_model_name) == 1 and hasattr(config.model, fusion_model_name[0]):
        delattr(config.model, fusion_model_name[0])

    config.model.names = selected_model_names
    logger.debug(f"selected models: {selected_model_names}")
    for model_name in selected_model_names:
        logger.debug(f"model dtypes: {getattr(config.model, model_name).data_types}")

    # clean up unused model configs
    model_keys = list(config.model.keys())
    for model_name in model_keys:
        if model_name not in selected_model_names + ["names"]:
            delattr(config.model, model_name)

    return config


def create_model(
    model_name: str,
    model_config: DictConfig,
    num_classes: Optional[int] = 0,
    classes: Optional[list] = None,
    num_numerical_columns: Optional[int] = None,
    num_categories: Optional[Dict] = None,
    numerical_fill_values: Optional[Dict] = None,
    pretrained: Optional[bool] = True,
    is_matching: Optional[bool] = False,
):
    """
    Create a single model.

    Parameters
    ----------
    model_name
        Name of the model.
    model_config
        Config of the model.
    num_classes
        The class number for a classification task. It should be 1 for a regression task.
    classes
        All classes in this dataset.
    num_numerical_columns
        The number of numerical columns in the training dataframe.
    num_categories
        The category number for each categorical column in the training dataframe.
    numerical_fill_values
        If numerical values are null, fill them with these.
    pretrained
        Whether using the pretrained timm models. If pretrained=True, download the pretrained model.
    is_matching
        Whether the model is used for semantic matching.

    Returns
    -------
    A model.
    """
    if model_name.lower().startswith(CLIP):
        from .clip import CLIPForImageText

        model = CLIPForImageText(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
            has_image=IMAGE in model_config.data_types,
            has_text=TEXT in model_config.data_types,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
            image_chan_num=model_config.image_chan_num,
            use_learnable_image=model_config.use_learnable_image,
            max_text_len=model_config.max_text_len,
            text_segment_num=model_config.text_segment_num,
            is_matching=is_matching,
        )
    elif model_name.lower().startswith(TIMM_IMAGE):
        from .timm_image import TimmAutoModelForImagePrediction

        model = TimmAutoModelForImagePrediction(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            mix_choice=model_config.mix_choice,
            pretrained=pretrained,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
            image_chan_num=model_config.image_chan_num,
            use_learnable_image=model_config.use_learnable_image,
        )
    elif model_name.lower().startswith(HF_TEXT):
        from .hf_text import HFAutoModelForTextPrediction

        model = HFAutoModelForTextPrediction(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pooling_mode=model_config.pooling_mode,
            gradient_checkpointing=model_config.gradient_checkpointing,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
            max_text_len=model_config.max_text_len,
            text_segment_num=model_config.text_segment_num,
            use_fast=model_config.use_fast,
        )
    elif model_name.lower().startswith(T_FEW):
        from .t_few import TFewModel

        model = TFewModel(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            length_norm=model_config.length_norm,  # Normalizes length to adjust for length bias in target template
            unlikely_loss=model_config.unlikely_loss,  # Adds loss term that lowers probability of incorrect outputs
            mc_loss=model_config.mc_loss,  # Adds multiple choice cross entropy loss
            num_classes=num_classes,
            gradient_checkpointing=model_config.gradient_checkpointing,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
            max_text_len=model_config.max_text_len,
            text_segment_num=model_config.text_segment_num,
        )
    elif model_name.lower().startswith(NUMERICAL_MLP):
        from .numerical_mlp import NumericalMLP

        model = NumericalMLP(
            prefix=model_name,
            in_features=num_numerical_columns,
            hidden_features=model_config.hidden_size,
            out_features=model_config.hidden_size,
            num_layers=model_config.num_layers,
            activation=model_config.activation,
            dropout=model_config.dropout,
            normalization=model_config.normalization,
            token_dim=model_config.token_dim,
            embedding_arch=model_config.embedding_arch,
            num_classes=num_classes,
            numerical_fill_values=numerical_fill_values,
        )
    elif model_name.lower().startswith(CATEGORICAL_MLP):
        from .categorical_mlp import CategoricalMLP

        model = CategoricalMLP(
            prefix=model_name,
            num_categories=num_categories,
            out_features=model_config.hidden_size,
            num_layers=model_config.num_layers,
            activation=model_config.activation,
            dropout=model_config.dropout,
            normalization=model_config.normalization,
            num_classes=num_classes,
        )
    elif model_name.lower().startswith(DOCUMENT_TRANSFORMER):
        from .document_transformer import DocumentTransformer

        model = DocumentTransformer(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pooling_mode=model_config.pooling_mode,
            gradient_checkpointing=model_config.gradient_checkpointing,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
        )
    elif model_name.lower().startswith(MMDET_IMAGE):
        from .mmdet_image import MMDetAutoModelForObjectDetection

        model = MMDetAutoModelForObjectDetection(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            config_file=model_config.config_file,
            classes=classes,
            pretrained=pretrained,
            output_bbox_format=model_config.output_bbox_format,
            frozen_layers=model_config.frozen_layers,
        )
    elif model_name.lower().startswith(MMOCR_TEXT_DET):
        from .mmocr_text_detection import MMOCRAutoModelForTextDetection

        model = MMOCRAutoModelForTextDetection(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
        )
    elif model_name.lower().startswith(MMOCR_TEXT_RECOG):
        from .mmocr_text_recognition import MMOCRAutoModelForTextRecognition

        model = MMOCRAutoModelForTextRecognition(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
        )
    elif model_name.lower().startswith(NER_TEXT):
        from .ner_text import HFAutoModelForNER

        model = HFAutoModelForNER(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            gradient_checkpointing=model_config.gradient_checkpointing,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
        )
    elif model_name.lower().startswith(FUSION_MLP):
        from .fusion import MultimodalFusionMLP

        model = functools.partial(
            MultimodalFusionMLP,
            prefix=model_name,
            hidden_features=model_config.hidden_sizes,
            num_classes=num_classes,
            adapt_in_features=model_config.adapt_in_features,
            activation=model_config.activation,
            dropout=model_config.dropout,
            normalization=model_config.normalization,
            aux_loss_weight=model_config.aux_loss_weight,
        )
    elif model_name.lower().startswith(FUSION_NER):
        from .fusion import MultimodalFusionNER

        model = functools.partial(
            MultimodalFusionNER,
            prefix=model_name,
            hidden_features=model_config.hidden_sizes,
            num_classes=num_classes,
            adapt_in_features=model_config.adapt_in_features,
            activation=model_config.activation,
            dropout_prob=model_config.drop_rate,
            normalization=model_config.normalization,
            loss_weight=model_config.weight if hasattr(model_config, "weight") else None,
        )
    elif model_name.lower().startswith(FUSION_TRANSFORMER):
        from .fusion import MultimodalFusionTransformer

        model = functools.partial(
            MultimodalFusionTransformer,
            prefix=model_name,
            hidden_features=model_config.hidden_size,
            num_classes=num_classes,
            num_blocks=model_config.num_blocks,
            attention_num_heads=model_config.attention_num_heads,
            ffn_hidden_size=model_config.ffn_hidden_size,
            attention_dropout=model_config.attention_dropout,
            residual_dropout=model_config.residual_dropout,
            ffn_dropout=model_config.ffn_dropout,
            attention_normalization=model_config.normalization,
            ffn_normalization=model_config.normalization,
            head_normalization=model_config.normalization,
            ffn_activation=model_config.ffn_activation,
            head_activation=model_config.head_activation,
            adapt_in_features=model_config.adapt_in_features,
            aux_loss_weight=model_config.aux_loss_weight,
            additive_attention=model_config.additive_attention,
            share_qv_weights=model_config.share_qv_weights,
        )
    elif model_name.lower().startswith(FT_TRANSFORMER):
        from .ft_transformer import FT_Transformer

        model = FT_Transformer(
            prefix=model_name,
            num_numerical_columns=num_numerical_columns,
            num_categories=num_categories,
            numerical_fill_values=numerical_fill_values,
            embedding_arch=model_config.embedding_arch,
            token_dim=model_config.token_dim,
            hidden_size=model_config.hidden_size,
            hidden_features=model_config.hidden_size,
            num_classes=num_classes,
            num_blocks=model_config.num_blocks,
            attention_num_heads=model_config.attention_num_heads,
            attention_dropout=model_config.attention_dropout,
            attention_normalization=model_config.normalization,
            ffn_hidden_size=model_config.ffn_hidden_size,
            ffn_dropout=model_config.ffn_dropout,
            ffn_normalization=model_config.normalization,
            ffn_activation=model_config.ffn_activation,
            residual_dropout=model_config.residual_dropout,
            head_normalization=model_config.normalization,
            head_activation=model_config.head_activation,
            additive_attention=model_config.additive_attention,
            share_qv_weights=model_config.share_qv_weights,
            pooling_mode=model_config.pooling_mode,
            checkpoint_name=model_config.checkpoint_name,
            pretrained=pretrained,
        )
    elif model_name.lower().startswith(SAM):
        from .sam import SAMForSemanticSegmentation

        model = SAMForSemanticSegmentation(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pretrained=pretrained,
            frozen_layers=model_config.frozen_layers,
            num_mask_tokens=model_config.num_mask_tokens,
            image_norm=model_config.image_norm,
        )
    elif model_name.lower().startswith(NNUNET):
        from .nnunet import ResEncUNetForSemanticSegmentation

        model = ResEncUNetForSemanticSegmentation(
            prefix=model_name,
            num_classes=num_classes,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
            pretrained=pretrained,
            n_stages=model_config.n_stages,
            features_per_stage=tuple(model_config.features_per_stage),
            n_blocks_per_stage=tuple(model_config.n_blocks_per_stage),
            n_conv_per_stage_decoder=tuple(model_config.n_conv_per_stage_decoder),
            deep_supervision=model_config.deep_supervision,
        )
    elif model_name.lower().startswith(META_TRANSFORMER):
        from .meta_transformer import MetaTransformer

        model = MetaTransformer(
            prefix=model_name,
            checkpoint_path=model_config.checkpoint_path,
            num_classes=num_classes,
            model_version=model_config.model_version,
            has_image=IMAGE in model_config.data_types,
            has_text=TEXT in model_config.data_types,
            num_numerical_columns=num_numerical_columns,
            num_categories=num_categories,
            numerical_fill_values=numerical_fill_values,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
            image_chan_num=model_config.image_chan_num,
            use_learnable_image=model_config.use_learnable_image,
            max_text_len=model_config.max_text_len,
            text_segment_num=model_config.text_segment_num,
        )
    else:
        raise ValueError(f"unknown model name: {model_name}")

    return model


def create_fusion_model(
    config: DictConfig,
    num_classes: Optional[int] = None,
    classes: Optional[list] = None,
    num_numerical_columns: Optional[int] = None,
    num_categories: Optional[Dict] = None,
    numerical_fill_values: Optional[Dict] = None,
    pretrained: Optional[bool] = True,
):
    """
    Create models. It supports the auto models of huggingface text and timm image.
    Multimodal models, e.g., CLIP, should be added case-by-case since their configs and usages
    may be different. It uses MLP for the numerical features, categorical features, and late-fusion.

    Parameters
    ----------
    config
        A DictConfig object. The model config should be accessible by "config.model".
    num_classes
        The class number for a classification task. It should be 1 for a regression task.
    classes
        All classes in this dataset.
    num_numerical_columns
        The number of numerical columns in the training dataframe.
    num_categories
        The category number for each categorical column in the training dataframe.
    numerical_fill_values
        If numerical values are null, fill them with these.
    pretrained
        Whether using the pretrained timm models. If pretrained=True, download the pretrained model.

    Returns
    -------
    A Pytorch model.
    """
    names = config.model.names
    if isinstance(names, str):
        names = [names]
    # make sure no duplicate model names
    assert len(names) == len(set(names))
    logger.debug(f"output_shape: {num_classes}")
    names = sorted(names)
    config.model.names = names
    single_models = []
    fusion_model = None

    for model_name in names:
        model_config = getattr(config.model, model_name)
        model = create_model(
            model_name=model_name,
            model_config=model_config,
            num_classes=num_classes,
            classes=classes,
            num_numerical_columns=num_numerical_columns,
            num_categories=num_categories,
            numerical_fill_values=numerical_fill_values,
            pretrained=pretrained,
        )

        if isinstance(model, functools.partial):  # fusion model
            if fusion_model is None:
                fusion_model = model
            else:
                raise ValueError(
                    f"More than one fusion models are detected in {names}. Only one fusion model is allowed."
                )
        else:  # single model
            if config.optim.peft is not None:
                model = apply_peft_adaptation(model, config)
            single_models.append(model)

    if len(single_models) > 1:
        # must have one fusion model if there are multiple independent models
        model = fusion_model(models=single_models)
    elif len(single_models) == 1:
        model = single_models[0]
    else:
        raise ValueError(f"No available models for {names}")

    # build augmenter for multimodal data augmentation
    if config.optim.lemda.turn_on:
        from .fusion import MultimodalFusionMLP

        assert isinstance(model, MultimodalFusionMLP)
        from .augmenter import Augmenter

        augmenter = Augmenter(
            arch_type=config.optim.lemda.arch_type,
            input_dim=model.augmenter_in_features,
            z_dim=config.optim.lemda.z_dim,
            num_layers=config.optim.lemda.num_layers,
            adv_weight=config.optim.lemda.adv_weight,
        )
        model.augmenter = augmenter

    return model


def apply_peft_adaptation(model: nn.Module, config: DictConfig) -> nn.Module:
    """
    Apply an adaptation to the model for efficient fine-tuning.

    Parameters
    ----------
    model
        A PyTorch model.
    config:
        A DictConfig object. The optimization config should be accessible by "config.optimization".
    """
    if config.optim.peft in PEFT_ADDITIVE_STRATEGIES:
        # 先注入 Linear（现有逻辑，SAM/Transformer 用）
        model = inject_adaptation_to_linear_layer(
            model=model,
            peft=config.optim.peft,
            lora_r=config.optim.lora.r,
            lora_alpha=config.optim.lora.alpha,
            module_filter=config.optim.lora.module_filter,
            filter=config.optim.lora.filter,
            extra_trainable_params=config.optim.extra_trainable_params,
            conv_lora_expert_num=config.optim.lora.conv_lora_expert_num,
        )
        # 再注入 Conv2d（新增逻辑，nnU-Net 等卷积网络用）
        model = inject_adaptation_to_conv2d_layer(
            model=model,
            peft=config.optim.peft,
            lora_r=config.optim.lora.r,
            lora_alpha=config.optim.lora.alpha,
            module_filter=config.optim.lora.module_filter,
            filter=config.optim.lora.filter,
            extra_trainable_params=config.optim.extra_trainable_params,
            conv_lora_expert_num=config.optim.lora.conv_lora_expert_num,
        )
        model.name_to_id = model.get_layer_ids()  # Need to update name to id dictionary.

    return model


def modify_duplicate_model_names(
    learner,
    postfix: str,
    blacklist: List[str],
):
    """
    Modify a learner's model names if they exist in a blacklist.

    Parameters
    ----------
    learner
        A BaseLearner object.
    postfix
        The postfix used to change the duplicate names.
    blacklist
        A list of names. The provided learner can't use model names in the list.

    Returns
    -------
    The learner guaranteed has no duplicate model names with the blacklist names.
    """
    model_names = []
    for n in learner._config.model.names:
        if n in blacklist:
            new_name = f"{n}_{postfix}"
            assert new_name not in blacklist
            assert new_name not in learner._config.model.names
            # modify model prefix
            if n == learner._model.prefix:
                learner._model.prefix = new_name
            else:
                assert isinstance(learner._model.model, nn.ModuleList)
                for per_model in learner._model.model:
                    if n == per_model.prefix:
                        per_model.prefix = new_name
                        break
            # modify data processor prefix
            for per_modality_processors in learner._data_processors.values():
                for per_processor in per_modality_processors:
                    if n == per_processor.prefix:
                        per_processor.prefix = new_name
            # modify model config keys
            setattr(learner._config.model, new_name, getattr(learner._config.model, n))
            delattr(learner._config.model, n)

            model_names.append(new_name)
        else:
            model_names.append(n)

    learner._config.model.names = model_names

    return learner


def list_timm_models(pretrained=True):
    return timm.list_models(pretrained=pretrained)


def is_lazy_weight_tensor(p: torch.Tensor) -> bool:
    from torch.nn.parameter import UninitializedParameter

    if isinstance(p, UninitializedParameter):
        warnings.warn(
            "A layer with UninitializedParameter was found. "
            "Thus, the total number of parameters detected may be inaccurate."
        )
        return True
    return False

```

```yaml
model:
  names:
  categorical_mlp:
    hidden_size: 64
    activation: "leaky_relu"
    num_layers: 1
    dropout: 0.1
    normalization: "layer_norm"
    data_types:
      - "categorical"

  numerical_mlp:
    hidden_size: 128
    activation: "leaky_relu"
    num_layers: 1
    dropout: 0.1
    normalization: "layer_norm"
    token_dim: 8
    embedding_arch:
    data_types:
      - "numerical"
    merge: "concat"

  hf_text:
    checkpoint_name: "google/electra-base-discriminator"
    gradient_checkpointing: False
    pooling_mode: 'cls'  # The pooling mode, can be 'cls' or 'mean'
    data_types:
      - "text"
    tokenizer_name: "hf_auto"
    use_fast: True  # Use a fast Rust-based tokenizer if it is supported for a given model. If a fast tokenizer is not available for a given model, a normal Python-based tokenizer is returned instead.
    max_text_len: 512  # If None or <=0, then use the max length of pretrained models.
    insert_sep: True
    low_cpu_mem_usage: False
    text_segment_num: 2
    stochastic_chunk: False
    text_aug_detect_length: 10                # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.1            # augmentation magnitude randomly drawn from [0, text_trivial_aug_maxscale]
    text_train_augment_types:        # specify augmentation space manually, will randomly select one from the following and identity
      # - "random_swap(0.05)"          # less than 0.1 based on eda paper
      # - "random_delete(0.05)"        # less than 0.1 based on eda paper
      # - "syn_replacement(0.05)"  # less than 0.1 based on eda paper
      # - "insert_punc(0.05)"

  ner_text:
    checkpoint_name: "bert-base-cased"
    max_text_len: 512
    gradient_checkpointing: False
    low_cpu_mem_usage: False
    data_types:
      - "text_ner"
    tokenizer_name: "hf_auto"
    insert_sep: False
    text_segment_num: 2
    stochastic_chunk: False
    special_tags: 
      - X # CLS, SEP, and non-first tokens of a word will be labelled as X
      - O # Outside of a named entity

  document_transformer:
    checkpoint_name: "microsoft/layoutlmv3-base" # document foundation models
    gradient_checkpointing: False
    pooling_mode: 'cls'  # The pooling mode, can be 'cls' or 'mean'
    data_types:
      - "document"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: 224
    tokenizer_name: "hf_auto"
    max_text_len: 512  # If None or <=0, then use the max length of pretrained models.
    insert_sep: True
    low_cpu_mem_usage: False
    text_segment_num: 2
    stochastic_chunk: False
    text_aug_detect_length: 10                # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.0            # augmentation magnitude randomly drawn from [0, text_trivial_aug_maxscale]

  t_few:
    checkpoint_name: "t5-small" #"bigscience/T0_3B"
    gradient_checkpointing: False
    data_types:
      - "text"
    tokenizer_name: "hf_auto"
    length_norm: 1.0 # Normalizes length to adjust for length bias in target template
    unlikely_loss: 1.0 # Adds loss term that lowers probability of incorrect outputs
    mc_loss: 1.0 # Adds multiple choice cross entropy loss
    max_text_len: 512  # If None or <=0, then use the max length of pretrained models.
    text_segment_num: 2
    insert_sep: True
    low_cpu_mem_usage: False
    stochastic_chunk: False
    text_aug_detect_length: 10                # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.0            # augmentation magnititude randomly drawn from [0, text_trivial_aug_maxscale]
    text_train_augment_types:

  timm_image:
    checkpoint_name: "swin_base_patch4_window7_224"
    mix_choice: "all_logits"
    data_types:
      - "image"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: null
    image_chan_num: 3
    use_learnable_image: False
    max_image_num_per_column: 1

  mmdet_image:
    checkpoint_name: "yolov3_mobilenetv2_8xb24-320-300e_coco"
    config_file: ""
    data_types:
      - "image"
    max_img_num_per_col: 1
    output_bbox_format: "xyxy"  # now support xyxy or xywh, for bbox format details see https://keras.io/api/keras_cv/bounding_box/formats/
    frozen_layers: null
    coco_root: null

  mmocr_text_detection:
    checkpoint_name: "TextSnake"
    data_types:
      - "image"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: 224
    max_img_num_per_col: 2

  mmocr_text_recognition:
    checkpoint_name: "ABINet"
    data_types:
      - "image"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: 224
    max_img_num_per_col: 2

  clip:
    checkpoint_name: "openai/clip-vit-base-patch32"
    data_types:
      - "image"
      - "text"
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "clip"
    image_size: 224
    image_chan_num: 3
    use_learnable_image: False
    max_image_num_per_column: 1
    tokenizer_name: "clip"
    max_text_len: 77  # The maximum possible length.
    insert_sep: False
    text_segment_num: 1
    stochastic_chunk: False
    text_aug_detect_length: 10                     # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.0                      # scale randomly drawn from [0, text_trivial_aug_maxscale]
    text_train_augment_types:        # specify augmentation space manually, will randomly select one from the following and identity
      # - "random_swap(0.05)"          # less than 0.1 based on eda paper
      # - "random_delete(0.05)"        # less than 0.1 based on eda paper
      # - "syn_replacement(0.05)"  # less than 0.1 based on eda paper
      # - "insert_punc(0.05)"

  fusion_mlp:
    aux_loss_weight:
    adapt_in_features: "max"
    hidden_sizes:
      - 128
    activation: "leaky_relu"
    dropout: 0.1
    normalization: "layer_norm"
    data_types:

  fusion_ner:
    weight:
    adapt_in_features: "max"
    hidden_sizes:
      - 128
    activation: "leaky_relu"
    drop_rate: 0.1
    normalization: "layer_norm"
    data_types:

  fusion_transformer:
    aux_loss_weight:
    hidden_size: 192
    num_blocks: 3
    attention_num_heads: 8
    adapt_in_features: "max"
    attention_dropout: 0.2
    residual_dropout: 0.0
    ffn_dropout: 0.1
    ffn_hidden_size: 192
    normalization: "layer_norm"
    ffn_activation: "geglu"
    head_activation: "relu"
    data_types:
    additive_attention: False # Whether to use lightweight additive attention, can be True, False or "auto"
    share_qv_weights: False # Whether to share weight for query and value, can be True, False or "auto"

  ft_transformer:
    data_types:
      - "categorical"
      - "numerical"
    embedding_arch:
      - 'linear'
    token_dim: 192
    hidden_size: 192
    num_blocks: 3
    attention_num_heads: 8
    attention_dropout: 0.2
    residual_dropout: 0.0
    ffn_dropout: 0.1
    ffn_hidden_size: 192
    ffn_activation: "geglu"
    head_activation: "relu"
    normalization: "layer_norm"
    merge: "concat"
    requires_all_dtypes: False
    additive_attention: False # Whether to use lightweight additive attention, can be True, False or "auto"
    share_qv_weights: False # Whether to share weight for query and value, can be True, False or "auto"
    pooling_mode: "cls"
    checkpoint_name: null

  sam:
    checkpoint_name: "facebook/sam-vit-huge"
    image_norm: "imagenet"
    data_types:
      - "semantic_segmentation_img"
    train_transforms:
      - "random_horizontal_flip"
    val_transforms: []
    img_transforms:
      - "resize_to_square"
    gt_transforms:
      - "resize_gt_to_square"
    max_img_num_per_col: 1
    frozen_layers: ["mask_decoder.iou_prediction_head", "prompt_encoder"]
    num_mask_tokens: 1
    ignore_label: 255

  nnunet:
    data_types:
      - "semantic_segmentation_img"
    image_norm: "imagenet"
    image_size: 512
    # ResEnc UNet
    n_stages: 7
    features_per_stage: [32, 64, 128, 256, 512, 512, 512]
    n_blocks_per_stage: [1, 3, 4, 6, 6, 6, 6]
    n_conv_per_stage_decoder: [1, 1, 1, 1, 1, 1]
    deep_supervision: False

  meta_transformer:
    data_types:
      - "image"
      - "text"
      - "categorical"
      - "numerical"
    checkpoint_path: null
    model_version: "base"
    requires_all_dtypes: False
    train_transforms:
      - "resize_shorter_side"
      - "center_crop"
      - "trivial_augment"
    val_transforms:
      - "resize_shorter_side"
      - "center_crop"
    image_norm: "imagenet"
    image_size: 224
    image_chan_num: 3
    use_learnable_image: False
    max_image_num_per_column: 1
    tokenizer_name: "hf_auto"
    max_text_len: 512  # If None or <=0, then use the max length of pretrained models.
    insert_sep: True
    text_segment_num: 2
    stochastic_chunk: False
    text_aug_detect_length: 10     # We perform text augmentation only if a text has more than text_detection_length words. It is used to differentiate text columns versus tabular columns that are treated as text.
    text_trivial_aug_maxscale: 0.1  # augmentation magnitude randomly drawn from [0, text_trivial_aug_maxscale]
    text_train_augment_types:
    merge: "concat"

```

```python
import argparse
import os

import pandas as pd

from autogluon.multimodal import MultiModalPredictor


def get_default_training_setting(dataset_name):
    validation_metric = "iou"
    loss = "structure_loss"
    max_epoch = 30
    lr = 1e-4

    if dataset_name == "SBU-shadow":
        validation_metric = "ber"
        loss = "balanced_bce"
        max_epoch = 10

    elif dataset_name == "polyp":
        validation_metric = "sm"

    elif dataset_name == "camo_sem_seg":
        validation_metric = "sm"
        max_epoch = 20

    elif dataset_name == "road_segmentation":
        validation_metric = "iou"
        max_epoch = 20
        lr = 3e-4

    elif dataset_name == "leaf_disease_segmentation":
        validation_metric = "iou"
        lr = 3e-4

    return validation_metric, loss, max_epoch, lr


def expand_path(df, dataset_dir):
    for col in ["image", "label"]:
        df[col] = df[col].apply(lambda ele: os.path.join(dataset_dir, ele))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script support converting voc format xmls to coco format json")
    parser.add_argument(
        "--task",
        type=str,
        default="leaf_disease_segmentation",
        choices=["polyp", "leaf_disease_segmentation", "camo_sem_seg", "isic2017", "road_segmentation", "SBU-shadow"],
    )
    parser.add_argument("--seed", type=int, default=42686693)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--expert_num", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--ckpt_path", type=str, default="outputs", help="Checkpoint path.")
    parser.add_argument("--per_gpu_batch_size", type=int, default=1, help="The batch size for each GPU.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="The effective batch size. If batch_size > per_gpu_batch_size * num_gpus, gradient accumulation would be used.",
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--backend",
        type=str,
        default="sam",
        choices=["sam", "nnunet"],
        help="选择语义分割后端：sam（默认）或 nnunet（ResEnc UNet）。",
    )
    args = parser.parse_args()

    dataset_name = args.task
    dataset_dir = os.path.join(f"datasets/{dataset_name}", dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare dataframes
    train_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"train.csv")), dataset_dir)
    val_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"val.csv")), dataset_dir)

    # get the validation metric
    validation_metric, loss, max_epoch, lr = get_default_training_setting(dataset_name)

    hyperparameters = {}
    hyperparameters.update(
        {
            "optim.lora.r": args.rank,
            "optim.peft": "conv_lora",
            "optim.lora.conv_lora_expert_num": args.expert_num,
            "env.num_gpus": args.num_gpus,
            "optim.loss_func": loss,
            "optim.max_epochs": max_epoch,
            "optim.lr": lr,
            "env.per_gpu_batch_size": args.per_gpu_batch_size,
            "env.batch_size": args.batch_size,
        }
    )

    # 选择模型后端
    if args.backend == "nnunet":
        hyperparameters.update(
            {
                "model.names": ["nnunet"],
                "model.nnunet.image_size": 512,
                "model.nnunet.image_norm": "imagenet",
            }
        )
    else:
        hyperparameters.update({"model.names": ["sam"]})

    if args.eval:  # load a checkpoint for evaluation
        predictor = MultiModalPredictor.load(args.ckpt_path)
    else:  # training
        predictor = MultiModalPredictor(
            problem_type="semantic_segmentation",
            validation_metric=validation_metric,
            eval_metric=validation_metric,
            hyperparameters=hyperparameters,
            label="label",
        )
        predictor.fit(train_data=train_df, tuning_data=val_df, seed=args.seed)

    # evaluation
    metric_file = os.path.join(args.output_dir, "metrics.txt")
    f = open(metric_file, "a")
    if dataset_name in ["isic2017", "SBU-shadow", "road_segmentation", "leaf_disease_segmentation"]:
        test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test.csv")), dataset_dir)
        if dataset_name == "SBU-shadow":
            eval_metrics = ["ber"]
        else:
            eval_metrics = ["iou"]

        res = predictor.evaluate(test_df, metrics=eval_metrics)
        print(f"Evaluation results for test dataset {dataset_name}: ", res)
        f.write(f"Evaluation results for test dataset {dataset_name}: {res} \n")
    elif dataset_name in ["polyp", "camo_sem_seg"]:
        if dataset_name == "polyp":
            test_datasets = ["CVC-ClinicDB", "Kvasir"]
        elif dataset_name == "camo_sem_seg":
            test_datasets = ["CAMO"]
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}.")
        for per_dataset in test_datasets:
            test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test_{per_dataset}.csv")), dataset_dir)
            res = predictor.evaluate(test_df, metrics=["sm", "fm", "em", "mae"])
            print(f"Evaluation results for test dataset {per_dataset}: ", res)
            f.write(f"Evaluation results for test dataset {per_dataset}: {res} \n")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}.")

    f.close()

```

我刚才用系统 Python 验证导入失败（因为 `autogluon.multimodal` 在当前 shell 没装到 system site-packages）；我会改用你正在用的 `conv-lora` 环境里的 Python 来做一次快速 import sanity check，然后再跑一个最小前向（随机张量）确认输出维度和 `MOE_LOSS` 线路没断。

定位到了问题：`dynamic_network_architectures` 当前版本里 `ResidualEncoderUNet` 的正确路径是 `dynamic_network_architectures.architectures.unet`，不是 `...residual_unet`。我现在修正 `nnunet` wrapper 的 import，然后再重跑最小前向，确认 `ConvLoRAConv2d` 注入与 `MOE_LOSS` 输出链路都正常。

```python
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..constants import COLUMN, IMAGE, LABEL, LOGITS, MOE_LOSS
from .adaptation_layers import ConvLoRAConv2d
from .utils import image_mean_std

logger = logging.getLogger(__name__)


class ResEncUNetForSemanticSegmentation(nn.Module):
    """
    ResEnc U-Net（dynamic-network-architectures）包装成 AutoGluon semantic_segmentation 的统一接口。

    目标：
    - 走 AutoGluon 的 data processor（resize/normalize）与 Lightning 训练循环
    - 允许通过 optim.peft=conv_lora 注入 ConvLoRAConv2d，并把 MOE_LOSS 透传到训练 loss
    - 输出二值 logits（shape: [B, 1, H, W]）
    """

    def __init__(
        self,
        prefix: str,
        num_classes: int = 1,
        image_size: int = 512,
        image_norm: Optional[str] = "imagenet",
        pretrained: bool = False,
        # ResEnc UNet hyperparams（先给出合理默认，后续可用 config 覆盖）
        n_stages: int = 7,
        features_per_stage=(32, 64, 128, 256, 512, 512, 512),
        n_blocks_per_stage=(1, 3, 4, 6, 6, 6, 6),
        n_conv_per_stage_decoder=(1, 1, 1, 1, 1, 1),
        deep_supervision: bool = False,
    ):
        super().__init__()
        if pretrained:
            logger.warning("ResEncUNetForSemanticSegmentation 当前不支持加载预训练权重，pretrained=True 将被忽略。")

        assert num_classes == 1, "当前选择的是二值 logits 形式，请将 num_classes 设为 1。"

        self.prefix = prefix
        self.num_classes = num_classes
        self.image_size = image_size
        self.image_mean, self.image_std = image_mean_std(image_norm)

        from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

        self.model = ResidualEncoderUNet(
            input_channels=3,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=nn.Conv2d,
            kernel_sizes=3,
            strides=(1, 2, 2, 2, 2, 2, 2),
            n_blocks_per_stage=n_blocks_per_stage,
            num_classes=num_classes,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=True,
            norm_op=nn.InstanceNorm2d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=deep_supervision,
        )

    @property
    def image_key(self):
        return f"{self.prefix}_{IMAGE}"

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    @property
    def image_column_prefix(self):
        return f"{self.image_key}_{COLUMN}"

    def _collect_moe_loss(self) -> Optional[torch.Tensor]:
        moe = None
        for m in self.modules():
            if isinstance(m, ConvLoRAConv2d) and getattr(m, "moe_loss", None) is not None:
                moe = m.moe_loss if moe is None else (moe + m.moe_loss)
        return moe

    def forward(self, batch):
        x = batch[self.image_key]
        # semantic seg processor 会产出 [B, N, C, H, W]，本任务默认 N=1
        if x.dim() == 5:
            x = x[:, 0]
        # 网络输出期望 [B, 1, H, W]
        logits = self.model(x)
        # deep_supervision=False 时 logits 是 Tensor；True 时为 list/tuple（暂不支持）
        if isinstance(logits, (list, tuple)):
            raise NotImplementedError("deep_supervision=True 暂未接入 AutoGluon loss，请先使用 deep_supervision=False。")

        # 对齐 label 的空间分辨率（保险起见）
        if logits.shape[-2:] != batch[self.label_key].shape[-2:]:
            logits = F.interpolate(logits, batch[self.label_key].shape[-2:], mode="bilinear", align_corners=False)

        if self.training:
            out = {self.prefix: {LOGITS: logits}}
        else:
            out = {self.prefix: {LOGITS: logits, LABEL: batch[self.label_key]}}

        moe_loss = self._collect_moe_loss()
        if moe_loss is not None:
            out[self.prefix][MOE_LOSS] = moe_loss
        return out

    def get_layer_ids(self):
        # 先给一个最简单的实现，确保 layerwise lr decay 的调用不报错
        return {n: 0 for n, _ in self.named_parameters()}



```

```python
import functools
import logging
import re
import warnings
from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch._dynamo
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from torch import nn
from torch.nn.modules.loss import _Loss
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertTokenizer, CLIPTokenizer, ElectraTokenizer
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss

from ..constants import (
    ALL_MODALITIES,
    CATEGORICAL,
    CATEGORICAL_MLP,
    CLASS_LOGITS,
    CLIP,
    CLIP_IMAGE_MEAN,
    CLIP_IMAGE_STD,
    DOCUMENT,
    DOCUMENT_TRANSFORMER,
    FT_TRANSFORMER,
    FUSION_MLP,
    FUSION_NER,
    FUSION_TRANSFORMER,
    HF_TEXT,
    IMAGE,
    LOGITS,
    META_TRANSFORMER,
    MMDET_IMAGE,
    MMOCR_TEXT_DET,
    MMOCR_TEXT_RECOG,
    NER_TEXT,
    NUMERICAL,
    NUMERICAL_MLP,
    OCR,
    PEFT_ADDITIVE_STRATEGIES,
    REGRESSION,
    SAM,
    NNUNET,
    SEMANTIC_MASK,
    SEMANTIC_SEGMENTATION,
    SEMANTIC_SEGMENTATION_IMG,
    T_FEW,
    TEXT,
    TEXT_NER,
    TIMM_IMAGE,
)
from .adaptation_layers import ConvLoRAConv2d, ConvLoRALinear, IA3Linear, IA3LoRALinear, LoRALinear

logger = logging.getLogger(__name__)


ALL_TOKENIZERS = {
    "bert": BertTokenizer,
    "clip": CLIPTokenizer,
    "electra": ElectraTokenizer,
    "hf_auto": AutoTokenizer,
}


class DummyLayer(nn.Module):
    """
    DummyLayer to ensure that the gradient checkpointing will assign output layer as require_grad=True.
    Reference: https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
    """

    def __init__(self):
        super().__init__()
        self.dummy_bias = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def forward(self, x):
        return x + self.dummy_bias.to(x) - self.dummy_bias.to(x)


def init_weights(module: nn.Module):
    """
    Initialize one module. It uses xavier_norm to initialize nn.Embedding
    and xavier_uniform to initialize nn.Linear's weight.

    Parameters
    ----------
    module
        A Pytorch nn.Module.
    """
    if isinstance(module, nn.Embedding):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def assign_encoder_layer_ids(
    encoder_names: List[List[str]],
):
    """
    Assign ids to encoder layers. The encoder may contain several blocks e.g., block1 and block2.
    This function iterates through all the layers of each block from the input end towards the output end.
    It increases 1 on the layer id when the detected digit in a layer name changes.

    Parameters
    ----------
    encoder_names
        Encoder layer names.

    Returns
    -------
    name_to_id
        The encoder layer-to-id mapping.
    encoder_layer_num
        The encoder layer number.
    """
    name_to_id = {}
    cur_id = 0
    for i, group_names in enumerate(encoder_names):
        last_inferred_id = -1
        for n in group_names:
            detect_id = False
            n_splits = n.split(".")

            for split in n_splits:
                # the first digit encountered is used to infer layer id
                if split.isdigit():
                    inferred_id = int(split)
                    # increase at most 1 one time
                    if inferred_id != last_inferred_id:
                        cur_id += 1  # layer.0 -> layer_id 1
                        last_inferred_id = inferred_id

                    name_to_id[n] = cur_id
                    detect_id = True
                    break

            if detect_id is False:
                raise ValueError(f"parameter name: {n} not has no id inside")

    if len(name_to_id) > 0:
        encoder_layer_num = max(name_to_id.values())
    else:
        encoder_layer_num = 0
    return name_to_id, encoder_layer_num


def assign_non_encoder_layer_ids(
    non_encoder_names: List[str],
    layer_id: int,
):
    """
    Assign the provided id to non-encoder layers.

    Parameters
    ----------
    non_encoder_names
        Names layers not belonging to an encoder.
    layer_id
        provided id.

    Returns
    -------
    A dictionary mapping the layer names (keys) to their ids (values).
    """
    name_to_id = {}
    for n in non_encoder_names:
        name_to_id[n] = layer_id
    return name_to_id


def split_encoder_non_encoder(names: List[str], post_encoder_patterns: Tuple[str, ...]):
    """
    Group layer names into two types: encoder and non-encoder.
    A layer belongs to encoder if its name contains at least one digit.
    It uses this rule since a model's encoder in Pytorch's implementation
    is generally wrapped by nn.Sequential() or nn.ModuleList(),
    which produce digits in layer names.

    Parameters
    ----------
    names
        Model layer names.
    Returns
    -------
    encoder_names
        A list of encoder layer names.
    non_encoder_names
        A list of non-encoder layer names.
    """
    encoder_names = []
    non_encoder_names = []
    for n in names:
        is_encoder = False
        if any(p in n for p in post_encoder_patterns):
            non_encoder_names.append(n)
            continue
        for i in n.split("."):
            if i.isdigit():
                encoder_names.append(n)
                is_encoder = True
                break
        if not is_encoder:
            non_encoder_names.append(n)

    return encoder_names, non_encoder_names


def group_param_names(
    names: List[str],
    pre_encoder_patterns: Tuple[str, ...],
    post_encoder_patterns: Tuple[str, ...],
    model_prefix: Optional[str] = None,
):
    """
    Group layer names into three types: pre-encoder, encoder, and post-encoder.
    If "model_prefix" is provided, the selected layer names must start with it.
    In this case, the left names will be returned for the next-time processing.
    This function first extracts the first-level children modules' names and
    classify them into encoder and non-encoder layers. Note that an encoder may
    consist of several manually named children modules, e.g., block1 and block2.
    The non-encoder layers are further subdivided into pre-encoder and post-encoder.

    Parameters
    ----------
    names
        Model layer names
    pre_encoder_patterns
        Patterns to identify a layer as a pre-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into pre-encoder layers.
    post_encoder_patterns
        Patterns to identify a layer as a post-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into post-encoder layers.
    model_prefix
        A prefix to filter layer names. Only layer names starting with it will be selected.
    Returns
    -------
    left_names
        The layer names left for the next-time processing.
    encoder_names_grouped
        Encoder layer names.
    pre_encoder_names
        Names of layers before the encoder.
    post_encoder_names
        Names of layers after the encoder.
    """
    # two set of patterns can't have intersections
    assert all(pre_p not in post_encoder_patterns for pre_p in pre_encoder_patterns)

    left_names = []
    # in case model_prefix is provided, e.g., the clip model with image and text branches
    selected_names = []
    for n in names:
        if model_prefix is not None and not n.startswith(model_prefix):
            left_names.append(n)
        else:
            selected_names.append(n)

    # split blocks at the first level
    children_prefix = []
    for n in selected_names:
        if model_prefix is not None:
            child_name = n[len(model_prefix) + 1 :].split(".")[0]
            child_prefix = f"{model_prefix}.{child_name}"
        else:
            child_prefix = n.split(".")[0]
        if child_prefix not in children_prefix:
            children_prefix.append(child_prefix)

    encoder_names_grouped = []
    non_encoder_names = []
    for child_prefix in children_prefix:
        per_names_group = [n for n in selected_names if n.startswith(child_prefix)]
        per_encoder_names, per_non_encoder_names = split_encoder_non_encoder(per_names_group, post_encoder_patterns)
        encoder_names_grouped.append(per_encoder_names)
        non_encoder_names.extend(per_non_encoder_names)

    pre_encoder_names = []
    post_encoder_names = []
    for n in non_encoder_names:
        if any(p in n for p in pre_encoder_patterns):
            pre_encoder_names.append(n)
        elif any(p in n for p in post_encoder_patterns):
            post_encoder_names.append(n)
        else:
            raise ValueError(f"parameter name: {n} belong to neither pre or post encoder names")

    # only process left names in next iteration
    return left_names, encoder_names_grouped, pre_encoder_names, post_encoder_names


def reverse_layer_ids(
    encoder_name_to_id: dict,
    pre_enocder_name_to_id: dict,
    post_enocder_name_to_id: dict,
):
    """
    The layer ids need to increase when going from the output end to the input end.
    We need to reverse the ids which were originally assigned in a decreasing order.

    Parameters
    ----------
    encoder_name_to_id
        The layer-to-id mapping of encoder layers.
    pre_enocder_name_to_id
        The layer-to-id mapping of pre-encoder layers.
    post_enocder_name_to_id
        The layer-to-id mapping of post-encoder layers.

    Returns
    -------
    The layer-to-id mapping of all layers with layer ids reversed.
    """
    name_to_id = {**pre_enocder_name_to_id, **encoder_name_to_id, **post_enocder_name_to_id}
    if len(name_to_id) > 0:
        layer_num = max(name_to_id.values())
        # if no post encoder layers, the minimum layer id should be 1
        if len(post_enocder_name_to_id) == 0:
            layer_num += 1
    for n, layer_id in name_to_id.items():
        name_to_id[n] = layer_num - layer_id

    return name_to_id


def assign_layer_ids(
    names: List[str],
    pre_encoder_patterns: Tuple[str, ...],
    post_encoder_patterns: Tuple[str, ...],
    model_pre: Optional[str] = None,
):
    """
    Assign ids to all layers. It splits a model into three parts: pre-encoder, encoder, and post-encoder.
    Encoder is generally a stack of multiple similar layers, such as transformer layers. Since encoder is
    generally wrapped by nn.Sequential() or nn.ModuleList(), its inside layer names contain digits.
    It sets 0 as the ids of all post-encoder layers and a maximum id (layer_num) for the all the pre-encoder
    layers. The encoder layers have decreasing ids from the input to the output ends.

    Parameters
    ----------
    names
        model layer names.
    pre_encoder_patterns
        Patterns to identify a layer as a pre-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into pre-encoder layers.
    post_encoder_patterns
        Patterns to identify a layer as a post-encoder layer. If a layer name contains one pattern,
        the layer will be grouped into post-encoder layers.
    model_pre
        The layer names' prefix. Only the layer names with this prefix will be assigned ids. The left
        layer names will be returned.

    Returns
    -------
    name_to_id
        A dictionary mapping the layer names (keys) to their ids (values).
    left_names
        The layer names not starting with the "model_pre".
    """
    try:
        left_names, encoder_names, pre_encoder_names, post_encoder_names = group_param_names(
            names=names,
            pre_encoder_patterns=pre_encoder_patterns,
            post_encoder_patterns=post_encoder_patterns,
            model_prefix=model_pre,
        )
        # add a constraint
        if len(encoder_names) == 0 and len(pre_encoder_names) != 0:
            raise ValueError(f"encoder_names is empty, but pre_encoder_names has values: {pre_encoder_names}")

        encoder_name_to_id, encoder_layer_num = assign_encoder_layer_ids(
            encoder_names=encoder_names,
        )

        pre_encoder_name_to_id = assign_non_encoder_layer_ids(non_encoder_names=pre_encoder_names, layer_id=0)

        post_encoder_name_to_id = assign_non_encoder_layer_ids(
            non_encoder_names=post_encoder_names, layer_id=encoder_layer_num + 1
        )

        name_to_id = reverse_layer_ids(
            encoder_name_to_id=encoder_name_to_id,
            pre_enocder_name_to_id=pre_encoder_name_to_id,
            post_enocder_name_to_id=post_encoder_name_to_id,
        )
    except Exception as e:
        logger.debug(
            f"When calling assign_layer_ids(), it catches exception: {e}. All the layers will use the same layer_id."
        )
        name_to_id = dict()
        left_names = names

    return name_to_id, left_names


def get_column_features(
    batch: Dict[str, torch.Tensor],
    column_name_prefix: str,
    features: torch.Tensor,
    valid_lengths: torch.Tensor,
    cls_feature: Optional[torch.Tensor] = None,
):
    """
    Index the features of one column defined by `column_name_prefix`.
    This function can be used to index both image and text features.
    The features have shape (b, n, d), where n can be the image number or
    text token number. One column corresponds to a subset of
    the n images or text tokens. One column name can only appear once in the return.

    Parameters
    ----------
    batch
        The batch input containing the feature column information, i.e., indexes.
    column_name_prefix
        The column name prefix of one modality (image or text).
    features
        The features of columns whose names starts with column_name_prefix.
    valid_lengths
        The valid image number or text token number of each sample in a batch.
    cls_feature
        The cls feature containing information from all feature columns.

    Returns
    -------
    The column features with masks. If the column has no valid features, its
    mask is 0.
    """
    column_features = {}
    feature_masks = {}

    cut_idx = len(column_name_prefix) + 1
    if cls_feature is not None:
        all_column_names = []
        # create a zero mask to do logical_or with each column's mask
        joint_mask = torch.zeros(features.shape[0]).to(features)  # (b,)
    for key in batch:
        if key.startswith(column_name_prefix):
            per_col_features = []
            per_col_masks = torch.zeros(features.shape[0]).to(features)  # (b,)
            assert batch[key].ndim == 2 and batch[key].shape[1] == 2
            for i, per_sample_col_idx in enumerate(batch[key]):
                start_idx = per_sample_col_idx[0]
                end_idx = per_sample_col_idx[1]
                if start_idx < end_idx:
                    assert end_idx <= valid_lengths[i]
                    per_col_features.append(features[i, start_idx:end_idx].mean(dim=0))
                    per_col_masks[i] = 1
                else:  # the column has no valid image/text.
                    per_col_features.append(torch.zeros_like(features[0, 0]))
                    per_col_masks[i] = 0
            column_name = key[cut_idx:]
            column_features[column_name] = torch.stack(per_col_features, dim=0)  # (b, num_features)
            feature_masks[column_name] = per_col_masks  # (b,)
            if cls_feature is not None:
                all_column_names.append(column_name)
                joint_mask = torch.logical_or(joint_mask, per_col_masks)

    # all the columns of one model's input share the model's cls feature
    if (
        cls_feature is not None and len(all_column_names) > 0
    ):  # some models', e.g, timm_image, output doesn't have the cls feature.
        # remove the individual column features since these column features not independent
        for column_name in all_column_names:
            column_features.pop(column_name)
            feature_masks.pop(column_name)

        joint_column_name = "_".join(all_column_names)
        column_features[joint_column_name] = cls_feature
        feature_masks[joint_column_name] = joint_mask.to(features)

    # print(f"column_features: {column_features}")
    return column_features, feature_masks


def create_adaptation(peft: str, layer: nn.Module, lora_r: int, lora_alpha: int, **kwargs):
    """
    Creates a model adaptation module (IA3, LoRA, IA3_LoRA) given a linear layer.

    Parameters
    ----------
    peft
        Name of the adaptation module.
    layer
       The layer the adaptation module should be applied to.
    lora_r
        The rank r of the low-rank decomposition.
    lora_alpha
        The scaling factor. Can be set to same value as r in
        most cases, as initialization is scaled already.
    filter
        Apply loRA only to linear layers filtered by name (e.g. "query.").
        If None, loRA is applied to all linear Layers in module.
    module_filter
        Apply loRA only to modules filtered by name (e.g. ".*EncDecAttention|.*DenseReluDense")
        If None, loRA is considered for all modules

    Returns
    -------
    Model with injected LoRA modules.
    """
    if "ia3_lora" in peft:
        return IA3LoRALinear(
            layer.in_features, layer.out_features, r=lora_r, lora_alpha=lora_alpha, merge_weights=False
        )
    elif "conv_lora" in peft:
        return ConvLoRALinear(
            layer.in_features,
            layer.out_features,
            r=lora_r,
            lora_alpha=lora_alpha,
            merge_weights=False,
            conv_lora_expert_num=kwargs["conv_lora_expert_num"],
        )
    elif "ia3" in peft:
        return IA3Linear(layer.in_features, layer.out_features, merge_weights=False)
    elif "lora" in peft:
        return LoRALinear(layer.in_features, layer.out_features, r=lora_r, lora_alpha=lora_alpha, merge_weights=False)
    elif peft is not None:
        raise NotImplementedError(
            f"The efficient finetuning strategy '{peft}'"
            f" is not supported. We only support"
            f" {', '.join(PEFT_ADDITIVE_STRATEGIES)}."
        )


def inject_adaptation_to_linear_layer(
    model: nn.Module,
    peft: str,
    lora_r: int = None,
    lora_alpha: int = None,
    filter: Optional[List[str]] = None,
    module_filter: Optional[List[str]] = None,
    extra_trainable_params: Optional[List[str]] = None,
    **kwargs,
) -> nn.Module:
    """
    Injects trainable adatio Low-Rank decomposition matrices (LoRA) into linear
    layers of a PyTorch model. Used for efficient fine-tuning of large
    pre-trained models.

    Parameters
    ----------
    model
        A PyTorch model.
    peft
        Efficient finetuning method that should be applied.
    lora_r
        The rank r of the low-rank decomposition.
    lora_alpha
        The scaling factor. Can be set to same value as r in
        most cases, as initialization is scaled already.
    filter
        Apply loRA only to linear layers filtered by name (e.g. "query.").
        If None, loRA is applied to all linear Layers in module.
    module_filter
        Apply loRA only to modules filtered by name (e.g. ".*EncDecAttention|.*DenseReluDense")
        If None, loRA is considered for all modules
    extra_trainable_params
        Not to apply loRA to modules filtered by name, and these modules are not frozen during training (e.g. "mask_decoder").
        If None, all the modules except for those applied loRA are frozen.
    Returns
    -------
    Model with injected LoRA modules.
    """
    for m_name, module in dict(model.named_modules()).items():
        if extra_trainable_params and any(re.match(filter_layer, m_name) for filter_layer in extra_trainable_params):
            continue
        if hasattr(model, "frozen_layers") and any(
            re.search(filter_layer, m_name) for filter_layer in model.frozen_layers
        ):
            continue
        if not module_filter or any(re.match(filter_module, m_name) for filter_module in module_filter):
            for c_name, layer in dict(module.named_children()).items():
                if not filter or any(re.match(filter_layer, c_name) for filter_layer in filter):
                    # 对非 Linear 直接跳过（让 conv-only 网络也能使用 peft=conv_lora，通过 Conv2d 注入生效）
                    if not isinstance(layer, nn.Linear):
                        continue
                    adaptation_layer = create_adaptation(peft, layer, lora_r, lora_alpha, **kwargs)
                    adaptation_layer.weight = layer.weight
                    adaptation_layer.bias = layer.bias
                    setattr(module, c_name, adaptation_layer)

    return model  # return model to enable method chaining


def inject_adaptation_to_conv2d_layer(
    model: nn.Module,
    peft: str,
    lora_r: int = None,
    lora_alpha: int = None,
    filter: Optional[List[str]] = None,
    module_filter: Optional[List[str]] = None,
    extra_trainable_params: Optional[List[str]] = None,
    **kwargs,
) -> nn.Module:
    """
    将 LoRA/Conv-LoRA 注入到 Conv2d（用于 nnU-Net 等卷积网络）。

    约定：
    - 仅对 `torch.nn.Conv2d` 生效
    - 目前 `peft` 仅支持 `conv_lora`（其它策略未来可扩展）
    - 注入后原卷积权重被冻结，新增参数以 `lora_*` 命名，便于 `get_peft_param_names` 选择可训练参数
    """
    if peft is None:
        return model
    if "conv_lora" not in peft:
        # 当前只扩展 conv_lora 到 Conv2d，其它 PEFT 保持原逻辑（linear-only）
        return model

    for m_name, module in dict(model.named_modules()).items():
        if extra_trainable_params and any(re.match(filter_layer, m_name) for filter_layer in extra_trainable_params):
            continue
        if hasattr(model, "frozen_layers") and any(re.search(filter_layer, m_name) for filter_layer in model.frozen_layers):
            continue
        if not module_filter or any(re.match(filter_module, m_name) for filter_module in module_filter):
            for c_name, layer in dict(module.named_children()).items():
                if not filter or any(re.match(filter_layer, c_name) for filter_layer in filter):
                    if not isinstance(layer, nn.Conv2d):
                        continue
                    adaptation_layer = ConvLoRAConv2d(
                        in_channels=layer.in_channels,
                        out_channels=layer.out_channels,
                        kernel_size=layer.kernel_size,
                        stride=layer.stride,
                        padding=layer.padding,
                        dilation=layer.dilation,
                        groups=layer.groups,
                        bias=layer.bias is not None,
                        padding_mode=layer.padding_mode,
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        merge_weights=False,
                        conv_lora_expert_num=kwargs["conv_lora_expert_num"],
                    )
                    # 复用原参数对象（weight/bias），确保 state_dict 兼容且冻结生效
                    adaptation_layer.weight = layer.weight
                    adaptation_layer.bias = layer.bias
                    setattr(module, c_name, adaptation_layer)

    return model


def get_model_head(model: nn.Module):
    """
    Return the model's head. Different models may have different head names.

    Parameters
    ----------
    model
        A Pytorch model.

    Returns
    -------
    The model's head.
    """
    if hasattr(model, "head"):
        head = model.head  # move the head outside
    elif hasattr(model, "last_linear"):
        head = model.last_linear
    elif hasattr(model, "fc"):
        head = model.fc
    elif hasattr(model, "classifier"):
        head = model.classifier
    else:
        raise ValueError(f"Model {type(model)} doesn't have head. Need to check its implementation.")

    return head.fc if hasattr(head, "fc") else head


def get_hf_config_and_model(
    checkpoint_name: str, pretrained: Optional[bool] = True, low_cpu_mem_usage: Optional[bool] = False
):
    """
    Get a Huggingface config and model based on a checkpoint name.

    Parameters
    ----------
    checkpoint_name
        A model checkpoint name or a local path that saves a custom checkpoint.
    pretrained
         Whether using the pretrained weights. If pretrained=True, download the pretrained model.
    low_cpu_mem_usage
        Whether to turn on the optimization of reducing the peak CPU memory usage when loading the pretrained model.

    Returns
    -------
    A Huggingface config and model.
    """
    config = AutoConfig.from_pretrained(checkpoint_name)    # nosec B615

    if pretrained:
        model = AutoModel.from_pretrained(checkpoint_name, low_cpu_mem_usage=low_cpu_mem_usage) # nosec B615
    else:
        model = AutoModel.from_config(config)
    # Explicitly set the model to train mode after loading as by default it is in eval mode
    # See issue #4965
    model.train()
    return config, model


def apply_sigmoid(output: Dict):
    """
    Apply the sigmoid to logits.

    Parameters
    ----------
    output
        The model output dict.

    Returns
    -------
    The output with logits transformed by sigmoid.
    """
    for k, v in output.items():
        output[k][LOGITS] = torch.sigmoid(v[LOGITS].float())
    return output


def apply_multi_class_semantic_seg_postprocess(output: Dict):
    """
    Apply the semantic postprocessing to logits.

    Parameters
    ----------
    output
        The model output dict.

    Returns
    -------
    The output with post-proceesed semantic masks.
    """

    def semantic_inference(mask_cls, mask_pred):
        """
        Post-processing mask prediction for multi-class semantic segmentation inference based on https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py

        Args:
            mask_cls (`torch.Tensor`):
                Class logits. A tensor of shape `(num_queries, num_classes + 1)` (include the "no object" category).

            mask_pred (`torch.Tensor`):
                Mask logits. A tensor of shape `(num_queries, height, width)`.

        Returns:
            semseg (`torch.Tensor`): The processed mask prediction. A tensor of shape `(num_classes, height, width)`.

        References:
        [1] https://arxiv.org/abs/2107.06278
        [2] https://arxiv.org/abs/2112.01527
        """
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    for k, v in output.items():
        pred_classes = output[k][CLASS_LOGITS]
        pred_masks = output[k][LOGITS]
        semantic_masks = []
        for mask_cls_result, mask_pred_result in zip(
            pred_classes, pred_masks
        ):  # bs, num_q, num_class and bs, num_q, h, w
            per_sample_semantic_masks = semantic_inference(mask_cls_result, mask_pred_result)  # num_class, h, w
            semantic_masks.append(per_sample_semantic_masks)
        semantic_masks = torch.stack(semantic_masks, dim=0)
        output[k][SEMANTIC_MASK] = semantic_masks
    return output


def get_model_postprocess_fn(problem_type: str, loss_func: _Loss):
    """
    Get the postprocessing function for the model outputs.

    Parameters
    ----------
    problem_type
        The problem type, e.g., classification or regression.
    loss_func
        The loss function used in training.

    Returns
    -------
    The postprocessing function.
    """
    postprocess_func = None
    if problem_type == REGRESSION:
        if isinstance(loss_func, nn.BCEWithLogitsLoss):
            postprocess_func = apply_sigmoid
    elif problem_type == SEMANTIC_SEGMENTATION:
        if isinstance(loss_func, Mask2FormerLoss):
            postprocess_func = apply_multi_class_semantic_seg_postprocess
        else:
            postprocess_func = apply_sigmoid

    return postprocess_func


def get_mmocr_config_and_model(checkpoint_name: str):
    """
    Get an MMOCR config and model based on a checkpoint name.

    Parameters
    ----------
    checkpoint_name
        A model checkpoint name.

    Returns
    -------
    An MMOCR config and model.
    """
    from ..utils import check_if_packages_installed

    check_if_packages_installed(package_names=["mmcv"])
    try:
        import mmocr
        from mmocr.models import build_detector
    except ImportError:
        mmocr = None
    from mim.commands.download import download

    checkpoints = download(package="mmocr", configs=[checkpoint_name], dest_root=".")

    # read config files
    check_if_packages_installed(problem_type=OCR)
    config_file = checkpoint_name + ".py"
    if isinstance(config_file, str):
        config = mmcv.Config.fromfile(config_file)

    # build model and load pretrained weights
    checkpoint = checkpoints[0]
    model = build_detector(config.model, test_cfg=config.get("test_cfg"))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")
    return config, model


def lookup_mmdet_config(key, config):
    if key in config:
        return config[key]
    for subconfig in config.values():
        if isinstance(subconfig, dict):
            result = lookup_mmdet_config(key, subconfig)
            if result is not None:
                return result
        elif isinstance(subconfig, list):
            for subsubconfig in subconfig:
                if isinstance(subsubconfig, dict):
                    result = lookup_mmdet_config(key, subsubconfig)
                    if result is not None:
                        return result
    return None


def update_mmdet_config(key, value, config):
    for k, subconfig in config.items():
        if key == k:
            config[k] = value
        elif isinstance(subconfig, dict):
            update_mmdet_config(key, value, subconfig)
        elif isinstance(subconfig, list):
            for subsubconfig in subconfig:
                if isinstance(subsubconfig, dict):
                    update_mmdet_config(key, value, subsubconfig)


def run_model(model: nn.Module, batch: dict, trt_model: Optional[nn.Module] = None):
    from ..utils.onnx import OnnxModule
    from .document_transformer import DocumentTransformer
    from .fusion.fusion_mlp import MultimodalFusionMLP
    from .hf_text import HFAutoModelForTextPrediction
    from .t_few import TFewModel
    from .timm_image import TimmAutoModelForImagePrediction

    supported_models = (
        TimmAutoModelForImagePrediction,
        HFAutoModelForTextPrediction,
        MultimodalFusionMLP,
        TFewModel,
        OnnxModule,
    )
    pure_model = model
    if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
        pure_model = model._orig_mod
    if isinstance(model, nn.DataParallel):
        pure_model = model.module
    if isinstance(pure_model, OnnxModule):
        for k in batch:
            # HACK input data types in ONNX
            if batch[k].dtype == torch.int32:
                batch[k] = batch[k].to(torch.int64)
    # DocumentTransformer inherited from HFAutoModelForTextPrediction
    if (not isinstance(pure_model, DocumentTransformer)) and isinstance(pure_model, supported_models):
        input_vec = [batch[k] for k in pure_model.input_keys]
        column_names, column_values = [], []
        for k in batch.keys():
            has_image_column_prefix = isinstance(pure_model, TimmAutoModelForImagePrediction) and k.startswith(
                pure_model.image_column_prefix
            )
            has_text_column_prefix = isinstance(pure_model, HFAutoModelForTextPrediction) and k.startswith(
                pure_model.text_column_prefix
            )
            if has_image_column_prefix or has_text_column_prefix:
                column_names.append(k)
                column_values.append(batch[k])
        if column_names != [] and column_values != []:
            input_vec.append(column_names)
            input_vec.append(column_values)
        if isinstance(pure_model, OnnxModule):
            # OnnxModule doesn't support multi-gpu yet
            output_vec = pure_model(*tuple(input_vec))
        else:
            output_vec = model(*tuple(input_vec))

        output = pure_model.get_output_dict(*output_vec)
    else:
        output = model(batch)
    return output


def freeze_model_layers(model, frozen_layers):
    """
    Freeze model layers with pattern in frozen_layers.

    Parameters
    ----------
    model
        The pytorch model.
    frozen_layers
        A list of substrings of frozen layers' names.

        e.g. if frozen_layers = ["backbone", "neck"],
            all layers including "backbone" or "neck" in the name will be frozen.
    """

    if not frozen_layers:
        return

    is_frozen_layer = lambda n: any(bb in n for bb in frozen_layers)

    for n, p in model.named_parameters():
        if is_frozen_layer(n):
            p.requires_grad = False


def get_pretrained_tokenizer(
    tokenizer_name: str,
    checkpoint_name: str,
    use_fast: Optional[bool] = True,
    add_prefix_space: Optional[bool] = None,
):
    """
    Load the tokenizer for a pre-trained huggingface checkpoint.

    Parameters
    ----------
    tokenizer_name
        The tokenizer type, e.g., "bert", "clip", "electra", and "hf_auto".
    checkpoint_name
        Name of a pre-trained checkpoint.
    use_fast
        Use a fast Rust-based tokenizer if it is supported for a given model.
        If a fast tokenizer is not available for a given model, a normal Python-based tokenizer is returned instead.
        See: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer.from_pretrained.use_fast

    Returns
    -------
    A tokenizer instance.
    """
    try:
        tokenizer_class = ALL_TOKENIZERS[tokenizer_name]
        if add_prefix_space is None:
            return tokenizer_class.from_pretrained(checkpoint_name, use_fast=use_fast)
        else:
            return tokenizer_class.from_pretrained(
                checkpoint_name, use_fast=use_fast, add_prefix_space=add_prefix_space
            )
    except TypeError as e:
        try:
            tokenizer = BertTokenizer.from_pretrained(checkpoint_name)  # nosec B615
            warnings.warn(
                f"Current checkpoint {checkpoint_name} does not support AutoTokenizer. "
                "Switch to BertTokenizer instead.",
                UserWarning,
            )
            return tokenizer
        except:
            raise e


def extract_value_from_config(
    config: Dict,
    keys: Tuple[str, ...],
):
    """
    Traverse a config dictionary to get some hyper-parameter's value.

    Parameters
    ----------
    config
        A config dictionary.
    keys
        The possible names of a hyper-parameter.

    Returns
    -------
    The hyper-parameter value.
    """
    result = []
    for k, v in config.items():
        if k in keys:
            result.append(v)
        elif isinstance(v, dict):
            result += extract_value_from_config(v, keys)
        else:
            pass

    return result


def extract_image_hparams_from_config(model_name: str, config):
    """
    Extract some default hyper-parameters, e.g., image size, mean, and std,
    from a pre-trained (timm or huggingface) checkpoint.

    Parameters
    ----------
    model_name
        Name of model.
    config
        Config of a pre-trained checkpoint.

    Returns
    -------
    image_size
        Image width/height.
    mean
        Image normalization mean.
    std
        Image normalizaiton std.
    """
    if model_name.lower().startswith((TIMM_IMAGE, META_TRANSFORMER)):
        image_size = config["input_size"][-1]
        image_mean = config["mean"]
        image_std = config["std"]
    elif model_name.lower().startswith((CLIP, DOCUMENT_TRANSFORMER)):
        extracted = extract_value_from_config(
            config=config.to_diff_dict(),
            keys=("image_size",),
        )
        if len(extracted) == 0:
            image_size = None
        elif len(extracted) >= 1:
            image_size = extracted[0]
            if isinstance(image_size, tuple):
                image_size = image_size[-1]
        else:
            raise ValueError(f" more than one image_size values are detected: {extracted}")
        image_mean = None
        image_std = None
    else:
        raise ValueError(f"Unknown image processor prefix: {model_name}")
    return image_size, image_mean, image_std


def image_mean_std(norm_type: str):
    """
    Get image normalization mean and std by its name.

    Parameters
    ----------
    norm_type
        Name of image normalization.

    Returns
    -------
    Normalization mean and std.
    """
    if norm_type == "inception":
        return IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    elif norm_type == "imagenet":
        return IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    elif norm_type == "clip":
        return CLIP_IMAGE_MEAN, CLIP_IMAGE_STD
    else:
        raise ValueError(f"unknown image normalization: {norm_type}")


def get_image_size_mean_std(
    model_name: str,
    config,
    provided_size: int,
    provided_norm_type: str,
    support_variable_input_size: Optional[bool] = False,
):
    image_size, image_mean, image_std = extract_image_hparams_from_config(
        model_name=model_name,
        config=config,
    )
    if support_variable_input_size and provided_size is not None:
        # We have detected that the model supports using an image size that is
        # different from the pretrained model, e.g., ConvNets with global pooling
        if provided_size < image_size:
            logger.warning(
                f"The provided image size={provided_size} is smaller than the default size "
                f"of the pretrained backbone, which is {image_size}. "
                f"Detailed configuration of the backbone is in {config}. "
                f"You may like to double check your configuration."
            )
        image_size = provided_size
    elif provided_size is not None and provided_size != image_size:
        logger.warning(
            f"The model does not support using an image size that is different from the default size. "
            f"Provided image size={provided_size}. Default size={image_size}. "
            f"Detailed model configuration={config}. We have ignored the provided image size."
        )

    if image_size is None:
        if provided_size is not None:
            image_size = provided_size
            logger.debug(f"using provided image size: {image_size}.")
        else:
            raise ValueError("image size is missing.")
    else:
        logger.debug(f"using detected image size: {image_size}")

    if image_mean is None or image_std is None:
        if provided_norm_type is not None:
            image_mean, image_std = image_mean_std(provided_norm_type)
            logger.debug(f"using provided normalization: {provided_norm_type}.")
        else:
            raise ValueError("image normalization mean and std are missing.")
    else:
        logger.debug(f"using detected image normalization: {image_mean} and {image_std}.")

    return image_size, image_mean, image_std


def get_text_segment_num(config, provided_segment_num: int, checkpoint_name: str):
    extracted = extract_value_from_config(config=config.to_diff_dict(), keys=("type_vocab_size",))
    if len(extracted) == 0:
        default_segment_num = 1
    elif len(extracted) == 1:
        default_segment_num = extracted[0]
    else:
        raise ValueError(f" more than one type_vocab_size values are detected: {extracted}")

    if default_segment_num <= 0:
        default_segment_num = 1

    if provided_segment_num < default_segment_num:
        warnings.warn(
            f"provided text_segment_num: {provided_segment_num} "
            f"is smaller than {checkpoint_name}'s default: {default_segment_num}"
        )
    text_segment_num = min(provided_segment_num, default_segment_num)
    assert text_segment_num >= 1
    logger.debug(f"text segment num: {text_segment_num}")

    return text_segment_num


def get_text_token_max_len(provided_max_len, config, tokenizer, checkpoint_name):
    """
    Compute the allowable max length of token sequences.

    Parameters
    ----------
    provided_max_len
        The provided max length.
    config
        Model config.
    tokenizer
        Text tokenizer.
    checkpoint_name
        Name of checkpoint.

    Returns
    -------
    Token sequence max length.
    """
    if hasattr(config, "relative_attention") and config.relative_attention:
        default_max_len = tokenizer.model_max_length
    elif hasattr(config, "position_embedding_type") and "relative" in config.position_embedding_type:
        default_max_len = tokenizer.model_max_length
    elif hasattr(config, "max_position_embeddings"):
        default_max_len = config.max_position_embeddings
    else:
        default_max_len = tokenizer.model_max_length

    if provided_max_len is None or provided_max_len <= 0:
        max_len = default_max_len
    else:
        if provided_max_len < default_max_len:
            if default_max_len < 10**6:  # Larger than this value usually means infinite.
                warnings.warn(
                    f"provided max length: {provided_max_len} "
                    f"is smaller than {checkpoint_name}'s default: {default_max_len}"
                )
        max_len = min(provided_max_len, default_max_len)

    logger.debug(f"text max length: {max_len}")

    return max_len


def replace_missing_images_with_learnable(
    images: torch.Tensor,
    image_masks,
    learnable_image: nn.Parameter,
):
    b, n, c, h, w = images.shape
    assert learnable_image.shape == (c, h, w)
    for i in range(b):
        for j in range(n):
            if not image_masks[i][j]:  # False means a missing image
                images[i][j] = learnable_image

    return images


def select_model(
    config: DictConfig,
    df_preprocessor,
    strict: Optional[bool] = True,
):
    """
    Filter model config through the detected modalities in the training data.
    If MultiModalFeaturePreprocessor can't detect some modality,
    this function will remove the models that use this modality. This function is to
    maximize the user flexibility in defining the config.
    For example, if one uses the default, including hf_text and timm_image, as the model config template
    but the training data don't have images, this function will filter out timm_image.

    Parameters
    ----------
    config
        A DictConfig object. The model config should be accessible by "config.model"
    df_preprocessor
        A MultiModalFeaturePreprocessor object, which has called .fit() on the training data.
        Column names of the same modality are grouped into one list. If a modality's list is empty,
        it means the training data don't have this modality.
    strict
        If False, allow retaining one model when partial modalities are available for that model.

    Returns
    -------
    Config with some unused models removed.
    """
    data_status = {}
    for per_modality in ALL_MODALITIES:
        data_status[per_modality] = False
    if len(df_preprocessor.image_feature_names) > 0:
        data_status[IMAGE] = True
    if len(df_preprocessor.text_feature_names) > 0:
        data_status[TEXT] = True
    if len(df_preprocessor.categorical_feature_names) > 0:
        data_status[CATEGORICAL] = True
    if len(df_preprocessor.numerical_feature_names) > 0:
        data_status[NUMERICAL] = True
    if len(df_preprocessor.ner_feature_names) > 0:
        data_status[TEXT_NER] = True
    if len(df_preprocessor.document_feature_names) > 0:
        data_status[DOCUMENT] = True
    if len(df_preprocessor.semantic_segmentation_feature_names) > 0:
        data_status[SEMANTIC_SEGMENTATION_IMG] = True

    names = config.model.names
    if isinstance(names, str):
        names = [names]
    selected_model_names = []
    fusion_model_name = []
    for model_name in names:
        model_config = getattr(config.model, model_name)
        strict = getattr(model_config, "requires_all_dtypes", strict)
        if not model_config.data_types:
            fusion_model_name.append(model_name)
            continue
        model_data_status = [data_status[d_type] for d_type in model_config.data_types]
        if all(model_data_status):
            selected_model_names.append(model_name)
        else:
            if any(model_data_status) and not strict:
                selected_model_names.append(model_name)
                # update data types to be consistent with detected
                model_config.data_types = [d_type for d_type in model_config.data_types if data_status[d_type]]
            else:
                delattr(config.model, model_name)

    if len(selected_model_names) == 0:
        raise ValueError("No model is available for this dataset.")
    # only allow no more than 1 fusion model
    if len(fusion_model_name) > 1:
        raise ValueError(f"More than one fusion models `{fusion_model_name}` are detected, but only one is allowed.")

    if len(selected_model_names) > 1:
        assert len(fusion_model_name) == 1
        selected_model_names.extend(fusion_model_name)
    elif len(fusion_model_name) == 1 and hasattr(config.model, fusion_model_name[0]):
        delattr(config.model, fusion_model_name[0])

    config.model.names = selected_model_names
    logger.debug(f"selected models: {selected_model_names}")
    for model_name in selected_model_names:
        logger.debug(f"model dtypes: {getattr(config.model, model_name).data_types}")

    # clean up unused model configs
    model_keys = list(config.model.keys())
    for model_name in model_keys:
        if model_name not in selected_model_names + ["names"]:
            delattr(config.model, model_name)

    return config


def create_model(
    model_name: str,
    model_config: DictConfig,
    num_classes: Optional[int] = 0,
    classes: Optional[list] = None,
    num_numerical_columns: Optional[int] = None,
    num_categories: Optional[Dict] = None,
    numerical_fill_values: Optional[Dict] = None,
    pretrained: Optional[bool] = True,
    is_matching: Optional[bool] = False,
):
    """
    Create a single model.

    Parameters
    ----------
    model_name
        Name of the model.
    model_config
        Config of the model.
    num_classes
        The class number for a classification task. It should be 1 for a regression task.
    classes
        All classes in this dataset.
    num_numerical_columns
        The number of numerical columns in the training dataframe.
    num_categories
        The category number for each categorical column in the training dataframe.
    numerical_fill_values
        If numerical values are null, fill them with these.
    pretrained
        Whether using the pretrained timm models. If pretrained=True, download the pretrained model.
    is_matching
        Whether the model is used for semantic matching.

    Returns
    -------
    A model.
    """
    if model_name.lower().startswith(CLIP):
        from .clip import CLIPForImageText

        model = CLIPForImageText(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
            has_image=IMAGE in model_config.data_types,
            has_text=TEXT in model_config.data_types,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
            image_chan_num=model_config.image_chan_num,
            use_learnable_image=model_config.use_learnable_image,
            max_text_len=model_config.max_text_len,
            text_segment_num=model_config.text_segment_num,
            is_matching=is_matching,
        )
    elif model_name.lower().startswith(TIMM_IMAGE):
        from .timm_image import TimmAutoModelForImagePrediction

        model = TimmAutoModelForImagePrediction(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            mix_choice=model_config.mix_choice,
            pretrained=pretrained,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
            image_chan_num=model_config.image_chan_num,
            use_learnable_image=model_config.use_learnable_image,
        )
    elif model_name.lower().startswith(HF_TEXT):
        from .hf_text import HFAutoModelForTextPrediction

        model = HFAutoModelForTextPrediction(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pooling_mode=model_config.pooling_mode,
            gradient_checkpointing=model_config.gradient_checkpointing,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
            max_text_len=model_config.max_text_len,
            text_segment_num=model_config.text_segment_num,
            use_fast=model_config.use_fast,
        )
    elif model_name.lower().startswith(T_FEW):
        from .t_few import TFewModel

        model = TFewModel(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            length_norm=model_config.length_norm,  # Normalizes length to adjust for length bias in target template
            unlikely_loss=model_config.unlikely_loss,  # Adds loss term that lowers probability of incorrect outputs
            mc_loss=model_config.mc_loss,  # Adds multiple choice cross entropy loss
            num_classes=num_classes,
            gradient_checkpointing=model_config.gradient_checkpointing,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
            max_text_len=model_config.max_text_len,
            text_segment_num=model_config.text_segment_num,
        )
    elif model_name.lower().startswith(NUMERICAL_MLP):
        from .numerical_mlp import NumericalMLP

        model = NumericalMLP(
            prefix=model_name,
            in_features=num_numerical_columns,
            hidden_features=model_config.hidden_size,
            out_features=model_config.hidden_size,
            num_layers=model_config.num_layers,
            activation=model_config.activation,
            dropout=model_config.dropout,
            normalization=model_config.normalization,
            token_dim=model_config.token_dim,
            embedding_arch=model_config.embedding_arch,
            num_classes=num_classes,
            numerical_fill_values=numerical_fill_values,
        )
    elif model_name.lower().startswith(CATEGORICAL_MLP):
        from .categorical_mlp import CategoricalMLP

        model = CategoricalMLP(
            prefix=model_name,
            num_categories=num_categories,
            out_features=model_config.hidden_size,
            num_layers=model_config.num_layers,
            activation=model_config.activation,
            dropout=model_config.dropout,
            normalization=model_config.normalization,
            num_classes=num_classes,
        )
    elif model_name.lower().startswith(DOCUMENT_TRANSFORMER):
        from .document_transformer import DocumentTransformer

        model = DocumentTransformer(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pooling_mode=model_config.pooling_mode,
            gradient_checkpointing=model_config.gradient_checkpointing,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
        )
    elif model_name.lower().startswith(MMDET_IMAGE):
        from .mmdet_image import MMDetAutoModelForObjectDetection

        model = MMDetAutoModelForObjectDetection(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            config_file=model_config.config_file,
            classes=classes,
            pretrained=pretrained,
            output_bbox_format=model_config.output_bbox_format,
            frozen_layers=model_config.frozen_layers,
        )
    elif model_name.lower().startswith(MMOCR_TEXT_DET):
        from .mmocr_text_detection import MMOCRAutoModelForTextDetection

        model = MMOCRAutoModelForTextDetection(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
        )
    elif model_name.lower().startswith(MMOCR_TEXT_RECOG):
        from .mmocr_text_recognition import MMOCRAutoModelForTextRecognition

        model = MMOCRAutoModelForTextRecognition(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
        )
    elif model_name.lower().startswith(NER_TEXT):
        from .ner_text import HFAutoModelForNER

        model = HFAutoModelForNER(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            gradient_checkpointing=model_config.gradient_checkpointing,
            low_cpu_mem_usage=model_config.low_cpu_mem_usage,
            pretrained=pretrained,
            tokenizer_name=model_config.tokenizer_name,
        )
    elif model_name.lower().startswith(FUSION_MLP):
        from .fusion import MultimodalFusionMLP

        model = functools.partial(
            MultimodalFusionMLP,
            prefix=model_name,
            hidden_features=model_config.hidden_sizes,
            num_classes=num_classes,
            adapt_in_features=model_config.adapt_in_features,
            activation=model_config.activation,
            dropout=model_config.dropout,
            normalization=model_config.normalization,
            aux_loss_weight=model_config.aux_loss_weight,
        )
    elif model_name.lower().startswith(FUSION_NER):
        from .fusion import MultimodalFusionNER

        model = functools.partial(
            MultimodalFusionNER,
            prefix=model_name,
            hidden_features=model_config.hidden_sizes,
            num_classes=num_classes,
            adapt_in_features=model_config.adapt_in_features,
            activation=model_config.activation,
            dropout_prob=model_config.drop_rate,
            normalization=model_config.normalization,
            loss_weight=model_config.weight if hasattr(model_config, "weight") else None,
        )
    elif model_name.lower().startswith(FUSION_TRANSFORMER):
        from .fusion import MultimodalFusionTransformer

        model = functools.partial(
            MultimodalFusionTransformer,
            prefix=model_name,
            hidden_features=model_config.hidden_size,
            num_classes=num_classes,
            num_blocks=model_config.num_blocks,
            attention_num_heads=model_config.attention_num_heads,
            ffn_hidden_size=model_config.ffn_hidden_size,
            attention_dropout=model_config.attention_dropout,
            residual_dropout=model_config.residual_dropout,
            ffn_dropout=model_config.ffn_dropout,
            attention_normalization=model_config.normalization,
            ffn_normalization=model_config.normalization,
            head_normalization=model_config.normalization,
            ffn_activation=model_config.ffn_activation,
            head_activation=model_config.head_activation,
            adapt_in_features=model_config.adapt_in_features,
            aux_loss_weight=model_config.aux_loss_weight,
            additive_attention=model_config.additive_attention,
            share_qv_weights=model_config.share_qv_weights,
        )
    elif model_name.lower().startswith(FT_TRANSFORMER):
        from .ft_transformer import FT_Transformer

        model = FT_Transformer(
            prefix=model_name,
            num_numerical_columns=num_numerical_columns,
            num_categories=num_categories,
            numerical_fill_values=numerical_fill_values,
            embedding_arch=model_config.embedding_arch,
            token_dim=model_config.token_dim,
            hidden_size=model_config.hidden_size,
            hidden_features=model_config.hidden_size,
            num_classes=num_classes,
            num_blocks=model_config.num_blocks,
            attention_num_heads=model_config.attention_num_heads,
            attention_dropout=model_config.attention_dropout,
            attention_normalization=model_config.normalization,
            ffn_hidden_size=model_config.ffn_hidden_size,
            ffn_dropout=model_config.ffn_dropout,
            ffn_normalization=model_config.normalization,
            ffn_activation=model_config.ffn_activation,
            residual_dropout=model_config.residual_dropout,
            head_normalization=model_config.normalization,
            head_activation=model_config.head_activation,
            additive_attention=model_config.additive_attention,
            share_qv_weights=model_config.share_qv_weights,
            pooling_mode=model_config.pooling_mode,
            checkpoint_name=model_config.checkpoint_name,
            pretrained=pretrained,
        )
    elif model_name.lower().startswith(SAM):
        from .sam import SAMForSemanticSegmentation

        model = SAMForSemanticSegmentation(
            prefix=model_name,
            checkpoint_name=model_config.checkpoint_name,
            num_classes=num_classes,
            pretrained=pretrained,
            frozen_layers=model_config.frozen_layers,
            num_mask_tokens=model_config.num_mask_tokens,
            image_norm=model_config.image_norm,
        )
    elif model_name.lower().startswith(NNUNET):
        from .nnunet import ResEncUNetForSemanticSegmentation

        model = ResEncUNetForSemanticSegmentation(
            prefix=model_name,
            num_classes=num_classes,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
            pretrained=pretrained,
            n_stages=model_config.n_stages,
            features_per_stage=tuple(model_config.features_per_stage),
            n_blocks_per_stage=tuple(model_config.n_blocks_per_stage),
            n_conv_per_stage_decoder=tuple(model_config.n_conv_per_stage_decoder),
            deep_supervision=model_config.deep_supervision,
        )
    elif model_name.lower().startswith(META_TRANSFORMER):
        from .meta_transformer import MetaTransformer

        model = MetaTransformer(
            prefix=model_name,
            checkpoint_path=model_config.checkpoint_path,
            num_classes=num_classes,
            model_version=model_config.model_version,
            has_image=IMAGE in model_config.data_types,
            has_text=TEXT in model_config.data_types,
            num_numerical_columns=num_numerical_columns,
            num_categories=num_categories,
            numerical_fill_values=numerical_fill_values,
            image_size=model_config.image_size,
            image_norm=model_config.image_norm,
            image_chan_num=model_config.image_chan_num,
            use_learnable_image=model_config.use_learnable_image,
            max_text_len=model_config.max_text_len,
            text_segment_num=model_config.text_segment_num,
        )
    else:
        raise ValueError(f"unknown model name: {model_name}")

    return model


def create_fusion_model(
    config: DictConfig,
    num_classes: Optional[int] = None,
    classes: Optional[list] = None,
    num_numerical_columns: Optional[int] = None,
    num_categories: Optional[Dict] = None,
    numerical_fill_values: Optional[Dict] = None,
    pretrained: Optional[bool] = True,
):
    """
    Create models. It supports the auto models of huggingface text and timm image.
    Multimodal models, e.g., CLIP, should be added case-by-case since their configs and usages
    may be different. It uses MLP for the numerical features, categorical features, and late-fusion.

    Parameters
    ----------
    config
        A DictConfig object. The model config should be accessible by "config.model".
    num_classes
        The class number for a classification task. It should be 1 for a regression task.
    classes
        All classes in this dataset.
    num_numerical_columns
        The number of numerical columns in the training dataframe.
    num_categories
        The category number for each categorical column in the training dataframe.
    numerical_fill_values
        If numerical values are null, fill them with these.
    pretrained
        Whether using the pretrained timm models. If pretrained=True, download the pretrained model.

    Returns
    -------
    A Pytorch model.
    """
    names = config.model.names
    if isinstance(names, str):
        names = [names]
    # make sure no duplicate model names
    assert len(names) == len(set(names))
    logger.debug(f"output_shape: {num_classes}")
    names = sorted(names)
    config.model.names = names
    single_models = []
    fusion_model = None

    for model_name in names:
        model_config = getattr(config.model, model_name)
        model = create_model(
            model_name=model_name,
            model_config=model_config,
            num_classes=num_classes,
            classes=classes,
            num_numerical_columns=num_numerical_columns,
            num_categories=num_categories,
            numerical_fill_values=numerical_fill_values,
            pretrained=pretrained,
        )

        if isinstance(model, functools.partial):  # fusion model
            if fusion_model is None:
                fusion_model = model
            else:
                raise ValueError(
                    f"More than one fusion models are detected in {names}. Only one fusion model is allowed."
                )
        else:  # single model
            if config.optim.peft is not None:
                model = apply_peft_adaptation(model, config)
            single_models.append(model)

    if len(single_models) > 1:
        # must have one fusion model if there are multiple independent models
        model = fusion_model(models=single_models)
    elif len(single_models) == 1:
        model = single_models[0]
    else:
        raise ValueError(f"No available models for {names}")

    # build augmenter for multimodal data augmentation
    if config.optim.lemda.turn_on:
        from .fusion import MultimodalFusionMLP

        assert isinstance(model, MultimodalFusionMLP)
        from .augmenter import Augmenter

        augmenter = Augmenter(
            arch_type=config.optim.lemda.arch_type,
            input_dim=model.augmenter_in_features,
            z_dim=config.optim.lemda.z_dim,
            num_layers=config.optim.lemda.num_layers,
            adv_weight=config.optim.lemda.adv_weight,
        )
        model.augmenter = augmenter

    return model


def apply_peft_adaptation(model: nn.Module, config: DictConfig) -> nn.Module:
    """
    Apply an adaptation to the model for efficient fine-tuning.

    Parameters
    ----------
    model
        A PyTorch model.
    config:
        A DictConfig object. The optimization config should be accessible by "config.optimization".
    """
    if config.optim.peft in PEFT_ADDITIVE_STRATEGIES:
        # 先注入 Linear（现有逻辑，SAM/Transformer 用）
        model = inject_adaptation_to_linear_layer(
            model=model,
            peft=config.optim.peft,
            lora_r=config.optim.lora.r,
            lora_alpha=config.optim.lora.alpha,
            module_filter=config.optim.lora.module_filter,
            filter=config.optim.lora.filter,
            extra_trainable_params=config.optim.extra_trainable_params,
            conv_lora_expert_num=config.optim.lora.conv_lora_expert_num,
        )
        # 再注入 Conv2d（新增逻辑，nnU-Net 等卷积网络用）
        model = inject_adaptation_to_conv2d_layer(
            model=model,
            peft=config.optim.peft,
            lora_r=config.optim.lora.r,
            lora_alpha=config.optim.lora.alpha,
            module_filter=config.optim.lora.module_filter,
            filter=config.optim.lora.filter,
            extra_trainable_params=config.optim.extra_trainable_params,
            conv_lora_expert_num=config.optim.lora.conv_lora_expert_num,
        )
        model.name_to_id = model.get_layer_ids()  # Need to update name to id dictionary.

    return model


def modify_duplicate_model_names(
    learner,
    postfix: str,
    blacklist: List[str],
):
    """
    Modify a learner's model names if they exist in a blacklist.

    Parameters
    ----------
    learner
        A BaseLearner object.
    postfix
        The postfix used to change the duplicate names.
    blacklist
        A list of names. The provided learner can't use model names in the list.

    Returns
    -------
    The learner guaranteed has no duplicate model names with the blacklist names.
    """
    model_names = []
    for n in learner._config.model.names:
        if n in blacklist:
            new_name = f"{n}_{postfix}"
            assert new_name not in blacklist
            assert new_name not in learner._config.model.names
            # modify model prefix
            if n == learner._model.prefix:
                learner._model.prefix = new_name
            else:
                assert isinstance(learner._model.model, nn.ModuleList)
                for per_model in learner._model.model:
                    if n == per_model.prefix:
                        per_model.prefix = new_name
                        break
            # modify data processor prefix
            for per_modality_processors in learner._data_processors.values():
                for per_processor in per_modality_processors:
                    if n == per_processor.prefix:
                        per_processor.prefix = new_name
            # modify model config keys
            setattr(learner._config.model, new_name, getattr(learner._config.model, n))
            delattr(learner._config.model, n)

            model_names.append(new_name)
        else:
            model_names.append(n)

    learner._config.model.names = model_names

    return learner


def list_timm_models(pretrained=True):
    return timm.list_models(pretrained=pretrained)


def is_lazy_weight_tensor(p: torch.Tensor) -> bool:
    from torch.nn.parameter import UninitializedParameter

    if isinstance(p, UninitializedParameter):
        warnings.warn(
            "A layer with UninitializedParameter was found. "
            "Thus, the total number of parameters detected may be inaccurate."
        )
        return True
    return False

```

### 已完成的融合实现（按你的 1/2/3 选择落地）
- **ResEnc 后端**：新增 AutoGluon 语义分割模型 `nnunet`（ResEnc UNet），输出 **二值 logits**（`[B,1,H,W]`），可被 `create_model("nnunet", ...)` 正常构建。
- **upsample_ratio 专家**：新增 `ConvLoRAConv2d`，在低秩特征上做 **MoE‑Conv + upsample_ratio 专家**，并产出 `moe_loss`。
- **接入 AutoGluon/Conv‑LoRA 脚手架**：扩展 `apply_peft_adaptation`，让 `optim.peft=conv_lora` 同时支持：
  - Linear 注入（原有 SAM/Transformer 逻辑）
  - **Conv2d 注入（新增，用于 nnunet）**
- **MOE_LOSS 端到端**：`nnunet` forward 会自动汇总所有 `ConvLoRAConv2d.moe_loss`，放到输出 dict 的 `MOE_LOSS`，训练时 `SemanticSegmentationLitModule` 会自动把它加到总 loss。

### 关键代码位置
- **Conv2d 版 Conv‑LoRA**：`multimodal/src/autogluon/multimodal/models/adaptation_layers.py`（新增 `ConvLoRAConv2d`）
- **Conv2d 注入**：`multimodal/src/autogluon/multimodal/models/utils.py`（新增 `inject_adaptation_to_conv2d_layer`，并在 `apply_peft_adaptation` 调用）
- **nnunet 后端模型**：`multimodal/src/autogluon/multimodal/models/nnunet.py`（`ResEncUNetForSemanticSegmentation`）
- **模型注册 + 默认配置**：
  - `multimodal/src/autogluon/multimodal/models/utils.py`（`create_model` 增加 `nnunet` 分支）
  - `multimodal/src/autogluon/multimodal/configs/model/default.yaml`（新增 `model.nnunet` 配置块）
- **示例脚本开关**：`examples/automm/Conv-LoRA/run_semantic_segmentation.py` 新增 `--backend {sam,nnunet}`

### 已做过的最小验证（无需跑训练）
- 在 `conv-lora` 环境中验证：
  - `nnunet` 可 import
  - `apply_peft_adaptation(peft=conv_lora)` 不会被 Linear 注入逻辑误伤（已改为跳过非 Linear）
  - 前向输出 logits 维度正确，且 `MOE_LOSS` 能进入输出 dict（`has_moe_loss True`）

### 你现在怎么跑第一轮实验（建议）
- **SAM（原基线）**：

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA
/root/autodl-tmp/envs/conv-lora/bin/python run_semantic_segmentation.py --task isic2017 --backend sam --rank 3 --expert_num 8
```

- **nnUNet(ResEnc) + Conv‑LoRA(Conv2d)**：

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA
/root/autodl-tmp/envs/conv-lora/bin/python run_semantic_segmentation.py --task isic2017 --backend nnunet --rank 4 --expert_num 4
```

如果你希望我下一步把 **nnUNet 的 Dice+CE（更接近 nnU‑Net loss）**、以及 **插入点筛选（只在 bottleneck / encoder 后半段注入）** 也做成可配置开关（便于性能冲顶和消融），你告诉我你想先跑哪个消融顺序（bottleneck → encoder 后半 → 全网络，还是直接 encoder 后半）。

```shellscript
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA
/root/autodl-tmp/envs/conv-lora/bin/python run_semantic_segmentation.py --task isic2017 --backend sam --rank 3 --expert_num 8
```

```shellscript
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA
/root/autodl-tmp/envs/conv-lora/bin/python run_semantic_segmentation.py --task isic2017 --backend nnunet --rank 4 --expert_num 4
```

---

**User**

跑nnUNet(ResEnc) + Conv‑LoRA(Conv2d) 需要先对nnUNet在ISIC2017上做训练才行吗？

---

**Cursor**

不需要。

### 为什么不需要先用 nnUNet CLI 训练
你现在跑的 **`nnUNet(ResEnc) + Conv‑LoRA(Conv2d)`** 是**作为 AutoGluon 的 `nnunet` 后端**来训练的：模型是 `ResEncUNetForSemanticSegmentation`（基于 `dynamic_network_architectures`），参数从随机初始化开始，训练/评估都走 AutoGluon 的 `SemanticSegmentationLearner` 流程，并不会读取 `external/nnUNet` 里 nnUNet CLI 产出的权重。

### 什么时候才“需要先训练”
- **只有当你想做 “加载 nnUNet CLI 训练出来的 checkpoint 再做 Conv‑LoRA 微调/对比”** 时，才需要先跑 nnUNet CLI 的训练（或者你希望用 nnUNet 的 `nnUNet_results` 里的预训练权重做初始化）。

### 你现在直接怎么跑
在 `conv-lora` 环境里直接：

```bash
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA
/root/autodl-tmp/envs/conv-lora/bin/python run_semantic_segmentation.py \
  --task isic2017 --backend nnunet --rank 4 --expert_num 4
```

如果你接下来希望“先跑一个 nnUNet(ResEnc) **不加 Conv‑LoRA** 的纯基线”，我们可以把 `--peft off` 之类的开关加进脚本（目前脚本默认 `optim.peft=conv_lora`）。

```shellscript
cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA
/root/autodl-tmp/envs/conv-lora/bin/python run_semantic_segmentation.py \
  --task isic2017 --backend nnunet --rank 4 --expert_num 4
```

