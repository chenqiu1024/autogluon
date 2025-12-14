"""
Train a lightweight bounding-box regressor on ISIC2017 (or similar) masks.

Usage (default paths assume datasets/isic2017/isic2017):
    python3 run_bbox_predictor.py --data_dir datasets/isic2017/isic2017 --output_dir outputs/bbox_isic2017
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from bbox_prompt_model import (
    BBoxDataset,
    BBoxPromptPredictor,
    BBoxRegressor,
    collate_fn,
    export_pred_csv,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
    validate,
)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Train a bbox regressor for SAM box prompts.")
    parser.add_argument("--data_dir", type=str, default="datasets/isic2017/isic2017")
    parser.add_argument("--output_dir", type=str, default="outputs/bbox_isic2017")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--rotation_deg", type=float, default=10.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of train used for val if val.csv missing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained_backbone", action="store_true", help="Use ImageNet pretrain for ResNet-18.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume.")
    parser.add_argument("--export_preds", action="store_true", help="Export bbox predictions on test set after training.")
    parser.add_argument("--test_csv", type=str, default=None, help="Optional custom test csv (defaults to test.csv under data_dir).")
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g., cuda:0 or cpu.")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    os.makedirs(args.output_dir, exist_ok=True)

    train_csv = os.path.join(args.data_dir, "train.csv")
    val_csv = os.path.join(args.data_dir, "val.csv")

    if os.path.exists(val_csv):
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
    else:
        full_df = pd.read_csv(train_csv)
        val_size = max(1, int(len(full_df) * args.val_split))
        train_size = len(full_df) - val_size
        train_df, val_df = random_split(full_df, [train_size, val_size])
        train_df = pd.DataFrame(train_df.dataset.iloc[train_df.indices])
        val_df = pd.DataFrame(val_df.dataset.iloc[val_df.indices])

    def expand(df):
        df = df.copy()
        df["image"] = df["image"].apply(lambda p: os.path.join(args.data_dir, p))
        df["label"] = df["label"].apply(lambda p: os.path.join(args.data_dir, p))
        return df

    train_df = expand(train_df)
    val_df = expand(val_df)

    train_dataset = BBoxDataset(
        train_df, image_size=args.image_size, augment=True, rotation_deg=args.rotation_deg
    )
    val_dataset = BBoxDataset(
        val_df, image_size=args.image_size, augment=False, rotation_deg=0.0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = BBoxRegressor(pretrained=args.pretrained_backbone).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_iou = -1.0
    best_path = os.path.join(args.output_dir, "best_bbox.pth")

    if args.resume:
        ckpt = load_checkpoint(model, args.resume, device)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        if "best_iou" in ckpt:
            best_iou = ckpt["best_iou"]
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, device)
        scheduler.step()

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_l1": val_metrics["l1"],
            "val_iou": val_metrics["iou"],
            "lr": scheduler.get_last_lr()[0],
        }
        print(json.dumps(log, indent=2))

        is_best = val_metrics["iou"] > best_iou
        if is_best:
            best_iou = val_metrics["iou"]
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_iou": best_iou,
                },
                best_path,
            )

    # Save final
    save_checkpoint(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": args.epochs - 1,
            "best_iou": best_iou,
        },
        os.path.join(args.output_dir, "last_bbox.pth"),
    )

    # Optionally export predictions on test set
    if args.export_preds:
        test_csv = args.test_csv or os.path.join(args.data_dir, "test.csv")
        if os.path.exists(test_csv):
            test_df = pd.read_csv(test_csv)
            test_df = expand(test_df)
            predictor = BBoxPromptPredictor(
                ckpt_path=best_path,
                device=device,
                image_size=args.image_size,
            )
            out_csv = os.path.join(args.output_dir, "test_bbox_preds.csv")
            export_pred_csv(test_df, predictor, out_csv)
            print(f"Exported test bbox predictions to {out_csv}")
        else:
            print(f"test.csv not found at {test_csv}, skip export.")

    # Save training summary
    summary = {
        "best_iou": best_iou,
        "best_ckpt": best_path,
        "epochs": args.epochs,
        "image_size": args.image_size,
        "rotation_deg": args.rotation_deg,
    }
    with open(os.path.join(args.output_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
