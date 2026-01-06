import argparse
import os

import numpy as np
import pandas as pd
import torch

from autogluon.multimodal import MultiModalPredictor
from bbox_prompt_model import BBoxPromptPredictor


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
        choices=["polyp", "leaf_disease_segmentation", "camo_sem_seg", "isic2017", "isic2018", "road_segmentation", "SBU-shadow"],
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
    
    # GSPO parameters
    parser.add_argument("--gspo_enable", action="store_true", help="Enable GSPO (Group Sequence Policy Optimization) training")
    parser.add_argument("--gspo_group_size", type=int, default=4, help="Number of predictions per group in GSPO")
    parser.add_argument("--gspo_warmup_epochs", type=int, default=5, help="Number of epochs before enabling GSPO")
    parser.add_argument("--gspo_contrastive_weight", type=float, default=0.1, help="Weight for contrastive loss in GSPO")
    parser.add_argument("--gspo_quality_momentum", type=float, default=0.9, help="Momentum for expert quality history")
    # GSPO reward / loss shaping hyper-parameters (可选，带默认值)
    parser.add_argument("--lambda_smooth", type=float, default=0.1, help="Smoothness loss weight (λ_smooth)")
    parser.add_argument("--lambda_boundary", type=float, default=0.3, help="Boundary loss weight (λ_boundary)")
    parser.add_argument("--w_boundary", type=float, default=0.3, help="Boundary reward weight (w_boundary)")
    parser.add_argument("--w_smooth", type=float, default=0.1, help="Smoothness reward weight (w_smooth)")
    parser.add_argument("--w_thin", type=float, default=0.05, help="Thin-structure reward weight (w_thin)")
    
    # Adapter parameters
    parser.add_argument("--adapter_enable", action="store_true", help="Enable standard Adapter modules")
    parser.add_argument("--adapter_dim", type=int, default=64, help="Dimension of the adapter bottleneck")
    
    # GSPO-Adapter extension parameters (Phase 1-2)
    parser.add_argument("--gspo_adapter_enable", action="store_true", 
                        help="Enable GSPO quality feedback for Encoder Adapters (Phase 1)")
    parser.add_argument("--gspo_adapter_momentum", type=float, default=0.9, 
                        help="Momentum for adapter quality history updates")
    parser.add_argument("--gspo_adapter_scale_adaptation", action="store_true",
                        help="Enable adaptive scale based on quality history (Phase 2)")
    
    # Decoder Attention LoRA parameters (LoRA on Attention)
    parser.add_argument("--decoder_attn_lora_enable", action="store_true", 
                        help="Enable LoRA on decoder attention Q/K/V projections")
    parser.add_argument("--decoder_attn_lora_r", type=int, default=8,
                        help="LoRA rank for decoder attention (default: 8)")
    parser.add_argument("--decoder_attn_lora_alpha", type=int, default=8,
                        help="LoRA alpha for decoder attention (default: 8, same as rank)")
    parser.add_argument("--decoder_attn_lora_dropout", type=float, default=0.0,
                        help="LoRA dropout for decoder attention (default: 0.0)")
    
    # GSPO-LoRA on Attention parameters (NEW)
    parser.add_argument("--gspo_lora_attention_enable", action="store_true",
                        help="Enable GSPO quality feedback for Decoder LoRA on Attention")
    parser.add_argument("--gspo_lora_attention_momentum", type=float, default=0.9,
                        help="Momentum for LoRA quality history updates (default: 0.9)")
    parser.add_argument("--gspo_lora_attention_scale_adaptation", action="store_true",
                        help="Enable adaptive scaling based on quality for LoRA")
    parser.add_argument("--gspo_allow_semisup", action="store_true",
                        help="Allow GSPO on labeled subset when semi_labeled_fraction<1.0 (unlabeled uses weak loss).")
    
    # TTA (Test-Time Augmentation) parameters
    parser.add_argument("--tta_enable", action="store_true", help="Enable Test-Time Augmentation for evaluation")
    parser.add_argument("--tta_scales", type=float, nargs="+", default=[0.75, 1.0, 1.25], 
                        help="Scale factors for multi-scale TTA (default: [0.75, 1.0, 1.25]). "
                             "Recommended: [0.75, 1.0, 1.25] or [0.8, 1.0, 1.2]")
    parser.add_argument("--tta_flips", type=str, nargs="+", default=["none", "horizontal"],
                        choices=["none", "horizontal", "vertical"],
                        help="Flip types for TTA (default: ['none', 'horizontal'])")
    parser.add_argument("--tta_rotations", type=float, nargs="+", default=[0],
                        help="Rotation angles in degrees for TTA (default: [0]). "
                             "Recommended: [0] for fast, or [-10, 0, 10] for better")
    parser.add_argument("--tta_fusion", type=str, default="mean", choices=["mean", "weighted_mean"],
                        help="Fusion method for TTA predictions (default: 'mean')")
    parser.add_argument("--tta_resize_method", type=str, default="bilinear", choices=["bilinear", "bicubic"],
                        help="Resize interpolation method for multi-scale TTA (default: 'bilinear'). "
                             "bilinear: faster, bicubic: higher quality")
    parser.add_argument("--tta_threshold", type=float, default=0.5,
                        help="Threshold for binary segmentation in TTA (default: 0.5)")
    parser.add_argument("--tta_min_area", type=float, default=0.001,
                        help="Remove components with area < tta_min_area * image_area (default: 0.001)")
    parser.add_argument("--tta_morphology", action="store_true",
                        help="Apply morphological closing in TTA post-processing")
    parser.add_argument("--tta_cache_dir", type=str, default=None,
                        help="Directory to cache TTA predictions for resume (default: None, no caching)")
    parser.add_argument("--tta_no_resume", action="store_true",
                        help="Disable resume from cache (start fresh even if cache exists)")
    parser.add_argument("--tta_box_prompt_mode", type=str, default="off",
                        choices=["off", "add", "replace"],
                        help="Use GT box as SAM box prompt (off/add/replace); box not transformed with TTA")
    # BBox prompt configuration
    parser.add_argument("--bbox_prompt_source", type=str, default="none",
                        choices=["none", "gt", "predict"],
                        help="Box prompt source: none (no prompt), gt (from mask), predict (from bbox model).")
    parser.add_argument("--bbox_model_ckpt", type=str, default=None,
                        help="Checkpoint for bbox predictor when bbox_prompt_source=predict.")
    parser.add_argument("--bbox_model_image_size", type=int, default=320,
                        help="Input size for bbox predictor inference.")
    parser.add_argument("--bbox_preds_csv", type=str, default=None,
                        help="Optional CSV with columns image,bbox_x1,bbox_y1,bbox_x2,bbox_y2 to attach prompts.")
    # Training-time box prompt mixing (default off)
    parser.add_argument("--train_box_prompt_mode", type=str, default="off",
                        choices=["off", "gt", "noisy", "mix", "predict"],
                        help="Training-time box prompt injection: off/gt/noisy/mix/predict (default off). "
                             "predict: use bbox predictor model to generate box prompts from images.")
    parser.add_argument("--train_box_prob_no_prompt", type=float, default=0.2,
                        help="When mode=mix, probability of no prompt.")
    parser.add_argument("--train_box_prob_gt", type=float, default=0.3,
                        help="When mode=mix, probability of GT box prompt.")
    parser.add_argument("--train_box_prob_noisy", type=float, default=0.5,
                        help="When mode=mix, probability of noisy GT box prompt.")
    parser.add_argument("--train_box_noise_frac", type=float, default=0.12,
                        help="Noisy box jitter fraction relative to box size (e.g., 0.12 => ±12%).")
    parser.add_argument("--train_bbox_model_ckpt", type=str, default=None,
                        help="Checkpoint for bbox predictor when train_box_prompt_mode=predict.")
    parser.add_argument("--train_bbox_model_image_size", type=int, default=320,
                        help="Input size for bbox predictor during training (when train_box_prompt_mode=predict).")
    
    # Semi/weak supervision simulation (ISIC2017 has full masks; we can hide most masks and supervise with coarse boxes)
    parser.add_argument("--semi_labeled_fraction", type=float, default=1.0,
                        help="Fraction of samples to use GT mask loss (e.g., 0.1 => only 10% supervised, rest weak).")
    parser.add_argument("--semi_labeled_seed", type=int, default=0,
                        help="Seed for deterministic labeled/unlabeled assignment via hashing image paths.")
    parser.add_argument("--weak_box_jitter_mode", type=str, default="box",
                        choices=["box", "image", "pixel"],
                        help="How to define weak box jitter amount: "
                             "box=fraction of GT bbox size; image=fraction of full image size; pixel=absolute pixels.")
    parser.add_argument("--weak_box_jitter_amount", type=float, default=0.12,
                        help="Weak box jitter amount (meaning depends on --weak_box_jitter_mode). "
                             "Typical: 0.05~0.3 for box/image, or 5~30 for pixel.")
    parser.add_argument("--weak_box_outward_only", action="store_true",
                        help="If set, weak boxes are generated by outward expansion only (never shrink vs GT bbox).")
    parser.add_argument("--weak_loss_outside_weight", type=float, default=1.0,
                        help="Weight for outside-box background constraint on weak samples.")
    parser.add_argument("--weak_loss_entropy_weight", type=float, default=0.05,
                        help="Weight for inside-box entropy minimization on weak samples.")
    parser.add_argument("--weak_loss_tv_weight", type=float, default=0.0,
                        help="Weight for inside-box total variation smoothness on weak samples.")
    
    # Quick test / Debug parameters
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: only process first 5 images for quick validation")
    parser.add_argument("--quick_test", type=int, default=None,
                        help="Quick test mode: only process first N images")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override default max_epoch for quick smoke tests.")
    
    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (default: auto based on task, typically 1e-4 or 3e-4)")
    
    # Decoder Adapter parameters
    parser.add_argument("--decoder_adapter_enable", action="store_true", help="Enable MLP-Adapter modules in the mask decoder")
    parser.add_argument("--decoder_adapter_dim", type=int, default=64, help="Dimension of the decoder adapter bottleneck")
    args = parser.parse_args()

    dataset_name = args.task
    dataset_dir = os.path.join(f"datasets/{dataset_name}", dataset_name)
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare dataframes
    train_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"train.csv")), dataset_dir)
    val_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"val.csv")), dataset_dir)

    # get the validation metric
    validation_metric, loss, max_epoch, lr = get_default_training_setting(dataset_name)
    if args.max_epochs is not None:
        max_epoch = args.max_epochs
    
    # Override learning rate if user specified it via CLI
    if args.lr is not None:
        lr = args.lr
        print(f"Using custom learning rate: {lr}")

    hyperparameters = {}
    hyperparameters.update(
        {
            "optim.lora.r": args.rank,
            # Paper alignment (Conv-LoRA):
            # - optim.peft: choose Conv-LoRA adaptation (Sec. 3)
            # - optim.lora.r: LoRA rank r (Sec. 3)
            # - optim.lora.conv_lora_expert_num: number of MoE-Conv experts M (Sec. 3.2)
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
    
    # GSPO configuration
    if args.gspo_enable:
        print(f"Enabling GSPO with group_size={args.gspo_group_size}, warmup_epochs={args.gspo_warmup_epochs}")
        hyperparameters.update({
            "optim.lora.gspo_enabled": True,
            "optim.lora.gspo_group_size": args.gspo_group_size,
            "optim.lora.gspo_quality_momentum": args.gspo_quality_momentum,
            "optim.lora.gspo_warmup_epochs": args.gspo_warmup_epochs,
            "optim.lora.gspo_contrastive_weight": args.gspo_contrastive_weight,
            # GSPO reward / loss shaping
            "optim.gspo.lambda_smooth": args.lambda_smooth,
            "optim.gspo.lambda_boundary": args.lambda_boundary,
            "optim.gspo.w_boundary": args.w_boundary,
            "optim.gspo.w_smooth": args.w_smooth,
            "optim.gspo.w_thin": args.w_thin,
            "optim.gspo.allow_semisup": bool(args.gspo_allow_semisup),
        })

    # Adapter configuration
    if args.adapter_enable:
        print(f"Enabling Standard Encoder Adapters with dim={args.adapter_dim}")
        hyperparameters.update({
            "model.sam.adapter_enabled": True,
            "model.sam.adapter_dim": args.adapter_dim,
        })
        
        # GSPO-Adapter extension configuration (Phase 1-2)
        if args.gspo_adapter_enable:
            if not args.gspo_enable:
                print("Warning: --gspo_adapter_enable requires --gspo_enable. Enabling GSPO automatically.")
                args.gspo_enable = True
            print(f"Enabling GSPO-Adapter extension with momentum={args.gspo_adapter_momentum}")
            hyperparameters.update({
                "optim.gspo.adapter_enabled": True,
                "optim.gspo.adapter_momentum": args.gspo_adapter_momentum,
            })
            # Pass scale adaptation to model config
            if args.gspo_adapter_scale_adaptation:
                print("Enabling GSPO-Adapter scale adaptation (Phase 2)")
                hyperparameters.update({
                    "optim.gspo.adapter_scale_adaptation": True,
                    "model.sam.adapter_gspo_enabled": True,
                    "model.sam.adapter_gspo_scale_adaptation": True,
                })
    
    # Decoder Attention LoRA configuration
    if args.decoder_attn_lora_enable:
        gspo_lora_info = ""
        if args.gspo_lora_attention_enable:
            gspo_lora_info = f", GSPO enabled (momentum={args.gspo_lora_attention_momentum}"
            if args.gspo_lora_attention_scale_adaptation:
                gspo_lora_info += ", scale_adaptation=True"
            gspo_lora_info += ")"
        print(f"Enabling Decoder Attention LoRA: r={args.decoder_attn_lora_r}, "
              f"alpha={args.decoder_attn_lora_alpha}, dropout={args.decoder_attn_lora_dropout}{gspo_lora_info}")
        hyperparameters.update({
            "model.sam.decoder_attention_lora_r": args.decoder_attn_lora_r,
            "model.sam.decoder_attention_lora_alpha": args.decoder_attn_lora_alpha,
            "model.sam.decoder_attention_lora_dropout": args.decoder_attn_lora_dropout,
        })
        
        # GSPO-LoRA on Attention configuration (NEW)
        if args.gspo_lora_attention_enable:
            hyperparameters.update({
                "model.sam.gspo_lora_attention_enabled": True,
                "model.sam.gspo_lora_attention_momentum": args.gspo_lora_attention_momentum,
                "model.sam.gspo_lora_attention_scale_adaptation": args.gspo_lora_attention_scale_adaptation,
            })
        
    # Decoder Adapter configuration
    if args.decoder_adapter_enable:
        print(f"Enabling Decoder MLP-Adapters with dim={args.decoder_adapter_dim}")
        hyperparameters.update({
            "model.sam.decoder_adapter_enabled": True,
            "model.sam.decoder_adapter_dim": args.decoder_adapter_dim,
        })

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

        # Configure training-time box prompt mixing
        predictor._learner._train_box_prompt_cfg = {
            "mode": args.train_box_prompt_mode,
            "p_no": args.train_box_prob_no_prompt,
            "p_gt": args.train_box_prob_gt,
            "p_noisy": args.train_box_prob_noisy,
            "noise_frac": args.train_box_noise_frac,
            # Semi/weak supervision simulation
            "semi_labeled_fraction": args.semi_labeled_fraction,
            "semi_labeled_seed": args.semi_labeled_seed,
            "weak_box_jitter_mode": args.weak_box_jitter_mode,
            "weak_box_jitter_amount": args.weak_box_jitter_amount,
            "weak_box_outward_only": bool(args.weak_box_outward_only),
            "weak_loss_outside_weight": args.weak_loss_outside_weight,
            "weak_loss_entropy_weight": args.weak_loss_entropy_weight,
            "weak_loss_tv_weight": args.weak_loss_tv_weight,
        }

        # Setup bbox predictor for training if mode=predict
        if args.train_box_prompt_mode == "predict":
            if not args.train_bbox_model_ckpt:
                raise ValueError("train_box_prompt_mode=predict requires --train_bbox_model_ckpt.")
            train_bbox_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_bbox_predictor = BBoxPromptPredictor(
                ckpt_path=args.train_bbox_model_ckpt,
                device=train_bbox_device,
                image_size=args.train_bbox_model_image_size,
            )
            predictor._learner._train_bbox_predictor = train_bbox_predictor
            print(f"\n[Info] Training-time bbox predictor loaded from: {args.train_bbox_model_ckpt}")
            print(f"       Input size: {args.train_bbox_model_image_size}\n")

        # 打印当前实验的模型保存目录，便于后续通过 AutogluonModels 路径追溯到具体实验日志
        try:
            # MultiModalPredictor 会在内部根据时间戳生成类似 AutogluonModels/ag-YYYYMMDD_HHMMSS 的目录
            model_path = predictor.path
        except AttributeError:
            # 兼容极端情况：如果未来接口变化，没有 path 属性，则显式提示
            model_path = None

        if model_path is not None:
            print("\n========================================")
            print("Training MultiModalPredictor")
            print(f"  Task        : {dataset_name}")
            print(f"  Save path   : {model_path}")
            print("  (You can use this directory name to link back to the training log.)")
            print("========================================\n")
        else:
            print("\n[Warning] MultiModalPredictor has no 'path' attribute. "
                  "Model save directory cannot be printed.\n")

        predictor.fit(train_data=train_df, tuning_data=val_df, seed=args.seed)

    # Enable TTA if requested
    if args.tta_enable:
        print(f"\n{'='*60}")
        print(f"Enabling Test-Time Augmentation (TTA)")
        print(f"  Scales: {args.tta_scales}")
        print(f"  Flips: {args.tta_flips}")
        print(f"  Rotations: {args.tta_rotations}")
        print(f"  Fusion: {args.tta_fusion}")
        print(f"  Resize method: {args.tta_resize_method}")
        print(f"  Total augmentations: {len(args.tta_scales) * len(args.tta_flips) * len(args.tta_rotations)}")
        print(f"{'='*60}\n")
        
        predictor.enable_tta(
            scales=args.tta_scales,
            flips=args.tta_flips,
            rotations=args.tta_rotations,
            fusion_method=args.tta_fusion,
            resize_method=args.tta_resize_method,
            threshold=args.tta_threshold,
            min_area_ratio=args.tta_min_area,
            use_morphology=args.tta_morphology,
            cache_dir=args.tta_cache_dir,
            resume_from_cache=not args.tta_no_resume,
            box_prompt_mode=args.tta_box_prompt_mode,
        )

    # Optional bbox predictor for predicted box prompts
    bbox_predictor = None
    if args.bbox_prompt_source == "predict":
        if not args.bbox_model_ckpt and not args.bbox_preds_csv:
            raise ValueError("bbox_prompt_source=predict requires --bbox_model_ckpt or --bbox_preds_csv.")
        if args.bbox_model_ckpt:
            bbox_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            bbox_predictor = BBoxPromptPredictor(
                ckpt_path=args.bbox_model_ckpt,
                device=bbox_device,
                image_size=args.bbox_model_image_size,
            )
        if args.tta_box_prompt_mode == "off":
            print("[Warning] bbox_prompt_source=predict set but tta_box_prompt_mode=off; box prompts will be ignored.")

    def attach_pred_boxes(df: pd.DataFrame) -> pd.DataFrame:
        """Attach predicted or precomputed boxes to dataframe."""
        df = df.copy()
        if args.bbox_preds_csv:
            preds = pd.read_csv(args.bbox_preds_csv)
            merged = df.merge(preds, on="image", how="left")
            if merged[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].isna().all().all():
                preds["image_basename"] = preds["image"].apply(lambda p: os.path.basename(p))
                merged["image_basename"] = merged["image"].apply(lambda p: os.path.basename(p))
                merged = merged.merge(
                    preds.drop(columns=["image"]), on="image_basename", how="left", suffixes=("", "_pred")
                )
                merged.drop(columns=["image_basename"], inplace=True)
                for col in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]:
                    pred_col = f"{col}_pred"
                    if pred_col in merged.columns:
                        merged[col] = merged[col].fillna(merged[pred_col])
                        merged.drop(columns=[pred_col], inplace=True)
            df = merged
        elif bbox_predictor is not None:
            boxes = []
            for _, r in df.iterrows():
                box = bbox_predictor.predict(r["image"])
                boxes.append(box)
            boxes = np.stack(boxes, axis=0)
            df["bbox_x1"] = boxes[:, 0]
            df["bbox_y1"] = boxes[:, 1]
            df["bbox_x2"] = boxes[:, 2]
            df["bbox_y2"] = boxes[:, 3]
        return df

    # evaluation
    metric_file = os.path.join(args.output_dir, "metrics.txt")
    f = open(metric_file, "a")
    if dataset_name in ["isic2017", "isic2018", "SBU-shadow", "road_segmentation", "leaf_disease_segmentation"]:
        test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test.csv")), dataset_dir)
        
        # Apply quick test / debug mode (方案 1: 快速验证模式)
        original_size = len(test_df)
        if args.debug:
            test_df = test_df.head(5)
            print(f"\n🔍 DEBUG MODE: Processing only first 5 images (out of {original_size})\n")
        elif args.quick_test is not None:
            test_df = test_df.head(args.quick_test)
            print(f"\n🔍 QUICK TEST MODE: Processing only first {args.quick_test} images (out of {original_size})\n")
        
        if dataset_name == "SBU-shadow":
            eval_metrics = ["ber"]
        else:
            eval_metrics = ["iou", "dice"]  # Evaluate both IoU and DICE

        # Attach predicted bbox prompts if requested
        if args.bbox_prompt_source == "predict" and (args.bbox_preds_csv or bbox_predictor):
            test_df = attach_pred_boxes(test_df)

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
            
            # Apply quick test / debug mode (方案 1: 快速验证模式)
            original_size = len(test_df)
            if args.debug:
                test_df = test_df.head(5)
                print(f"\n🔍 DEBUG MODE: Processing only first 5 images (out of {original_size}) for {per_dataset}\n")
            elif args.quick_test is not None:
                test_df = test_df.head(args.quick_test)
                print(f"\n🔍 QUICK TEST MODE: Processing only first {args.quick_test} images (out of {original_size}) for {per_dataset}\n")
            
            if args.bbox_prompt_source == "predict" and (args.bbox_preds_csv or bbox_predictor):
                test_df = attach_pred_boxes(test_df)

            res = predictor.evaluate(test_df, metrics=["sm", "fm", "em", "mae"])
            print(f"Evaluation results for test dataset {per_dataset}: ", res)
            f.write(f"Evaluation results for test dataset {per_dataset}: {res} \n")
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}.")

    f.close()
