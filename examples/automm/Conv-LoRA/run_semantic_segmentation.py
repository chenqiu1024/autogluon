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
    
    # GSPO parameters
    parser.add_argument("--gspo_enable", action="store_true", help="Enable GSPO (Group Sequence Policy Optimization) training")
    parser.add_argument("--gspo_group_size", type=int, default=4, help="Number of predictions per group in GSPO")
    parser.add_argument("--gspo_warmup_epochs", type=int, default=5, help="Number of epochs before enabling GSPO")
    parser.add_argument("--gspo_contrastive_weight", type=float, default=0.1, help="Weight for contrastive loss in GSPO")
    parser.add_argument("--gspo_quality_momentum", type=float, default=0.9, help="Momentum for expert quality history")
    
    # Adapter parameters
    parser.add_argument("--adapter_enable", action="store_true", help="Enable standard Adapter modules")
    parser.add_argument("--adapter_dim", type=int, default=64, help="Dimension of the adapter bottleneck")
    
    # AdapterFusion parameters
    parser.add_argument("--adapter_fusion_enable", action="store_true", help="Enable AdapterFusion gating module")
    parser.add_argument("--adapter_fusion_proj_dim", type=int, default=256, help="Projection dim inside fusion")
    parser.add_argument("--adapter_fusion_dropout", type=float, default=0.05, help="Dropout used in fusion output")
    parser.add_argument("--adapter_fusion_gate_init", type=float, default=-4.0, help="Init value for fusion gates")
    parser.add_argument(
        "--adapter_fusion_disable_ffn",
        action="store_true",
        help="Disable optional FFN inside fusion (default: enabled)",
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
        })

    # Adapter configuration
    if args.adapter_enable:
        print(f"Enabling Standard Adapters with dim={args.adapter_dim}")
        hyperparameters.update({
            "model.sam.adapter_enabled": True,
            "model.sam.adapter_dim": args.adapter_dim,
        })
        if args.adapter_fusion_enable:
            print(
                f"Enabling AdapterFusion (proj_dim={args.adapter_fusion_proj_dim}, "
                f"dropout={args.adapter_fusion_dropout}, gate_init={args.adapter_fusion_gate_init}, "
                f"use_ffn={not args.adapter_fusion_disable_ffn})"
            )
            hyperparameters.update({
                "model.sam.adapter_fusion_enabled": True,
                "model.sam.adapter_fusion_proj_dim": args.adapter_fusion_proj_dim,
                "model.sam.adapter_fusion_dropout": args.adapter_fusion_dropout,
                "model.sam.adapter_fusion_gate_init": args.adapter_fusion_gate_init,
                "model.sam.adapter_fusion_use_ffn": not args.adapter_fusion_disable_ffn,
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
        predictor.fit(train_data=train_df, tuning_data=val_df, seed=args.seed)

    # evaluation
    metric_file = os.path.join(args.output_dir, "metrics.txt")
    f = open(metric_file, "a")
    if dataset_name in ["isic2017", "SBU-shadow", "road_segmentation", "leaf_disease_segmentation"]:
        test_df = expand_path(pd.read_csv(os.path.join(dataset_dir, f"test.csv")), dataset_dir)
        if dataset_name == "SBU-shadow":
            eval_metrics = ["ber"]
        else:
            eval_metrics = ["iou", "dice"]  # Evaluate both IoU and DICE

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
