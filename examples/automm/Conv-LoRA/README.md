# Conv-LoRA: Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model (ICLR 2024)

Examples showing how to use `Conv-LoRA` for parameter efficient fine-tuning SAM.

## 1. Installation
The installation may take a while since AutoGluon Multimodal has multiple dependencies.
```shell
  conda create -n conv-lora python=3.10
  conda activate conv-lora
  pip install -U pip
  pip install -U setuptools wheel
  git clone https://github.com/autogluon/autogluon
  cd autogluon && pip install -e multimodal/[tests]
  ```

## 2. Dataset

Enter the `autogluon/examples/automm/Conv-LoRA` directory and run the following script to download the datasets.

`python prepare_semantic_segmentation_datasets.py`

## 3. Training

`python run_semantic_segmentation.py --<flag> <value>`

- `task` refers to the dataset name, i.e., one of the datasets we have downloaded. Options are `polyp, leaf_disease_segmentation, camo_sem_seg, isic2017, road_segmentation, or SBU-shadow`.
- `seed` determines the random seed.
- `rank` determines the rank of Conv-LoRA. Default is 3.
- `expert_num` determines the used expert number of Conv-LoRA. Default is 8.
- `num_gpus` determines the number of gpu used for training. Default is 1.
- `output_dir` determines the path of output directory. Default is "outputs" folder.
- `ckpt_path` determines the path of model for evaluation. Default is "outputs" folder.
- `per_gpu_batch_size` is the batch size for each GPU. Default is 1.
- `batch_size` effective batch size. If batch_size > per_gpu_batch_size * num_gpus, gradient accumulation would be used. Default is 4.

## 4. Evaluation

After running the benchmark, the evaluation results of test set are stored in "{output_dir}/metrics.txt".

You can also run the following command to evaluate a checkpoint:

`python3 run_semantic_segmentation.py --task {dataset_name} --output_dir {output_dir} --ckpt_path {ckpt_path} --eval`

## 5. RL-based Dynamic Layer Selection (NEW!)

We now support using Reinforcement Learning to dynamically select which Conv-LoRA layers to activate for each input image, improving segmentation performance.

### Quick Start

**中文用户请查看**: [中文使用指南.md](中文使用指南.md) - 详细的中文教程，从训练baseline到评估RL模型

**English users see**: [RL_LAYER_SELECTION_README.md](RL_LAYER_SELECTION_README.md) - Complete guide

### Three-Step Workflow

1. **Train baseline Conv-LoRA model** (all 32 layers active)
```bash
python run_semantic_segmentation.py --task isic2017 --output_dir baseline_conv_lora
```

2. **Train RL policy** to learn layer selection
```bash
python train_rl_layer_selection.py \
    --task isic2017 \
    --ckpt_path AutogluonModels/ag-YYYYMMDD_HHMMSS \
    --output_dir rl_layer_selection \
    --num_episodes 1000
```

3. **Evaluate RL policy** on test set
```bash
python evaluate_rl_policy.py \
    --task isic2017 \
    --ckpt_path AutogluonModels/ag-YYYYMMDD_HHMMSS \
    --policy_path rl_layer_selection/checkpoints/best.pt \
    --output_dir rl_evaluation
```

### Expected Results

- **Performance**: +1-2% IoU/DICE improvement over baseline
- **Efficiency**: ~70% of layers activated (22/32 on average)
- **Adaptivity**: Different images use different layer combinations

### Documentation

- **中文详细指南**: [中文使用指南.md](中文使用指南.md) - 完整的中文教程
- **English Guide**: [RL_LAYER_SELECTION_README.md](RL_LAYER_SELECTION_README.md)
- **Implementation Details**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **实现完成报告**: [实现完成报告.txt](实现完成报告.txt)

### Citation

```
@article{zhong2024convolution,
  title={Convolution Meets LoRA: Parameter Efficient Finetuning for Segment Anything Model},
  author={Zhong, Zihan and Tang, Zhiqiang and He, Tong and Fang, Haoyang and Yuan, Chun},
  journal={arXiv preprint arXiv:2401.17868},
  year={2024}
}
```