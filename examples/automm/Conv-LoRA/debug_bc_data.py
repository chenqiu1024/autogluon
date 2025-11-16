"""
Debug script to analyze BC data collection.

Usage:
    python debug_bc_data.py --model_path AutogluonModels/.../epoch-X.ckpt --max_samples 20
"""

import argparse
import sys
from pathlib import Path

import torch
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "multimodal" / "src"))

from rl_utils import load_trained_conv_lora_model, prepare_dataset
from rl_bc_routing_policy import collect_noisy_topk_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="isic2017")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    predictor, sam_model, conv_lora_layers = load_trained_conv_lora_model(
        args.task,
        args.model_path,
        device=args.device
    )
    
    # Load data
    train_df, _ = prepare_dataset(args.task, split='train')
    
    # Collect BC data
    print(f"\nCollecting BC data from {args.max_samples} samples...")
    bc_dataset = collect_noisy_topk_actions(
        sam_model=sam_model,
        conv_lora_layers=conv_lora_layers,
        train_df=train_df,
        predictor=predictor,
        max_samples=args.max_samples,
        device=args.device,
    )
    
    # Analyze
    print("\n" + "="*60)
    print("BC Data Analysis")
    print("="*60)
    
    print(f"\nTotal samples: {len(bc_dataset)}")
    print(f"Samples per layer: ~{len(bc_dataset) // len(conv_lora_layers)}")
    
    # Feature shapes
    shapes = Counter([tuple(s['feats'].shape) for s in bc_dataset])
    print(f"\nFeature shapes:")
    for shape, count in shapes.most_common():
        print(f"  {shape}: {count} samples ({count/len(bc_dataset)*100:.1f}%)")
    
    # Expert distribution overall
    actions = [s['action'] for s in bc_dataset]
    action_dist = Counter(actions)
    print(f"\nExpert usage (overall):")
    for expert, count in sorted(action_dist.items()):
        print(f"  Expert {expert}: {count} ({count/len(actions)*100:.1f}%)")
    
    # Expert distribution per layer
    print(f"\nExpert usage per layer:")
    for layer_idx in range(len(conv_lora_layers)):
        layer_samples = [s for s in bc_dataset if s['layer_idx'] == layer_idx]
        if layer_samples:
            layer_actions = [s['action'] for s in layer_samples]
            layer_dist = Counter(layer_actions)
            most_common = layer_dist.most_common(3)
            print(f"  Layer {layer_idx}: {len(layer_samples)} samples, "
                  f"top experts: {[(e, c/len(layer_samples)*100) for e, c in most_common]}")
    
    # Uniformity check
    import numpy as np
    action_counts = np.array([action_dist.get(i, 0) for i in range(8)])
    uniformity = action_counts.std() / (action_counts.mean() + 1e-8)
    print(f"\nUniformity (CV): {uniformity:.3f}")
    print(f"  (0.0 = perfectly uniform, >1.0 = highly skewed)")
    
    # Expected BC accuracy
    if len(actions) > 0:
        most_common_action = action_dist.most_common(1)[0]
        majority_baseline = most_common_action[1] / len(actions)
        print(f"\nExpected BC accuracy:")
        print(f"  Random baseline: {1/8*100:.1f}%")
        print(f"  Majority class baseline: {majority_baseline*100:.1f}%")
        print(f"  Good BC (if learning patterns): 60-80%")


if __name__ == "__main__":
    main()

