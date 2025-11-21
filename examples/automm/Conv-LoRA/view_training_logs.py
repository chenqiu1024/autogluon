#!/usr/bin/env python3
"""Quick script to view training logs from RL routing policy training."""

import os
import sys
from pathlib import Path

try:
    from tensorboard.backend.event_processing import event_accumulator
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    MISSING_DEP = str(e)


def load_tb_logs(log_dir):
    """Load TensorBoard logs."""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    return ea


def extract_scalar(ea, tag):
    """Extract scalar values from event accumulator."""
    try:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    except KeyError:
        return [], []


def main():
    if not HAS_DEPS:
        print(f"❌ Missing dependencies: {MISSING_DEP}")
        print("\nPlease install required packages:")
        print("  pip install tensorboard matplotlib numpy")
        return 1
    
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "rl_routing_schemeB-251119/logs"
    
    # Convert to absolute path if relative
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(os.path.dirname(__file__), log_dir)
    
    if not os.path.exists(log_dir):
        print(f"❌ Log directory not found: {log_dir}")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} <log_directory>")
        print("\nExample:")
        print(f"  python {sys.argv[0]} rl_routing_schemeB-251119/logs")
        return 1
    
    print(f"📊 Loading logs from: {log_dir}")
    ea = load_tb_logs(log_dir)
    
    # Print available tags
    available_tags = ea.Tags()['scalars']
    print(f"\n✅ Available metrics ({len(available_tags)} total):")
    for tag in sorted(available_tags):
        print(f"  - {tag}")
    
    # Extract key metrics
    metrics = {
        'IoU': 'train/iou',
        'Reward': 'train/reward',
        'FLOPs': 'train/flops',
        'Imbalance': 'train/imbalance',
        'Total Loss': 'loss/total',
        'KL Loss': 'loss/kl',
    }
    
    print("\n" + "="*80)
    print("📈 TRAINING SUMMARY STATISTICS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (name, tag) in enumerate(metrics.items()):
        steps, values = extract_scalar(ea, tag)
        if len(steps) > 0:
            axes[idx].plot(steps, values, linewidth=2, color='#2E86AB')
            axes[idx].set_title(name, fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Training Step', fontsize=11)
            axes[idx].set_ylabel(name, fontsize=11)
            axes[idx].grid(True, alpha=0.3, linestyle='--')
            
            # Add smoothed line for noisy metrics
            if len(values) > 50:
                from scipy.ndimage import gaussian_filter1d
                try:
                    smoothed = gaussian_filter1d(values, sigma=10)
                    axes[idx].plot(steps, smoothed, linewidth=2.5, color='#A23B72', 
                                 alpha=0.8, label='Smoothed')
                    axes[idx].legend(fontsize=9)
                except:
                    pass
            
            # Print summary statistics
            print(f"\n{name} ({tag}):")
            print(f"  📍 Initial:  {values[0]:>12.6f}  (step {steps[0]})")
            print(f"  🎯 Final:    {values[-1]:>12.6f}  (step {steps[-1]})")
            
            if 'iou' in tag.lower() or 'reward' in tag.lower():
                best_val = max(values)
                best_step = steps[values.index(best_val)]
                print(f"  ⭐ Best:     {best_val:>12.6f}  (step {best_step})")
            else:
                best_val = min(values)
                best_step = steps[values.index(best_val)]
                print(f"  ⭐ Best:     {best_val:>12.6f}  (step {best_step})")
            
            print(f"  📊 Mean:     {np.mean(values):>12.6f}")
            print(f"  📏 Std:      {np.std(values):>12.6f}")
            
            # Calculate improvement
            if len(values) >= 100:
                initial_avg = np.mean(values[:100])
                final_avg = np.mean(values[-100:])
                improvement = final_avg - initial_avg
                improvement_pct = (improvement / abs(initial_avg)) * 100 if initial_avg != 0 else 0
                
                if 'loss' in tag.lower() or 'imbalance' in tag.lower() or 'flops' in tag.lower():
                    # Lower is better
                    print(f"  📉 Change:   {improvement:>12.6f}  ({improvement_pct:+.2f}%)")
                else:
                    # Higher is better
                    print(f"  📈 Change:   {improvement:>12.6f}  ({improvement_pct:+.2f}%)")
        else:
            axes[idx].text(0.5, 0.5, f'No data for {name}', 
                          ha='center', va='center', fontsize=12)
            axes[idx].set_title(name, fontsize=14, fontweight='bold')
            print(f"\n{name} ({tag}):")
            print(f"  ❌ No data available")
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(log_dir), "training_summary.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    print("\n" + "="*80)
    print(f"✅ Training curves saved to: {output_path}")
    print("="*80)
    
    # Additional metrics if available
    print("\n📋 Additional Metrics:")
    other_metrics = ['train/dual_alpha', 'loss/entropy', 'loss/adapter_l2']
    for tag in other_metrics:
        steps, values = extract_scalar(ea, tag)
        if len(steps) > 0:
            print(f"  {tag}: {values[0]:.6f} → {values[-1]:.6f}")
    
    print("\n💡 Tip: To view interactively, run:")
    print(f"   tensorboard --logdir {log_dir} --port 6006")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

