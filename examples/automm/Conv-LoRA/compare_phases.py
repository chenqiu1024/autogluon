#!/usr/bin/env python3
"""Compare training curves across all three phases."""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def load_scalar(log_dir, tag):
    """Load scalar data from TensorBoard logs."""
    try:
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    except Exception as e:
        print(f"Warning: Failed to load {tag} from {log_dir}: {e}")
        return [], []


def main():
    # Define phase directories
    phases = {
        'Phase 1: RoutingPolicy': 'rl_routing_schemeB-251119/logs',
        'Phase 2: LayerPolicy': 'rl_hier_layer_only-251119/logs',
        'Phase 3: Joint': 'rl_hier_joint-251119/logs',
    }
    
    # Metrics to compare
    metrics = [
        ('train/iou', 'Training IoU', 'IoU'),
        ('train/reward', 'Training Reward', 'Reward'),
        ('train/imbalance', 'Expert Imbalance', 'Imbalance'),
        ('train/active_layers', 'Active Layers', 'Layers'),
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, (tag, title, ylabel) in enumerate(metrics):
        ax = axes[idx]
        
        for phase_idx, (phase_name, log_dir) in enumerate(phases.items()):
            if not os.path.exists(log_dir):
                print(f"Warning: {log_dir} not found, skipping {phase_name}")
                continue
            
            steps, values = load_scalar(log_dir, tag)
            
            if len(steps) > 0:
                # Plot raw data
                ax.plot(steps, values, alpha=0.3, color=colors[phase_idx], linewidth=1)
                
                # Plot smoothed data
                if len(values) > 50:
                    from scipy.ndimage import gaussian_filter1d
                    try:
                        smoothed = gaussian_filter1d(values, sigma=20)
                        ax.plot(steps, smoothed, label=phase_name, 
                               color=colors[phase_idx], linewidth=2.5)
                    except:
                        ax.plot(steps, values, label=phase_name, 
                               color=colors[phase_idx], linewidth=2)
                else:
                    ax.plot(steps, values, label=phase_name, 
                           color=colors[phase_idx], linewidth=2)
                
                # Print summary
                print(f"\n{phase_name} - {title}:")
                print(f"  Initial: {values[0]:.4f}")
                print(f"  Final: {values[-1]:.4f}")
                print(f"  Best: {max(values) if 'iou' in tag or 'reward' in tag else min(values):.4f}")
                print(f"  Mean: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
        ax.set_xlabel('Training Step', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = 'phase_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved to: {output_path}")
    
    # Create a summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Metric':<20} {'Phase 1':<15} {'Phase 2':<15} {'Phase 3':<15} {'Best':<10}")
    print("-"*80)
    
    summary_metrics = [
        ('train/iou', 'Final IoU', 'higher'),
        ('train/iou', 'Mean IoU', 'higher'),
        ('train/imbalance', 'Imbalance', 'lower'),
        ('train/active_layers', 'Active Layers', 'none'),
    ]
    
    for tag, name, better in summary_metrics:
        values_by_phase = []
        for log_dir in phases.values():
            if os.path.exists(log_dir):
                _, values = load_scalar(log_dir, tag)
                if len(values) > 0:
                    if 'Mean' in name:
                        val = np.mean(values)
                    elif 'Final' in name:
                        val = values[-1]
                    else:
                        val = np.mean(values)
                    values_by_phase.append(val)
                else:
                    values_by_phase.append(0)
            else:
                values_by_phase.append(0)
        
        # Determine best
        if better == 'higher':
            best_idx = np.argmax(values_by_phase)
        elif better == 'lower':
            best_idx = np.argmin(values_by_phase)
        else:
            best_idx = -1
        
        row = f"{name:<20}"
        for idx, val in enumerate(values_by_phase):
            marker = " 🥇" if idx == best_idx else ""
            row += f"{val:.4f}{marker:<7}"
        
        if best_idx >= 0:
            row += f"Phase {best_idx + 1}"
        else:
            row += "-"
        
        print(row)
    
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

