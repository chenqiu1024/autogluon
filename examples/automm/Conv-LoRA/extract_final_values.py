#!/usr/bin/env python3
"""Extract final IoU values from training log files."""

import re
import os


def extract_final_iou(log_file):
    """Extract the final IoU value from training log."""
    if not os.path.exists(log_file):
        return None, None, None
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Find the last line with IoU
    for line in reversed(lines):
        # Match pattern like: [5000/5000] phase=layer reward=0.9026 iou=0.9126 flops=1.03e+08 active_layers=32.0
        match = re.search(r'\[(\d+)/(\d+)\].*iou=([\d.]+).*active_layers=([\d.]+)', line)
        if match:
            step = int(match.group(1))
            total = int(match.group(2))
            iou = float(match.group(3))
            active_layers = float(match.group(4))
            return step, iou, active_layers
    
    return None, None, None


def main():
    log_files = {
        'Phase 1 (RoutingPolicy)': 'out_rl_train_routing-251119.log',
        'Phase 2 (LayerPolicy)': 'train_hirl_layer_policy-251119.log',
        'Phase 3 (Joint)': 'rl_hier_joint-251119.log',
    }
    
    print("="*80)
    print("FINAL VALUES FROM TRAINING LOG FILES")
    print("="*80)
    print(f"{'Phase':<25} {'Step':<10} {'Final IoU':<15} {'Active Layers':<15}")
    print("-"*80)
    
    results = {}
    for phase_name, log_file in log_files.items():
        step, iou, active_layers = extract_final_iou(log_file)
        results[phase_name] = (step, iou, active_layers)
        
        if step is not None:
            print(f"{phase_name:<25} {step:<10} {iou:<15.4f} {active_layers:<15.1f}")
        else:
            print(f"{phase_name:<25} {'N/A':<10} {'N/A':<15} {'N/A':<15}")
    
    print("="*80)
    
    # Find best
    valid_results = [(name, iou) for name, (step, iou, _) in results.items() if iou is not None]
    if valid_results:
        best_phase, best_iou = max(valid_results, key=lambda x: x[1])
        print(f"\n🥇 Best Final IoU: {best_iou:.4f} ({best_phase})")
    
    print("\n📝 Note: These are the actual final values from training logs,")
    print("   which may differ from TensorBoard's last recorded value due to")
    print("   different logging frequencies (log file: every 50 steps,")
    print("   TensorBoard: every 10 steps with step % 10 == 0).")
    print()
    print("   For Phase 3, the training log shows step 3000 with IoU=0.9442,")
    print("   but TensorBoard's last record is step 2990 with IoU=0.9260.")


if __name__ == "__main__":
    main()

