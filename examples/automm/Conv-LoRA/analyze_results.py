"""
Results Analysis Tool for GSPO-ConvLoRA Experiments

This script parses metrics from different experiment runs and generates
comparison tables and visualizations.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


def parse_metrics_file(metrics_path: str) -> Dict:
    """
    Parse metrics.txt file and extract evaluation results.
    
    Args:
        metrics_path: Path to metrics.txt file
        
    Returns:
        Dictionary containing parsed metrics
    """
    if not os.path.exists(metrics_path):
        return {}
    
    metrics = {}
    with open(metrics_path, 'r') as f:
        content = f.read()
        
        # Extract IoU
        iou_match = re.search(r"'iou':\s*([\d.]+)", content)
        if iou_match:
            metrics['iou'] = float(iou_match.group(1))
        
        # Extract DICE
        dice_match = re.search(r"'dice':\s*([\d.]+)", content)
        if dice_match:
            metrics['dice'] = float(dice_match.group(1))
    
    return metrics


def collect_all_results(base_dir: str = "outputs") -> Dict[str, Dict]:
    """
    Collect results from all experiment directories.
    
    Args:
        base_dir: Base directory containing experiment outputs
        
    Returns:
        Dictionary mapping experiment names to their metrics
    """
    results = {}
    
    experiment_dirs = [
        ("Baseline Conv-LoRA", "baseline_convlora"),
        ("GSPO (group=3)", "gspo_convlora_g3"),
        ("GSPO (group=4)", "gspo_convlora_g4"),
        ("GSPO (group=6)", "gspo_convlora_g6"),
    ]
    
    for exp_name, dir_name in experiment_dirs:
        metrics_path = os.path.join(base_dir, dir_name, "metrics.txt")
        metrics = parse_metrics_file(metrics_path)
        if metrics:
            results[exp_name] = metrics
    
    return results


def generate_comparison_table(results: Dict[str, Dict]) -> str:
    """
    Generate a formatted comparison table.
    
    Args:
        results: Dictionary of experiment results
        
    Returns:
        Formatted table string
    """
    if not results:
        return "No results found."
    
    # Table header
    table = "=" * 70 + "\n"
    table += "GSPO-ConvLoRA Experimental Results Comparison\n"
    table += "=" * 70 + "\n\n"
    
    # Column headers
    table += f"{'Method':<25} | {'IoU':<10} | {'DICE':<10} | {'Improvement':<15}\n"
    table += "-" * 70 + "\n"
    
    # Baseline metrics
    baseline_name = "Baseline Conv-LoRA"
    baseline_metrics = results.get(baseline_name, {})
    baseline_iou = baseline_metrics.get('iou', 0.0)
    baseline_dice = baseline_metrics.get('dice', 0.0)
    
    # Add baseline row
    if baseline_metrics:
        table += f"{baseline_name:<25} | {baseline_iou:>9.4f} | {baseline_dice:>9.4f} | {'(baseline)':<15}\n"
    
    # Add GSPO results with improvements
    for method_name, metrics in results.items():
        if method_name == baseline_name:
            continue
        
        iou = metrics.get('iou', 0.0)
        dice = metrics.get('dice', 0.0)
        
        # Calculate improvements
        if baseline_iou > 0:
            iou_improve = ((iou - baseline_iou) / baseline_iou) * 100
            improvement_str = f"+{iou_improve:.2f}% IoU"
        else:
            improvement_str = "N/A"
        
        table += f"{method_name:<25} | {iou:>9.4f} | {dice:>9.4f} | {improvement_str:<15}\n"
    
    table += "=" * 70 + "\n"
    
    return table


def plot_comparison_chart(results: Dict[str, Dict], output_path: str = "outputs/comparison_chart.png"):
    """
    Generate a bar chart comparing different methods.
    
    Args:
        results: Dictionary of experiment results
        output_path: Path to save the chart
    """
    if not results:
        print("No results to plot.")
        return
    
    methods = list(results.keys())
    iou_scores = [results[m].get('iou', 0) for m in methods]
    dice_scores = [results[m].get('dice', 0) for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, iou_scores, width, label='IoU', color='#3498db')
    bars2 = ax.bar(x + width/2, dice_scores, width, label='DICE', color='#e74c3c')
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('GSPO-ConvLoRA Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)
    
    autolabel(bars1)
    autolabel(bars2)
    
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison chart saved to: {output_path}")


def generate_ablation_analysis(results: Dict[str, Dict]) -> str:
    """
    Generate ablation study analysis focusing on group size impact.
    
    Args:
        results: Dictionary of experiment results
        
    Returns:
        Formatted analysis string
    """
    analysis = "\n" + "=" * 70 + "\n"
    analysis += "Ablation Study: Impact of Group Size\n"
    analysis += "=" * 70 + "\n\n"
    
    gspo_results = {k: v for k, v in results.items() if 'GSPO' in k}
    
    if len(gspo_results) < 2:
        return analysis + "Insufficient GSPO results for ablation analysis.\n"
    
    # Extract group sizes and metrics
    group_data = []
    for method, metrics in gspo_results.items():
        match = re.search(r'group=(\d+)', method)
        if match:
            group_size = int(match.group(1))
            iou = metrics.get('iou', 0)
            dice = metrics.get('dice', 0)
            group_data.append((group_size, iou, dice))
    
    # Sort by group size
    group_data.sort(key=lambda x: x[0])
    
    analysis += f"{'Group Size':<15} | {'IoU':<10} | {'DICE':<10}\n"
    analysis += "-" * 40 + "\n"
    
    for group_size, iou, dice in group_data:
        analysis += f"{group_size:<15} | {iou:>9.4f} | {dice:>9.4f}\n"
    
    # Find optimal group size
    if group_data:
        best_iou_idx = max(range(len(group_data)), key=lambda i: group_data[i][1])
        best_group_size, best_iou, _ = group_data[best_iou_idx]
        
        analysis += "\n"
        analysis += f"Optimal group size: {best_group_size} (IoU: {best_iou:.4f})\n"
    
    analysis += "=" * 70 + "\n"
    
    return analysis


def save_results_json(results: Dict[str, Dict], output_path: str = "outputs/results_summary.json"):
    """
    Save results to JSON for programmatic access.
    
    Args:
        results: Dictionary of experiment results
        output_path: Path to save JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to JSON: {output_path}")


def main():
    """Main analysis function."""
    print("Analyzing GSPO-ConvLoRA Experimental Results...")
    print("-" * 70)
    
    # Collect results
    results = collect_all_results()
    
    if not results:
        print("ERROR: No results found in outputs/ directory.")
        print("Please run experiments first using run_gspo_experiments.sh")
        return
    
    print(f"Found results for {len(results)} experiments.")
    print()
    
    # Generate comparison table
    table = generate_comparison_table(results)
    print(table)
    
    # Generate ablation analysis
    ablation = generate_ablation_analysis(results)
    print(ablation)
    
    # Save to file
    summary_path = "outputs/results_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(table)
        f.write("\n")
        f.write(ablation)
    print(f"Summary saved to: {summary_path}")
    
    # Generate visualization
    try:
        plot_comparison_chart(results)
    except Exception as e:
        print(f"Warning: Could not generate chart: {e}")
        print("(matplotlib may not be available)")
    
    # Save JSON
    save_results_json(results)
    
    print()
    print("Analysis complete!")


if __name__ == "__main__":
    main()

