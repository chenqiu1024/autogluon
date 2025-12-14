#!/usr/bin/env python3
"""
TTA 实验结果分析脚本

用法:
    python analyze_tta_results.py --results_dir outputs/ --output results_summary.md
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_metrics_file(filepath: str) -> Dict[str, float]:
    """解析 metrics.txt 文件"""
    metrics = {}
    
    if not os.path.exists(filepath):
        return metrics
    
    with open(filepath, 'r') as f:
        content = f.read()
        
        # 匹配 'iou': 0.7860, 'dice': 0.8639 格式
        iou_match = re.search(r"'iou':\s*([\d.]+)", content)
        dice_match = re.search(r"'dice':\s*([\d.]+)", content)
        
        if iou_match:
            metrics['iou'] = float(iou_match.group(1))
        if dice_match:
            metrics['dice'] = float(dice_match.group(1))
    
    return metrics


def parse_experiment_name(exp_name: str) -> Dict[str, str]:
    """从实验名称中解析配置信息"""
    config = {
        'name': exp_name,
        'scales': '-',
        'resize': '-',
        'flips': '-',
        'rotations': '-',
        'threshold': '-',
        'morphology': 'N',
        'fusion': '-',
    }
    
    # 解析 scales
    if 'scales_' in exp_name:
        scales_match = re.search(r'scales_([\d_]+)', exp_name)
        if scales_match:
            scales_str = scales_match.group(1).replace('_', ', ')
            config['scales'] = f"[{scales_str}]"
    
    # 解析 resize method
    if 'bilinear' in exp_name:
        config['resize'] = 'bilinear'
    elif 'bicubic' in exp_name:
        config['resize'] = 'bicubic'
    
    # 解析 flips
    if 'flip_h' in exp_name and 'flip_hv' not in exp_name:
        config['flips'] = '[n, h]'
    elif 'flip_hv' in exp_name:
        config['flips'] = '[n, h, v]'
    
    # 解析 rotations
    if 'rot_' in exp_name:
        if 'rot_10' in exp_name:
            config['rotations'] = '[-10, 0, 10]'
        elif 'rot_5' in exp_name:
            config['rotations'] = '[-5, 0, 5]'
    
    # 解析 threshold
    thresh_match = re.search(r'thresh_(\d+)', exp_name)
    if thresh_match:
        thresh_str = thresh_match.group(1)
        config['threshold'] = f"0.{thresh_str}"
    
    # 解析 morphology
    if 'morphology' in exp_name:
        config['morphology'] = 'Y'
    
    # 解析 fusion
    if 'weighted' in exp_name:
        config['fusion'] = 'weighted'
    elif any(key in exp_name for key in ['exp_', 'quick_']):
        config['fusion'] = 'mean'
    
    return config


def collect_results(results_dir: str) -> List[Tuple[str, Dict[str, str], Dict[str, float]]]:
    """收集所有实验结果"""
    results = []
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"错误: 结果目录不存在: {results_dir}")
        return results
    
    for exp_dir in sorted(results_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        metrics_file = exp_dir / "metrics.txt"
        
        if not metrics_file.exists():
            print(f"警告: {exp_name} 没有 metrics.txt 文件，跳过")
            continue
        
        metrics = parse_metrics_file(str(metrics_file))
        if not metrics:
            print(f"警告: {exp_name} 的 metrics.txt 无法解析，跳过")
            continue
        
        config = parse_experiment_name(exp_name)
        results.append((exp_name, config, metrics))
    
    return results


def generate_markdown_report(results: List[Tuple[str, Dict[str, str], Dict[str, float]]], 
                            output_file: str):
    """生成 Markdown 格式的报告"""
    
    if not results:
        print("错误: 没有找到任何实验结果")
        return
    
    # 按 IoU 降序排序
    results_sorted = sorted(results, key=lambda x: x[2].get('iou', 0), reverse=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# TTA 实验结果汇总\n\n")
        f.write(f"共 {len(results)} 个实验\n\n")
        
        # 写入表格
        f.write("## 实验结果对比\n\n")
        f.write("| Rank | 实验名称 | Scales | Resize | Flips | Rotations | Threshold | Morph | Fusion | IoU | Dice |\n")
        f.write("|------|----------|--------|--------|-------|-----------|-----------|-------|--------|-----|------|\n")
        
        for rank, (exp_name, config, metrics) in enumerate(results_sorted, 1):
            iou = metrics.get('iou', 0)
            dice = metrics.get('dice', 0)
            
            f.write(f"| {rank} | {exp_name} | {config['scales']} | {config['resize']} | "
                   f"{config['flips']} | {config['rotations']} | {config['threshold']} | "
                   f"{config['morphology']} | {config['fusion']} | "
                   f"{iou:.4f} | {dice:.4f} |\n")
        
        f.write("\n")
        
        # 写入 Top 5
        f.write("## Top 5 配置\n\n")
        for rank, (exp_name, config, metrics) in enumerate(results_sorted[:5], 1):
            iou = metrics.get('iou', 0)
            dice = metrics.get('dice', 0)
            
            f.write(f"### {rank}. {exp_name}\n\n")
            f.write(f"- **IoU**: {iou:.4f}\n")
            f.write(f"- **Dice**: {dice:.4f}\n")
            f.write(f"- **配置**:\n")
            f.write(f"  - Scales: {config['scales']}\n")
            f.write(f"  - Resize: {config['resize']}\n")
            f.write(f"  - Flips: {config['flips']}\n")
            f.write(f"  - Rotations: {config['rotations']}\n")
            f.write(f"  - Threshold: {config['threshold']}\n")
            f.write(f"  - Morphology: {config['morphology']}\n")
            f.write(f"  - Fusion: {config['fusion']}\n")
            f.write("\n")
        
        # 分析各因素的影响
        f.write("## 超参数影响分析\n\n")
        
        # 分析 resize method
        f.write("### Resize Method 对比\n\n")
        bilinear_results = [(n, m) for n, c, m in results if c['resize'] == 'bilinear']
        bicubic_results = [(n, m) for n, c, m in results if c['resize'] == 'bicubic']
        
        if bilinear_results and bicubic_results:
            bilinear_avg = sum(m['iou'] for _, m in bilinear_results) / len(bilinear_results)
            bicubic_avg = sum(m['iou'] for _, m in bicubic_results) / len(bicubic_results)
            
            f.write(f"- **Bilinear** 平均 IoU: {bilinear_avg:.4f} ({len(bilinear_results)} 个实验)\n")
            f.write(f"- **Bicubic** 平均 IoU: {bicubic_avg:.4f} ({len(bicubic_results)} 个实验)\n")
            f.write(f"- **差异**: {abs(bicubic_avg - bilinear_avg):.4f}\n\n")
        
        # 分析 morphology
        f.write("### Morphology 后处理影响\n\n")
        morph_yes = [(n, m) for n, c, m in results if c['morphology'] == 'Y']
        morph_no = [(n, m) for n, c, m in results if c['morphology'] == 'N']
        
        if morph_yes and morph_no:
            morph_yes_avg = sum(m['iou'] for _, m in morph_yes) / len(morph_yes)
            morph_no_avg = sum(m['iou'] for _, m in morph_no) / len(morph_no)
            
            f.write(f"- **使用 Morphology**: {morph_yes_avg:.4f} ({len(morph_yes)} 个实验)\n")
            f.write(f"- **不使用 Morphology**: {morph_no_avg:.4f} ({len(morph_no)} 个实验)\n")
            f.write(f"- **提升**: {(morph_yes_avg - morph_no_avg):.4f}\n\n")
        
        # 基线对比
        baseline_result = next(((n, c, m) for n, c, m in results if 'baseline' in n), None)
        if baseline_result:
            baseline_iou = baseline_result[2]['iou']
            baseline_dice = baseline_result[2]['dice']
            best_result = results_sorted[0]
            best_iou = best_result[2]['iou']
            best_dice = best_result[2]['dice']
            
            f.write("### 相对 Baseline 的提升\n\n")
            f.write(f"- **Baseline IoU**: {baseline_iou:.4f}\n")
            f.write(f"- **Baseline Dice**: {baseline_dice:.4f}\n")
            f.write(f"- **最佳 IoU**: {best_iou:.4f} (提升: +{(best_iou - baseline_iou):.4f}, +{(best_iou - baseline_iou) / baseline_iou * 100:.2f}%)\n")
            f.write(f"- **最佳 Dice**: {best_dice:.4f} (提升: +{(best_dice - baseline_dice):.4f}, +{(best_dice - baseline_dice) / baseline_dice * 100:.2f}%)\n")
            f.write("\n")
        
        # 推荐配置
        f.write("## 推荐配置\n\n")
        if results_sorted:
            best = results_sorted[0]
            f.write(f"基于实验结果，推荐使用以下配置：\n\n")
            f.write("```bash\n")
            f.write("python run_semantic_segmentation.py \\\n")
            f.write("  --task isic2017 \\\n")
            f.write("  --eval \\\n")
            f.write("  --ckpt_path AutogluonModels/ag-20251208_054753 \\\n")
            f.write("  --tta_enable \\\n")
            
            if best[1]['scales'] != '-':
                scales_str = best[1]['scales'].strip('[]').replace(', ', ' ')
                f.write(f"  --tta_scales {scales_str} \\\n")
            
            if best[1]['resize'] != '-':
                f.write(f"  --tta_resize_method {best[1]['resize']} \\\n")
            
            if best[1]['flips'] == '[n, h]':
                f.write(f"  --tta_flips none horizontal \\\n")
            elif best[1]['flips'] == '[n, h, v]':
                f.write(f"  --tta_flips none horizontal vertical \\\n")
            
            if best[1]['rotations'] == '[-10, 0, 10]':
                f.write(f"  --tta_rotations -10 0 10 \\\n")
            elif best[1]['rotations'] == '[-5, 0, 5]':
                f.write(f"  --tta_rotations -5 0 5 \\\n")
            
            if best[1]['threshold'] != '-':
                f.write(f"  --tta_threshold {best[1]['threshold']} \\\n")
            
            if best[1]['morphology'] == 'Y':
                f.write(f"  --tta_morphology \\\n")
            
            if best[1]['fusion'] != '-':
                f.write(f"  --tta_fusion {best[1]['fusion']} \\\n")
            
            f.write("  --output_dir outputs/final_best\n")
            f.write("```\n\n")
            
            f.write(f"**预期性能**: IoU={best[2]['iou']:.4f}, Dice={best[2]['dice']:.4f}\n")
    
    print(f"✅ 报告已生成: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="分析 TTA 实验结果")
    parser.add_argument("--results_dir", type=str, default="outputs", 
                       help="实验结果目录 (默认: outputs)")
    parser.add_argument("--output", type=str, default="TTA_Results_Summary.md",
                       help="输出报告文件名 (默认: TTA_Results_Summary.md)")
    
    args = parser.parse_args()
    
    print(f"📊 收集实验结果: {args.results_dir}")
    results = collect_results(args.results_dir)
    
    if not results:
        print("❌ 没有找到任何实验结果")
        return
    
    print(f"✅ 找到 {len(results)} 个实验结果")
    print(f"📝 生成报告: {args.output}")
    
    generate_markdown_report(results, args.output)
    
    print("\n完成！")


if __name__ == "__main__":
    main()
