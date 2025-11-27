"""
批量评估和对比 RLOO 训练前后的模型性能
"""
import argparse
import os
import json
from datetime import datetime
import subprocess


def run_evaluation(checkpoint_path, base_predictor_path, task, output_file):
    """运行单个评估"""
    cmd = [
        "python", "evaluate_rloo_lightning.py",
        "--checkpoint_path", checkpoint_path,
        "--base_predictor_path", base_predictor_path,
        "--task", task,
        "--output_file", output_file,
    ]
    
    print(f"\n运行评估: {os.path.basename(checkpoint_path)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        return None
    
    # 读取结果
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            return json.load(f)
    return None


def compare_models(
    rloo_output_dir: str,
    base_checkpoint_path: str,
    task: str = "isic2017",
):
    """
    对比 RLOO 训练前后的模型
    
    Parameters
    ----------
    rloo_output_dir : str
        RLOO 训练输出目录（包含 checkpoints/ 文件夹）
    base_checkpoint_path : str
        基础模型路径（RLOO 训练前）
    task : str
        任务名称
    """
    print("="*80)
    print("RLOO 模型性能对比")
    print("="*80)
    print(f"RLOO 输出目录: {rloo_output_dir}")
    print(f"基础模型: {base_checkpoint_path}")
    print(f"任务: {task}")
    print("="*80)
    
    # 准备结果目录
    results_dir = os.path.join(rloo_output_dir, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 收集所有要评估的 checkpoint
    checkpoints_to_eval = []
    
    # 1. 基础模型（RLOO 训练前）
    base_model_ckpt = os.path.join(base_checkpoint_path, "model.ckpt")
    if os.path.exists(base_model_ckpt):
        checkpoints_to_eval.append({
            'name': 'Baseline (Before RLOO)',
            'path': base_model_ckpt,
            'type': 'baseline'
        })
    
    # 2. RLOO 训练后的 checkpoints
    ckpt_dir = os.path.join(rloo_output_dir, "checkpoints")
    if os.path.exists(ckpt_dir):
        for ckpt_file in sorted(os.listdir(ckpt_dir)):
            if ckpt_file.endswith('.ckpt'):
                ckpt_path = os.path.join(ckpt_dir, ckpt_file)
                
                if 'last' in ckpt_file:
                    name = 'RLOO - Last Epoch'
                elif 'epoch=' in ckpt_file:
                    # 提取 epoch 和 reward
                    parts = ckpt_file.replace('.ckpt', '').split('-')
                    epoch_info = [p for p in parts if 'epoch=' in p][0] if any('epoch=' in p for p in parts) else ''
                    reward_info = [p for p in parts if 'reward=' in p][0] if any('reward=' in p for p in parts) else ''
                    name = f'RLOO - {epoch_info}'
                    if reward_info:
                        name += f' ({reward_info})'
                else:
                    name = f'RLOO - {ckpt_file}'
                
                checkpoints_to_eval.append({
                    'name': name,
                    'path': ckpt_path,
                    'type': 'rloo'
                })
    
    print(f"\n找到 {len(checkpoints_to_eval)} 个 checkpoint 待评估:\n")
    for i, ckpt in enumerate(checkpoints_to_eval, 1):
        print(f"  {i}. {ckpt['name']}")
    print()
    
    # 运行评估
    all_results = []
    
    for ckpt_info in checkpoints_to_eval:
        output_file = os.path.join(
            results_dir, 
            f"eval_{ckpt_info['type']}_{os.path.basename(ckpt_info['path']).replace('.ckpt', '.json')}"
        )
        
        result = run_evaluation(
            checkpoint_path=ckpt_info['path'],
            base_predictor_path=base_checkpoint_path,
            task=task,
            output_file=output_file,
        )
        
        if result:
            result['name'] = ckpt_info['name']
            result['type'] = ckpt_info['type']
            all_results.append(result)
    
    # 生成对比报告
    if all_results:
        print("\n" + "="*80)
        print("评估结果对比")
        print("="*80)
        print(f"{'模型':<40} {'IoU':<12} {'Dice':<12} {'样本数':<8}")
        print("-"*80)
        
        for result in all_results:
            name = result['name'][:38]
            iou = result['mean_iou']
            dice = result['mean_dice']
            n_samples = result['num_samples']
            
            print(f"{name:<40} {iou:>6.4f}±{result.get('std_iou', 0):>4.4f}  {dice:>6.4f}±{result.get('std_dice', 0):>4.4f}  {n_samples:>6}")
        
        print("="*80)
        
        # 计算改进
        baseline_result = next((r for r in all_results if r['type'] == 'baseline'), None)
        rloo_results = [r for r in all_results if r['type'] == 'rloo']
        
        if baseline_result and rloo_results:
            print("\nRLOO 训练改进:")
            print("-"*80)
            baseline_iou = baseline_result['mean_iou']
            baseline_dice = baseline_result['mean_dice']
            
            for rloo_result in rloo_results:
                improvement_iou = rloo_result['mean_iou'] - baseline_iou
                improvement_dice = rloo_result['mean_dice'] - baseline_dice
                improvement_iou_pct = (improvement_iou / baseline_iou) * 100
                improvement_dice_pct = (improvement_dice / baseline_dice) * 100
                
                print(f"\n{rloo_result['name']}:")
                print(f"  IoU:  {improvement_iou:+.4f} ({improvement_iou_pct:+.2f}%)")
                print(f"  Dice: {improvement_dice:+.4f} ({improvement_dice_pct:+.2f}%)")
            
            print("="*80)
        
        # 保存完整报告
        report_file = os.path.join(results_dir, f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w') as f:
            json.dump({
                'task': task,
                'base_checkpoint': base_checkpoint_path,
                'rloo_output_dir': rloo_output_dir,
                'timestamp': datetime.now().isoformat(),
                'results': all_results,
            }, f, indent=2)
        
        print(f"\n✓ 完整报告已保存到: {report_file}")
    else:
        print("\n⚠ 没有成功的评估结果")


def main():
    parser = argparse.ArgumentParser(
        description="批量评估和对比 RLOO 训练前后的模型"
    )
    parser.add_argument(
        "--rloo_output_dir",
        type=str,
        required=True,
        help="RLOO 训练输出目录（包含 checkpoints 文件夹）"
    )
    parser.add_argument(
        "--base_checkpoint_path",
        type=str,
        required=True,
        help="基础模型路径（RLOO 训练前）"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="isic2017",
        choices=["polyp", "leaf_disease_segmentation", "camo_sem_seg", 
                 "isic2017", "road_segmentation", "SBU-shadow"],
        help="任务名称"
    )
    
    args = parser.parse_args()
    
    compare_models(
        rloo_output_dir=args.rloo_output_dir,
        base_checkpoint_path=args.base_checkpoint_path,
        task=args.task,
    )


if __name__ == "__main__":
    main()

