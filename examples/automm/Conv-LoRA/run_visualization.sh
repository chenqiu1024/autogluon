#!/bin/bash
# 一键运行可视化脚本

cd /root/autodl-tmp/works/autogluon/examples/automm/Conv-LoRA

echo "============================================"
echo "半监督数据可视化"
echo "============================================"
echo ""

if [ ! -f "datasets/isic2017/isic2017/train_labeled_10pct.csv" ]; then
    echo "❌ 错误: 找不到 train_labeled_10pct.csv"
    exit 1
fi

if [ ! -f "datasets/isic2017/isic2017/train_weak_90pct.csv" ]; then
    echo "❌ 错误: 找不到 train_weak_90pct.csv"
    exit 1
fi

echo "✅ 数据文件检查通过"
echo ""

echo "请选择可视化模式:"
echo "  1) 前10个样本（快速测试）"
echo "  2) 前50个样本（推荐预览）"
echo "  3) 前200个样本"
echo "  4) 全部2000个样本"
echo ""
read -p "请输入选项 [1-4]: " choice

case $choice in
    1)
        python visualize_semi_supervised_data.py --max_samples 10
        ;;
    2)
        python visualize_semi_supervised_data.py --max_samples 50
        ;;
    3)
        python visualize_semi_supervised_data.py --max_samples 200
        ;;
    4)
        read -p "是否在后台运行? [y/n]: " bg_choice
        if [ "$bg_choice" = "y" ] || [ "$bg_choice" = "Y" ]; then
            nohup python visualize_semi_supervised_data.py > viz_full.log 2>&1 &
            echo "✅ 已在后台启动（PID: $!）"
            echo "   查看进度: tail -f viz_full.log"
        else
            python visualize_semi_supervised_data.py
        fi
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo "可视化完成！"
echo "============================================"
