# ISIC2017 SAM segmentation with predicted bboxes
_Exported on 12/12/2025 at 23:36:00 PST from Cursor (2.2.20)_

---

**User**

通过 docs/cursor_bbox_predict-251211.md 恢复之前的会话，然后继续我们新的对话。我现在想知道，如果使用我们已经训练好的 预测bounding box 的模型（使用的训练命令是`nohup python3 run_bbox_predictor.py   \
--data_dir datasets/isic2017/isic2017   \
--output_dir outputs/bbox_predict-251212   \
--epochs 60   --pretrained_backbone   --export_preds   --image_size 320   \
--num_workers 4   > outputs/bbox_predict-251212.log 2>&1 &`），通过输入待分割图像来预测得到bbox，再把这个bbox输入未经任何改进的SAM（不使用任何Adapter，也包括不使用本项目所基于的Conv-LoRA方法，仅使用原始的SAM）去做分割，那么在ISIC2017 测试集上能够获得的平均分割性能是多少（DICE）？请为此写出脚本来实现它，告诉我如何使用

---

**Cursor**

已新增评估脚本 `examples/automm/Conv-LoRA/eval_sam_with_pred_bbox.py`：用已训练的 bbox 模型生成 box prompt，再用“原始 SAM（无 Adapter / 无 Conv-LoRA）”在 ISIC2017 测试集计算平均 Dice。脚本支持直接读取已有 bbox CSV，或在线调用 `BBoxPromptPredictor` 预测框。

**依赖/准备**
- 本地 SAM 权重（如 `sam_vit_h_4b8939.pth`），安装 `segment-anything`。
- 已训练好的 bbox 模型（如 `outputs/bbox_predict-251212/best_bbox.pth`），或对应的预测 CSV。
- 数据集 CSV 位于 `datasets/isic2017/isic2017/test.csv`，含 `image,label` 列。

**运行示例**
1) 训练 bbox（你已有同款命令，仅供确认）：
```
nohup python3 run_bbox_predictor.py \
  --data_dir datasets/isic2017/isic2017 \
  --output_dir outputs/bbox_predict-251212 \
  --epochs 60 --pretrained_backbone --export_preds \
  --image_size 320 --num_workers 4 \
  > outputs/bbox_predict-251212.log 2>&1 &
```
2) 用原始 SAM 评估（在线预测框，保存框 CSV 与掩码）：
```
python3 examples/automm/Conv-LoRA/eval_sam_with_pred_bbox.py \
  --data_dir datasets/isic2017/isic2017 \
  --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
  --sam_model_type vit_h \
  --bbox_model_ckpt outputs/bbox_predict-251212/best_bbox.pth \
  --bbox_image_size 320 \
  --save_pred_csv outputs/bbox_predict-251212/test_bbox_preds.csv \
  --save_mask_dir outputs/bbox_predict-251212/sam_masks
```
3) 若已有 bbox 预测 CSV（离线生成）：
```
python3 examples/automm/Conv-LoRA/eval_sam_with_pred_bbox.py \
  --data_dir datasets/isic2017/isic2017 \
  --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
  --sam_model_type vit_h \
  --bbox_preds_csv outputs/bbox_predict-251212/test_bbox_preds.csv
```
执行后会输出“测试集平均 Dice: X.XXXX（样本数 N）”，即你要的平均分割性能。当前未在本地跑实测，请按上面命令结合你的 SAM 权重与数据集执行即可获得具体 Dice。

```plaintext
nohup python3 run_bbox_predictor.py \
  --data_dir datasets/isic2017/isic2017 \
  --output_dir outputs/bbox_predict-251212 \
  --epochs 60 --pretrained_backbone --export_preds \
  --image_size 320 --num_workers 4 \
  > outputs/bbox_predict-251212.log 2>&1 &
```

```plaintext
python3 examples/automm/Conv-LoRA/eval_sam_with_pred_bbox.py \
  --data_dir datasets/isic2017/isic2017 \
  --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
  --sam_model_type vit_h \
  --bbox_model_ckpt outputs/bbox_predict-251212/best_bbox.pth \
  --bbox_image_size 320 \
  --save_pred_csv outputs/bbox_predict-251212/test_bbox_preds.csv \
  --save_mask_dir outputs/bbox_predict-251212/sam_masks
```

```plaintext
python3 examples/automm/Conv-LoRA/eval_sam_with_pred_bbox.py \
  --data_dir datasets/isic2017/isic2017 \
  --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
  --sam_model_type vit_h \
  --bbox_preds_csv outputs/bbox_predict-251212/test_bbox_preds.csv
```

