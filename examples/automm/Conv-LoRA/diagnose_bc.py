"""
Diagnostic script to analyze BC training in detail.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "multimodal" / "src"))

from autogluon.multimodal.rl.policies.routing_policy import RoutingPolicy
from rl_utils import load_trained_conv_lora_model, prepare_dataset
from rl_bc_routing_policy import collect_noisy_topk_actions


def main():
    # Load model
    print("Loading model...")
    predictor, sam_model, conv_lora_layers = load_trained_conv_lora_model(
        'isic2017',
        'AutogluonModels/ag-20251116_011442/epoch=3-step=2000.ckpt',
        device='cuda'
    )
    
    # Collect small BC dataset
    train_df, _ = prepare_dataset('isic2017', split='train')
    
    print("\nCollecting BC data (10 samples)...")
    bc_dataset = collect_noisy_topk_actions(
        sam_model=sam_model,
        conv_lora_layers=conv_lora_layers,
        train_df=train_df,
        predictor=predictor,
        max_samples=10,
        device='cuda',
    )
    
    print(f"\n{'='*60}")
    print("BC Data Analysis")
    print(f"{'='*60}")
    
    # Analyze expert distribution
    actions = [s['action'] for s in bc_dataset]
    action_dist = Counter(actions)
    
    print(f"\nExpert usage:")
    for expert in range(8):
        count = action_dist.get(expert, 0)
        pct = count / len(actions) * 100 if actions else 0
        print(f"  Expert {expert}: {count:4d} ({pct:5.2f}%)")
    
    # Check if data makes sense
    print(f"\nData validation:")
    print(f"  Total samples: {len(bc_dataset)}")
    print(f"  Unique layer indices: {sorted(set(s['layer_idx'] for s in bc_dataset))[:5]}...")
    print(f"  Feature shapes: {set(tuple(s['feats'].shape) for s in bc_dataset)}")
    
    # Test policy forward
    print(f"\n{'='*60}")
    print("Testing Policy")
    print(f"{'='*60}")
    
    policy = RoutingPolicy(
        num_experts=8,
        feature_dim=3,
        hidden_dim=256,
        layer_embed_dim=32,
        num_layers=32,
    ).cuda()
    
    # Test on first sample
    sample = bc_dataset[0]
    feat = sample['feats'].unsqueeze(0).cuda()  # (1, C, H, W)
    layer_idx = sample['layer_idx']
    true_action = sample['action']
    
    # Resize if needed
    if feat.size(2) != 16 or feat.size(3) != 16:
        feat = F.interpolate(feat, size=(16, 16), mode='bilinear', align_corners=False)
    
    with torch.no_grad():
        logits = policy(feat, layer_idx)
        pred_action = logits.argmax(dim=1).item()
        probs = F.softmax(logits, dim=1)[0]
    
    print(f"\nSample test:")
    print(f"  True expert: {true_action}")
    print(f"  Predicted (untrained): {pred_action}")
    print(f"  Probability distribution:")
    for i, p in enumerate(probs):
        print(f"    Expert {i}: {p.item():.3f}")
    
    # Train for 1 epoch and retest
    print(f"\n{'='*60}")
    print("Quick Training Test (1 epoch)")
    print(f"{'='*60}")
    
    from torch.utils.data import DataLoader, TensorDataset
    
    # Prepare tensors
    feats_list = []
    layer_idx_list = []
    actions_list = []
    
    for sample in bc_dataset[:1000]:  # Use first 1000
        feat = sample['feats']
        if feat.size(1) != 16 or feat.size(2) != 16:
            feat = F.interpolate(
                feat.unsqueeze(0),
                size=(16, 16),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        feats_list.append(feat)
        layer_idx_list.append(sample['layer_idx'])
        actions_list.append(sample['action'])
    
    feats_tensor = torch.stack(feats_list).cuda()
    layer_idx_tensor = torch.tensor(layer_idx_list, dtype=torch.long, device='cuda')
    actions_tensor = torch.tensor(actions_list, dtype=torch.long, device='cuda')
    
    dataset = TensorDataset(feats_tensor, layer_idx_tensor, actions_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    policy.train()
    
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    for feats, layer_idxs, actions in dataloader:
        # Forward
        logits_list = []
        for i in range(feats.size(0)):
            logits = policy(feats[i:i+1], layer_idxs[i].item())
            logits_list.append(logits)
        
        logits = torch.cat(logits_list, dim=0)
        
        # Loss and acc
        loss = F.cross_entropy(logits, actions)
        acc = (logits.argmax(dim=1) == actions).float().mean()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += acc.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    print(f"\nAfter 1 epoch:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {avg_acc:.4f}")
    
    # Retest on sample
    policy.eval()
    with torch.no_grad():
        feat = bc_dataset[0]['feats'].unsqueeze(0).cuda()
        if feat.size(2) != 16 or feat.size(3) != 16:
            feat = F.interpolate(feat, size=(16, 16), mode='bilinear', align_corners=False)
        logits = policy(feat, bc_dataset[0]['layer_idx'])
        pred_action = logits.argmax(dim=1).item()
        probs = F.softmax(logits, dim=1)[0]
    
    print(f"\nSame sample after training:")
    print(f"  True expert: {bc_dataset[0]['action']}")
    print(f"  Predicted: {pred_action}")
    print(f"  Top 3 probabilities:")
    top3 = torch.topk(probs, 3)
    for i, (prob, idx) in enumerate(zip(top3.values, top3.indices)):
        print(f"    Expert {idx.item()}: {prob.item():.3f}")
    
    # Conclusion
    print(f"\n{'='*60}")
    print("Diagnosis")
    print(f"{'='*60}")
    
    if avg_acc > 0.20:
        print("✓ BC training works - policy can learn patterns")
    elif avg_acc > 0.15:
        print("⚠ BC learning is weak but functional")
        print("  Likely cause: Expert distribution is very uniform in data")
    else:
        print("✗ BC training may have issues")
        print("  Accuracy too close to random (12.5%)")
    
    print(f"\nBased on {avg_acc*100:.1f}% accuracy after 1 epoch:")
    if avg_acc < 0.15:
        print("→ Expert selection in Noisy-TopK is nearly random")
        print("→ BC cannot learn meaningful patterns from random data")
        print("→ This is OK for RL! RL will learn from IoU rewards directly")
    else:
        print("→ BC successfully learning patterns")
        print("→ Good initialization for RL")


if __name__ == "__main__":
    main()

