#!/usr/bin/env python3
"""
Runner: visualize aerial-optimized model predictions and save outputs into model output folder
"""
import os, sys
import torch
import numpy as np

# add project to path
sys.path.insert(0, os.path.dirname(__file__))

from pointnet2_aerial_optimized import PointNet2AerialSSG, DiceLoss
from visualize_predictions import visualize_tile, create_class_distribution_plot
from lidar_dataloader import LiDARTileDataset


def main():
    # Paths
    ckpt = '/cfs/earth/scratch/nogernic/PA2/src/PointnetPP/outputs_advanced/aerial_optimized_20251203_213512/final_model_aerial_optimized.pth'
    model_output_dir = os.path.dirname(ckpt)
    output_dir = os.path.join(model_output_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading checkpoint: {ckpt}")
    # Ensure unpickler can find the AerialOptimizedConfig class if it was saved under __main__
    try:
        import __main__ as _main
        if not hasattr(_main, 'AerialOptimizedConfig'):
            class AerialOptimizedConfig:
                pass
            _main.AerialOptimizedConfig = AerialOptimizedConfig
    except Exception:
        pass

    checkpoint = torch.load(ckpt, map_location='cpu')

    # Instantiate model (match training params)
    model = PointNet2AerialSSG(num_classes=5, input_channels=4, use_xyz=True, dropout=0.6)
    # Handle possible 'model.' prefix from LightningModule state_dict
    raw_state = checkpoint['model_state_dict']
    new_state = {}
    for k, v in raw_state.items():
        if k.startswith('model.'):
            new_state[k[len('model.'):]] = v
        else:
            new_state[k] = v
    # Load with non-strict in case there are small mismatches
    model.load_state_dict(new_state, strict=False)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # Dataset
    data_dir = '/cfs/earth/scratch/nogernic/PA2/data/lidar/pointnet_tiles'
    test_dataset = LiDARTileDataset(data_dir, split='test')

    num_tiles = 5
    tile_indices = np.linspace(0, len(test_dataset)-1, num_tiles, dtype=int)

    all_gt = []
    all_preds = []
    accuracies = []

    for i, tile_idx in enumerate(tile_indices, 1):
        print(f"[{i}/{num_tiles}] Tile {tile_idx}")
        features, labels = test_dataset[tile_idx]
        # features: (N,7) torch, labels: (N,)
        # prepare input
        pc = features.unsqueeze(0)  # (1,N,7)
        if torch.cuda.is_available():
            pc = pc.cuda().float().contiguous()
            labels_gpu = labels.unsqueeze(0).cuda()
        else:
            pc = pc.float().contiguous()
            labels_gpu = labels.unsqueeze(0)

        with torch.no_grad():
            logits = model(pc)  # (1,C,N)
            preds = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        gt = labels.numpy()
        all_gt.append(gt)
        all_preds.append(preds)

        acc = visualize_tile(features, gt, preds, tile_idx, output_dir)
        accuracies.append(acc)
        print(f"  Saved visualization for tile {tile_idx}, accuracy {acc:.4f}")

    # global class distribution
    all_gt = np.concatenate(all_gt)
    all_preds = np.concatenate(all_preds)
    create_class_distribution_plot(all_gt, all_preds, output_dir)

    print('\nDone. Visualizations saved to:', output_dir)

if __name__ == '__main__':
    main()
