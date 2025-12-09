#!/usr/bin/env python3
"""
Intelligent Fine-tuning for Advanced PointNet++ Model
======================================================

Strategy:
1. Load best Advanced V2 model (Water=0.59, Buildings=0.49)
2. Freeze early layers (SA modules 0-2)
3. Train only late layers with focused weights
4. Goal: Keep Water high + improve Buildings

Target Improvements:
- Water: 0.59 → 0.57-0.59 (maintain)
- Buildings: 0.49 → 0.55-0.58 (+6-9%)
- Overall: 77.3% → 77.8-78.2%
"""

import os
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add PointNet2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PointNet2_PyTorch'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PointNet2_PyTorch', 'pointnet2_ops_lib'))

from lidar_dataloader import get_dataloaders, Compose, RandomRotation, RandomJitter, RandomScale
from pointnet_train_advanced import PointNet2SemanticSegmentationAdvanced, AdvancedConfig


# ============================================================================
# Fine-tuning Configuration
# ============================================================================

class FinetuneAdvancedConfig(AdvancedConfig):
    """Intelligent fine-tuning configuration"""
    
    # Source checkpoint - best Advanced V2 model
    CHECKPOINT_PATH = '/cfs/earth/scratch/nogernic/PA2/src/PointnetPP/outputs_advanced/final_model_advanced.pth'
    
    # Output directory with timestamp
    RUN_NAME = f"finetune_intelligent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    OUTPUT_DIR = f'/cfs/earth/scratch/nogernic/PA2/src/PointnetPP/outputs_advanced/{RUN_NAME}'
    
    # Fine-tuning hyperparameters
    LEARNING_RATE = 1e-4      # 10x lower than training (1e-3 → 1e-4)
    MAX_EPOCHS = 20           # Additional epochs
    BATCH_SIZE = 24
    
    # Balanced class weights - focus on Water + Buildings
    # Water: maintain high (2.0)
    # Buildings: boost significantly (0.45 → 1.5)
    # Impervious: slight boost (0.4 → 0.7)
    CLASS_WEIGHTS = [2.0, 0.6, 0.2, 0.7, 1.5]  # [Water, Tree, LowVeg, Impervious, Buildings]
    
    # Keep augmentation for generalization
    USE_AUGMENTATION = True
    AUG_ROTATION_RANGE = 180
    AUG_JITTER_SIGMA = 0.02
    AUG_JITTER_CLIP = 0.05
    AUG_SCALE_MIN = 0.8
    AUG_SCALE_MAX = 1.2
    
    # Focal Loss settings
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA = 2.0
    USE_SPG_LOSS = False
    
    # Layer freezing - freeze early feature extractors
    FREEZE_EARLY_LAYERS = True
    FREEZE_SA_MODULES = [0, 1, 2]  # Freeze SA modules 0, 1, 2 (keep 3 trainable)


# ============================================================================
# Fine-tuning with Layer Freezing
# ============================================================================

def freeze_layers(model, config):
    """Freeze early layers to preserve learned features"""
    if not config.FREEZE_EARLY_LAYERS:
        print("✓ All layers trainable (no freezing)")
        return
    
    print("\n" + "="*80)
    print("FREEZING EARLY LAYERS")
    print("="*80)
    
    # Freeze specified SA modules
    for idx in config.FREEZE_SA_MODULES:
        if idx < len(model.model.SA_modules):
            for param in model.model.SA_modules[idx].parameters():
                param.requires_grad = False
            print(f"✓ Frozen SA_module[{idx}]")
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nParameter Summary:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"  Frozen parameters:    {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    print("="*80 + "\n")


def finetune_advanced():
    """Main fine-tuning function"""
    
    config = FinetuneAdvancedConfig()
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*80)
    print("INTELLIGENT FINE-TUNING - ADVANCED MODEL")
    print("="*80)
    print(f"Source checkpoint: {config.CHECKPOINT_PATH}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print(f"Learning rate: {config.LEARNING_RATE} (10x lower)")
    print(f"Max epochs: {config.MAX_EPOCHS}")
    print(f"Class weights: {config.CLASS_WEIGHTS}")
    print(f"  - Water (maintain):    {config.CLASS_WEIGHTS[0]}")
    print(f"  - Buildings (boost):   {config.CLASS_WEIGHTS[4]} ⬆️")
    print(f"Freezing: SA modules {config.FREEZE_SA_MODULES}")
    print("="*80)
    
    # CUDA check
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available!")
        return
    print(f"\n✓ CUDA: {torch.cuda.get_device_name(0)}\n")
    
    # Setup augmentation
    train_transform = None
    if config.USE_AUGMENTATION:
        train_transform = Compose([
            RandomRotation(max_angle=config.AUG_ROTATION_RANGE),
            RandomJitter(sigma=config.AUG_JITTER_SIGMA, clip=config.AUG_JITTER_CLIP),
            RandomScale(scale_low=config.AUG_SCALE_MIN, scale_high=config.AUG_SCALE_MAX)
        ])
        print("✓ Data augmentation enabled\n")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, metadata = get_dataloaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        train_transform=train_transform
    )
    
    print("✓ Data loaded:")
    print(f"  Train: {len(train_loader.dataset)} tiles ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_loader.dataset)} tiles ({len(val_loader)} batches)")
    print(f"  Test:  {len(test_loader.dataset)} tiles ({len(test_loader)} batches)")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from: {config.CHECKPOINT_PATH}")
    
    # Handle pickle loading
    import sys
    from pointnet_train_advanced import AdvancedConfig as LoadConfig
    sys.modules['__main__'].AdvancedConfig = LoadConfig
    
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location='cpu')
    
    # Create model with NEW config (updated weights)
    print("Creating model with updated configuration...")
    model = PointNet2SemanticSegmentationAdvanced(config)
    
    # Load state dict from checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Checkpoint loaded successfully")
    print(f"  Original epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Freeze layers
    freeze_layers(model, config)
    
    # Trainer
    logger = pl.loggers.TensorBoardLogger(
        save_dir=config.OUTPUT_DIR,
        name='tensorboard_logs'
    )
    
    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=logger,
        log_every_n_steps=config.LOG_EVERY_N_STEPS,
        gradient_clip_val=1.0,
        deterministic=False,
    )
    
    # Fine-tune
    print("\nStarting fine-tuning...")
    print("="*80)
    trainer.fit(model, train_loader, val_loader)
    
    # Save fine-tuned model
    print("\n" + "="*80)
    print("Saving fine-tuned model...")
    print("="*80)
    final_path = os.path.join(config.OUTPUT_DIR, 'finetuned_model.pth')
    torch.save({
        'epoch': config.MAX_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizers[0].state_dict(),
        'config': config,
        'base_checkpoint': config.CHECKPOINT_PATH,
    }, final_path)
    print(f"✓ Model saved to: {final_path}")
    
    # Test evaluation
    print("\n" + "="*80)
    print("Running final test evaluation...")
    print("="*80)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()
            
            logits = model(features)
            preds = logits.argmax(dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Concatenate results
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    class_names = ['Water', 'Tree canopy', 'Low vegetation', 'Impervious', 'Buildings']
    
    print("\n" + "="*80)
    print("TEST RESULTS (FINE-TUNED)")
    print("="*80)
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"F1 Score (weighted): {f1_weighted:.4f}")
    print("\nPer-class F1 scores:")
    for i, (name, f1) in enumerate(zip(class_names, f1_per_class)):
        print(f"  Class {i} ({name:20s}): {f1:.4f}")
    
    # Detailed report
    print("\nDetailed Classification Report:")
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0,
        digits=4
    )
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(config.NUM_CLASSES))
    
    # Save confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix - Fine-tuned Advanced', fontsize=14)
    
    cm_path = os.path.join(config.OUTPUT_DIR, 'confusion_matrix_finetuned.png')
    fig.tight_layout()
    fig.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Confusion matrix saved: {cm_path}")
    
    # Save results
    results_path = os.path.join(config.OUTPUT_DIR, 'test_results_finetuned.txt')
    with open(results_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TEST RESULTS (FINE-TUNED ADVANCED)\n")
        f.write("="*80 + "\n")
        f.write(f"Base model: {config.CHECKPOINT_PATH}\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score (macro): {f1_macro:.4f}\n")
        f.write(f"F1 Score (weighted): {f1_weighted:.4f}\n")
        f.write("\nPer-class F1 scores:\n")
        for i, (name, f1) in enumerate(zip(class_names, f1_per_class)):
            f.write(f"  Class {i} ({name:20s}): {f1:.4f}\n")
        f.write("\n" + report)
    print(f"✓ Results saved: {results_path}")
    
    print("\n" + "="*80)
    print("INTELLIGENT FINE-TUNING COMPLETE!")
    print("="*80)
    print(f"Output directory: {config.OUTPUT_DIR}")
    print(f"Model: {final_path}")
    print(f"Results: {results_path}")
    print(f"Confusion matrix: {cm_path}")
    print("="*80)


if __name__ == "__main__":
    finetune_advanced()
