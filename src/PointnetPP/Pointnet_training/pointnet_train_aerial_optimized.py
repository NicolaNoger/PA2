#!/usr/bin/env python3
"""
PointNet++ Training Script - AERIAL OPTIMIZED (V3)
===================================================

Major architectural improvements for 25x25m outdoor tiles:
✅ Radii scaled 10x (1.0m-10.0m vs 0.1m-0.8m indoor)
✅ More sampling points (2048 vs 1024 in SA1)
✅ Dice Loss + Focal Loss combination
✅ Higher dropout (0.6 vs 0.5)
✅ Lower weight decay (5e-5 vs 1e-4)

Expected: +2-3% accuracy improvement over V2
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from pointnet2_aerial_optimized import PointNet2AerialSSG, DiceLoss


# ============================================================================
# Focal Loss (from Advanced)
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, N)
            targets: (B, N)
        """
        logits = logits.permute(0, 2, 1).contiguous().view(-1, logits.size(1))  # (B*N, C)
        targets = targets.view(-1)  # (B*N,)
        
        # Cross entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        
        # Get probabilities
        pt = torch.exp(-ce_loss)
        
        # Focal term
        focal_term = (1 - pt) ** self.gamma
        
        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_term = alpha_t * focal_term
        
        # Final loss
        loss = focal_term * ce_loss
        
        return loss.mean()


# ============================================================================
# Configuration
# ============================================================================

class AerialOptimizedConfig:
    """Configuration for aerial-optimized training"""
    
    # Data
    DATA_DIR = '/cfs/earth/scratch/nogernic/PA2/data/lidar/pointnet_tiles'
    NUM_CLASSES = 5
    NUM_FEATURES = 7  # XYZ + Intensity + ReturnNumber + NumberOfReturns + ScanAngle
    INPUT_CHANNELS = 4  # Features without XYZ
    
    # Architecture
    USE_XYZ = True
    DROPOUT = 0.6  # Increased from 0.5
    
    # Training
    BATCH_SIZE = 24
    MAX_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 5e-5  # Reduced from 1e-4
    
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Augmentation
    USE_AUGMENTATION = True
    AUG_ROTATION_RANGE = 180
    AUG_JITTER_SIGMA = 0.02
    AUG_JITTER_CLIP = 0.05
    AUG_SCALE_MIN = 0.8
    AUG_SCALE_MAX = 1.2
    
    # Loss Configuration
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA = 2.0
    
    USE_DICE_LOSS = True  # NEW!
    DICE_WEIGHT = 0.2     # Weight for Dice Loss
    
    # Optimized class weights (from Advanced V2)
    CLASS_WEIGHTS = [2.0, 0.6, 0.2, 0.4, 0.45]  # [Water, Tree, LowVeg, Impervious, Buildings]
    
    # Output
    RUN_NAME = f"aerial_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    OUTPUT_DIR = f'/cfs/earth/scratch/nogernic/PA2/src/PointnetPP/outputs_advanced/{RUN_NAME}'
    
    # Early stopping
    PATIENCE = 10
    
    # Logging
    LOG_EVERY_N_STEPS = 10


# ============================================================================
# Lightning Module
# ============================================================================

class PointNet2AerialOptimized(pl.LightningModule):
    """PyTorch Lightning module with aerial-optimized architecture"""
    
    def __init__(self, config: AerialOptimizedConfig):
        super().__init__()
        self.config = config
        
        # Create optimized model
        self.model = PointNet2AerialSSG(
            num_classes=config.NUM_CLASSES,
            input_channels=config.INPUT_CHANNELS,
            use_xyz=config.USE_XYZ,
            dropout=config.DROPOUT
        )
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        print("\n" + "="*80)
        print("AERIAL-OPTIMIZED ARCHITECTURE INITIALIZED")
        print("="*80)
        print(f"Radii: 1.0m, 2.5m, 5.0m, 10.0m (vs 0.1-0.8m indoor)")
        print(f"Sampling: 2048, 512, 128, 32 points")
        print(f"Dropout: {config.DROPOUT} (increased from 0.5)")
        print(f"Weight Decay: {config.WEIGHT_DECAY} (reduced from 1e-4)")
        print("="*80 + "\n")
        
        # Setup losses
        self._setup_losses()
        
        # Metrics storage
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        # Batch counter for logging
        self.train_batch_count = 0
        self.val_batch_count = 0
    
    def _setup_losses(self):
        """Setup loss functions"""
        class_weights = torch.tensor(self.config.CLASS_WEIGHTS, dtype=torch.float32)
        if torch.cuda.is_available():
            class_weights = class_weights.cuda()
        
        # Focal Loss
        if self.config.USE_FOCAL_LOSS:
            self.focal_loss = FocalLoss(
                gamma=self.config.FOCAL_GAMMA,
                alpha=class_weights
            )
            print(f"✓ Focal Loss (gamma={self.config.FOCAL_GAMMA})")
        else:
            self.focal_loss = None
        
        # Dice Loss (NEW!)
        if self.config.USE_DICE_LOSS:
            self.dice_loss = DiceLoss()
            print(f"✓ Dice Loss (weight={self.config.DICE_WEIGHT})")
        else:
            self.dice_loss = None
        
        print(f"✓ Class Weights: {self.config.CLASS_WEIGHTS}\n")
    
    def forward(self, x):
        """Forward pass"""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        features, labels = batch
        
        # Move to GPU if available
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        
        # Forward pass
        logits = self(features)
        
        # Combined loss
        loss = 0.0
        
        # Focal Loss
        if self.focal_loss is not None:
            focal_loss = self.focal_loss(logits, labels)
            loss += focal_loss
        
        # Dice Loss
        if self.dice_loss is not None:
            dice_loss = self.dice_loss(logits, labels)
            loss += self.config.DICE_WEIGHT * dice_loss
        
        # Calculate accuracy
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        # Print every 50 batches
        self.train_batch_count += 1
        if self.train_batch_count % 50 == 0:
            print(f"[Train] Epoch {self.current_epoch} | Batch {batch_idx} | "
                  f"Loss: {loss.item():.4f} | Acc: {acc.item():.4f}")
        
        # Return dict for pytorch-lightning 0.7.1
        return {'loss': loss, 'train_loss': loss.detach(), 'train_acc': acc.detach()}
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        features, labels = batch
        
        # Move to GPU if available
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        
        # Forward pass
        logits = self(features)
        
        # Combined loss
        loss = 0.0
        
        if self.focal_loss is not None:
            focal_loss = self.focal_loss(logits, labels)
            loss += focal_loss
        
        if self.dice_loss is not None:
            dice_loss = self.dice_loss(logits, labels)
            loss += self.config.DICE_WEIGHT * dice_loss
        
        # Calculate accuracy
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        # Print first batch
        if batch_idx == 0:
            print(f"[Val] Epoch {self.current_epoch} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}")
        
        # Return dict for pytorch-lightning 0.7.1
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}
    
    def validation_epoch_end(self, outputs):
        """Called at the end of validation epoch - OLD API"""
        if len(outputs) == 0:
            return
        
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        
        # Just print, no logging in 0.7.1
        print(f"[Val Epoch End] Epoch {self.current_epoch} | Avg Loss: {avg_loss.item():.4f} | Avg Acc: {avg_acc.item():.4f}")
        
        return {'val_loss': avg_loss, 'val_acc': avg_acc}
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # OneCycleLR scheduler (best from V2)
        steps_per_epoch = 255  # Approximate for 6125 samples, batch 24
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.LEARNING_RATE,
            epochs=self.config.MAX_EPOCHS,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,  # 10% warmup
            div_factor=25.0,
            final_div_factor=1e4
        )
        
        # Return list format for pytorch-lightning 0.7.1
        # Old API doesn't support dict format for step-level schedulers
        return [optimizer], [scheduler]


# ============================================================================
# Manual Test Evaluation
# ============================================================================

def evaluate_on_test_set(model, test_loader, config):
    """Manual test evaluation"""
    print("\n" + "="*80)
    print("RUNNING TEST EVALUATION")
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
    print("TEST RESULTS - AERIAL OPTIMIZED")
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
    ax.set_title('Confusion Matrix - Aerial Optimized', fontsize=14)
    
    cm_path = os.path.join(config.OUTPUT_DIR, 'confusion_matrix_aerial_optimized.png')
    fig.tight_layout()
    fig.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Confusion matrix saved: {cm_path}")
    
    # Save results
    results_path = os.path.join(config.OUTPUT_DIR, 'test_results_aerial_optimized.txt')
    with open(results_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TEST RESULTS - AERIAL OPTIMIZED\n")
        f.write("="*80 + "\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score (macro): {f1_macro:.4f}\n")
        f.write(f"F1 Score (weighted): {f1_weighted:.4f}\n")
        f.write("\nPer-class F1 scores:\n")
        for i, (name, f1) in enumerate(zip(class_names, f1_per_class)):
            f.write(f"  Class {i} ({name:20s}): {f1:.4f}\n")
        f.write("\n" + report)
    print(f"✓ Results saved: {results_path}")
    
    print("="*80)


# ============================================================================
# Main Training Function
# ============================================================================

def train_aerial_optimized():
    """Main training function"""
    
    config = AerialOptimizedConfig()
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*80)
    print("POINTNET++ TRAINING - AERIAL OPTIMIZED (V3)")
    print("="*80)
    print(f"Run name: {config.RUN_NAME}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Weight decay: {config.WEIGHT_DECAY} (reduced)")
    print(f"Dropout: {config.DROPOUT} (increased)")
    print(f"Loss: Focal + {config.DICE_WEIGHT}*Dice")
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
    
    # Create model
    print("\nCreating aerial-optimized model...")
    model = PointNet2AerialOptimized(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
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
    
    # Train
    print("\nStarting training...")
    print("="*80)
    trainer.fit(model, train_loader, val_loader)
    
    # Save model
    print("\n" + "="*80)
    print("Saving model...")
    print("="*80)
    final_path = os.path.join(config.OUTPUT_DIR, 'final_model_aerial_optimized.pth')
    torch.save({
        'epoch': config.MAX_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizers[0].state_dict(),
        'config': config,
    }, final_path)
    print(f"✓ Model saved to: {final_path}")
    
    # Test evaluation
    evaluate_on_test_set(model, test_loader, config)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Output directory: {config.OUTPUT_DIR}")
    print(f"Model: {final_path}")
    print("="*80)


if __name__ == "__main__":
    train_aerial_optimized()
