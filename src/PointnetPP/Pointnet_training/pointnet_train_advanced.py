"""
PointNet++ Training Script - Advanced Version (SSG + Focal Loss + Augmentation)
===============================================================================

Advanced training with better techniques for class imbalance:
- SSG architecture (simpler, better for uniform point clouds)
- Focal Loss (focuses on hard examples like rare Water class)
- Data Augmentation (improves robustness)
- Optional SPG Loss (feature clustering - can be disabled)

Key differences from baseline:
- Focal Loss instead of CrossEntropy (better for extreme imbalance)
- Augmentation enabled (rotation, jitter, scale)
- Optional SPG Loss for feature learning

Dataset: Wädenswil LiDAR tiles (25x25m, 16384 points, 7 features)
Classes: 5 (Water, Tree canopy, Low vegetation, Impervious, Buildings)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add PointNet2 and pointnet2_ops to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PointNet2_PyTorch'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PointNet2_PyTorch', 'pointnet2_ops_lib'))

from lidar_dataloader import get_dataloaders, Compose, RandomRotation, RandomJitter, RandomScale

# ============================================================================
# Advanced Loss Functions
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focuses training on hard examples by down-weighting easy examples.
    Particularly useful for extreme imbalance (Water is only 0.75%!)
    
    Args:
        gamma: Focusing parameter (default 2). Higher = more focus on hard examples
        alpha: Class weights (optional)
        reduction: 'mean' or 'sum'
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C, N) logits
            targets: (B, N) labels
        """
        B, C, N = inputs.shape
        
        # Reshape for cross entropy
        inputs_flat = inputs.permute(0, 2, 1).reshape(-1, C)  # (B*N, C)
        targets_flat = targets.view(-1)  # (B*N,)
        
        # Calculate CE loss
        ce_loss = F.cross_entropy(inputs_flat, targets_flat, reduction='none')
        
        # Calculate pt (probability of correct class)
        p = F.softmax(inputs_flat, dim=1)
        pt = p.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        
        # Calculate focal term
        focal_term = (1 - pt) ** self.gamma
        
        # Calculate focal loss
        loss = focal_term * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            # Move alpha to same device as inputs
            alpha_t = self.alpha.to(inputs.device)[targets_flat]
            loss = alpha_t * loss
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SPGLoss(nn.Module):
    """
    Subspace Prototype Guidance Loss
    
    Encourages features of the same class to cluster together in feature space.
    Complements classification loss by improving feature representations.
    
    From your old training script - helps with class separation!
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, features, labels):
        """
        Args:
            features: (B*N, D) point features
            labels: (B*N,) point labels
        """
        loss = torch.tensor(0.0, device=features.device)
        valid_classes = 0
        
        # For each class, calculate within-class variance
        for cls in range(self.num_classes):
            mask = labels == cls
            if torch.any(mask):
                cls_features = features[mask]
                if cls_features.shape[0] > 1:  # Need at least 2 points
                    # Calculate class prototype (center)
                    prototype = cls_features.mean(dim=0, keepdim=True)
                    # Calculate variance from prototype
                    variance = ((cls_features - prototype).pow(2).sum()) / cls_features.shape[0]
                    loss = loss + variance
                    valid_classes += 1
        
        # Normalize by number of classes
        if valid_classes > 0:
            loss = loss / valid_classes
        
        return loss


# ============================================================================
# Configuration
# ============================================================================

class AdvancedConfig:
    """Advanced training configuration with Focal Loss and Augmentation"""
    
    # Paths (will be overwritten for HPC)
    DATA_DIR = '/cfs/earth/scratch/nogernic/PA2/data/lidar/pointnet_tiles' 
    OUTPUT_DIR = '/cfs/earth/scratch/nogernic/PA2/src/PointnetPP/outputs_advanced' 
    
    # Model - USE SSG (not MSG!)
    NUM_CLASSES = 5
    NUM_POINTS = 16384
    NUM_FEATURES = 7
    USE_XYZ = True
    USE_SSG = True  # ✅ Use Single-Scale Grouping (better for uniform point clouds)
    
    # Training
    BATCH_SIZE = 24
    MAX_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0  # No regularization (baseline showed weight_decay hurt)
    
    # Data
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Augmentation (moderate - not too aggressive)
    USE_AUGMENTATION = True
    AUG_ROTATION_RANGE = 180  # Full rotation for better generalization
    AUG_JITTER_SIGMA = 0.02   # Increased noise for robustness
    AUG_JITTER_CLIP = 0.05
    AUG_SCALE_MIN = 0.8       # More aggressive scaling
    AUG_SCALE_MAX = 1.2
    
    # Loss Configuration
    USE_FOCAL_LOSS = True     # ✅ Focal Loss for extreme imbalance
    FOCAL_GAMMA = 2.0         # Standard gamma (try 1.5, 2.0, 2.5)
    
    USE_SPG_LOSS = False      # ⚠️ SPG Loss (set True to enable feature clustering)
    SPG_WEIGHT = 0.1          # Weight for SPG loss (only if USE_SPG_LOSS=True)
    
    # Class weights - Using sqrt-based weighting (softer than inverse frequency)
    # Original inverse freq: [4.279, 0.371, 0.034, 0.129, 0.187]
    # Sqrt approach: more balanced, less extreme
    CLASS_WEIGHTS = [2.0, 0.6, 0.2, 0.4, 0.45]  # [Water, Tree, LowVeg, Impervious, Buildings]
    
    # Early stopping
    PATIENCE = 10
    
    # Logging
    LOG_EVERY_N_STEPS = 10


# ============================================================================
# Lightning Module
# ============================================================================

class PointNet2SemanticSegmentationAdvanced(pl.LightningModule):
    """
    Advanced PyTorch Lightning module with experimental loss functions.
    """
    
    def __init__(self, config: AdvancedConfig):
        super().__init__()
        self.config = config
        # self.save_hyperparameters()  # Not available in pytorch-lightning 0.7.1
        
        # Initialize model
        hparams = {
            'model.use_xyz': config.USE_XYZ,
            'optimizer.lr': config.LEARNING_RATE,
            'optimizer.weight_decay': config.WEIGHT_DECAY,
        }
        
        # Use SSG (Single-Scale Grouping) - better for uniform point clouds
        if config.USE_SSG:
            from pointnet2.models.pointnet2_ssg_sem import PointNet2SemSegSSG
            self.model = PointNet2SemSegSSG(hparams)
            print("Using SSG architecture (Single-Scale Grouping)")
        else:
            from pointnet2.models.pointnet2_msg_sem import PointNet2SemSegMSG
            self.model = PointNet2SemSegMSG(hparams)
            print("Using MSG architecture (Multi-Scale Grouping)")
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # Setup loss functions
        self._setup_losses()
        
        # Metrics storage
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def _setup_losses(self):
        """Setup loss functions based on config"""
        class_weights = torch.tensor(self.config.CLASS_WEIGHTS, dtype=torch.float32)
        if torch.cuda.is_available():
            class_weights = class_weights.cuda()
        
        # Focal Loss for classification
        if self.config.USE_FOCAL_LOSS:
            self.classification_loss = FocalLoss(
                gamma=self.config.FOCAL_GAMMA,
                alpha=class_weights
            )
            print(f"Using Focal Loss (gamma={self.config.FOCAL_GAMMA}) with class weights")
        else:
            self.classification_loss = nn.CrossEntropyLoss(weight=class_weights)
            print("Using CrossEntropy Loss with class weights")
        
        # Optional SPG Loss for feature clustering
        if self.config.USE_SPG_LOSS:
            self.spg_loss_fn = SPGLoss(num_classes=self.config.NUM_CLASSES)
            print(f"Using SPG Loss (weight={self.config.SPG_WEIGHT}) for feature clustering")
        else:
            self.spg_loss_fn = None
            print("SPG Loss disabled")
    
    def forward(self, x):
        """Forward pass - returns logits"""
        return self.model(x)  # (B, C, N)
    
    def extract_features(self, points):
        """
        Extract point features before final classification layer.
        Used for SPG loss calculation.
        
        Returns:
            features: (B*N, D) point features
        """
        # Separate XYZ and features
        xyz = points[..., :3]  # (B, N, 3)
        if points.size(-1) > 3:
            feats_input = points[..., 3:].permute(0, 2, 1).contiguous()  # (B, C_in, N)
        else:
            feats_input = None
        
        # Apply Set Abstraction modules
        l_xyz = [xyz]
        l_features = [feats_input]
        
        for sa in self.model.SA_modules:
            cur_xyz = l_xyz[-1].contiguous()
            cur_feats = l_features[-1]
            if cur_feats is not None:
                cur_feats = cur_feats.contiguous()
            li_xyz, li_feats = sa(cur_xyz, cur_feats)
            l_xyz.append(li_xyz.contiguous())
            l_features.append(li_feats.contiguous())
        
        # Apply Feature Propagation modules
        for i in range(len(self.model.FP_modules)):
            idx = -1 - i
            before = l_xyz[idx-1].contiguous()
            after = l_xyz[idx].contiguous()
            f1 = l_features[idx-1].contiguous() if l_features[idx-1] is not None else None
            f2 = l_features[idx].contiguous()
            l_features[idx-1] = self.model.FP_modules[idx](before, after, f1, f2).contiguous()
        
        # l_features[0] has shape (B, C_feat, N)
        point_feats = l_features[0]
        
        # Apply FC layers up to second-to-last
        fc_layers = list(self.model.fc_lyaer.children())
        feat_extractor = nn.Sequential(*fc_layers[:-1])
        features = feat_extractor(point_feats)  # (B, C_feat, N)
        
        # Reshape to (B*N, C_feat)
        B, C_feat, N = features.shape
        features = features.permute(0, 2, 1).reshape(-1, C_feat)
        
        return features
    
    def training_step(self, batch, batch_idx):
        """Training step with Focal Loss and optional SPG Loss"""
        features, labels = batch  # (B, N, 7), (B, N)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        
        # Forward pass
        logits = self(features)  # (B, 5, N)
        
        # Classification loss (Focal or CE)
        cls_loss = self.classification_loss(logits, labels)
        
        # Optional SPG loss for feature clustering
        if self.config.USE_SPG_LOSS and self.spg_loss_fn is not None:
            point_features = self.extract_features(features)  # (B*N, D)
            labels_flat = labels.view(-1)  # (B*N,)
            spg_loss = self.spg_loss_fn(point_features, labels_flat)
            
            # Combine losses
            total_loss = cls_loss + self.config.SPG_WEIGHT * spg_loss
        else:
            total_loss = cls_loss
        
        # Calculate accuracy for logging
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean()
        
        # Print progress every N batches
        if batch_idx % 50 == 0:
            print(f"[Train] Epoch {self.current_epoch} | Batch {batch_idx} | Loss: {total_loss.item():.4f} | Acc: {acc.item():.4f}")
        
        # Return dict for pytorch-lightning 0.7.1
        return {'loss': total_loss, 'train_loss': total_loss.detach(), 'train_acc': acc.detach()}
    
    def on_train_epoch_end(self, outputs):
        """Print training summary at end of each epoch"""
        if not outputs:
            return
        
        avg_loss = torch.stack([x['train_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['train_acc'] for x in outputs]).mean()
        
        print(f"\n{'='*80}")
        print(f"[TRAINING] Epoch {self.current_epoch}")
        print(f"  Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
        print(f"{'='*80}\n")
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        features, labels = batch
        
        # Move to GPU if available
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        
        # Forward pass
        logits = self(features)
        
        # Calculate loss (classification only for validation)
        loss = self.classification_loss(logits, labels)
        
        # Calculate metrics
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        # Print progress
        if batch_idx == 0:  # Print first batch of validation
            print(f"[Val] Epoch {self.current_epoch} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}")
        
        # Store for epoch-level metrics
        self.validation_step_outputs.append({
            'loss': loss,
            'acc': acc,
            'preds': preds.detach().cpu(),
            'labels': labels.detach().cpu()
        })
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def on_validation_epoch_end(self):
        """Aggregate validation metrics"""
        if not self.validation_step_outputs:
            return
        
        # Average metrics
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in self.validation_step_outputs]).mean()
        
        # Per-class metrics
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).numpy().flatten()
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs]).numpy().flatten()
        
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # Print summary
        class_names = ['Water', 'Tree canopy', 'Low vegetation', 'Impervious', 'Buildings']
        print(f"\n{'='*80}")
        print(f"[VALIDATION] Epoch {self.current_epoch}")
        print(f"  Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | F1 (macro): {f1_macro:.4f}")
        print(f"{'-'*80}")
        for i, (name, f1) in enumerate(zip(class_names, f1_per_class)):
            print(f"  Class {i} ({name:20s}): F1 = {f1:.4f}")
        print(f"{'='*80}\n")
        
        self.validation_step_outputs.clear()
        return {'val_loss': avg_loss, 'val_acc': avg_acc, 'val_f1': f1_macro}
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        features, labels = batch
        
        # Move to GPU if available
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        
        logits = self(features)
        preds = logits.argmax(dim=1)
        
        self.test_step_outputs.append({
            'preds': preds.detach().cpu(),
            'labels': labels.detach().cpu()
        })
        
        return {'preds': preds, 'labels': labels}
    
    def on_test_epoch_end(self):
        """Generate test report"""
        if not self.test_step_outputs:
            return
        
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs]).numpy().flatten()
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs]).numpy().flatten()
        
        # Metrics
        acc = accuracy_score(all_labels, all_preds)
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Report
        class_names = ['Water', 'Tree canopy', 'Low vegetation', 'Impervious', 'Buildings']
        report = classification_report(
            all_labels, all_preds,
            target_names=class_names,
            zero_division=0,
            digits=4
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=range(self.config.NUM_CLASSES))
        
        # Print
        print("\n" + "="*80)
        print("TEST RESULTS (ADVANCED: SSG + Focal Loss + Augmentation)")
        print("="*80)
        loss_desc = f"Focal (γ={self.config.FOCAL_GAMMA})" if self.config.USE_FOCAL_LOSS else "CrossEntropy"
        spg_desc = f" + SPG (w={self.config.SPG_WEIGHT})" if self.config.USE_SPG_LOSS else ""
        print(f"Loss: {loss_desc}{spg_desc}")
        print(f"Architecture: {'SSG' if self.config.USE_SSG else 'MSG'}")
        print(f"Augmentation: {'Enabled' if self.config.USE_AUGMENTATION else 'Disabled'}")
        print(f"\nOverall Accuracy: {acc:.4f}")
        print(f"F1 Score (macro): {f1_macro:.4f}")
        print(f"F1 Score (weighted): {f1_weighted:.4f}")
        print("\nPer-class F1 scores:")
        for i, (name, f1) in enumerate(zip(class_names, f1_per_class)):
            print(f"  Class {i} ({name:20s}): {f1:.4f}")
        print("\nDetailed Classification Report:")
        print(report)
        
        # Save
        self._save_confusion_matrix(cm, class_names)
        
        self.test_step_outputs.clear()
        print("="*80)
        print("Test evaluation completed!")
        print("="*80 + "\n")
    
    def _save_confusion_matrix(self, cm, class_names):
        """Save confusion matrix"""
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        title = f"Confusion Matrix - Advanced ({'SSG' if self.config.USE_SSG else 'MSG'}"
        title += f" + Focal γ={self.config.FOCAL_GAMMA}"
        if self.config.USE_SPG_LOSS:
            title += f" + SPG"
        title += ")"
        ax.set_title(title, fontsize=14)
        
        save_path = os.path.join(self.config.OUTPUT_DIR, 'confusion_matrix_advanced.png')
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"\nConfusion matrix saved to: {save_path}")
    
    def test_dataloader(self):
        """Return test dataloader - required for pytorch-lightning 0.7.1"""
        if hasattr(self, '_test_loader'):
            return self._test_loader
        return None
    
    def configure_optimizers(self):
        """Configure optimizer with OneCycleLR scheduler for better convergence"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # OneCycleLR: starts low, peaks at max_lr, then anneals to very low
        # Better than CosineAnnealing for finding optimal learning rates
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.LEARNING_RATE,
            epochs=self.config.MAX_EPOCHS,
            steps_per_epoch=255,  # Number of training batches (6125 tiles / 24 batch_size)
            pct_start=0.1,        # 10% warmup
            anneal_strategy='cos',
            div_factor=25.0,      # Initial lr = max_lr / 25
            final_div_factor=1e4  # Final lr = max_lr / 10000
        )
        
        # Return list format for pytorch-lightning 0.7.1
        return [optimizer], [scheduler]


# ============================================================================
# Main Training Function
# ============================================================================

def train_advanced():
    """Main training function with advanced features"""
    
    config = AdvancedConfig()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print("POINTNET++ TRAINING - ADVANCED")
    print("="*80)
    print(f"Architecture: {'SSG (Single-Scale)' if config.USE_SSG else 'MSG (Multi-Scale)'}")
    print(f"Loss: {'Focal (γ=' + str(config.FOCAL_GAMMA) + ')' if config.USE_FOCAL_LOSS else 'CrossEntropy'}")
    if config.USE_SPG_LOSS:
        print(f"SPG Loss: Enabled (weight={config.SPG_WEIGHT})")
    else:
        print("SPG Loss: Disabled")
    print(f"Augmentation: {'Enabled' if config.USE_AUGMENTATION else 'Disabled'}")
    if config.USE_AUGMENTATION:
        print(f"  - Rotation: ±{config.AUG_ROTATION_RANGE}°")
        print(f"  - Jitter: σ={config.AUG_JITTER_SIGMA}, clip={config.AUG_JITTER_CLIP}")
        print(f"  - Scale: [{config.AUG_SCALE_MIN}, {config.AUG_SCALE_MAX}]")
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Max epochs: {config.MAX_EPOCHS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Weight decay: {config.WEIGHT_DECAY}")
    print("="*80)
    
    # CUDA check
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available!")
    else:
        print(f"\n✓ CUDA: {torch.cuda.get_device_name(0)}\n")
    
    # Setup augmentation
    train_transform = None
    if config.USE_AUGMENTATION:
        train_transform = Compose([
            RandomRotation(max_angle=config.AUG_ROTATION_RANGE),
            RandomJitter(sigma=config.AUG_JITTER_SIGMA, clip=config.AUG_JITTER_CLIP),
            RandomScale(scale_low=config.AUG_SCALE_MIN, scale_high=config.AUG_SCALE_MAX)
        ])
        print("✓ Data augmentation enabled (training set only)\n")
    
    # Load data with augmentation
    print("Loading data...")
    train_loader, val_loader, test_loader, metadata = get_dataloaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        train_transform=train_transform  # ✅ Apply augmentation!
    )
    
    print("✓ Data loaded:")
    print(f"  Train: {len(train_loader.dataset)} tiles ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_loader.dataset)} tiles ({len(val_loader)} batches)")
    print(f"  Test:  {len(test_loader.dataset)} tiles ({len(test_loader)} batches)")
    
    # Model
    print("\nInitializing advanced model...")
    model = PointNet2SemanticSegmentationAdvanced(config)
    print("✓ Model initialized")
    
    # Callbacks - DISABLED for pytorch-lightning 0.7.1 compatibility
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(config.OUTPUT_DIR, 'checkpoints', 'pointnet2-adv-{epoch:02d}-{val/loss:.4f}'),
        monitor='val/loss',
        mode='min',
        save_top_k=3
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        patience=config.PATIENCE,
        mode='min',
        verbose=True
    )
    
    logger = TensorBoardLogger(
        save_dir=config.OUTPUT_DIR,
        name='tensorboard_logs'
    )
    
    # Trainer (simplified for pytorch-lightning 0.7.1)
    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        # callbacks=[checkpoint_callback, early_stop_callback],  # Disabled for 0.7.1
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
    final_path = os.path.join(config.OUTPUT_DIR, 'final_model_advanced.pth')
    torch.save({
        'epoch': trainer.current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizers[0].state_dict(),
        'config': config,
    }, final_path)
    print(f"✓ Model saved to: {final_path}")
    
    # Test - Manual evaluation because pytorch-lightning 0.7.1 test() doesn't call callbacks properly
    print("\n" + "="*80)
    print("Running final test...")
    print("="*80)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(test_loader):
            if torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()
            
            # Forward pass
            logits = model(features)  # (B, 5, N)
            preds = logits.argmax(dim=1)  # (B, N)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Concatenate all predictions
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    class_names = ['Water', 'Tree canopy', 'Low vegetation', 'Impervious', 'Buildings']
    
    print("\n" + "="*80)
    print("TEST RESULTS (ADVANCED)")
    print("="*80)
    loss_desc = "Focal Loss" if config.USE_FOCAL_LOSS else "CE Loss"
    if config.USE_SPG_LOSS:
        loss_desc += " + SPG Loss"
    print(f"Loss: {loss_desc}")
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
    ax.set_title(f'Confusion Matrix - Advanced ({loss_desc})', fontsize=14)
    
    cm_path = os.path.join(config.OUTPUT_DIR, 'confusion_matrix_advanced.png')
    fig.tight_layout()
    fig.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Confusion matrix saved: {cm_path}")
    
    # Save results to text file
    results_path = os.path.join(config.OUTPUT_DIR, 'test_results_advanced.txt')
    with open(results_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TEST RESULTS (ADVANCED)\n")
        f.write("="*80 + "\n")
        f.write(f"Loss: {loss_desc}\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score (macro): {f1_macro:.4f}\n")
        f.write(f"F1 Score (weighted): {f1_weighted:.4f}\n")
        f.write("\nPer-class F1 scores:\n")
        for i, (name, f1) in enumerate(zip(class_names, f1_per_class)):
            f.write(f"  Class {i} ({name:20s}): {f1:.4f}\n")
        f.write("\n" + report)
    print(f"✓ Results saved: {results_path}")
    
    print("\n" + "="*80)
    print("ADVANCED TRAINING COMPLETE!")
    print("="*80)
    print(f"Model: {final_path}")
    print(f"Confusion matrix: {cm_path}")
    print(f"Test results: {results_path}")
    

if __name__ == "__main__":
    train_advanced()
