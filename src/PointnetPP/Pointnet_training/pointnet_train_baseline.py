"""
PointNet++ Training Script - Baseline Version
==============================================

Clean, modern training script for LiDAR semantic segmentation.

Key improvements over old version:
- Uses preprocessed data (no on-the-fly FPS needed)
- Class weights in loss function (more direct than sampling)
- Simpler, more maintainable code
- Better logging and metrics

Dataset: Wädenswil LiDAR tiles (25x25m, 16384 points, 7 features)
Classes: 5 (Water, Tree canopy, Low vegetation, Impervious, Buildings)
"""

import os
import sys
import torch
import torch.nn as nn
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

from lidar_dataloader import get_dataloaders

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Training configuration"""
    
    # Paths (will be overwritten for HPC)
    DATA_DIR = '/cfs/earth/scratch/nogernic/PA2/data/lidar/pointnet_tiles' 
    OUTPUT_DIR = '/cfs/earth/scratch/nogernic/PA2/src/PointnetPP/outputs' 
    
    # Model
    NUM_CLASSES = 5
    NUM_POINTS = 16384
    NUM_FEATURES = 7  # X, Y, Z, Intensity, ReturnNumber, NumberOfReturns, ScanAngle
    USE_XYZ = True  # Use XYZ coordinates as additional input (PointNet++ standard approach)
    
    # Training
    BATCH_SIZE = 24  # Optimized for A100 40GB / L40S 48GB
    MAX_EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0
    
    # Data
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Class weights for handling severe imbalance
    # From dataloader analysis: Water is very rare (0.75%)
    USE_CLASS_WEIGHTS = True
    CLASS_WEIGHTS = [4.279, 0.371, 0.034, 0.129, 0.187]  # From test_dataloader.py
    
    # Early stopping
    PATIENCE = 10  # Stop if no improvement for 10 epochs
    
    # Logging
    LOG_EVERY_N_STEPS = 10
    

# ============================================================================
# Lightning Module
# ============================================================================

class PointNet2SemanticSegmentation(pl.LightningModule):
    """
    PyTorch Lightning module for PointNet++ semantic segmentation.
    
    Simplified version focusing on clean code and good practices.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # self.save_hyperparameters()  # Optional, causes issues with older pytorch-lightning
        
        # Initialize model
        hparams = {
            'model.use_xyz': config.USE_XYZ,
            'optimizer.lr': config.LEARNING_RATE,
            'optimizer.weight_decay': config.WEIGHT_DECAY,
        }
        
        # Use SSG (Single-Scale Grouping) for uniform point density aerial LiDAR
        # SSG is faster, uses less memory, and works better for uniform point clouds
        from pointnet2.models.pointnet2_ssg_sem import PointNet2SemSegSSG
        self.model = PointNet2SemSegSSG(hparams)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # Loss function with class weights
        if config.USE_CLASS_WEIGHTS:
            class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32)
            if torch.cuda.is_available():
                class_weights = class_weights.cuda()
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"Using class weights: {config.CLASS_WEIGHTS}")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Metrics storage for epoch aggregation
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, N, 7) point cloud with features
            
        Returns:
            logits: (B, num_classes, N) class scores per point
        """
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step - called for each batch"""
        features, labels = batch  # (B, N, 7), (B, N)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        
        # Forward pass
        logits = self(features)  # (B, 5, N)
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        
        # Calculate accuracy (for logging)
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean()
        
        # In pytorch-lightning 0.7.1, return dict with loss and optional log metrics
        return {'loss': loss, 'train_loss': loss.detach(), 'train_acc': acc.detach()}
    
    def on_train_epoch_end(self, outputs):
        """Print training summary at end of each epoch"""
        # In pytorch-lightning 0.7.1, outputs is a list of dicts from training_step
        if not outputs:
            return
        
        avg_loss = torch.stack([x['train_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['train_acc'] for x in outputs]).mean()
        
        print(f"\n{'='*80}")
        print(f"[TRAINING] Epoch {self.current_epoch}")
        print(f"  Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
        print(f"{'='*80}\n")
    
    def validation_step(self, batch, batch_idx):
        """Validation step - called for each validation batch"""
        features, labels = batch
        
        # Move to GPU if available
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        
        # Forward pass
        logits = self(features)
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        # Store for epoch-level metrics
        self.validation_step_outputs.append({
            'loss': loss,
            'acc': acc,
            'preds': preds.detach().cpu(),
            'labels': labels.detach().cpu()
        })
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def on_validation_epoch_end(self):
        """Aggregate validation metrics at epoch end"""
        if not self.validation_step_outputs:
            return
        
        # Average metrics
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in self.validation_step_outputs]).mean()
        
        # Calculate per-class metrics
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).numpy().flatten()
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs]).numpy().flatten()
        
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        # Print summary (logging not available in pytorch-lightning 0.7.1)
        class_names = ['Water', 'Tree canopy', 'Low vegetation', 'Impervious', 'Buildings']
        print(f"\n{'='*80}")
        print(f"[VALIDATION] Epoch {self.current_epoch}")
        print(f"  Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | F1 (macro): {f1_macro:.4f}")
        print(f"{'-'*80}")
        for i, (name, f1) in enumerate(zip(class_names, f1_per_class)):
            print(f"  Class {i} ({name:20s}): F1 = {f1:.4f}")
        print(f"{'='*80}\n")
        
        # Clear for next epoch
        self.validation_step_outputs.clear()
        
        # Return dict for progress bar (pytorch-lightning 0.7.1 style)
        return {'val_loss': avg_loss, 'val_acc': avg_acc, 'val_f1': f1_macro}
    
    def test_step(self, batch, batch_idx):
        """Test step - called for each test batch"""
        features, labels = batch
        
        # Move to GPU if available
        if torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        
        # Forward pass
        logits = self(features)
        preds = logits.argmax(dim=1)
        
        # Store for epoch-level analysis
        self.test_step_outputs.append({
            'preds': preds.detach().cpu(),
            'labels': labels.detach().cpu()
        })
        
        return {'preds': preds, 'labels': labels}
    
    def on_test_epoch_end(self):
        """Generate comprehensive test report"""
        if not self.test_step_outputs:
            return
        
        # Aggregate all predictions
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs]).numpy().flatten()
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs]).numpy().flatten()
        
        # Calculate metrics
        acc = accuracy_score(all_labels, all_preds)
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Classification report
        class_names = ['Water', 'Tree canopy', 'Low vegetation', 'Impervious', 'Buildings']
        report = classification_report(
            all_labels, all_preds, 
            target_names=class_names,
            zero_division=0, 
            digits=4
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=range(self.config.NUM_CLASSES))
        
        # Print results
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)
        print(f"Overall Accuracy: {acc:.4f}")
        print(f"F1 Score (macro): {f1_macro:.4f}")
        print(f"F1 Score (weighted): {f1_weighted:.4f}")
        print("\nPer-class F1 scores:")
        for i, (name, f1) in enumerate(zip(class_names, f1_per_class)):
            print(f"  Class {i} ({name:20s}): {f1:.4f}")
        print("\nDetailed Classification Report:")
        print(report)
        
        # Save confusion matrix
        self._save_confusion_matrix(cm, class_names)
        
        # Logging not available in pytorch-lightning 0.7.1
        # Results are printed above and saved to tensorboard via logger
        
        self.test_step_outputs.clear()
        print("="*80)
        print("Test evaluation completed!")
        print("="*80 + "\n")
    
    def _save_confusion_matrix(self, cm, class_names):
        """Save confusion matrix as image"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        ax.set_title('Confusion Matrix - Test Set', fontsize=14)
        
        # Save
        save_path = os.path.join(self.config.OUTPUT_DIR, 'confusion_matrix.png')
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
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler: cosine annealing (doesn't need validation metrics)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.MAX_EPOCHS,
            eta_min=1e-6
        )
        
        # For pytorch-lightning 0.7.1, return as list
        return [optimizer], [scheduler]


# ============================================================================
# Main Training Function
# ============================================================================

def train():
    """Main training function"""
    
    # Configuration
    config = Config()
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print("POINTNET++ TRAINING - BASELINE")
    print("="*80)
    print(f"Data directory: {config.DATA_DIR}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Max epochs: {config.MAX_EPOCHS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Number of classes: {config.NUM_CLASSES}")
    print(f"Points per tile: {config.NUM_POINTS}")
    print(f"Use class weights: {config.USE_CLASS_WEIGHTS}")
    print("="*80)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available! Training will be very slow.")
        print("This script is designed for GPU training on HPC.\n")
    else:
        print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}\n")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, metadata = get_dataloaders(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    print(f"✓ Data loaded:")
    print(f"  Train: {len(train_loader.dataset)} tiles ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_loader.dataset)} tiles ({len(val_loader)} batches)")
    print(f"  Test:  {len(test_loader.dataset)} tiles ({len(test_loader)} batches)")
    
    # Initialize model
    print("\nInitializing model...")
    model = PointNet2SemanticSegmentation(config)
    print("✓ Model initialized")
    
    # Callbacks (pytorch-lightning 0.7.1 uses filepath instead of dirpath)
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(config.OUTPUT_DIR, 'checkpoints', 'pointnet2-{epoch:02d}-{val/loss:.4f}'),
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
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=config.OUTPUT_DIR,
        name='tensorboard_logs'
    )
    
    # Trainer (callbacks disabled for pytorch-lightning 0.7.1 compatibility)
    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        # callbacks=[checkpoint_callback, early_stop_callback],  # Disabled
        logger=logger,
        log_every_n_steps=config.LOG_EVERY_N_STEPS,
        gradient_clip_val=1.0,  # Prevent exploding gradients
        deterministic=False,  # Faster training
    )
    
    # Train
    print("\nStarting training...")
    print("="*80)
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model manually (checkpoints were disabled for pytorch-lightning 0.7.1)
    print("\n" + "="*80)
    print("Saving model...")
    print("="*80)
    final_model_path = os.path.join(config.OUTPUT_DIR, 'final_model.pth')
    torch.save({
        'epoch': trainer.current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizers[0].state_dict(),
        'config': config,
    }, final_model_path)
    print(f"✓ Model saved to: {final_model_path}")
    
    # Test - in pytorch-lightning 0.7.1, test() only takes model
    # The model already has the test_dataloader via test_dataloader() method
    print("\n" + "="*80)
    print("Running final test...")
    print("="*80)
    
    # Set the test dataloader on the model for pytorch-lightning 0.7.1
    model._test_loader = test_loader
    trainer.test(model)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Checkpoints: {os.path.join(config.OUTPUT_DIR, 'checkpoints')}")
    print(f"Tensorboard logs: {os.path.join(config.OUTPUT_DIR, 'tensorboard_logs')}")
    print(f"Confusion matrix: {os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png')}")
    print("\nTo view tensorboard:")
    print(f"  tensorboard --logdir {os.path.join(config.OUTPUT_DIR, 'tensorboard_logs')}")
    

if __name__ == "__main__":
    train()
