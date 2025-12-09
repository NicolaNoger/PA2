#!/usr/bin/env python3
"""
Fine-tuning script for PointNet++ on LiDAR semantic segmentation.
Loads a pre-trained checkpoint and continues training with adjusted hyperparameters.
"""

import os
import sys
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add PointNet2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PointNet2_PyTorch'))

from lidar_dataloader import get_dataloaders
from pointnet_train_baseline import PointNet2SemanticSegmentation, Config


# ============================================================================
# Fine-tuning Configuration
# ============================================================================

class FinetuneConfig(Config):
    """Fine-tuning configuration - inherits from baseline Config"""
    
    # Fine-tuning specific
    CHECKPOINT_PATH = '/cfs/earth/scratch/nogernic/PA2/src/PointnetPP/outputs/final_model.pth'
    
    # Create unique output directory with timestamp
    RUN_NAME = f"finetune_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    OUTPUT_DIR = f'/cfs/earth/scratch/nogernic/PA2/src/PointnetPP/outputs/{RUN_NAME}'
    
    # Fine-tuning hyperparameters
    LEARNING_RATE = 1e-4  # Lower LR for fine-tuning
    MAX_EPOCHS = 20  # Additional epochs
    BATCH_SIZE = 24
    
    # Adjust class weights - increase Water (class 0) weight
    WATER_WEIGHT_MULTIPLIER = 3.0  # Increase focus on Water class
    USE_CLASS_WEIGHTS = True
    CLASS_WEIGHTS = [
        4.279 * WATER_WEIGHT_MULTIPLIER,  # Water - boosted!
        0.371,  # Tree canopy
        0.034,  # Low vegetation
        0.129,  # Impervious
        0.187   # Buildings
    ]
    
    # Evaluation option
    EVALUATE_AFTER_TRAINING = False  # Set to True to run test evaluation after training


# ============================================================================
# Evaluation Function (optional - runs after training if enabled)
# ============================================================================

def evaluate_on_test_set(model, test_loader, config):
    """Evaluate model on test set and save results"""
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print(f"  Processing batch {batch_idx+1}/{len(test_loader)}...")
            
            features = features.cuda()
            labels = labels.cuda()
            
            # Forward pass - LightningModule forward() takes full features (B, N, 7)
            logits = model(features)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    class_names = ['Water', 'Tree canopy', 'Low vegetation', 'Impervious', 'Buildings']
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    
    # Print results
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro F1 Score:   {f1_macro:.4f}")
    print(f"Weighted F1:      {f1_weighted:.4f}")
    print("\nPer-Class F1 Scores:")
    for name, f1 in zip(class_names, f1_per_class):
        print(f"  {name:20s}: {f1:.4f}")
    print("\n" + "="*80)
    print(report)
    
    # Save results
    results_path = os.path.join(config.OUTPUT_DIR, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TEST RESULTS - FINE-TUNED MODEL\n")
        f.write("="*80 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1 Score: {f1_macro:.4f}\n")
        f.write(f"Weighted F1 Score: {f1_weighted:.4f}\n\n")
        f.write("Per-Class F1 Scores:\n")
        for name, f1 in zip(class_names, f1_per_class):
            f.write(f"  {name:20s}: {f1:.4f}\n")
        f.write("\n" + "="*80 + "\n")
        f.write(report)
        f.write("\n" + "="*80 + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Fine-tuned Model', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    
    print(f"✓ Results saved to: {results_path}")
    print(f"✓ Confusion matrix saved to: {cm_path}")
    
    return accuracy, f1_macro, f1_per_class


# ============================================================================
# Main Fine-tuning Function
# ============================================================================

def finetune(evaluate_after=False):
    """Main fine-tuning function"""
    
    # Configuration
    config = FinetuneConfig()
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT_DIR, 'checkpoints'), exist_ok=True)
    
    print("\n" + "="*80)
    print("POINTNET++ FINE-TUNING")
    print("="*80)
    print(f"Run name: {config.RUN_NAME}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print(f"Checkpoint to load: {config.CHECKPOINT_PATH}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Additional epochs: {config.MAX_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Water weight multiplier: {config.WATER_WEIGHT_MULTIPLIER}")
    print(f"Adjusted class weights: {config.CLASS_WEIGHTS}")
    print("="*80)
    
    # Check if checkpoint exists
    if not os.path.exists(config.CHECKPOINT_PATH):
        print(f"\n❌ Error: Checkpoint not found at {config.CHECKPOINT_PATH}")
        sys.exit(1)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, train_dataset, val_dataset = get_dataloaders(
        config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    print(f"✓ Data loaded:")
    print(f"  Train: {len(train_dataset)} tiles ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_dataset)} tiles ({len(val_loader)} batches)")
    
    # Load pre-trained model
    print(f"\nLoading pre-trained model from: {config.CHECKPOINT_PATH}")
    
    # Create model instance
    model = PointNet2SemanticSegmentation(config)
    
    # Load checkpoint
    checkpoint = torch.load(config.CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to GPU
    if torch.cuda.is_available():
        model = model.cuda()
    
    print(f"✓ Model loaded successfully!")
    print(f"  Previously trained for: {checkpoint['epoch']} epochs")
    print(f"  Now fine-tuning with adjusted Water weight (x{config.WATER_WEIGHT_MULTIPLIER})")
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=config.OUTPUT_DIR,
        name='tensorboard_logs'
    )
    
    # Trainer
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
    finetuned_model_path = os.path.join(config.OUTPUT_DIR, 'finetuned_model.pth')
    torch.save({
        'epoch': trainer.current_epoch + checkpoint['epoch'],  # Total epochs
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizers[0].state_dict(),
        'config': config,
        'base_checkpoint': config.CHECKPOINT_PATH,
        'finetune_epochs': config.MAX_EPOCHS,
    }, finetuned_model_path)
    print(f"✓ Fine-tuned model saved to: {finetuned_model_path}")
    
    # Optional: Evaluate on test set
    if evaluate_after or config.EVALUATE_AFTER_TRAINING:
        print("\n" + "="*80)
        print("Running test evaluation...")
        print("="*80)
        
        # Load test data
        from lidar_dataloader import LiDARTileDataset
        test_dataset = LiDARTileDataset(
            data_dir=config.DATA_DIR,
            split='test'
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        
        # Run evaluation
        evaluate_on_test_set(model, test_loader, config)
    
    print("\n" + "="*80)
    print("FINE-TUNING COMPLETE!")
    print("="*80)
    print(f"Run directory: {config.OUTPUT_DIR}")
    print(f"Model: {finetuned_model_path}")
    print(f"TensorBoard logs: {os.path.join(config.OUTPUT_DIR, 'tensorboard_logs')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune PointNet++ model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation on test set after training')
    args = parser.parse_args()
    
    finetune(evaluate_after=args.evaluate)
