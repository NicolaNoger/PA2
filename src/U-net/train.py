import tensorflow as tf
import numpy as np
import os
from datetime import datetime

# Set matplotlib backend BEFORE importing pyplot (for headless systems)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers without display
import matplotlib.pyplot as plt

# Try to import sklearn, but make it optional
try:
    from sklearn.metrics import confusion_matrix, classification_report, f1_score
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except (ImportError, TypeError) as e:
    print(f"Warning: sklearn/scipy not available: {e}")
    print("  Confusion matrix and F1 scores will be skipped")
    SKLEARN_AVAILABLE = False

# Disable XLA to avoid libdevice.10.bc error
# XLA compilation causes issues with CUDA 11.6.2 on this system
tf.config.optimizer.set_jit(False)

# Configure GPU memory growth to avoid OOM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")

# Importiere deine Module
from U_net import build_unet
from dataloader import load_npy_dataset, split_dataset, prepare_dataset


class Config:
    # Paths
    DATA_PATH = "A:/STUDIUM/05_Herbstsemester25/PA2/data/aerial/training_data"
    IMG_TILES_PATH = os.path.join(DATA_PATH, "img_tiles")
    MASK_TILES_PATH = os.path.join(DATA_PATH, "mask_tiles")
    OUTPUT_DIR = "models"
    
    # Dataset Parameter
    NUM_CLASSES = 5  
    TRAIN_RATIO = 0.8

    # Training Hyperparameter
    BATCH_SIZE = 4  # Reduced from 8 to 4 to avoid OOM
    EPOCHS = 20  
    LEARNING_RATE = 1e-4 
    
    # Model Parameter
    INPUT_SHAPE = (512, 512, 4)  # NIR + RGB Channels
    STEPS_PER_EPOCH = None
    VALIDATION_STEPS = None


def create_output_directory():
    """creates output directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(Config.OUTPUT_DIR, f"unet_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_data():
    """Loads and prepares data"""
    print("\n" + "="*70)
    print("STEP 1: LOAD DATA")
    print("="*70)
    print(f"Loading data from:")
    print(f"  - Images: {Config.IMG_TILES_PATH}")
    print(f"  - Masks:  {Config.MASK_TILES_PATH}")
    
    dataset = load_npy_dataset(Config.IMG_TILES_PATH, Config.MASK_TILES_PATH)
    
    total_samples = dataset.cardinality().numpy()
    print(f"Total Samples: {total_samples}")
    
    # 2. Split into Training and Validation
    print(f"\nSplitting dataset (Train/Val: {Config.TRAIN_RATIO:.0%}/{1-Config.TRAIN_RATIO:.0%})...")
    train_dataset, val_dataset = split_dataset(dataset, train_ratio=Config.TRAIN_RATIO)
    train_size = train_dataset.cardinality().numpy()
    val_size = val_dataset.cardinality().numpy()
    print(f"Training Samples:   {train_size}")
    print(f"Validation Samples: {val_size}")
    
    # 3. Berechne Steps für Training
    Config.STEPS_PER_EPOCH = max(1, train_size // Config.BATCH_SIZE)
    # Reduce validation steps to save memory (only validate on subset)
    Config.VALIDATION_STEPS = min(100, max(1, val_size // Config.BATCH_SIZE))
    print(f"Steps per Epoch:    {Config.STEPS_PER_EPOCH}")
    print(f"Validation Steps:   {Config.VALIDATION_STEPS} (limited to save memory)")
    
    # 4. Prepare Datasets (with Batching, Augmentation, etc.)
    print("\nPreparing Training Dataset...")
    train_batches = prepare_dataset(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        num_classes=Config.NUM_CLASSES,
        is_training=True
    )
    print("Training Dataset ready")
    
    print("Preparing Validation Dataset...")
    val_batches = prepare_dataset(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        num_classes=Config.NUM_CLASSES,
        is_training=False  # No Shuffle, No Augment, No Repeat
    )
    print("Validation Dataset ready")
    
    # 5. Check Data Format
    print("\nChecking Data Format...")
    for images, masks in train_batches.take(1):
        print(f"Batch Images Shape: {images.shape}")  # (batch_size, 512, 512, 4)
        print(f"Batch Masks Shape:  {masks.shape}")   # (batch_size, 512, 512, num_classes)
        print(f"  Image dtype:        {images.dtype}")
        print(f"  Mask dtype:         {masks.dtype}")
        print(f"  Image value range:  [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")
        print(f"  Mask unique values: {np.unique(masks.numpy())}")  
    
    return train_batches, val_batches


def build_model():
    """Builds and compiles the U-Net model"""
    print("\n" + "="*70)
    print("STEP 2: BUILD MODEL")
    print("="*70)
    
    # 1. Create Model
    model = build_unet(
        input_shape=Config.INPUT_SHAPE,
        num_classes=Config.NUM_CLASSES
    )
    print("Model created")
    
    # 2. compile Model
    print("\nCompiling Model...")
    # Use legacy Adam optimizer to avoid XLA issues with libdevice
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=Config.LEARNING_RATE),
        loss='categorical_crossentropy',  # For multi-class segmentation
        metrics=[
            'accuracy',
            tf.keras.metrics.CategoricalAccuracy(name='cat_accuracy'),
            tf.keras.metrics.MeanIoU(num_classes=Config.NUM_CLASSES, name='iou')
        ]
    )
    print("Model compiled")

    
    # 3. show Model Summary
    print("\nModel Architecture:")
    print("-" * 70)
    model.summary()
    
    total_params = model.count_params()
    print(f"\nTotal Parameters: {total_params:,}")
    
    return model


def setup_callbacks(output_dir):
    """Creates callbacks for training"""
    print("\n" + "="*70)
    print("STEP 3: SETUP CALLBACKS")
    print("="*70)
    
    callbacks = []
    
    # 0. Memory Cleanup Callback - Clear cache after each epoch to avoid OOM
    class MemoryCleanupCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            import gc
            gc.collect()
            # Don't clear_session as it would reset the model!
            # Just force garbage collection
    
    callbacks.append(MemoryCleanupCallback())
    print("MemoryCleanupCallback: Forces garbage collection after each epoch")
    
    # 1. ModelCheckpoint - saves best model
    # Use SavedModel format instead of .h5 to avoid h5py precision issues
    checkpoint_path = os.path.join(output_dir, "best_model")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_format='tf',  # Use SavedModel format instead of HDF5
        mode='min',
        verbose=1
    )
    callbacks.append(checkpoint)
    print(f"ModelCheckpoint: {checkpoint_path} (SavedModel format)")
    
    # 2. EarlyStopping - Stops training if no progress
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,  # Waits 5 epochs without improvement
        mode='min',
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stop)
    print("EarlyStopping: patience=5")
    
    # 3. ReduceLROnPlateau - Reduces learning rate on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Halves learning rate
        patience=3,
        min_lr=1e-7,
        mode='min',
        verbose=1
    )
    callbacks.append(reduce_lr)
    print("ReduceLROnPlateau: factor=0.5, patience=3")
    
    # 4. CSVLogger - Saves training history
    csv_path = os.path.join(output_dir, "training_history.csv")
    csv_logger = tf.keras.callbacks.CSVLogger(csv_path)
    callbacks.append(csv_logger)
    print(f"CSVLogger: {csv_path}")
    
    # 5. TensorBoard - For visualization 
    tensorboard_dir = os.path.join(output_dir, "tensorboard_logs")
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_dir,
        histogram_freq=1,
        write_graph=True
    )
    callbacks.append(tensorboard)
    print(f"TensorBoard: {tensorboard_dir}")
    print("  (Start with: tensorboard --logdir=models)")
    
    return callbacks


def train_model(model, train_batches, val_batches, callbacks):
    """Trains the model"""
    print("\n" + "="*70)
    print("STEP 4: START TRAINING")
    print("="*70)

    print(f"\nTraining Configuration:")
    print(f"  Epochs:             {Config.EPOCHS}")
    print(f"  Batch Size:         {Config.BATCH_SIZE}")
    print(f"  Steps per Epoch:    {Config.STEPS_PER_EPOCH}")
    print(f"  Validation Steps:   {Config.VALIDATION_STEPS}")
    print(f"  Number of Classes:  {Config.NUM_CLASSES}")
    print("\nTraining starts...\n")
    
    # GPU Info
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Training with GPU: {gpus[0].name}")
    else:
        print("Warning: No GPU found, training on CPU!")
    print()
    
    # Perform training
    history = model.fit(
        train_batches,
        epochs=Config.EPOCHS,
        steps_per_epoch=Config.STEPS_PER_EPOCH,
        validation_data=val_batches,
        validation_steps=Config.VALIDATION_STEPS,
        callbacks=callbacks,
        verbose=1  # shows Progress Bar
    )
    
    print("\nTraining completed!")
    return history


def plot_training_history(history, output_dir):
    """Plots training history"""
    print("\n" + "="*70)
    print("STEP 5: VISUALIZATION")
    print("="*70)
    
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Convert history values to plain Python lists to avoid numpy/tensor issues
        # 1. Loss
        loss = [float(x) for x in history.history['loss']]
        val_loss = [float(x) for x in history.history['val_loss']]
        axes[0].plot(loss, label='Training Loss', linewidth=2)
        axes[0].plot(val_loss, label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Accuracy
        acc = [float(x) for x in history.history['accuracy']]
        val_acc = [float(x) for x in history.history['val_accuracy']]
        axes[1].plot(acc, label='Training Accuracy', linewidth=2)
        axes[1].plot(val_acc, label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. IoU
        if 'iou' in history.history:
            iou = [float(x) for x in history.history['iou']]
            val_iou = [float(x) for x in history.history['val_iou']]
            axes[2].plot(iou, label='Training IoU', linewidth=2)
            axes[2].plot(val_iou, label='Validation IoU', linewidth=2)
            axes[2].set_xlabel('Epoch', fontsize=12)
            axes[2].set_ylabel('IoU', fontsize=12)
            axes[2].set_title('Intersection over Union', fontsize=14, fontweight='bold')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save with error handling
        plot_path = os.path.join(output_dir, "training_history.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved: {plot_path}")
        
    except Exception as e:
        print(f"Warning: Could not save training plot: {e}")
    finally:
        plt.close('all')


def visualize_predictions(model, val_batches, output_dir, num_samples=3):
    """Visualizes predictions"""
    print("\nCreating prediction visualizations...")
    
    try:
        # Get one batch
        for images, masks in val_batches.take(1):
            predictions = model.predict(images, verbose=0)
            
            # Convert one-hot back to classes
            true_masks = np.argmax(masks.numpy(), axis=-1)
            pred_masks = np.argmax(predictions, axis=-1)
            
            # Plot first num_samples
            for i in range(min(num_samples, images.shape[0])):
                try:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Convert to plain numpy arrays to avoid matplotlib issues
                    # Original image (show only RGB channels)
                    rgb_img = np.array(images[i, :, :, 1:4].numpy(), dtype=np.float32)
                    # Normalize for display
                    rgb_min = float(rgb_img.min())
                    rgb_max = float(rgb_img.max())
                    rgb_img = (rgb_img - rgb_min) / (rgb_max - rgb_min + 1e-7)
                    
                    axes[0].imshow(rgb_img)
                    axes[0].set_title('Original Image (RGB)', fontsize=12, fontweight='bold')
                    axes[0].axis('off')
                    
                    # Ground Truth (convert to int for clean display)
                    true_mask_int = np.array(true_masks[i], dtype=np.int32)
                    axes[1].imshow(true_mask_int, cmap='tab10', vmin=0, vmax=Config.NUM_CLASSES-1)
                    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
                    axes[1].axis('off')
                    
                    # Prediction (convert to int for clean display)
                    pred_mask_int = np.array(pred_masks[i], dtype=np.int32)
                    im = axes[2].imshow(pred_mask_int, cmap='tab10', vmin=0, vmax=Config.NUM_CLASSES-1)
                    axes[2].set_title('Prediction', fontsize=12, fontweight='bold')
                    axes[2].axis('off')
                    
                    # Colorbar
                    cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
                    cbar.set_label('Class', rotation=270, labelpad=15)
                    
                    plt.tight_layout()
                    
                    # Save
                    pred_path = os.path.join(output_dir, f"prediction_sample_{i+1}.png")
                    plt.savefig(pred_path, dpi=300, bbox_inches='tight')
                    plt.close('all')
                    
                    print(f"Prediction {i+1} saved: {pred_path}")
                    
                except Exception as e:
                    print(f"Warning: Could not save prediction {i+1}: {e}")
                    plt.close('all')
                    
    except Exception as e:
        print(f"Warning: Could not create prediction visualizations: {e}")


def evaluate_model(model, val_batches, output_dir):
    """Evaluates model with confusion matrix and F1 score"""
    
    if not SKLEARN_AVAILABLE:
        print("\n" + "="*70)
        print("STEP 5b: MODEL EVALUATION - SKIPPED")
        print("="*70)
        print("sklearn/scipy not available - skipping confusion matrix and F1 scores")
        print("Training metrics are still available in training_history.csv")
        return
    
    print("\n" + "="*70)
    print("STEP 5b: MODEL EVALUATION")
    print("="*70)
    
    print("Collecting predictions for evaluation...")
    
    all_true_labels = []
    all_pred_labels = []
    
    # Collect predictions from validation set
    num_batches = min(100, Config.VALIDATION_STEPS)  # Limit for speed
    for i, (images, masks) in enumerate(val_batches.take(num_batches)):
        if i % 20 == 0:
            print(f"  Processing batch {i+1}/{num_batches}...")
        
        predictions = model.predict(images, verbose=0)
        
        # Convert one-hot to class labels
        true_labels = np.argmax(masks.numpy(), axis=-1).flatten()
        pred_labels = np.argmax(predictions, axis=-1).flatten()
        
        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_labels)
    
    all_true_labels = np.array(all_true_labels)
    all_pred_labels = np.array(all_pred_labels)
    
    print(f"\nEvaluated on {len(all_true_labels):,} pixels")
    
    # 1. Confusion Matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=range(Config.NUM_CLASSES),
                    yticklabels=range(Config.NUM_CLASSES))
        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('True Class', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Confusion matrix saved: {cm_path}")
    except Exception as e:
        print(f"⚠ Could not save confusion matrix plot: {e}")
    
    # 2. Classification Report
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    report = classification_report(
        all_true_labels, 
        all_pred_labels,
        target_names=[f'Class {i}' for i in range(Config.NUM_CLASSES)],
        digits=4
    )
    print(report)
    
    # Save report to file
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*70 + "\n")
        f.write(report)
    print(f"✓ Classification report saved: {report_path}")
    
    # 3. F1 Scores
    print("\n" + "="*70)
    print("F1 SCORES")
    print("="*70)
    
    # Per-class F1
    f1_per_class = f1_score(all_true_labels, all_pred_labels, average=None)
    print("\nPer-Class F1 Scores:")
    for i, f1 in enumerate(f1_per_class):
        print(f"  Class {i}: {f1:.4f}")
    
    # Overall F1 scores
    f1_macro = f1_score(all_true_labels, all_pred_labels, average='macro')
    f1_weighted = f1_score(all_true_labels, all_pred_labels, average='weighted')
    
    print(f"\nMacro-averaged F1:    {f1_macro:.4f}")
    print(f"Weighted-averaged F1: {f1_weighted:.4f}")
    
    # Save F1 scores
    f1_path = os.path.join(output_dir, "f1_scores.txt")
    with open(f1_path, 'w') as f:
        f.write("F1 SCORES\n")
        f.write("="*70 + "\n\n")
        f.write("Per-Class F1 Scores:\n")
        for i, f1 in enumerate(f1_per_class):
            f.write(f"  Class {i}: {f1:.4f}\n")
        f.write(f"\nMacro-averaged F1:    {f1_macro:.4f}\n")
        f.write(f"Weighted-averaged F1: {f1_weighted:.4f}\n")
    print(f"✓ F1 scores saved: {f1_path}")
    
    print("="*70)


def save_model(model, output_dir):
    """Saves final model"""
    print("\n" + "="*70)
    print("STEP 6: SAVE MODEL")
    print("="*70)
    
    # Save complete model (SavedModel format to avoid h5py issues)
    model_path = os.path.join(output_dir, "final_model")
    model.save(model_path, save_format='tf')
    print(f"Final model saved: {model_path} (SavedModel format)")
    
    # Save only weights (can still use .h5 for weights only)
    weights_path = os.path.join(output_dir, "model_weights.h5")
    try:
        model.save_weights(weights_path)
        print(f"Model weights saved: {weights_path}")
    except Exception as e:
        print(f"Warning: Could not save weights as .h5: {e}")
        weights_path_tf = os.path.join(output_dir, "model_weights")
        model.save_weights(weights_path_tf)
        print(f"Model weights saved: {weights_path_tf} (TF format)")


def main():
    """Main function - Executes complete training"""
    print("\n" + "="*70)
    print("U-NET TRAINING PIPELINE")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directory
    output_dir = create_output_directory()
    
    try:
        # 1. Load data
        train_batches, val_batches = load_data()
        
        # 2. Build model
        model = build_model()
        
        # 3. Setup callbacks
        callbacks = setup_callbacks(output_dir)
        
        # 4. Train model
        history = train_model(model, train_batches, val_batches, callbacks)
        
        # 5. Visualize results
        plot_training_history(history, output_dir)
        evaluate_model(model, val_batches, output_dir)  # Confusion Matrix & F1
        visualize_predictions(model, val_batches, output_dir, num_samples=3)
        
        # 6. Save model
        save_model(model, output_dir)
        
        print("\n" + "="*70)
        print("✓ TRAINING SUCCESSFUL!")
        print("="*70)
        print(f"All files saved in: {output_dir}")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print("\n" + "="*70)
        print("ERROR DURING TRAINING!")
        print("="*70)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"\nPartial results may be in: {output_dir}")


if __name__ == "__main__":
    main()

