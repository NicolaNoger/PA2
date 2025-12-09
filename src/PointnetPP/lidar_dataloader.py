"""
Custom PyTorch Dataset for LiDAR Semantic Segmentation
Loads preprocessed tiles from numpy arrays
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os


class LiDARTileDataset(Dataset):
    """
    Dataset for LiDAR point cloud tiles
    Loads features and labels from preprocessed numpy arrays
    
    Expected structure:
    data_dir/
        train_features.npy  (N_train, 16384, 7)
        train_labels.npy    (N_train, 16384)
        val_features.npy    (N_val, 16384, 7)
        val_labels.npy      (N_val, 16384)
        test_features.npy   (N_test, 16384, 7)
        test_labels.npy     (N_test, 16384)
        metadata.json
    """
    
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir: Path to directory containing the numpy arrays
            split: 'train', 'val', or 'test'
            transform: Optional transform to apply to point clouds
        """
        assert split in ['train', 'val', 'test'], f"Split must be 'train', 'val', or 'test', got {split}"
        
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load metadata (allow several common paths / fallbacks to avoid relative-path issues on HPC)
        metadata_path = os.path.join(data_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            # Try absolute project locations as fallbacks
            candidates = [
                os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'lidar', 'pointnet_tiles', 'metadata.json'),
                os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed', 'pointnet_tiles', 'metadata.json'),
                os.path.join('/cfs/earth/scratch/nogernic/PA2', 'data', 'lidar', 'pointnet_tiles', 'metadata.json')
            ]
            found = False
            for cand in candidates:
                cand = os.path.normpath(cand)
                if os.path.exists(cand):
                    metadata_path = cand
                    # set data_dir to the directory containing the metadata so subsequent loads use the same folder
                    data_dir = os.path.dirname(metadata_path)
                    found = True
                    print(f"Metadata not found at provided path; using fallback: {metadata_path}")
                    break
            if not found:
                raise FileNotFoundError(f"Could not find metadata.json in {data_dir} or fallbacks: {candidates}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loading {split} data from {data_dir}...")
        
        # Load features and labels
        features_path = os.path.join(data_dir, f'{split}_features.npy')
        labels_path = os.path.join(data_dir, f'{split}_labels.npy')
        
        # Load with memory mapping for efficiency (doesn't load all into RAM)
        self.features = np.load(features_path, mmap_mode='r')
        self.labels = np.load(labels_path, mmap_mode='r')
        
        print(f"Loaded {split} data:")
        print(f"  Features: {self.features.shape} ({self.features.dtype})")
        print(f"  Labels: {self.labels.shape} ({self.labels.dtype})")
        
        # Verify shapes
        assert self.features.shape[0] == self.labels.shape[0], "Features and labels must have same number of tiles"
        assert self.features.shape[1] == self.metadata['points_per_tile'], f"Expected {self.metadata['points_per_tile']} points per tile"
        assert self.features.shape[2] == self.metadata['num_features'], f"Expected {self.metadata['num_features']} features"
        
        self.num_tiles = self.features.shape[0]
        self.num_points = self.features.shape[1]
        self.num_features = self.features.shape[2]
        self.num_classes = self.metadata['num_classes']
        
        print(f"  Classes: {self.num_classes}")
        print(f"  Tiles: {self.num_tiles}")
    
    def __len__(self):
        return self.num_tiles
    
    def __getitem__(self, idx):
        """
        Returns:
            features: (N, 7) tensor - point cloud with features
            labels: (N,) tensor - semantic labels
        """
        # Load single tile (memory mapped, so efficient)
        features = self.features[idx]  # (16384, 7)
        labels = self.labels[idx]      # (16384,)
        
        # Convert to torch tensors
        features = torch.from_numpy(features.copy()).float()
        labels = torch.from_numpy(labels.copy()).long()
        
        # Apply transform if any
        if self.transform is not None:
            features, labels = self.transform(features, labels)
        
        return features, labels
    
    def get_class_weights(self):
        """
        Calculate class weights for weighted loss
        Useful for handling class imbalance
        """
        print(f"Calculating class weights for {self.split} set...")
        
        # Sample 10% of data to estimate class distribution
        sample_size = max(1, self.num_tiles // 10)
        sample_indices = np.random.choice(self.num_tiles, sample_size, replace=False)
        
        class_counts = np.zeros(self.num_classes, dtype=np.int64)
        
        for idx in sample_indices:
            labels = self.labels[idx]
            for c in range(self.num_classes):
                class_counts[c] += np.sum(labels == c)
        
        # Calculate weights (inverse frequency)
        total = class_counts.sum()
        class_weights = total / (self.num_classes * class_counts + 1e-6)
        
        # Normalize
        class_weights = class_weights / class_weights.sum() * self.num_classes
        
        print(f"Class distribution (sampled):")
        for c, name in self.metadata['classes'].items():
            print(f"  Class {c} ({name}): {class_counts[int(c)]:,} points, weight: {class_weights[int(c)]:.3f}")
        
        return torch.from_numpy(class_weights).float()


def get_dataloaders(data_dir, batch_size=16, num_workers=4, pin_memory=True, train_transform=None):
    """
    Convenience function to create train, val, test dataloaders
    
    Args:
        data_dir: Path to processed data directory
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        pin_memory: Pin memory for faster GPU transfer
        train_transform: Optional transform for training data augmentation (not applied to val/test)
    
    Returns:
        train_loader, val_loader, test_loader, metadata
    """
    # Create datasets (apply transform only to training data!)
    train_dataset = LiDARTileDataset(data_dir, split='train', transform=train_transform)
    val_dataset = LiDARTileDataset(data_dir, split='val', transform=None)  # No augmentation
    test_dataset = LiDARTileDataset(data_dir, split='test', transform=None)  # No augmentation
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for stable training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    print("\n" + "="*80)
    print("DATALOADERS CREATED")
    print("="*80)
    print(f"Train: {len(train_loader)} batches ({len(train_dataset)} tiles)")
    print(f"Val:   {len(val_loader)} batches ({len(val_dataset)} tiles)")
    print(f"Test:  {len(test_loader)} batches ({len(test_dataset)} tiles)")
    print(f"Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader, train_dataset.metadata


# Optional: Data augmentation transforms
class RandomRotation:
    """Random rotation around Z axis"""
    def __init__(self, max_angle=180):
        self.max_angle = max_angle
    
    def __call__(self, features, labels):
        # Random angle
        angle = np.random.uniform(-self.max_angle, self.max_angle) * np.pi / 180.0
        
        # Rotation matrix (around Z axis)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # Rotate X and Y coordinates
        x = features[:, 0].clone()
        y = features[:, 1].clone()
        features[:, 0] = cos_a * x - sin_a * y
        features[:, 1] = sin_a * x + cos_a * y
        
        return features, labels


class RandomJitter:
    """Add random jitter to point positions"""
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip
    
    def __call__(self, features, labels):
        # Add jitter to X, Y, Z
        jitter = torch.clamp(
            torch.randn_like(features[:, :3]) * self.sigma,
            -self.clip, self.clip
        )
        features[:, :3] += jitter
        return features, labels


class RandomScale:
    """Random scaling"""
    def __init__(self, scale_low=0.8, scale_high=1.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
    
    def __call__(self, features, labels):
        scale = torch.rand(1) * (self.scale_high - self.scale_low) + self.scale_low
        features[:, :3] *= scale
        return features, labels


class Compose:
    """Compose multiple transforms"""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, features, labels):
        for t in self.transforms:
            features, labels = t(features, labels)
        return features, labels


# Example usage
if __name__ == "__main__":
    # Test the dataloader
    data_dir = "/cfs/earth/scratch/nogernic/PA2/data/lidar/pointnet_tiles"
    
    # Without augmentation
    train_loader, val_loader, test_loader, metadata = get_dataloaders(
        data_dir=data_dir,
        batch_size=4,
        num_workers=0  # 0 for debugging
    )
    
    # Test loading a batch
    print("\n" + "="*80)
    print("Testing data loading...")
    print("="*80)
    
    for features, labels in train_loader:
        print(f"Batch features: {features.shape} ({features.dtype})")
        print(f"Batch labels: {labels.shape} ({labels.dtype})")
        print(f"Feature range: X=[{features[:,:,0].min():.2f}, {features[:,:,0].max():.2f}], "
              f"Z=[{features[:,:,2].min():.2f}, {features[:,:,2].max():.2f}]")
        print(f"Label range: [{labels.min()}, {labels.max()}]")
        break
    
    # Test with augmentation
    print("\n" + "="*80)
    print("Testing with augmentation...")
    print("="*80)
    
    augmentation = Compose([
        RandomRotation(max_angle=180),
        RandomJitter(sigma=0.01, clip=0.05),
        RandomScale(scale_low=0.8, scale_high=1.2)
    ])
    
    train_dataset_aug = LiDARTileDataset(data_dir, split='train', transform=augmentation)
    train_loader_aug = DataLoader(train_dataset_aug, batch_size=4, shuffle=True)
    
    for features, labels in train_loader_aug:
        print(f"Augmented batch: {features.shape}")
        break
    
    # Calculate class weights
    print("\n" + "="*80)
    print("Calculating class weights...")
    print("="*80)
    
    class_weights = train_dataset_aug.get_class_weights()
    print(f"Class weights tensor: {class_weights}")
    
    print("\n✓ Dataloader test complete!")
