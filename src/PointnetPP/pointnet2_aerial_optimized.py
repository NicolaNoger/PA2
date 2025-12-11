"""
Optimized PointNet2 Architecture for Aerial LiDAR
=================================================

Key Optimizations for 25x25m Aerial Tiles:
1. Larger radii (1.0-10.0m vs 0.1-0.8m indoor)
2. More sampling points for better coverage
3. Dice Loss for boundary optimization
4. Increased dropout for better generalization

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule


# ============================================================================
# Dice Loss for Better Boundary Segmentation
# ============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for semantic segmentation.
    
    Optimizes directly for IoU/F1 score, particularly good for:
    - Sharp boundaries (building edges)
    - Small regions (water bodies)
    - Class imbalance
    """
    
    def __init__(self, smooth=1.0, ignore_index=-100):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, N) - raw predictions
            targets: (B, N) - ground truth labels
        
        Returns:
            dice_loss: scalar
        """
        # Get probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, N)
        
        # Flatten
        B, C, N = probs.shape
        probs = probs.permute(0, 2, 1).contiguous().view(-1, C)  # (B*N, C)
        targets_flat = targets.view(-1)  # (B*N,)
        
        # Mask out ignore_index
        if self.ignore_index >= 0:
            mask = targets_flat != self.ignore_index
            probs = probs[mask]
            targets_flat = targets_flat[mask]
        
        # One-hot encode targets
        targets_one_hot = F.one_hot(targets_flat, num_classes=C).float()  # (B*N, C)
        
        # Calculate Dice coefficient per class
        intersection = (probs * targets_one_hot).sum(dim=0)  # (C,)
        cardinality = probs.sum(dim=0) + targets_one_hot.sum(dim=0)  # (C,)
        
        dice_coef = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Dice loss = 1 - mean dice coefficient
        dice_loss = 1.0 - dice_coef.mean()
        
        return dice_loss


# ============================================================================
# Optimized Architecture for Aerial LiDAR
# ============================================================================

class PointNet2AerialSSG(nn.Module):
    """
    PointNet2 Semantic Segmentation optimized for Aerial LiDAR.
    
    Differences from standard SSG:
    - Larger radii adapted for 25x25m tiles (1m-10m vs 0.1m-0.8m)
    - More sampling points for dense aerial scans
    - Higher dropout for better generalization
    - 7 input features (XYZ + Intensity + 3 return info)
    """
    
    def __init__(self, num_classes=5, input_channels=4, use_xyz=True, dropout=0.6):
        super(PointNet2AerialSSG, self).__init__()
        
        self.num_classes = num_classes
        self.use_xyz = use_xyz
        
        # Set Abstraction Modules - Optimized for 25x25m tiles
        self.SA_modules = nn.ModuleList()
        
        # SA1: Fine details (1m radius, 2048 points)
        # Captures small structures: thin trees, building details, small water bodies
        self.SA_modules.append(
            PointnetSAModule(
                npoint=2048,
                radius=1.0,
                nsample=64,
                mlp=[input_channels, 32, 32, 64],
                use_xyz=use_xyz,
            )
        )
        
        # SA2: Local patterns (2.5m radius, 512 points)
        # Captures: tree clusters, building sections, vegetation patches
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=2.5,
                nsample=48,
                mlp=[64, 64, 64, 128],
                use_xyz=use_xyz,
            )
        )
        
        # SA3: Medium context (5m radius, 128 points)
        # Captures: whole buildings, large trees, roads
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=5.0,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz,
            )
        )
        
        # SA4: Global context (10m radius, 32 points)
        # Captures: tile-level patterns, neighborhood structure
        self.SA_modules.append(
            PointnetSAModule(
                npoint=32,
                radius=10.0,
                nsample=24,
                mlp=[256, 256, 256, 512],
                use_xyz=use_xyz,
            )
        )
        
        # Feature Propagation Modules - Upsample features back to original resolution
        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(
            PointnetFPModule(mlp=[128 + input_channels, 128, 128, 128])
        )
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))
        
        # Final classification layer with increased dropout
        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(dropout),  # Increased from 0.5 to 0.6
            nn.Conv1d(128, num_classes, kernel_size=1),
        )
    
    def _break_up_pc(self, pc):
        """Separate XYZ coordinates from features"""
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features
    
    def forward(self, pointcloud):
        """
        Forward pass
        
        Args:
            pointcloud: (B, N, 3+C) - XYZ + features
        
        Returns:
            logits: (B, num_classes, N)
        """
        xyz, features = self._break_up_pc(pointcloud)
        
        # Encoder: Set Abstraction
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        
        # Decoder: Feature Propagation
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )
        
        # Final classification
        return self.fc_layer(l_features[0])




if __name__ == "__main__":
    
    # Test model
    print("Testing model instantiation...")
    model = PointNet2AerialSSG(num_classes=5, input_channels=4, dropout=0.6)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created successfully")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 16384, 7)  # (B, N, 3+4)
    output = model(dummy_input)
    print(f"Forward pass successful")
    print(f"   Input shape:  {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Test Dice Loss
    dice_loss = DiceLoss()
    dummy_targets = torch.randint(0, 5, (2, 16384))
    loss = dice_loss(output, dummy_targets)
    print(f"Dice Loss computed: {loss.item():.4f}")
    
    print("\n All tests passed! Ready for training.")
