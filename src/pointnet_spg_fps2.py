import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib
matplotlib.use("Agg")              # ← prevents any GUI backend
import matplotlib.pyplot as plt
import seaborn as sns              # nicer heat-map; install if missing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Datenladen and Sampling-Utilities
# -------------------------------

def farthest_point_sampling(xyz, npoint):
    """
    Farthest Point Sampling (FPS)
    Eingabe:
        xyz: (N, 3) Array mit Punktkoordinaten
        npoint: Anzahl der zu samplenden Punkte
    Rückgabe:
        centroids: (npoint,) Array der Indizes der ausgewählten Punkte
    """
    # if torch.Tensor, convert to NumPy
    if isinstance(xyz, torch.Tensor):
        pts = xyz.cpu().numpy()
    else:
        pts = xyz
    N, _ = pts.shape
    centroids = np.zeros(npoint, dtype=np.int64)
    distance = np.ones(N) * 1e10
    # Choose a random starting point
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = pts[farthest, :3]
        # Calculate squared distance to the last centroid
        dist = np.sum((pts[:, :3] - centroid) ** 2, axis=1)
        # Update next distance
        mask = dist < distance
        distance[mask] = dist[mask]
        # Select the new farthest point
        farthest = np.argmax(distance)
    return centroids

# -------------------------------
class LiDARPointCloudDataset(Dataset):
    """
    Dataset für LiDAR-Punktwolken-Segmentierung.
    Erwartet Punkt- und Label-.npy-Dateien mit mindestens num_points.
    Integriert FPS- und instanz-basiertes Sampling.
    """
    def __init__(self, data_dir, pattern_points="*_points.npy", pattern_labels="*_labels.npy", num_points=4096, ignore_label=7):
        super().__init__()
        # Gather file paths
        self.point_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith("_points.npy")])
        self.label_files = [pf.replace("_points.npy", "_labels.npy") for pf in self.point_files]
        self.num_points = num_points
        self.ignore_label = ignore_label
        all_labels = [] #-------------------------
        for f in self.label_files:
            raw = np.load(f)
            all_labels.extend(raw.tolist())
        uniques = sorted(set(all_labels) - {ignore_label}) #-------------------------
        self.class_map = {orig: i for i, orig in enumerate(uniques)}
        self.num_classes = len(uniques) #-------------------------


    def __len__(self):
        return len(self.point_files)

    def __getitem__(self, idx):
        # Load point cloud and labels
        points = np.load(self.point_files[idx])  # Form: (N, 3 [+ Features])
        labels = np.load(self.label_files[idx]).astype(int)  # Rohlabels

        # 2) Map to 0...C-1 (ignore_label remains untouched)
        mapped = np.full_like(labels, fill_value=self.ignore_label, dtype=int)
        for orig, new in self.class_map.items():
            mapped[labels == orig] = new
        labels = mapped  # Now all values ∈ [0…num_classes-1] ∪ {ignore_label}

        N = points.shape[0]
        # If there are more points than num_points, apply FPS
        if N > self.num_points:
            inds = farthest_point_sampling(points[:, :3], self.num_points)
            points = points[inds, :]
            labels = labels[inds]
        # If there are fewer points, pad (here by repetition)
        elif N < self.num_points:
            dup_inds = np.random.choice(N, self.num_points - N, replace=True)
            points = np.concatenate([points, points[dup_inds, :]], axis=0)
            labels = np.concatenate([labels, labels[dup_inds]], axis=0)
            # Instance-based Sampling:
            # Ensure that each class appears at least once in the sample
        unique_cls = np.unique(labels)
        for cls in unique_cls:
            if cls == self.ignore_label:
                continue
            if np.sum(labels == cls) == 0:
                # If the class exists in the original, add a point
                orig_class_inds = np.where(labels == cls)[0]
                if len(orig_class_inds) == 0:
                    continue
                add_idx = np.random.choice(orig_class_inds)
                # Replace point from the current majority class
                class_counts = {c: np.sum(labels == c) for c in unique_cls if c != self.ignore_label}
                major = max(class_counts, key=class_counts.get)
                major_inds = np.where(labels == major)[0]
                remove_idx = np.random.choice(major_inds) if len(major_inds) > 1 else np.random.randint(0, self.num_points)
                points[remove_idx] = points[add_idx]
                labels[remove_idx] = labels[add_idx]

        # conver to Torch-Tensor 
        points = torch.from_numpy(points).float().contiguous()  # (num_points, D)
        labels = torch.from_numpy(labels).long()   # (num_points,)

        return points, labels

# -------------------------------
# Lightning-Modul for Training
# -------------------------------

class PointNet2Segmentation(pl.LightningModule):
    """
    PyTorch Lightning Modul für PointNet++ Semantische Segmentierung mit SPG-Loss.
    """
    def __init__(self, model, num_classes=13, ignore_label=7, spg_weight=0.1, lr=1e-3):
        """
        Args:
            model: PointNet2-Segmentierungsmodell (z.B. PointNet2SemSegSSG)
            num_classes: Anzahl der Klassen (inklusive Ignorierklasse)
            ignore_label: Label, das im Loss ignoriert wird
            spg_weight: Gewichtungsfaktor für den SPG-Loss
            lr: Lernrate
        """
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.ignore_label)
        self.spg_weight = spg_weight
        self.lr = lr

    def forward(self, x):
        # Forward pass through the model
        # Expects input (B, N, 3 [+ Features])

        logits = self.model(x)  # output: (B, num_classes, N)
        return logits

    def training_step(self, batch, batch_idx):
        points, labels = batch
        device = next(self.parameters()).device
        points = points.to(device).contiguous()
        labels = labels.to(device)

        print("[DEBUG] points.device =", points.device, flush=True)


        logits = self(points)  # (B, C, N)

        # Cross-Entropy loss (ignores the outlier class)
        ce_loss = self.ce_loss(logits, labels)

        # Subspace Prototype Guidance (SPG) Loss
        # Extract features before the last classification layer
        xyz = points[..., :3]  # (B, N, 3)
        if points.size(-1) > 3:
            feats_input = points[..., 3:].permute(0, 2, 1).contiguous()  # (B, C_in, N)
        else:
            feats_input = None

        l_xyz = [xyz]
        l_features = [feats_input]
        # Apply Set Abstraction Modules
        for sa in self.model.SA_modules:
            cur_xyz   = l_xyz[-1].contiguous()                  
            cur_feats = l_features[-1]
            if cur_feats is not None:
                cur_feats = cur_feats.contiguous()               
            li_xyz, li_feats = sa(cur_xyz, cur_feats)
            l_xyz.append(li_xyz.contiguous())                  
            l_features.append(li_feats.contiguous())              

        # FP Modules in reverse order
        for i in range(len(self.model.FP_modules)):
            idx = -1 - i
            before = l_xyz[idx-1].contiguous()                 
            after  = l_xyz[idx].contiguous()                    
            f1     = l_features[idx-1].contiguous()                
            f2     = l_features[idx].contiguous()                  
            l_features[idx-1] = self.model.FP_modules[idx](before, after, f1, f2).contiguous()
        # l_features[0] has dim(B, 128, N)
        point_feats = l_features[0]  # (B, C_feat, N)

        # Apply to FC layers (up to second last Conv1d)
        fc_layers = list(self.model.fc_lyaer.children())
        feat_extractor = nn.Sequential(*fc_layers[:-1])
        feats = feat_extractor(point_feats)  # (B, C_feat, N)
        B, C_feat, N = feats.shape
        feats = feats.permute(0, 2, 1).reshape(-1, C_feat)  # (B*N, C_feat)
        labels_flat = labels.view(-1)  # (B*N,)

        # Calculate class prototypes and SPG loss
        spg_loss = torch.tensor(0.0, device=logits.device)
        for cls in range(self.num_classes):
            if cls == self.ignore_label:
                continue
            mask = labels_flat == cls
            if torch.any(mask):
                cls_feats = feats[mask]
                center = cls_feats.mean(dim=0, keepdim=True)  # (1, C_feat)
                spg_loss = spg_loss + ((cls_feats - center).pow(2).sum()) / (cls_feats.shape[0] + 1e-6)
        # normalize SPG-Loss 
        spg_loss = spg_loss / (self.num_classes - 1)

        total_loss = ce_loss + self.spg_weight * spg_loss
        # Log the losses
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            print(f"[TRAIN] acc = {acc:.4f}")


        return {
            "loss":            total_loss,
            "train_ce_loss":   ce_loss.detach(),
            "train_spg_loss":  spg_loss.detach(),
            "train_total_loss": total_loss.detach()
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def validation_step(self, batch, batch_idx):
        points, labels = batch
        device = next(self.parameters()).device
        points = points.to(device).contiguous()
        labels = labels.to(device)

        logits = self(points)             # (B, C, N)
        loss = self.ce_loss(logits, labels)
        preds = logits.argmax(dim=1)
        acc   = (preds == labels).float().mean()
        print(f"[VAL] batch acc = {acc.item():.4f}")
        # Return as dict: Lightning automatically collects these
        return {"val_loss": loss.detach(),
                 "val_acc": acc.detach()}

    def validation_epoch_end(self, outputs):
        # outputs: list of dictionaries returned by validation_step
        val_loss = torch.stack([o["val_loss"] for o in outputs]).mean().item()
        val_acc  = torch.stack([o["val_acc"]  for o in outputs]).mean().item()
    
        print(f"\n[VAL] loss={val_loss:.4f}  acc={val_acc:.4f}")
        return {"val_loss": val_loss, "val_acc": val_acc}

    
    def test_dataloader(self):
        test_ds = LiDARPointCloudDataset(data_dir=TEST_DIR, num_points=NUM_POINTS, ignore_label=IGNORE_LABEL)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        return test_loader

    def test_step(self, batch, batch_idx):
        points, labels = batch
        device = next(self.parameters()).device
        points = points.to(device).contiguous()
        labels = labels.to(device)
        logits = self.forward(points)                # (B, C, N)
        preds  = logits.argmax(dim=1)                  # (B, N)
        # Returns NumPy arrays for easier aggregation in the end hook:
        return {
            "preds": preds.detach().cpu(),
            "labels": labels.detach().cpu()
        }

    def test_epoch_end(self, outputs):
        # 1. stack and flatten 
        all_preds  = torch.cat([x["preds"]  for x in outputs], dim=0).cpu().numpy().ravel()
        all_labels = torch.cat([x["labels"] for x in outputs], dim=0).cpu().numpy().ravel()

        # 2. Standard Metrics 
        acc = accuracy_score(all_labels, all_preds)
        f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
        report = classification_report(all_labels, all_preds, zero_division=0, digits=4)

        # 3. Calculate and save confusion matrix 
        cm = confusion_matrix(all_labels, all_preds, labels=range(self.num_classes))

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=[f"C{i}" for i in range(self.num_classes)],
            yticklabels=[f"C{i}" for i in range(self.num_classes)],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        # Adjust path as needed:
        save_path = os.path.join(os.getcwd(), "confusion_matrix.png")
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)            

        print("\n===== Test Results =====")
        print(f"Overall Accuracy: {acc:.4f}")
        print("F1 per class:", f1_per_class)
        print("\nClassification Report:\n", report)
        print(f"Confusion matrix saved to: {save_path}")


        # (optional) Lightning-Logger
        metrics = {"test_acc": acc}
        for i, f1c in enumerate(f1_per_class):
            metrics[f"test_f1_class_{i}"] = f1c
        self.logger.log_metrics(metrics, step=self.global_step)




        return metrics

# -------------------------------
# Prepare DataLoader and Trainer
# -------------------------------

if __name__ == "__main__":
    import torch
    import pointnet2_ops._ext as _ext

    print("CUDA available:", torch.cuda.is_available())
    print("raw op:", _ext.furthest_point_sampling)
    # Parameter
    DATA_DIR = '/cfs/earth/scratch/nogernic/Pointnet2_PyTorch/train'
    VAL_DIR =  '/cfs/earth/scratch/nogernic/Pointnet2_PyTorch/val'
    TEST_DIR = '/cfs/earth/scratch/nogernic/Pointnet2_PyTorch/test'
    BATCH_SIZE = 12
    NUM_POINTS = 4096
    IGNORE_LABEL = 7
    NUM_CLASSES = 13  
    LR = 1e-3
    SPG_WEIGHT = 1
    MAX_EPOCHS = 10
    MODEL_SAVE_PATH = '/cfs/earth/scratch/nogernic/Pointnet2_PyTorch/models/pointnet2_ssg_sem.pth'
    INPUT_CHANNELS = 3 


    # Create dataset
    train_ds = LiDARPointCloudDataset(data_dir=DATA_DIR, num_points=NUM_POINTS, ignore_label=IGNORE_LABEL)
    val_ds = LiDARPointCloudDataset(data_dir=VAL_DIR, num_points=NUM_POINTS, ignore_label=IGNORE_LABEL)


    # (Optional) WeightedRandomSampler for class balancing
    #sample_weights = [1.0] * len(train_ds)  
    #sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    ignore_label = 7                         # bleibt, wie im Dataset
    class_weights = np.array([16.24, 1.12, 41.40, 8.46, 1.77, 1.93, 131.60, 50.19])
    sample_weights = []
    for lbl_file in train_ds.label_files:
        lbl = np.load(lbl_file)

        # ignore-Label ausblenden
        cls_present = np.unique(lbl[lbl != ignore_label])

        # Gewicht des Samples = Mittelwert der Gewichte der vorkommenden Klassen
        # (oder max(), je nachdem was du möchtest)
        w = class_weights[cls_present].mean()
        sample_weights.append(float(w))

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)


    NUM_CLASSES = train_ds.num_classes 
    hparams = {
        'num_points': NUM_POINTS,
        'num_classes': NUM_CLASSES,
        'input_channels': INPUT_CHANNELS,
        'model.use_xyz': True,
    }
    # PointNet2 model (Erik Wijmans implementation)
    # Requires the installation and availability of the library
    from pointnet2.models.pointnet2_ssg_sem import PointNet2SemSegSSG
    model = PointNet2SemSegSSG(hparams)

    # Instantiate Lightning module
    lit_model = PointNet2Segmentation(model=model, num_classes=NUM_CLASSES,
                                      ignore_label=IGNORE_LABEL,
                                      spg_weight=SPG_WEIGHT, lr=LR)

    # Trainer
    trainer = pl.Trainer(gpus=1,  max_epochs=MAX_EPOCHS, log_every_n_steps=10)
    trainer.fit(lit_model, train_loader, val_loader)

    trainer.test(lit_model)

        # save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print('Modell gespeichert unter:', MODEL_SAVE_PATH)
