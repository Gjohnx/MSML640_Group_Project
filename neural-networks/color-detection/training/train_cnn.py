import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class DatasetStats:
    kept: int
    skipped_missing_image: int
    skipped_invalid_label: int


class CubeFaceDataset(Dataset):
    
    def __init__(self, image_dir: Path, labels_csv: Path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples: List[Tuple[Path, List[int]]] = []
        skipped_missing_image = 0
        skipped_invalid_label = 0

        with labels_csv.open() as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 10:
                    skipped_invalid_label += 1
                    continue

                image_id_raw, tile_values = row[0], row[1:]

                if not image_id_raw.isdigit():
                    skipped_invalid_label += 1
                    continue

                try:
                    labels = [int(v) for v in tile_values]
                except ValueError:
                    skipped_invalid_label += 1
                    continue

                if len(labels) != 9 or any(v < 0 or v > 5 for v in labels):
                    skipped_invalid_label += 1
                    continue

                image_path = image_dir / f"captured_{int(image_id_raw):04d}.png"
                if not image_path.exists():
                    skipped_missing_image += 1
                    continue

                self.samples.append((image_path, labels))

        self.stats = DatasetStats(
            kept=len(self.samples),
            skipped_missing_image=skipped_missing_image,
            skipped_invalid_label=skipped_invalid_label,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, labels = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return image, labels_tensor


class CubeCNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.02),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.03),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(512, 9 * 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        return logits.view(-1, 9, 6)


def build_transforms() -> transforms.Compose:
    # Images are already 100x100, but resizing guards against mismatches.
    return transforms.Compose(
        [
            transforms.Resize((100, 100)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=7, translate=(0.05, 0.05)),
            transforms.RandomPerspective(distortion_scale=0.04, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def split_train_val(num_samples: int, val_ratio: float, seed: int) -> Tuple[Sequence[int], Sequence[int]]:
    rng = random.Random(seed)
    indices = list(range(num_samples))
    rng.shuffle(indices)

    val_size = max(1, int(num_samples * val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_tiles = 0
    correct_tiles = 0
    correct_faces = 0
    total_faces = 0

    torch.set_grad_enabled(is_train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits.view(-1, 6), labels.view(-1))

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=-1)
        correct_tiles += (preds == labels).sum().item()
        total_tiles += labels.numel()
        correct_faces += (preds == labels).all(dim=1).sum().item()
        total_faces += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / max(total_faces, 1)
    tile_acc = correct_tiles / max(total_tiles, 1)
    face_acc = correct_faces / max(total_faces, 1)
    return {"loss": avg_loss, "tile_acc": tile_acc, "face_acc": face_acc}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    output_dir: Path,
    label_smoothing: float,
    patience: int,
):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.05
    )

    best_val_loss = float("inf")
    best_path = output_dir / "cube_cnn.pt"
    epochs_since_improve = 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)

        print(
            f"Validation Epoch {epoch:02d}/{epochs} "
            f"Train loss {train_metrics['loss']:.4f} | "
            f"Train tile acc {train_metrics['tile_acc']:.3f} | "
            f"Val loss {val_metrics['loss']:.4f} | "
            f"Val tile acc {val_metrics['tile_acc']:.3f} | "
            f"Val face acc {val_metrics['face_acc']:.3f}"
        )

        scheduler.step()

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epochs_since_improve >= patience:
            print(
                f"Early stopping: no val loss improvement for {patience} epochs "
                f"(stopped after epoch {epoch})."
            )
            break

    return best_path, best_val_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN to classify Rubik cube face colors.")
    parser.add_argument("--data-dir",type=Path,default=Path(__file__).resolve().parent / "synthetic-data" / "resized")
    parser.add_argument("--labels",type=Path,default=Path(__file__).resolve().parent / "synthetic-data" / "labels" / "labels_numeric.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-ratio",type=float,default=0.15)
    parser.add_argument("--label-smoothing",type=float,default=0.01)
    parser.add_argument("--patience",type=int,default=40)
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent / "models")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = build_transforms()
    dataset = CubeFaceDataset(args.data_dir, args.labels, transform=transform)
    print(
        f"Loaded dataset: {dataset.stats.kept} images | "
        f"skipped missing images: {dataset.stats.skipped_missing_image} | "
        f"skipped invalid labels: {dataset.stats.skipped_invalid_label}"
    )

    if len(dataset) < 5:
        raise RuntimeError("Not enough samples to perform a split. Add more data first.")

    train_idx, val_idx = split_train_val(len(dataset), args.val_ratio, args.seed)
    print(f"\n--- Single split | train={len(train_idx)} | val={len(val_idx)} ---")

    model = CubeCNN().to(device)
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    best_path, best_loss = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
    )
    print(f"Best val loss {best_loss:.4f} | saved to {best_path}")

    print("Training complete")


if __name__ == "__main__":
    main()
