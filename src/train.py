import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR / "datasets"
DEFAULT_MODEL_DIR = PROJECT_DIR / "models"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cropie: Train ResNet-18 Tomato Leaf Disease Classifier"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to datasets directory containing train/ and validation/",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=0.0001,
        help="Learning rate for backbone layer4 (default: 0.0001)",
    )
    parser.add_argument(
        "--lr-head",
        type=float,
        default=0.001,
        help="Learning rate for classification head (default: 0.001)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def train(args):
    device = get_device()
    print("Training Device:", device)
    if device.type == "cuda":
        print("GPU Device:", torch.cuda.get_device_name(0))

    args.model_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Pretrained weights and normalization parameters
    weights = ResNet18_Weights.DEFAULT
    preprocessing = weights.transforms()
    mean = preprocessing.mean
    std = preprocessing.std

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    validation_transform = preprocessing

    # Load datasets
    train_dataset = datasets.ImageFolder(
        args.data_dir / "train",
        transform=train_transform,
    )
    validation_dataset = datasets.ImageFolder(
        args.data_dir / "validation",
        transform=validation_transform,
    )

    if train_dataset.classes != validation_dataset.classes:
        raise ValueError("Training and validation classes do not match.")

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"\nNumber of Classes: {num_classes}")
    print("Class Labels:", class_names)
    print(f"Training Samples:   {len(train_dataset)}")
    print(f"Validation Samples: {len(validation_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Model definition
    model = models.resnet18(weights=weights)

    # Freeze earlier layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4 for domain-specific fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace classifier head
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Inverse frequency class weighting for class imbalance
    class_counts = torch.bincount(
        torch.tensor(train_dataset.targets),
        minlength=num_classes,
    ).float()
    class_weights = len(train_dataset) / (num_classes * class_counts)
    class_weights = class_weights.to(device)

    print("\nClass Balancing Weights:")
    for name, count, weight in zip(
        class_names,
        class_counts.tolist(),
        class_weights.tolist(),
    ):
        print(f"  {name:<26} count: {int(count):<6} weight: {weight:.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        [
            {"params": model.layer4.parameters(), "lr": args.lr_backbone},
            {"params": model.fc.parameters(), "lr": args.lr_head},
        ]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_validation_accuracy = 0.0
    best_validation_loss = float("inf")
    checkpoint_path = args.model_dir / "cropie_tomato10_v2_resnet18.pth"

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        train_loss_total = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * images.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()

        train_loss = train_loss_total / len(train_dataset)
        train_accuracy = train_correct / len(train_dataset)

        model.eval()
        val_loss_total = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in validation_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_total += loss.item() * images.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()

        val_loss = val_loss_total / len(validation_dataset)
        val_accuracy = val_correct / len(validation_dataset)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2%} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2%}"
        )

        if val_accuracy > best_validation_accuracy:
            best_validation_accuracy = val_accuracy
            best_validation_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "validation_accuracy": best_validation_accuracy,
                    "validation_loss": best_validation_loss,
                },
                checkpoint_path,
            )
            print(f"  Checkpoint saved with validation accuracy: {val_accuracy:.2%}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nTraining complete.")
    print(f"Best Validation Accuracy: {best_validation_accuracy:.2%}")
    print(f"Model saved to: {checkpoint_path}")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
