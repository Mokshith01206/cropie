import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.models import ResNet18_Weights

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TEST_DIR = PROJECT_DIR / "datasets" / "test"
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "cropie_tomato10_v2_resnet18.pth"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cropie: Model Evaluation on Test Dataset"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the model checkpoint file (.pth)",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help="Path to the test images directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size (default: 32)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    print("Evaluation Device:", device)
    if device.type == "cuda":
        print("GPU Device:", torch.cuda.get_device_name(0))

    if not args.model.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")

    if not args.test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {args.test_dir}")

    # Load checkpoint
    checkpoint = torch.load(
        args.model,
        map_location=device,
        weights_only=True,
    )
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    # Load test dataset
    weights = ResNet18_Weights.DEFAULT
    test_dataset = datasets.ImageFolder(
        args.test_dir,
        transform=weights.transforms(),
    )

    if test_dataset.classes != class_names:
        raise ValueError(
            f"Test classes ({test_dataset.classes}) do not match "
            f"checkpoint classes ({class_names})."
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"\nEvaluated Model:  {args.model.name}")
    print(f"Target Classes:   {num_classes}")
    print(f"Test Image Count: {len(test_dataset)}")

    # Initialize model
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            for actual, predicted in zip(
                labels.cpu().tolist(),
                predictions.cpu().tolist(),
            ):
                confusion[actual][predicted] += 1

    accuracy = correct / total if total > 0 else 0.0

    print("\n================== Evaluation Results ==================")
    print(f"Overall Accuracy: {accuracy:.2%} ({correct}/{total} correct)")
    print("========================================================\n")

    print("Per-Class Metrics:")
    print(f"{'Class Name':<28} {'Precision':<12} {'Recall':<12} {'Support':<8}")
    print("-" * 62)

    for i, name in enumerate(class_names):
        true_positive = confusion[i][i]
        actual_count = sum(confusion[i])
        predicted_count = sum(confusion[row][i] for row in range(num_classes))

        precision = (
            true_positive / predicted_count if predicted_count > 0 else 0.0
        )
        recall = (
            true_positive / actual_count if actual_count > 0 else 0.0
        )

        print(
            f"{name:<28} {precision:>9.2%}   {recall:>9.2%}   {actual_count:>7}"
        )

    print("\nConfusion Matrix (Rows: Actual, Columns: Predicted):")
    header = f"{'Actual':<22}" + "".join([f"{n[:8]:>9}" for n in class_names])
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        row_str = f"{name[:20]:<22}" + "".join(
            [f"{val:>9}" for val in confusion[i]]
        )
        print(row_str)


if __name__ == "__main__":
    main()
