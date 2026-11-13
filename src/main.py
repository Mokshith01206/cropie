import argparse
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.models import ResNet18_Weights

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "cropie_tomato10_v2_resnet18.pth"
DEFAULT_IMAGE_PATH = PROJECT_DIR / "sample_images" / "sample_healthy.jpg"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_path, device):
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_names


def predict(image_path, model, class_names, device):
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    preprocessing = ResNet18_Weights.DEFAULT.transforms()

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = preprocessing(image).unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_index = int(probabilities.argmax().item())
    predicted_class = class_names[predicted_index]
    confidence = probabilities[predicted_index].item() * 100.0

    scores = sorted(
        zip(class_names, probabilities.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )

    return predicted_class, confidence, scores


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cropie: Tomato Leaf Disease Inference"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help="Path to the input leaf image (default: sample_images/sample_healthy.jpg)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of top prediction scores to display (default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    print("Execution Device:", device)
    if device.type == "cuda":
        print("GPU Device:", torch.cuda.get_device_name(0))

    model, class_names = load_model(args.model, device)
    predicted_class, confidence, scores = predict(
        args.image, model, class_names, device
    )

    print("\n---------------- Cropie Prediction ----------------")
    print(f"Image:            {args.image.name}")
    print(f"Predicted Class:  {predicted_class}")
    print(f"Confidence Score: {confidence:.2f}%")
    print("---------------------------------------------------")

    print(f"\nTop {min(args.topk, len(scores))} Class Probabilities:")
    for class_name, score in scores[: args.topk]:
        print(f"  {class_name:<26} {score * 100.0:>6.2f}%")


if __name__ == "__main__":
    main()
