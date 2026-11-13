import argparse
from pathlib import Path
import random
import shutil

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = PROJECT_DIR / "plantvillage-source" / "raw" / "color"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "datasets"

CLASSES = {
    "Tomato___healthy": "healthy",
    "Tomato___Bacterial_spot": "bacterial_spot",
    "Tomato___Early_blight": "early_blight",
    "Tomato___Late_blight": "late_blight",
    "Tomato___Leaf_Mold": "leaf_mold",
    "Tomato___Septoria_leaf_spot": "septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "spider_mites",
    "Tomato___Target_Spot": "target_spot",
    "Tomato___Tomato_mosaic_virus": "tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "yellow_leaf_curl_virus",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cropie: Dataset Preparation and Train/Val/Test Split"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Path to raw PlantVillage dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for processed splits",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proportion of training set (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of validation set (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def prepare_dataset(source_dir, output_dir, train_ratio, val_ratio, seed):
    random.seed(seed)

    for source_class in CLASSES:
        source_folder = source_dir / source_class
        if not source_folder.is_dir():
            raise FileNotFoundError(
                f"Source folder not found: {source_folder}"
            )

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"The output directory '{output_dir}' is not empty. "
            "Please back up or clear it before rebuilding the dataset."
        )

    print("Beginning dataset preparation...")
    print(f"Source Directory: {source_dir}")
    print(f"Output Directory: {output_dir}")
    print(
        f"Splits: Train {train_ratio:.0%}, Validation {val_ratio:.0%}, "
        f"Test {1.0 - train_ratio - val_ratio:.0%}"
    )

    for source_class, target_class in CLASSES.items():
        source_folder = source_dir / source_class
        images = [
            file
            for file in source_folder.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not images:
            raise ValueError(f"No images found in {source_folder}")

        random.shuffle(images)
        total = len(images)
        train_end = int(total * train_ratio)
        validation_end = train_end + int(total * val_ratio)

        splits = {
            "train": images[:train_end],
            "validation": images[train_end:validation_end],
            "test": images[validation_end:],
        }

        print(f"\nClass: {target_class} (Total: {total} images)")
        for split_name, split_images in splits.items():
            dest = output_dir / split_name / target_class
            dest.mkdir(parents=True, exist_ok=True)

            for img in split_images:
                shutil.copy2(img, dest / img.name)

            print(f"  {split_name:<12} {len(split_images)} images")

    print("\nDataset preparation completed successfully.")


def main():
    args = parse_args()
    prepare_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
