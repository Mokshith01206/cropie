# Cropie: Tomato Leaf Disease Classification

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Backbone-ResNet--18-success.svg)](https://pytorch.org/vision/stable/models.html)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.43%25-brightgreen.svg)](#benchmark-results)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

Cropie is a computer vision framework designed for automated, high-precision identification and classification of tomato leaf diseases from digital imagery. Powered by deep convolutional neural networks and transfer learning on ResNet-18, Cropie provides automated disease diagnosis across 10 distinct health and pathological conditions affecting tomato crops.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Supported Classes](#supported-classes)
- [Model Architecture & Training Methodology](#model-architecture--training-methodology)
- [Benchmark Results](#benchmark-results)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference & Usage](#inference--usage)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Foliar diseases represent a major vector of yield loss in global tomato cultivation. Early and accurate detection allows targeted agricultural interventions, reducing pesticide overuse and preserving crop productivity. 

Cropie addresses this challenge by providing an end-to-end deep learning pipeline encompassing:
1. Automated stratified data partitioning and preprocessing.
2. Transfer learning with domain-specific fine-tuning on deep residual representations.
3. Class-weighted optimization mitigating dataset imbalance.
4. Comprehensive test-set evaluation including confusion matrices, precision, and recall.
5. Command-line inference supporting arbitrary leaf images.

---

## Key Features

- **High Accuracy**: Attains 98.43% test accuracy across 2,734 unseen test images and 98.90% peak validation accuracy.
- **Deep Residual Architecture**: Leverages ResNet-18 pretrained on ImageNet, fine-tuning high-level semantic feature representations in residual block 4.
- **Class-Imbalance Compensation**: Employs dynamic inverse-frequency class weighting in the cross-entropy loss function to ensure balanced performance across rare and frequent classes.
- **Robust Data Augmentation**: Utilizes random resized cropping, horizontal reflections, rotations, and channel-wise normalization during training to prevent overfitting.
- **Modular Codebase**: Strict separation of concerns across data preparation, training, evaluation, and inference modules with CLI parameterization via `argparse`.
- **Reproducible Pipeline**: Deterministic random seeds across NumPy, Python standard library, and PyTorch CUDA kernels.

---

## Supported Classes

The system categorizes input leaf imagery into 10 target classes:

| Class Identifier | Common Name | Pathogen / Condition |
|---|---|---|
| `healthy` | Healthy Leaf | Normal physiological condition |
| `bacterial_spot` | Bacterial Spot | *Xanthomonas spp.* |
| `early_blight` | Early Blight | *Alternaria solani* |
| `late_blight` | Late Blight | *Phytophthora infestans* |
| `leaf_mold` | Leaf Mold | *Passalora fulva* |
| `septoria_leaf_spot` | Septoria Leaf Spot | *Septoria lycopersici* |
| `spider_mites` | Two-Spotted Spider Mites | *Tetranychus urticae* |
| `target_spot` | Target Spot | *Corynespora casiicola* |
| `tomato_mosaic_virus` | Tomato Mosaic Virus | *Tomato mosaic virus (ToMV)* |
| `yellow_leaf_curl_virus` | Yellow Leaf Curl Virus | *Tomato yellow leaf curl virus (TYLCV)* |

---

## Model Architecture & Training Methodology

### Architecture Breakdown

- **Backbone**: ResNet-18 (Residual Network with 18 weighted layers).
- **Frozen Stages**: Initial convolution, batch normalization, max pooling, and residual blocks 1 through 3 are frozen to retain foundational visual primitives (edges, textures, low-level patterns).
- **Fine-Tuning Stage**: Residual block 4 (`layer4`) is unfrozen to adapt high-level visual representations to tomato leaf pathologies.
- **Classification Head**: Fully connected layer replacing original 1,000-class head with a linear mapping `Linear(in_features=512, out_features=10)`.

### Training Configuration

- **Input Dimensions**: 224 x 224 x 3 (RGB)
- **Batch Size**: 32
- **Epochs**: 20
- **Loss Function**: Weighted Cross-Entropy Loss:
  $$\mathcal{L} = - \sum_{c=1}^{C} w_c \, y_c \log(\hat{y}_c)$$
  where $w_c = \frac{N}{C \cdot N_c}$ compensates for class sample disparity.
- **Optimizer**: Adam with parameter-group specific learning rates:
  - Backbone `layer4`: $\eta = 10^{-4}$
  - Classifier `fc`: $\eta = 10^{-3}$
- **Learning Rate Scheduler**: `ReduceLROnPlateau` (factor = 0.5, patience = 2, monitoring validation loss).

---

## Benchmark Results

The model was evaluated against an independent held-out test split of 2,734 images.

### Quantitative Summary

- **Total Test Samples**: 2,734 images
- **Overall Accuracy**: **98.43%** (2,691 / 2,734 correct predictions)
- **Best Validation Accuracy**: **98.90%**

### Per-Class Performance Breakdown

| Class | Precision | Recall | Test Samples (Support) |
|---|---|---|---|
| `bacterial_spot` | 98.15% | 99.69% | 320 |
| `early_blight` | 98.54% | 90.00% | 150 |
| `healthy` | 97.56% | 100.00% | 240 |
| `late_blight` | 97.25% | 98.61% | 287 |
| `leaf_mold` | 97.26% | 98.61% | 144 |
| `septoria_leaf_spot` | 97.77% | 98.50% | 267 |
| `spider_mites` | 98.00% | 97.22% | 252 |
| `target_spot` | 97.18% | 97.64% | 212 |
| `tomato_mosaic_virus` | 100.00% | 100.00% | 57 |
| `yellow_leaf_curl_virus` | 100.00% | 99.38% | 805 |

---

## Project Structure

```text
cropie/
├── .gitignore               # Ignored data directories, caches, and environments
├── environment.yml          # Conda environment definition
├── requirements.txt         # Pip dependency requirements
├── LICENSE                  # MIT License
├── README.md                # Project documentation
├── models/                  # Serialized PyTorch model checkpoints (.pth)
│   ├── cropie_tomato10_v2_resnet18.pth
│   ├── cropie_tomato10_resnet18.pth
│   └── cropie_resnet18.pth
├── sample_images/           # Verification samples for quickstart inference
│   ├── sample_healthy.jpg
│   ├── sample_early_blight.jpg
│   └── sample_yellow_leaf_curl_virus.jpg
└── src/                     # Source code modules
    ├── data_prep.py         # Automated dataset splitting and organization
    ├── train.py             # Transfer learning training script with class weighting
    ├── evaluate.py          # Detailed test evaluation and metric reporting
    └── main.py              # CLI inference utility for single-image diagnosis
```

---

## Installation & Setup

### Prerequisites

- Python 3.10 or 3.11
- CUDA-compatible GPU (recommended for training, optional for inference)

### 1. Clone the Repository

```bash
git clone https://github.com/Mokshith01206/cropie.git
cd cropie
```

### 2. Set Up Virtual Environment

#### Using Conda:

```bash
conda env create -f environment.yml
conda activate cropie
```

#### Using venv and pip:

```bash
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

---

## Dataset Preparation

Cropie is trained on the tomato pathology subsets from the open-source PlantVillage dataset.

1. Download the raw PlantVillage dataset and place the color images under:
   ```text
   plantvillage-source/raw/color/
   ```
2. Execute the automated data preparation script:
   ```bash
   python src/data_prep.py
   ```
   This stratifies the dataset into:
   - `datasets/train/` (70%)
   - `datasets/validation/` (15%)
   - `datasets/test/` (15%)

Optional parameters:
```bash
python src/data_prep.py --source-dir <path> --output-dir <path> --train-ratio 0.70 --val-ratio 0.15
```

---

## Training

To train the ResNet-18 model with transfer learning and class weighting:

```bash
python src/train.py
```

### Customizable Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--data-dir` | Path | `datasets` | Path to dataset directory containing train/val splits |
| `--model-dir` | Path | `models` | Directory to save checkpoint files |
| `--epochs` | int | `20` | Maximum number of training epochs |
| `--batch-size` | int | `32` | Training and validation batch size |
| `--lr-backbone` | float | `0.0001` | Learning rate for residual layer 4 |
| `--lr-head` | float | `0.001` | Learning rate for classifier head |
| `--seed` | int | `42` | Random seed for reproducibility |

---

## Evaluation

To evaluate a saved model checkpoint against the test dataset and generate accuracy, per-class precision, recall, and confusion matrix:

```bash
python src/evaluate.py --model models/cropie_tomato10_v2_resnet18.pth
```

---

## Inference & Usage

You can run disease inference on any single tomato leaf image using `src/main.py`.

### Quick Test with Bundled Samples

```bash
# Test on healthy sample
python src/main.py --image sample_images/sample_healthy.jpg

# Test on early blight sample
python src/main.py --image sample_images/sample_early_blight.jpg

# Test on yellow leaf curl virus sample
python src/main.py --image sample_images/sample_yellow_leaf_curl_virus.jpg
```

### Example Output

```text
Execution Device: cuda
GPU Device: NVIDIA GeForce RTX 4050 Laptop GPU

---------------- Cropie Prediction ----------------
Image:            sample_healthy.jpg
Predicted Class:  healthy
Confidence Score: 100.00%
---------------------------------------------------

Top 5 Class Probabilities:
  healthy                    100.00%
  septoria_leaf_spot           0.00%
  target_spot                  0.00%
  yellow_leaf_curl_virus       0.00%
  leaf_mold                    0.00%
```

### Custom Inference

```bash
python src/main.py --image path/to/leaf_image.jpg --model models/cropie_tomato10_v2_resnet18.pth --topk 5
```

---

## Contributing

Contributions to improve accuracy, add support for mobile deployment (ONNX / TorchScript / TFLite), or expand to additional crops are welcome:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/improvement`).
3. Commit your changes (`git commit -m "Add feature description"`).
4. Push to the branch (`git push origin feature/improvement`).
5. Open a Pull Request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
