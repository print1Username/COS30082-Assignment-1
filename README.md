# CUB-200 Bird Species Classification

<div align="center">

[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](README.md)
[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-red?style=for-the-badge)](README.zh-CN.md)
</div>

Image classification for 200 bird species using the Caltech-UCSD Birds 200 (CUB-200) dataset. The project compares ResNet18, ResNet50 and EfficientNet-B0 with transfer learning. **ResNet50 is the recommended final model.**

## Links

- GitHub: [COS30082-Assignment-1](https://github.com/print1Username/COS30082-Assignment-1)
- Hugging Face profile: [print1Username](https://huggingface.co/print1Username)
- Dataset: [CUB-200 dataset](https://huggingface.co/datasets/print1Username/COS30082-Assignment-1-dataset)
- Model files: [trained model repository](https://huggingface.co/print1Username/COS30082-Assignment-1-model)

## Results

| Model | Best validation accuracy | Test Top-1 | Test average per-class |
|---|---:|---:|---:|
| ResNet18 | 70.19% | 69.60% | 69.18% |
| EfficientNet-B0 | 72.57% | 70.93% | 69.87% |
| **ResNet50** | **78.99%** | **76.91%** | **76.16%** |

All results use 1,204 test images and 200 evaluated classes. See [model_result.md](model_result.md) for details.

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/print1Username/COS30082-Assignment-1.git
cd COS30082-Assignment-1
```

### 2. Create an environment

Using `venv`:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

Using Conda:

```bash
conda create -n cub200 python=3.10 -y
conda activate cub200
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The requirements file installs the CPU PyTorch build. For a CUDA build, install the matching PyTorch and torchvision versions from the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install the remaining packages from `requirements.txt`.

## Download the data

The downloader retrieves `Train.zip`, `Test.zip`, `train.txt` and `test.txt` from the [Hugging Face dataset repository](https://huggingface.co/datasets/print1Username/COS30082-Assignment-1-dataset), extracts the images and checks the expected structure:

```bash
python download_data.py
```

> The script cleans the existing `data/` contents before downloading. Back up any local data you want to keep.

## Train a model

Run one model at a time:

```bash
python src/train.py --model resnet18
python src/train.py --model resnet50
python src/train.py --model efficientnet_b0
```

Training uses a stratified 80/20 train-validation split, ImageNet-pretrained weights, data augmentation, two-stage fine-tuning, AdamW, label smoothing, learning-rate scheduling, early stopping and best-checkpoint saving. Checkpoints are written to `models/checkpoints/`; histories are written to `models/history/`.

During the five warm-up epochs, the frozen backbone stays in evaluation mode so BatchNorm statistics do not change. Fine-tuning uses a backbone learning rate of `1e-5`, classifier rate of `1e-4`, and weight decay of `1e-3`. Scheduling and early stopping monitor validation loss (patience: 5); checkpoints retain the highest validation accuracy for model selection. The official test set is used only for final evaluation.

## Notebook version (standalone)

The `notebooks/` folder contains a complete notebook workflow. The notebooks do not import the training implementation from `src/*.py`; each notebook contains the logic needed for its own stage:

1. `notebooks/01_data_preparation.ipynb`: combined data exploration and preparation. It checks paths, annotations, classes, missing/corrupted images, image sizes, class distributions, PyTorch `Dataset`/`DataLoader`, and augmentation. It also includes summary charts.
2. `notebooks/02_training.ipynb`: full training pipeline (train/validation/fine-tune/checkpoint/history) with loss, accuracy, learning-rate, and training-status charts.
3. `notebooks/03_evaluation.ipynb`: test evaluation with Top-1 accuracy and average accuracy per class, plus per-class performance charts.

Run them in numeric order. The original separate data-exploration notebook was merged into `01_data_preparation.ipynb` to avoid duplicated functionality.

Notebook training and evaluation histories remain in memory and are displayed as charts and metrics. The notebooks save model checkpoints but do not write JSON files under `models/`; those JSON files belong to the local `src` workflow. The training notebook uses `num_workers=0` for Windows/Jupyter compatibility.

## Evaluate on the test set

Evaluate a trained model after its checkpoint has been created:

```bash
python src/evaluate.py --model resnet50
```

Results are saved under `models/results/`. The required metrics are Top-1 accuracy and average accuracy per class.

## Run the local web application

Ensure `gradio` and the other dependencies are installed, then run:

```bash
python app.py
```

Open the local URL shown in the terminal, usually `http://localhost:7860`. The app accepts an image upload and returns the top-5 bird-species predictions. It uses a local checkpoint when available; otherwise it downloads a checkpoint and `class_mapping.json` from the [Hugging Face model repository](https://huggingface.co/print1Username/COS30082-Assignment-1-model).

At startup, `app.py` compares the `validation_accuracy` stored inside available local checkpoints and automatically loads the highest-scoring model. If no local checkpoints exist, it downloads and compares the repository's root-level checkpoint files instead. Each checkpoint must include `model_name` and `validation_accuracy`; no history or test-result JSON is used for selection. The interface displays the actual architecture and validation accuracy. Restart the app after retraining to refresh its selection.

## Project structure

```text
├── app.py                 # Gradio web application
├── download_data.py       # Download and extract CUB-200
├── requirements.txt       # Python dependencies
├── class_mapping.json     # Class index to bird-species mapping
├── src/
│   ├── dataset.py         # Dataset loading and preprocessing
│   ├── models.py          # Model definitions
│   ├── train.py           # Training and checkpoint saving
│   └── evaluate.py        # Test-set evaluation
├── data/                  # Local dataset files
└── models/                # Local checkpoints, histories and results
```

## Assignment metrics

- **Top-1 Accuracy:** correctly classified test images divided by all test images.
- **Average Accuracy per Class:** calculate accuracy for each bird species, then average the class accuracies.

## Academic use

This repository was created for COS30082 Assignment 1. The CUB-200 dataset remains subject to its original terms of use.
