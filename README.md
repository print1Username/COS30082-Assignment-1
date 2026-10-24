# CUB-200 Bird Species Classification

<div align="center">

[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](README.md)
[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-red?style=for-the-badge)](README.zh-CN.md)
</div>

Image classification for 200 bird species using the Caltech-UCSD Birds 200 (CUB-200) dataset. The project compares ResNet18, ResNet50 and EfficientNet-B0 with transfer learning. It includes local Python scripts, local Jupyter notebooks, and a separate Google Colab version. EfficientNet-B0 leads the recorded results below; the app automatically selects its model by validation accuracy.

## Links

- GitHub: [COS30082-Individual-Assignment](https://github.com/print1Username/COS30082-Individual-Assignment)
- Hugging Face profile: [print1Username](https://huggingface.co/print1Username)
- Dataset: [CUB-200 dataset](https://huggingface.co/datasets/print1Username/COS30082-Individual-Assignment.dataset)
- Model files: [trained model repository](https://huggingface.co/print1Username/COS30082-Individual-Assignment.model)
- Data and Google Colab version: [shared Google Drive folder](https://drive.google.com/drive/folders/1vLkbNAVSo3dt7mVHacQpqIANziIXmqB_?usp=sharing)

## Results

| Model | Best validation accuracy | Test Top-1 | Test average per-class |
|---|---:|---:|---:|
| ResNet18 | 66.77% | 65.12% | 64.45% |
| **EfficientNet-B0** | **74.64%** | **74.75%** | **74.21%** |
| ResNet50 | 70.91% | 70.02% | 69.09% |

Results are taken directly from the local Python evaluation files `models/results/<model>_test_results.json`. All three models use 1,204 test images and 200 evaluated classes. The training histories in `models/history/<model>_history.json` record 35 epochs for ResNet18 and ResNet50, and 33 for EfficientNet-B0. The best validation accuracies in the result files match the maxima in the training histories.

EfficientNet-B0 has the highest best validation accuracy (74.64%), test Top-1 accuracy (74.75%) and test average per-class accuracy (74.21%). These JSON files are the source for the table above and [model_result.md](model_result.md). The current `src/train.py` and `src/evaluate.py` use `models/checkpoints/`, `models/history/`, and `models/results/` for model weights, training histories, and evaluation results, respectively. New runs can overwrite existing files for the same model.

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
conda create -n cos30082a1 python=3.10 -y
conda activate cos30082a1
```

### 3. Install dependencies

Upgrade pip, then choose **one** of the following PyTorch installation options:

```bash
python -m pip install --upgrade pip
```

**Option A: CPU (Windows/Linux, no NVIDIA GPU required)**

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

**Option B: CUDA (Windows/Linux with a compatible NVIDIA GPU and driver)**

The following example uses the official CUDA 12.6 wheel index:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Use the [official PyTorch selector](https://pytorch.org/get-started/locally/) to choose a CUDA build compatible with your system. On macOS, use the selector's macOS installation command instead. Run the selected option in a fresh environment. `requirements.txt` includes a CPU package index, so install your chosen PyTorch build first; the subsequent requirements command keeps the already installed packages that satisfy its unpinned requirements.

Check the installation:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available: True` means PyTorch can use the NVIDIA GPU; `False` is expected for the CPU option.

## Download the data

The downloader retrieves `Train.zip`, `Test.zip`, `train.txt` and `test.txt` from the [Hugging Face dataset repository](https://huggingface.co/datasets/print1Username/COS30082-Assignment-1-dataset), extracts the images and checks the expected structure:

```bash
python download_data.py
```

> The script cleans the existing `data/` contents before downloading. Back up any local data you want to keep.

## Check the model setup before training

After installing dependencies and downloading the data, run this check from the repository root before starting training and evaluation:

```bash
python src/models.py
```

This creates an ImageNet-pretrained ResNet18, checks backbone freezing and unfreezing, and prints parameter counts. Wait for `Model test completed successfully.` before continuing. The first run requires internet access if the ResNet18 pretrained weights are not already cached; training another architecture may download its own weights.

`train.py` and `evaluate.py` import `models.py` automatically, so this standalone check is not a strict code dependency and does not create a trained checkpoint. It helps identify model/dependency or pretrained-weight download problems before training. Use the order: model check → training → evaluation; evaluation requires the matching trained checkpoint.

## Train a model

Run one model at a time:

```bash
python src/train.py --model resnet18
python src/train.py --model resnet50
python src/train.py --model efficientnet_b0
```

Training uses a stratified 80/20 train-validation split, ImageNet-pretrained weights, data augmentation, two-stage fine-tuning, AdamW, label smoothing, learning-rate scheduling, early stopping and best-checkpoint saving. Checkpoints are written to `models/checkpoints/`; histories are written to `models/history/`.

The local Python and local notebook training implementations use the same overfitting controls:

| Setting | Revised behavior |
|---|---|
| Architectures | Keep ResNet18, ResNet50 and EfficientNet-B0 unchanged |
| Data split | Stratified 80/20 split of the official training set; seed `42` |
| Classifier warm-up | 5 epochs at learning rate `1e-3`; frozen backbone remains in evaluation mode so BatchNorm statistics do not change |
| Fine-tuning | Backbone learning rate `1e-5`; classifier learning rate `1e-4` |
| Regularization | AdamW weight decay `1e-3`, label smoothing `0.1`, and training-only image augmentation |
| Learning-rate scheduling | Reduce the learning rate when validation loss stops improving |
| Early stopping | Stop after 5 consecutive epochs without a lower validation loss; reset patience when fine-tuning starts |
| Checkpoint selection | Save the checkpoint with the highest validation accuracy |
| Training limit | Batch size `32`; maximum 35 epochs |

Loss controls scheduling and stopping; accuracy selects the saved model. The official test set is used only for final evaluation, not for tuning or app model selection.

## Local notebook version (standalone)

The `notebooks/` folder contains a complete notebook workflow. The notebooks do not import the training implementation from `src/*.py`; each notebook contains the logic needed for its own stage:

1. `notebooks/01_data_preparation.ipynb`: combined data exploration and preparation. It checks paths, annotations, classes, missing/corrupted images, image sizes, class distributions, PyTorch `Dataset`/`DataLoader`, and augmentation. It also includes summary charts.
2. `notebooks/02_training.ipynb`: full training pipeline (train/validation/fine-tune/checkpoint/history) with loss, accuracy, learning-rate, and training-status charts.
3. `notebooks/03_evaluation.ipynb`: test evaluation with Top-1 accuracy and average accuracy per class, plus per-class performance charts.

Run them in numeric order. The original separate data-exploration notebook was merged into `01_data_preparation.ipynb` to avoid duplicated functionality.

Notebook training and evaluation histories remain in memory and are displayed as charts and metrics. The notebooks save model checkpoints but do not write JSON files under `models/`; those JSON files belong to the local `src` workflow. The training notebook uses `num_workers=0` for Windows/Jupyter compatibility.

After installing the project dependencies and downloading the data, install the notebook tools and start Jupyter from the `notebooks/` directory:

```bash
pip install notebook matplotlib numpy
cd notebooks
jupyter notebook
```

The notebooks resolve the project root as the parent of their working directory, so keep the kernel working directory at `notebooks/`. In `02_training.ipynb`, choose `MODEL` and uncomment `hist=train_model(MODEL)` to start training. In `03_evaluation.ipynb`, select the matching model name before evaluation. Notebook checkpoints use the same filenames as local Python checkpoints and can replace an existing checkpoint for that model.

## Google Colab version

The separate Google Colab version and the dataset are available in the [shared Google Drive folder](https://drive.google.com/drive/folders/1vLkbNAVSo3dt7mVHacQpqIANziIXmqB_?usp=sharing). Use that folder to access the cloud notebooks and their data; the local notebook files are in this repository's `notebooks/` directory. Follow the setup and data-path instructions in the Colab notebooks.

The code changes documented here apply to the local Python scripts and local notebooks in this repository. The separate Google Drive/Colab copies have not been edited or verified as part of these fixes.

## Evaluate on the test set

Evaluate a trained model after its checkpoint has been created:

```bash
python src/evaluate.py --model resnet50
python src/evaluate.py --model resnet18
python src/evaluate.py --model efficientnet_b0
```

Results are saved under `models/results/`. The required metrics are Top-1 accuracy and average accuracy per class.

## Model output folders

Normal local training and evaluation use only these folders under `models/`:

| Folder | Contents | Written by |
|---|---|---|
| `models/checkpoints/` | `<model>_best.pth`: weights, architecture, validation accuracy, optimizer/scheduler states, and training configuration for newly trained checkpoints | Local Python training or the local training notebook |
| `models/history/` | `<model>_history.json`: epoch loss, accuracy, and learning-rate history, updated after each completed epoch | `src/train.py` only |
| `models/results/` | `<model>_test_results.json`: final test Top-1 and average per-class accuracy | `src/evaluate.py` only |

The code does not automatically create `baseline`, `interrupted_trial`, or `retraining` folders. Running training or evaluation again for the same model can overwrite its existing files; automatic backup folders are not created.

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
├── notebooks/            # Local standalone Jupyter workflow
│   ├── 01_data_preparation.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── data/                  # Local dataset files
└── models/
    ├── checkpoints/       # Model checkpoints
    ├── history/           # Local src training-history JSON
    └── results/           # Local src test-result JSON
```

## Assignment metrics

- **Top-1 Accuracy:** correctly classified test images divided by all test images.
- **Average Accuracy per Class:** calculate accuracy for each bird species, then average the class accuracies.

## Academic use

This repository was created for COS30082 Assignment 1. The CUB-200 dataset remains subject to its original terms of use.
