# CUB-200 鸟类物种分类

[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](README.md)
[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-red?style=for-the-badge)](README.zh-CN.md)

使用 Caltech-UCSD Birds 200（CUB-200）数据集，对 200 种鸟类进行图像分类。项目比较 ResNet18、ResNet50 和 EfficientNet-B0 的迁移学习效果，提供本地 Python、本地 Jupyter notebook 和独立的 Google Colab 版本。EfficientNet-B0 在下方已有结果中领先；网页应用会根据验证准确率自动选择模型。

## 项目链接

- GitHub：[COS30082-Assignment-1](https://github.com/print1Username/COS30082-Assignment-1)
- Hugging Face 主页：[print1Username](https://huggingface.co/print1Username)
- 数据集：[CUB-200 dataset](https://huggingface.co/datasets/print1Username/COS30082-Individual-Assignment.dataset)
- 模型文件：[训练模型仓库](https://huggingface.co/print1Username/COS30082-Individual-Assignment.model)
- 数据和 Google Colab 版本：[Google Drive 共享文件夹](https://drive.google.com/drive/folders/1vLkbNAVSo3dt7mVHacQpqIANziIXmqB_?usp=sharing)

## 训练结果

| 模型 | 最佳验证准确率 | 测试 Top-1 | 测试每类平均准确率 |
|---|---:|---:|---:|
| ResNet18 | 66.77% | 65.12% | 64.45% |
| **EfficientNet-B0** | **74.64%** | **74.75%** | **74.21%** |
| ResNet50 | 70.91% | 70.02% | 69.09% |

结果直接取自本地 Python 评估文件 `models/results/<model>_test_results.json`。三个模型均使用 1,204 张测试图像和 200 个评估类别。`models/history/<model>_history.json` 中，ResNet18 和 ResNet50 各记录了 35 轮训练，EfficientNet-B0 记录了 33 轮。结果文件中的最佳验证准确率与训练历史中的最大值一致。

EfficientNet-B0 的最佳验证准确率（74.64%）、测试 Top-1 准确率（74.75%）和测试每类平均准确率（74.21%）均为最高。上表和 [model_result.md](model_result.md) 均以这些 JSON 文件为准。当前 `src/train.py` 和 `src/evaluate.py` 分别使用 `models/checkpoints/`、`models/history/` 和 `models/results/` 保存模型权重、训练历史和评估结果。新运行可能覆盖同一模型的已有文件。

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/print1Username/COS30082-Assignment-1.git
cd COS30082-Assignment-1
```

### 2. 创建 Python 环境

使用 `venv`：

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

使用 Conda：

```bash
conda create -n cos30082a1 python=3.10 -y
conda activate cos30082a1
```

### 3. 安装依赖

先升级 pip，再从下面两种 PyTorch 安装方式中**选择一种**：

```bash
python -m pip install --upgrade pip
```

**方案 A：CPU（Windows/Linux，无需 NVIDIA 显卡）**

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

**方案 B：CUDA（Windows/Linux，需要兼容的 NVIDIA 显卡及驱动）**

下面以官方 CUDA 12.6 软件包索引为例：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

请通过 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 选择与系统兼容的 CUDA 版本。macOS 用户请使用选择器提供的 macOS 安装命令。请在新建的环境中执行所选方案。由于 `requirements.txt` 包含 CPU 软件包索引，应先安装所选版本的 PyTorch；随后安装项目依赖时，会保留已安装且满足未锁定版本要求的软件包。

检查安装结果：

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

输出 `CUDA available: True` 表示 PyTorch 可以使用 NVIDIA 显卡；CPU 方案输出 `False` 属于正常情况。

## 下载本地数据

下载脚本会从 [Hugging Face 数据集仓库](https://huggingface.co/datasets/print1Username/COS30082-Assignment-1-dataset) 获取 `Train.zip`、`Test.zip`、`train.txt` 和 `test.txt`，解压图像并检查目录结构：

```bash
python download_data.py
```

> 脚本下载前会清空现有 `data/` 内容。请先备份需要保留的本地数据。

## 训练前检查模型环境

安装依赖并下载数据后，先在项目根目录执行以下检查，再继续训练和评估：

```bash
python src/models.py
```

该命令会创建使用 ImageNet 预训练权重的 ResNet18，检查骨干网络的冻结与解冻，并输出参数数量。看到 `Model test completed successfully.` 后再继续。如果本地尚未缓存 ResNet18 预训练权重，首次运行需要联网下载；训练其他架构时，可能还需要下载对应的权重。

`train.py` 和 `evaluate.py` 会自动导入 `models.py`，因此单独执行此检查并非代码层面的强制依赖，也不会生成训练后的 checkpoint。它可以帮助提前发现模型、依赖或预训练权重下载问题。建议执行顺序为：模型检查 → 训练 → 评估；评估需要对应模型训练后生成的 checkpoint。

## 训练模型

一次训练一个模型：

```bash
python src/train.py --model resnet18
python src/train.py --model resnet50
python src/train.py --model efficientnet_b0
```

训练流程使用分层 80/20 训练-验证划分、ImageNet 预训练权重、数据增强、两阶段微调、AdamW、标签平滑、学习率调度、early stopping 和最佳 checkpoint 保存。模型权重保存在 `models/checkpoints/`，训练历史保存在 `models/history/`。

本地 Python 和本地 notebook 的训练实现采用相同的过拟合控制措施：

| 设置 | 修正后的行为 |
|---|---|
| 模型架构 | 保留 ResNet18、ResNet50 和 EfficientNet-B0，不更换架构 |
| 数据划分 | 从官方训练集进行分层 80/20 划分，随机种子为 `42` |
| 分类层预热 | 前 5 轮只训练分类层，学习率为 `1e-3`；冻结部分保持评估模式，防止 BatchNorm 统计量变化 |
| 微调学习率 | 预训练部分为 `1e-5`，分类层为 `1e-4` |
| 训练约束 | AdamW 权重衰减为 `1e-3`，标签平滑为 `0.1`，数据增强仅用于训练图片 |
| 学习率调度 | 验证 loss 不再改善时降低学习率 |
| 提前停止 | 连续 5 轮验证 loss 没有下降就停止；开始微调时重新计算等待轮数 |
| 最佳权重选择 | 按最高验证准确率保存 checkpoint |
| 训练上限 | Batch size 为 `32`，最多训练 35 轮 |

验证 loss 决定何时降低学习率和停止训练，验证准确率决定保存哪个模型。测试集只用于最终评估，不用于调参或网页应用的模型选择。

## 本地 Notebook 版本（独立运行）

`notebooks/` 目录包含完整的 notebook 工作流程。Notebook 不会导入 `src/*.py` 中的训练实现，每个 notebook 都包含自己阶段所需的代码：

1. `notebooks/01_data_preparation.ipynb`：合并后的数据探索与准备 notebook，检查路径、标注、类别、缺失/损坏图片、图片尺寸、类别分布、PyTorch `Dataset`/`DataLoader` 和数据增强，并包含数据摘要图表。
2. `notebooks/02_training.ipynb`：完整训练流程（train/validation/fine-tune/checkpoint/history），包含 loss、accuracy、学习率和训练状态图表。
3. `notebooks/03_evaluation.ipynb`：测试集评估（Top-1 与每类平均准确率），并包含每类性能图表。

请按照数字顺序执行。原本独立的数据探索 notebook 已合并到 `01_data_preparation.ipynb`，避免功能重复。

Notebook 的训练历史和评估结果保留在内存中，通过图表和指标展示。Notebook 会保存模型权重，但不写入 `models/` 中的 JSON；这些 JSON 只由本地 `src` 流程保存。训练 notebook 使用 `num_workers=0`，兼容 Windows/Jupyter。

安装项目依赖、下载数据后，安装 notebook 工具，并从 `notebooks/` 目录启动 Jupyter：

```bash
pip install notebook matplotlib numpy
cd notebooks
jupyter notebook
```

Notebook 将工作目录的上一级作为项目根目录，因此请保持内核工作目录为 `notebooks/`。在 `02_training.ipynb` 中选择 `MODEL`，取消 `hist=train_model(MODEL)` 的注释即可开始训练；在 `03_evaluation.ipynb` 中选择对应模型后再评估。Notebook 与本地 Python 使用相同的 checkpoint 文件名，因此可能覆盖同名模型的已有权重。

## Google Colab 版本

独立的 Google Colab 版本和数据集统一放在 [Google Drive 共享文件夹](https://drive.google.com/drive/folders/1vLkbNAVSo3dt7mVHacQpqIANziIXmqB_?usp=sharing)。通过该文件夹访问云端 notebook 和所需数据；本地 notebook 则位于此仓库的 `notebooks/` 目录。Colab 的环境准备和数据路径设置请按对应云端 notebook 内的说明操作。

本 README 记录的代码修正已应用于此仓库的本地 Python 和本地 notebook；本次没有修改或验证 Google Drive 中独立的 Colab 副本。

## 测试集评估

模型训练并生成 checkpoint 后，再运行评估：

```bash
python src/evaluate.py --model resnet50
python src/evaluate.py --model resnet18
python src/evaluate.py --model efficientnet_b0
```

结果保存在 `models/results/`。作业要求的指标为 Top-1 accuracy 和 Average Accuracy per Class。

## 模型输出文件夹

正常运行本地训练和评估时，`models/` 只使用以下三个文件夹：

| 文件夹 | 内容 | 写入来源 |
|---|---|---|
| `models/checkpoints/` | `<model>_best.pth`：模型权重、架构、验证准确率、优化器和调度器状态；新训练的 checkpoint 还记录训练配置 | 本地 Python 训练或本地训练 notebook |
| `models/history/` | `<model>_history.json`：每轮 loss、准确率和学习率，完成每轮训练后更新 | 仅 `src/train.py` |
| `models/results/` | `<model>_test_results.json`：最终测试 Top-1 和每类平均准确率 | 仅 `src/evaluate.py` |

代码不会自动创建 `baseline`、`interrupted_trial` 或 `retraining` 文件夹。同一模型再次训练或评估时，可能覆盖它原有的对应文件；程序不会自动创建备份文件夹。

## 本地运行网页端

确认已安装 Gradio 和其他依赖后运行：

```bash
python app.py
```

打开终端显示的本地地址，通常是 `http://localhost:7860`。网页端支持上传图片，并返回 Top-5 鸟类预测结果。如果本地存在 checkpoint，程序会优先使用本地文件；否则会从 [Hugging Face 模型仓库](https://huggingface.co/print1Username/COS30082-Assignment-1-model) 下载 checkpoint 和 `class_mapping.json`。

`app.py` 启动时会比较本地 checkpoint 内保存的 `validation_accuracy`，自动加载验证准确率最高的模型。没有本地权重时，才下载并比较 Hugging Face 仓库根目录中的 checkpoint。每份 checkpoint 必须包含 `model_name` 和 `validation_accuracy`；选择过程不读取训练历史或测试结果 JSON。页面显示实际使用的模型名称和验证准确率。重新训练后，需要重启 App 才会重新选择模型。

## 项目结构

```text
├── app.py                 # Gradio 网页应用
├── download_data.py       # 下载并解压 CUB-200
├── requirements.txt       # Python 依赖
├── class_mapping.json     # 类别编号与鸟类名称映射
├── src/
│   ├── dataset.py         # 数据集加载与预处理
│   ├── models.py          # 模型定义
│   ├── train.py           # 模型训练与 checkpoint 保存
│   └── evaluate.py        # 测试集评估
├── notebooks/            # 本地独立 Jupyter 工作流程
│   ├── 01_data_preparation.ipynb
│   ├── 02_training.ipynb
│   └── 03_evaluation.ipynb
├── data/                  # 本地数据集文件
└── models/
    ├── checkpoints/       # 模型权重
    ├── history/           # 仅本地 src 的训练历史 JSON
    └── results/           # 仅本地 src 的测试结果 JSON
```

## 作业评估指标

- **Top-1 Accuracy：** 正确分类的测试图像数除以测试图像总数。
- **Average Accuracy per Class：** 先分别计算每个鸟类的准确率，再对所有类别准确率取平均值。

## 学术用途

本项目用于 COS30082 Assignment 1。CUB-200 数据集仍受其原始使用条款约束。
