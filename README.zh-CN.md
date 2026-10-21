# CUB-200 鸟类物种分类

[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](README.md)
[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-red?style=for-the-badge)](README.zh-CN.md)

使用 Caltech-UCSD Birds 200（CUB-200）数据集，对 200 种鸟类进行图像分类。项目比较 ResNet18、ResNet50 和 EfficientNet-B0 的迁移学习效果。**最终推荐模型为 ResNet50。**

## 项目链接

- GitHub：[COS30082-Assignment-1](https://github.com/print1Username/COS30082-Assignment-1)
- Hugging Face 主页：[print1Username](https://huggingface.co/print1Username)
- 数据集：[CUB-200 dataset](https://huggingface.co/datasets/print1Username/COS30082-Assignment-1-dataset)
- 模型文件：[训练模型仓库](https://huggingface.co/print1Username/COS30082-Assignment-1-model)

## 训练结果

| 模型 | 最佳验证准确率 | 测试 Top-1 | 测试每类平均准确率 |
|---|---:|---:|---:|
| ResNet18 | 70.19% | 69.60% | 69.18% |
| EfficientNet-B0 | 72.57% | 70.93% | 69.87% |
| **ResNet50** | **78.99%** | **76.91%** | **76.16%** |

所有结果均基于 1,204 张测试图像和 200 个评估类别。详细结果请查看 [model_result.md](model_result.md)。

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
conda create -n cub200 python=3.10 -y
conda activate cub200
```

安装依赖：

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` 默认安装 CPU 版本的 PyTorch。如果使用 CUDA，请先根据 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 安装匹配版本的 PyTorch 和 torchvision，再安装 `requirements.txt` 中的其他依赖。

## 下载本地数据

下载脚本会从 [Hugging Face 数据集仓库](https://huggingface.co/datasets/print1Username/COS30082-Assignment-1-dataset) 获取 `Train.zip`、`Test.zip`、`train.txt` 和 `test.txt`，解压图像并检查目录结构：

```bash
python download_data.py
```

> 脚本下载前会清空现有 `data/` 内容。请先备份需要保留的本地数据。

## 训练模型

一次训练一个模型：

```bash
python src/train.py --model resnet18
python src/train.py --model resnet50
python src/train.py --model efficientnet_b0
```

训练流程使用分层 80/20 训练-验证划分、ImageNet 预训练权重、数据增强、两阶段微调、AdamW、标签平滑、学习率调度、early stopping 和最佳 checkpoint 保存。模型权重保存在 `models/checkpoints/`，训练历史保存在 `models/history/`。

## Notebook 版本（独立运行）

`notebooks/` 目录包含完整的 notebook 工作流程。Notebook 不会导入 `src/*.py` 中的训练实现，每个 notebook 都包含自己阶段所需的代码：

1. `notebooks/01_data_preparation.ipynb`：合并后的数据探索与准备 notebook，检查路径、标注、类别、缺失/损坏图片、图片尺寸、类别分布、PyTorch `Dataset`/`DataLoader` 和数据增强，并包含数据摘要图表。
2. `notebooks/02_training.ipynb`：完整训练流程（train/validation/fine-tune/checkpoint/history），包含 loss、accuracy、学习率和训练状态图表。
3. `notebooks/03_evaluation.ipynb`：测试集评估（Top-1 与每类平均准确率），并包含每类性能图表。

请按照数字顺序执行。原本独立的数据探索 notebook 已合并到 `01_data_preparation.ipynb`，避免功能重复。

## 测试集评估

模型训练并生成 checkpoint 后，再运行评估：

```bash
python src/evaluate.py --model resnet50
```

结果保存在 `models/results/`。作业要求的指标为 Top-1 accuracy 和 Average Accuracy per Class。

## 本地运行网页端

确认已安装 Gradio 和其他依赖后运行：

```bash
python app.py
```

打开终端显示的本地地址，通常是 `http://localhost:7860`。网页端支持上传图片，并返回 Top-5 鸟类预测结果。如果本地存在 checkpoint，程序会优先使用本地文件；否则会从 [Hugging Face 模型仓库](https://huggingface.co/print1Username/COS30082-Assignment-1-model) 下载 checkpoint 和 `class_mapping.json`。

当前 `app.py` 的推理配置使用 EfficientNet-B0。最终评估表现最好的模型是 ResNet50；如果要在网页端部署 ResNet50，必须同步修改网页端的模型架构配置，并确保上传的模型文件与架构一致。

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
├── data/                  # 本地数据集文件
└── models/                # checkpoint、训练历史和评估结果
```

## 作业评估指标

- **Top-1 Accuracy：** 正确分类的测试图像数除以测试图像总数。
- **Average Accuracy per Class：** 先分别计算每个鸟类的准确率，再对所有类别准确率取平均值。

## 学术用途

本项目用于 COS30082 Assignment 1。CUB-200 数据集仍受其原始使用条款约束。
