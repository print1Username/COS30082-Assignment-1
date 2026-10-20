# Model Results / 模型表现总结

## Overview / 概览

This document summarises the final evaluation results of three image-classification models. All metrics below are taken directly from the JSON result files in `models/results/`.

本文汇总三个图像分类模型的最终评估结果。以下所有指标均直接来自 `models/results/` 中的 JSON 文件。

**Evaluation setting / 评估设定**

- Test set: **1,204 images**
- Classes evaluated: **200 classes**
- Top-1 Accuracy: overall proportion of correctly classified test images / 所有测试图像中预测正确的比例。
- Average Accuracy per Class: mean of the class-wise accuracies / 先分别计算每个类别的准确率，再取平均值。

## Results Comparison / 结果对比

| Model / 模型 | Best Validation Accuracy / 最佳验证准确率 | Test Top-1 Accuracy / 测试 Top-1 准确率 | Test Average Accuracy per Class / 测试每类平均准确率 | Test Samples / 测试样本 | Classes / 类别数 |
|---|---:|---:|---:|---:|---:|
| ResNet18 | 70.19% | 69.60% | 69.18% | 1,204 | 200 |
| EfficientNet-B0 | 72.57% | 70.93% | 69.87% | 1,204 | 200 |
| **ResNet50** | **78.99%** | **76.91%** | **76.16%** | 1,204 | 200 |

## Model-by-Model Summary / 各模型总结

### 1. ResNet18

- Best validation accuracy / 最佳验证准确率: **70.19%**
- Test Top-1 accuracy / 测试 Top-1 准确率: **69.60%**
- Test average per-class accuracy / 测试每类平均准确率: **69.18%**
- Training history / 训练历史: 32 epochs

ResNet18 provides a solid baseline. Its validation-to-test drop is only **0.59 percentage points**, indicating that the model generalises reasonably well. However, it records the lowest test metrics among the three models.

ResNet18 可作为可靠的 baseline。其最佳验证准确率与测试 Top-1 准确率相差仅 **0.59 个百分点**，显示泛化能力合理；但三者中它的测试指标最低。

### 2. EfficientNet-B0

- Best validation accuracy / 最佳验证准确率: **72.57%**
- Test Top-1 accuracy / 测试 Top-1 准确率: **70.93%**
- Test average per-class accuracy / 测试每类平均准确率: **69.87%**
- Training history / 训练历史: 35 epochs

EfficientNet-B0 improves over ResNet18 by **1.33 percentage points** in Test Top-1 accuracy and **0.69 percentage points** in average per-class accuracy. Its validation-to-test Top-1 gap is **1.64 percentage points**, which remains modest.

EfficientNet-B0 的测试 Top-1 准确率比 ResNet18 高 **1.33 个百分点**，每类平均准确率高 **0.69 个百分点**。验证集与测试集的 Top-1 差距为 **1.64 个百分点**，仍属合理范围。

### 3. ResNet50 — Best Model / 最佳模型

- Best validation accuracy / 最佳验证准确率: **78.99%**
- Test Top-1 accuracy / 测试 Top-1 准确率: **76.91%**
- Test average per-class accuracy / 测试每类平均准确率: **76.16%**
- Training history / 训练历史: 35 epochs

ResNet50 is the best-performing model in this experiment. Compared with ResNet18, it improves Test Top-1 accuracy by **7.31 percentage points** and average per-class accuracy by **6.99 percentage points**. Compared with EfficientNet-B0, it improves these metrics by **5.98** and **6.29 percentage points**, respectively.

ResNet50 是本实验中表现最好的模型。相较 ResNet18，它的测试 Top-1 准确率提升 **7.31 个百分点**，每类平均准确率提升 **6.99 个百分点**；相较 EfficientNet-B0，这两项指标分别提升 **5.98** 与 **6.29 个百分点**。

Its validation-to-test Top-1 gap is **2.07 percentage points**. This is slightly larger than the other models, but the final test results remain clearly superior; therefore, ResNet50 is the recommended model for deployment or further development.

其验证集与测试集的 Top-1 差距为 **2.07 个百分点**，略高于另外两个模型；但最终测试表现仍明显领先，因此建议选择 ResNet50 用于部署或后续开发。

## Ranking / 排名

| Rank / 排名 | Model / 模型 | Selection reason / 选择理由 |
|---:|---|---|
| 1 | **ResNet50** | Highest Test Top-1 accuracy and highest average per-class accuracy / 两项测试指标均为最高。 |
| 2 | EfficientNet-B0 | Better test performance than ResNet18 / 测试表现优于 ResNet18。 |
| 3 | ResNet18 | Stable baseline, but lowest final test performance / 泛化稳定，但最终测试表现最低。 |

## Final Conclusion / 最终结论

**Recommended model / 推荐模型: ResNet50.** It achieves the strongest overall and class-balanced performance: **76.91% Test Top-1 Accuracy** and **76.16% Test Average Accuracy per Class**. Because both metrics lead the comparison on the same 1,204-image, 200-class test set, ResNet50 is the most suitable final model based on the available results.

**推荐使用 ResNet50。** 它同时取得最高的整体分类表现与类别均衡表现：**76.91% 测试 Top-1 准确率** 和 **76.16% 测试每类平均准确率**。在相同的 1,204 张图像、200 类测试集上，两项指标都领先，因此根据现有结果，ResNet50 最适合作为最终模型。

## Source Files / 数据来源

- `models/results/resnet18_test_results.json`
- `models/results/efficientnet_b0_test_results.json`
- `models/results/resnet50_test_results.json`
- `models/history/resnet18_history.json`
- `models/history/efficientnet_b0_history.json`
- `models/history/resnet50_history.json`
