# Model Results / 模型表现总结

## Overview / 概览

This document summarises the final evaluation results of three image-classification models. Evaluation metrics come from the JSON result files in `models/results/`, and epoch counts come from `models/history/`.

本文汇总三个图像分类模型的最终评估结果。评估指标来自 `models/results/` 中的 JSON 文件，训练轮数来自 `models/history/`。

**Evaluation setting / 评估设定**

- Test set: **1,204 images**
- Classes evaluated: **200 classes**
- Top-1 Accuracy: overall proportion of correctly classified test images / 所有测试图像中预测正确的比例。
- Average Accuracy per Class: mean of the class-wise accuracies / 先分别计算每个类别的准确率，再取平均值。

## Results Comparison / 结果对比

| Model / 模型 | Best Validation Accuracy / 最佳验证准确率 | Test Top-1 Accuracy / 测试 Top-1 准确率 | Test Average Accuracy per Class / 测试每类平均准确率 | Test Samples / 测试样本 | Classes / 类别数 |
|---|---:|---:|---:|---:|---:|
| ResNet18 | 66.77% | 65.12% | 64.45% | 1,204 | 200 |
| **EfficientNet-B0** | **74.64%** | **74.75%** | **74.21%** | 1,204 | 200 |
| ResNet50 | 70.91% | 70.02% | 69.09% | 1,204 | 200 |

## Model-by-Model Summary / 各模型总结

### 1. ResNet18

- Best validation accuracy / 最佳验证准确率: **66.77%**
- Test Top-1 accuracy / 测试 Top-1 准确率: **65.12%**
- Test average per-class accuracy / 测试每类平均准确率: **64.45%**
- Training history / 训练历史: 35 epochs

ResNet18 provides a baseline. Its validation-to-test Top-1 drop is **1.65 percentage points**, and it records the lowest test metrics among the three models. Its training accuracy is much higher than its validation accuracy at the best validation epoch, indicating overfitting.

ResNet18 可作为 baseline。其最佳验证准确率比测试 Top-1 准确率高 **1.65 个百分点**，且三者中它的测试指标最低。在最佳验证轮次，训练准确率明显高于验证准确率，显示出过拟合迹象。

### 2. EfficientNet-B0 — Best Model / 最佳模型

- Best validation accuracy / 最佳验证准确率: **74.64%**
- Test Top-1 accuracy / 测试 Top-1 准确率: **74.75%**
- Test average per-class accuracy / 测试每类平均准确率: **74.21%**
- Training history / 训练历史: 33 epochs

EfficientNet-B0 improves over ResNet18 by **9.63 percentage points** in Test Top-1 accuracy and **9.76 percentage points** in average per-class accuracy. Its test Top-1 accuracy is **0.11 percentage points** higher than its best validation accuracy.

EfficientNet-B0 的测试 Top-1 准确率比 ResNet18 高 **9.63 个百分点**，每类平均准确率高 **9.76 个百分点**。其测试 Top-1 准确率比最佳验证准确率高 **0.11 个百分点**。

### 3. ResNet50

- Best validation accuracy / 最佳验证准确率: **70.91%**
- Test Top-1 accuracy / 测试 Top-1 准确率: **70.02%**
- Test average per-class accuracy / 测试每类平均准确率: **69.09%**
- Training history / 训练历史: 35 epochs

ResNet50 ranks second in this experiment. Compared with ResNet18, it improves Test Top-1 accuracy by **4.90 percentage points** and average per-class accuracy by **4.64 percentage points**. It trails EfficientNet-B0 by **4.73** and **5.12 percentage points** on these metrics, respectively.

ResNet50 在本实验中排名第二。相较 ResNet18，它的测试 Top-1 准确率提升 **4.90 个百分点**，每类平均准确率提升 **4.64 个百分点**；相较 EfficientNet-B0，这两项指标分别低 **4.73** 与 **5.12 个百分点**。

Its best validation accuracy exceeds its test Top-1 accuracy by **0.89 percentage points**. Its test results are stronger than ResNet18's but lower than EfficientNet-B0's.

其最佳验证准确率比测试 Top-1 准确率高 **0.89 个百分点**。它的测试表现优于 ResNet18，但低于 EfficientNet-B0。

## Ranking / 排名

| Rank / 排名 | Model / 模型 | Selection reason / 选择理由 |
|---:|---|---|
| 1 | **EfficientNet-B0** | Highest Test Top-1 accuracy and highest average per-class accuracy / 两项测试指标均为最高。 |
| 2 | ResNet50 | Better test performance than ResNet18 / 测试表现优于 ResNet18。 |
| 3 | ResNet18 | Baseline with the lowest final test performance / 最终测试表现最低的 baseline。 |

## Final Conclusion / 最终结论

**Recommended model / 推荐模型: EfficientNet-B0.** It achieves the strongest overall and class-balanced performance: **74.75% Test Top-1 Accuracy** and **74.21% Test Average Accuracy per Class**. Because both metrics lead the comparison on the same 1,204-image, 200-class test set, EfficientNet-B0 is the most suitable final model based on the available results.

**推荐使用 EfficientNet-B0。** 它同时取得最高的整体分类表现与类别均衡表现：**74.75% 测试 Top-1 准确率** 和 **74.21% 测试每类平均准确率**。在相同的 1,204 张图像、200 类测试集上，两项指标都领先，因此根据现有结果，EfficientNet-B0 最适合作为最终模型。

## Source Files / 数据来源

- `models/results/resnet18_test_results.json`
- `models/results/efficientnet_b0_test_results.json`
- `models/results/resnet50_test_results.json`
- `models/history/resnet18_history.json`
- `models/history/efficientnet_b0_history.json`
- `models/history/resnet50_history.json`
