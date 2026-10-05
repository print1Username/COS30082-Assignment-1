# Assignment 1 Requirements / 作业 1 要求整理

> Course Assignment: Multi-class Classification on CUB-200  
> 作业主题：使用 CUB-200 数据集进行多类分类

---

## 1. Task Overview / 任务概览

- Work on **multi-class classification**.
- 进行 **多类分类** 任务。
- Dataset: **Caltech-UCSD Birds 200 (CUB-200)**.
- 数据集：**Caltech-UCSD Birds 200 (CUB-200)**。
- CUB-200 contains photos of **200 bird species**.
- CUB-200 包含 **200 种鸟类** 的图片。
- Training images: **4829 images**.
- 训练图像数量：**4829 张**。
- You may use any technique learned so far.
- 可以使用目前学过的任意技术。
- Important: There is a high probability of **overfitting**.
- 注意：很可能会出现 **过拟合** 问题。
- Your model design should consider how to **minimize overfitting**.
- 模型设计需要考虑如何 **减少过拟合**。

---

## 2. Dataset and Annotation Files / 数据集与标注文件

Download from Canvas Assignment information page:

- `Train.zip`
- `train.txt`
- `Test.zip`
- `test.txt`

从 Canvas 的 Assignment information page 下载以上文件。

### Annotation Format / 标注格式

```text
image's name[space]class label
```

Example / 示例：

```text
bird_image_001.jpg 12
```

- Training and testing annotation files have the same format.
- 训练集和测试集标注文件格式相同。

---

## 3. Evaluation Metrics / 评估指标

You must report these two evaluation metrics:

必须报告以下两个评估指标：

1. **Top-1 Accuracy**
2. **Average Accuracy per Class**

You may include additional evaluation metrics if properly justified.

如果合理说明理由，也可以加入其他评估指标。

### Top-1 Accuracy Formula / Top-1 准确率公式

\[
\mathrm{Top\text{-}1\ accuracy} =
\frac{1}{N}
\sum_{k=1}^{N}
1\{\arg\max(y) == \text{groundtruth}\}
\]

Where / 其中：

- \(N\) = total number of testing images / 测试图像总数
- \(y\) = output probabilities of 200 classes / 200 个类别的输出概率

### Average Accuracy per Class Formula / 每类平均准确率公式

\[
\mathrm{Ave} =
\frac{1}{C}
\sum_{i=1}^{C} T_i
\]

Where / 其中：

- \(T_i\) = average accuracy for all test images related to class \(C_i\)  
  第 \(C_i\) 类所有测试图像的平均准确率
- \(C\) = total number of classes / 类别总数

---

## 4. Required Submission / 必须提交内容

Submit to Canvas under the Assignment submission link.

提交到 Canvas 的 Assignment submission link。

### 4.1 Report PDF / 报告 PDF

- A report PDF consisting of two sections:
  - **Methodology**
  - **Result and Discussion**
- 报告 PDF 包含两个部分：
  - **Methodology / 方法**
  - **Result and Discussion / 结果与讨论**
- Limit the report to **4 pages**.
- 报告限制在 **4 页以内**。
- Must include your **name and ID**.
- 必须包含 **姓名和 ID**。
- Must include your **YouTube link**.
- 必须包含 **YouTube 链接**。
- Must include your **Hugging Face link**.
- 必须包含 **Hugging Face 链接**。
- Describe:
  - Model architecture
  - Loss function
  - Hyperparameters
  - Other details of interest
- 需要描述：
  - 模型架构
  - 损失函数
  - 超参数
  - 其他重要细节
- Discuss overfitting and how your model design minimizes it.
- 讨论过拟合问题，以及你的模型设计如何减少过拟合。
- Discuss performance differences between models.
- 讨论不同模型之间的性能差异。
- Justify which model gave you the best result.
- 说明为什么某个模型取得了最佳结果。

### 4.2 Python Program Source Code / Python 程序源代码

- Create a **single zip file** with your code.
- 将代码打包成一个 **zip 文件**。
- Submit together with the report.
- 与报告一起提交。
- Please comment generously to show that you understand your code.
- 请大量注释代码，以表明你理解自己的代码。

### 4.3 Video Presentation / 视频演示

- Max **10 mins**.
- 最长 **10 分钟**。
- Explain key concepts you are using and your model design.
- 解释你使用的关键概念和模型设计。
- Slides are recommended.
- 推荐使用 Slides。
- Present and explain your code.
- 展示并解释你的代码。
- Demonstrate the training and results generated.
- 演示训练过程和生成的结果。
- Present your AI model web application via Hugging Face.
- 展示通过 Hugging Face 部署的 AI 模型 Web 应用。
- Attach the video link in your report.
- 在报告中附上视频链接。
- Attach the Hugging Face link in your report too.
- 同时在报告中附上 Hugging Face 链接。

### 4.4 Submission Location / 提交位置

- Submit to Canvas page under Assignment submission link.
- 提交到 Canvas 的 Assignment submission link。

---

## 5. To-Do Checklist / 待办清单

- [ ] Download and unzip `Train.zip`, `Test.zip`, `train.txt`, `test.txt`.
- [ ] 下载并解压训练集、测试集和标注文件。
- [ ] Parse `train.txt` and `test.txt`, build data loaders.
- [ ] 读取 `train.txt` 和 `test.txt`，构建数据加载器。
- [ ] Preprocess images and apply data augmentation.
- [ ] 进行图像预处理和数据增强。
- [ ] Choose a model, e.g., transfer learning model.
- [ ] 选择模型，例如迁移学习模型。
- [ ] Design loss function and hyperparameters.
- [ ] 设计损失函数和超参数。
- [ ] Train model and monitor validation performance.
- [ ] 训练模型并监控验证集表现。
- [ ] Use anti-overfitting strategies:
  - Data augmentation
  - Regularization
  - Dropout
  - Early stopping
  - Transfer learning
- [ ] 使用防过拟合策略：
  - 数据增强
  - 正则化
  - Dropout
  - 早停
  - 迁移学习
- [ ] Compute **Top-1 accuracy** and **Average accuracy per class** on test set.
- [ ] 在测试集上计算 **Top-1 accuracy** 和 **Average accuracy per class**。
- [ ] Compare different models/configurations and analyze performance.
- [ ] 比较不同模型或配置，并分析性能差异。
- [ ] Select and justify the best model.
- [ ] 选择最佳模型并说明理由。
- [ ] Deploy the best model to **Hugging Face Space** as a web application.
- [ ] 将最佳模型部署到 **Hugging Face Space**，做成 Web 应用。
- [ ] Write PDF report, max 4 pages, with Methodology and Result and Discussion.
- [ ] 写 PDF 报告，最多 4 页，包含 Methodology 和 Result and Discussion。
- [ ] Include name, ID, YouTube link, and Hugging Face link in the report.
- [ ] 在报告中包含姓名、ID、YouTube 链接和 Hugging Face 链接。
- [ ] Zip source code with generous comments.
- [ ] 将源代码打包成 zip，并充分注释。
- [ ] Record video, max 10 mins, upload to YouTube.
- [ ] 录制最长 10 分钟的视频，上传到 YouTube。
- [ ] Submit report and code zip on Canvas.
- [ ] 在 Canvas 提交报告和代码 zip。

---

## 6. Suggested Report Structure / 建议报告结构

### Methodology / 方法

- Data processing / 数据处理
- Model architecture / 模型架构
- Loss function / 损失函数
- Hyperparameters / 超参数
- Training strategy / 训练策略
- Anti-overfitting methods / 防过拟合方法
- Deployment method / 部署方法

### Result and Discussion / 结果与讨论

- Top-1 accuracy / Top-1 准确率
- Average accuracy per class / 每类平均准确率
- Model comparison table / 模型对比表
- Overfitting analysis / 过拟合分析
- Best model justification / 最佳模型理由
- Additional metrics if any / 其他评估指标

### Links / 链接

- YouTube video link / YouTube 视频链接
- Hugging Face link / Hugging Face 链接

---

## 7. Suggested Video Structure / 建议视频结构

- Key concepts / 关键概念
- Model design / 模型设计
- Code walkthrough / 代码讲解
- Training and results / 训练与结果
- Hugging Face web app demo / Hugging Face Web 应用演示
- Summary / 总结

---

## 8. Final Deliverables / 最终交付清单

- [ ] Report PDF, max 4 pages, includes name and ID
- [ ] 报告 PDF，最多 4 页，包含姓名和 ID
- [ ] YouTube link in report
- [ ] 报告中包含 YouTube 链接
- [ ] Hugging Face link in report
- [ ] 报告中包含 Hugging Face 链接
- [ ] Code zip with generous comments
- [ ] 代码 zip，充分注释
- [ ] Video max 10 mins, uploaded to YouTube
- [ ] 视频最长 10 分钟，已上传 YouTube
- [ ] Hugging Face web app deployed
- [ ] Hugging Face Web 应用已部署
- [ ] Submitted on Canvas
- [ ] 已在 Canvas 提交