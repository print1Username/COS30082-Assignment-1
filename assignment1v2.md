# **COS30082 Assignment–** 

# **Bird Species Classification** 

**_Weighting: 20% Individual Assignment_** 

## **1. Details** 

In this assignment, you will work on multi-class classification. You can use any of the techniques you have learned so far to solve this problem. 

## **2. Dataset** 

Caltech-UCSD Birds 200 (CUB-200) is an image dataset with photos of **200 bird species** (mostly North American). The training images consists of 4829 number of images. You may download the images and annotation file in Canvas’s _Assignment information page_ . 

### **Caltect-UCSD Birds 200** 



<!-- Start of picture text -->
http://www.vision.caltech.edu/visipedia/CUB-200.html<br>« oS Sap "4 ee<br><!-- End of picture text -->

## **3. Image and Annotation files** 

You can find the training images in the **Train.zip** folder and the annotation file in **train.txt** . The format of the **train.txt** file is as follows: 

_image’s name{space}class label_ 

Here is an example: 

_Black_footed_Albatross_0019_416160254.jpg 0 Black_footed_Albatross_0005_2755588934.jpg 0 Laysan_Albatross_0014_174432783.jpg 1 Sooty_Albatross_0005_340127050.jpg 2_ 

COS30082 Assignment 1 

You can also find the testing images in the **Test.zip** folder and the annotation file in **test.txt** . Its format is the same as the one of the training. 

## **4. Evaluation metric** 

You have to report your results with these two evaluation metrics: the _Top-1 accuracy_ and the _Average accuracy per class_ . You can also include additional evaluation metrics subject to proper justification. 

The _Top-1 accuracy_ is used to evaluate the overall classification performance of the models in this competition. **𝑎 𝑎 𝑎 𝑔 𝑔 𝑔** 



𝑖𝑖=1 𝐶𝐶 total number of classes. where 𝑇𝑇𝑖𝑖 is the average accuracy for all test images related to the 𝐶𝐶𝑖𝑖 class and the 𝐶𝐶 is the 

## **5. Required Submission** 

Submit to Canvas page under Assignment submission link. 

### **_Report:_** 

A report (PDF) consisting of two sections: _Methodology_ and _Result and Discussion._ 

There is a high probability that you will run into _overfitting_ problems, so if this is the case, your model design should take into account how to minimize this problem. You should describe your models' architecture, loss function, hyperparameters, and any other details of interest, and discuss the performance differences between them. You should also justify which model gave you the best result. 

Please limit the report to _4 pages_ . And, report must include your _name_ and _ID_ . 

### **_Python program source code:_** 

Create a single zip file with your code, and submit together with the report. 

*Please comment generously to show that you understand your code. 

COS30082 Assignment 1 

### **_Video Presentation:_** 

Your presentation should: 

- Explain key concepts you are using and your model design.  Slides are Recommended. 

- Present and explain your code. 

- Demonstrate the training and results generated. 

- Present your AI model web application via Hugging face. 

- Max 10 mins 

Please attach the video link (Youtube link) in your report. Also, please attach the hugging face link in your report too. 

COS30082 Assignment 1 

