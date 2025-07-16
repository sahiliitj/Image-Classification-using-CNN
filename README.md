# 🧠 CNN from Scratch using PyTorch – Multi-Class Image Classification
 
> **Author:** Sahil Sharma  
> **Program:** M.Sc - M.Tech (Data and Computational Sciences)  
> **Institute:** Indian Institute of Technology, Jodhpur  

---

## 📁 Project Overview

This project involves implementing a **Convolutional Neural Network (CNN)** from scratch using the **PyTorch** framework to classify handwritten digits. The dataset used contains 70,000 grayscale images of digits (0–9). The task is divided into two experiments:

- **Experiment 1:** Classify images into 10 original digit classes (0–9)
- **Experiment 2:** Map digits into 4 custom classes and evaluate performance

A bonus section includes advanced techniques to improve generalization and prevent overfitting.

---

## 🧾 Dataset Summary

- **Total Images:** 70,000  
- **Image Size:** 28 × 28 pixels  
- **Train Set:** 60,000 images  
- **Test Set:** 10,000 images  
- **Labels:** 0–9 (10 digit classes)

### 🧪 Class Mapping for Experiment 2:

| New Class | Mapped Digits      |
|-----------|--------------------|
| Class 0   | {0, 6}             |
| Class 1   | {1, 7}             |
| Class 2   | {2, 3, 5, 8}       |
| Class 3   | {4, 9}             |

---

## 📊 Exploratory Data Analysis

- **Visualized sample images**
- **Histogram of digit label distribution**
- **Pixel intensity histogram**

---

## 🧱 CNN Architecture (Layer-wise Description)

The model uses **3 convolutional layers** followed by **1 fully connected (linear) layer**.

| Layer             | Configuration                                    |
|------------------|--------------------------------------------------|
| Conv Layer 1      | 1 → 16 filters, 7×7 kernel, stride=1, padding=3 |
| Max Pooling       | 2×2, stride=2                                   |
| Conv Layer 2      | 16 → 8 filters, 5×5 kernel, stride=1, padding=2 |
| Max Pooling       | 2×2, stride=2                                   |
| Conv Layer 3      | 8 → 4 filters, 3×3 kernel, stride=1, padding=1  |
| Avg Pooling       | 2×2, stride=2                                   |
| Fully Connected   | Flattened output → 10 (or 4) classes            |

### Additional Details:
- **Activation:** ReLU (Conv layers), Softmax (Output)
- **Loss Function:** Cross-Entropy
- **Optimizer:** Adam (LR = 0.001)
- **Batch Size:** 32 (based on roll no.)
- **Epochs:** 10

---

## 🧪 Experiments & Results

### 🧼 Experiment 1 – 10-Class Classification

- **Train Accuracy:** 96.91%  
- **Test Accuracy:** 97.11%  
- **Total Parameters:** 4,670  
- **Confusion Matrix:** Plotted  
- **Classification Report:** Precision, Recall, F1 for each digit class

### 📉 Experiment 2 – 4-Class Classification

- **Train Accuracy:** 97.80%  
- **Test Accuracy:** 96.10%  
- **Total Parameters:** 4,448  
- **Confusion Matrix:** Plotted  
- **Classification Report:** Precision, Recall, F1 for each custom class

---

## 🎁 Bonus: Regularization & Performance Enhancement

To reduce overfitting and enhance generalization:

1. **L2 Regularization (Weight Decay):** `1e-4` applied in optimizer  
2. **Dropout Regularization:** Applied to the fully connected layer  
3. **Learning Rate Scheduler:** LR reduced by 0.9 every 5 epochs  
4. **Batch Normalization:** (Recommended for further improvement)  
5. **Hyperparameter Tuning:** Used to optimize batch size, LR, etc.


---

## 📚 Resources Used

- [PyTorch Official Tutorials](https://pytorch.org/tutorials/beginner/nlp/pytorch_tutorial.html)
- [PyTorch NN Docs](https://pytorch.org/docs/stable/nn.html)
- [CNN – GeeksforGeeks](https://www.geeksforgeeks.org/convolutional-neural-network-cnn-in-machine-learning/)
- Lecture slides provided by the course instructor

---

## 👨‍🎓 Author Info

```text
Name       : Sahil Sharma
Program    : M.Sc - M.Tech (Data and Computational Sciences)
Institute  : Indian Institute of Technology, Jodhpur
