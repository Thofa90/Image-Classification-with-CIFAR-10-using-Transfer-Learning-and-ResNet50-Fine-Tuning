# 🖼️ Image Classification with CIFAR-10 using Transfer Learning (ResNet50)

## 📌 Project Overview

This project focuses on building a deep learning-based **image classifier** for the CIFAR-10 dataset using **transfer learning** with **ResNet50** pre-trained model.  
We start with a **baseline model** (31.76% accuracy) and progressively fine-tune it using:
- Unfreezing ResNet50 layers
- Adding custom dense layers with dropout
- Lowering the learning rate
- Applying data augmentation
- Using early stopping to prevent overfitting

The final model achieved **88.82% accuracy** on the test set.

---

## 🎯 Goal

To develop a **highly accurate, generalizable** image classification model by leveraging **transfer learning** and advanced training techniques, making it applicable to real-world scenarios such as product categorization, surveillance, and autonomous systems.

---

## 💼 Business Context

Image classification is critical in:
- **Retail** (automatic product tagging)
- **Automotive** (object detection in self-driving cars)
- **Healthcare** (medical imaging classification)
- **Security** (identifying threats in real time)

By applying deep learning with transfer learning, companies can leverage pre-trained models on massive datasets and adapt them to their own classification tasks without starting from scratch — saving both time and computational resources.

---

## 🌍 Real-World Applications

- **E-commerce**: Categorizing products from images automatically.
- **Autonomous Vehicles**: Detecting traffic objects and hazards.
- **Medical Imaging**: Classifying medical scans for diagnostics.
- **Social Media**: Filtering inappropriate image content.

---

## 📂 Dataset

The project uses the **CIFAR-10** dataset:
This **dataset is also available** in TensorFlow & Keras. 
Here is how you can import it: from tensorflow.keras.datasets import cifar10
- **60,000** 32x32 RGB images
- **10 classes**, each with **6,000 images**
- **50,000 training** images and **10,000 test** images

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

---

## 📊 Project Workflow

1. **EDA Summary**
   
   **- Dataset shape & class distribution**
     
       •    Training set: 10,000 RGB images, size 32×32×3
     
	   •	Test set: 10,000 RGB images, size 32×32×3

	   •	Classes: 10 mutually exclusive categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

       •	Balanced across categories, ranging from 937 to 1,032 images per class.
       
  **- Sample image visualization**
   **- Pixel intensity analysis**
     
     	•	Most pixel values range from 50 to 150 (moderate brightness).
     
	    •	Few pixels are near 0 (pure black) or 255 (pure white).
     
	    •	Indicates well-lit, balanced images without extreme contrast.
     
        •	Red & Green channels peak around 100–120, fairly symmetric.
     
	    •	Blue channel slightly darker overall, peaking near 100.
     
	    •	No strong color bias — suitable for CNN training.
     
   **- Data Quality and Basic Statistics**
     
     •	No missing, NaN, or Inf values.
     
	 •	All images have consistent size (32×32×3).

     •	Mean pixel value: 121.04
     
	 •	Pixel standard deviation: 64.39

3. **Data Preprocessing**
   - Image normalization
   - Data augmentation
4. **Baseline Model**
   - ResNet50 feature extraction (frozen layers)
   - Dense layers for classification
5. **Fine-Tuning**
   - Unfreezing layers
   - Adding more hidden layers & dropout
   - Reducing learning rate
   - Applying early stopping
6. **Evaluation**
   - Accuracy/Loss curves
   - Confusion matrix
   - Classification report

---

## 🚀 Usage

### 1️⃣ Run in Google Colab
Click below to run the notebook directly in Colab:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YourGitHubUser/Image-Classification-CIFAR10-ResNet50/blob/main/ImageClassification_CIFAR10_CV_Project.ipynb)

---

### 2️⃣ Run Locally
```bash
git clone https://github.com/YourGitHubUser/Image-Classification-CIFAR10-ResNet50.git
cd Image-Classification-CIFAR10-ResNet50
pip install -r requirements.txt
jupyter notebook ImageClassification_CIFAR10_CV_Project.ipynb
