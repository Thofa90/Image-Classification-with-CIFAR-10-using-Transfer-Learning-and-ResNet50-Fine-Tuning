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

2. **Data Preprocessing**
   
    1.	Normalization
    
	  •	All images were normalized by dividing pixel values by 255.0, scaling them to the range [0, 1].

	  •	Benefits: Improves convergence speed and stability during training.

	  •	Post-normalization stats:

	    •	Train images shape: (10,000, 32, 32, 3)
	    •	Test images shape: (10,000, 32, 32, 3)
	    •	Mean pixel value: 0.4747
	    •	Pixel standard deviation: 0.2525

    2.	Label Preparation
       
	•	Flattened label arrays to ensure compatibility with sparse loss functions during training.

    ✅ Outcome: Data is clean, scaled, and formatted correctly for CNN-based image classification.

3. **Baseline Model**
   
   **🔹 Pre-trained Model: ResNet50 Setup**

   We used ResNet50 (pre-trained on ImageNet) as the base model for CIFAR-10 classification, following these steps:

	 1.	Load Pre-trained ResNet50
 
	    •	Imported ResNet50 with weights='imagenet', excluding the top classification layer (include_top=False) to use it as a feature extractor.
    	
	    •	Input shape set to (32, 32, 3) for CIFAR-10 images.
	
	 2.	Freeze Base Layers
 
	    •	Set base_model.trainable = False to retain the pre-trained weights and avoid updating them in the initial training phase.
	
	 3.	Add Custom Classification Head
 
	    •	GlobalAveragePooling2D → Converts feature maps into a single vector.
    	
	    •	Dense(512, relu) → First fully connected layer for feature learning.
    	
	    •	Dense(256, relu) → Second hidden layer for deeper representation.
    	
	    •	Dense(10, softmax) → Output layer for 10 CIFAR-10 classes.
	
	 4. Compile the Model
 
	    •  Optimizer: Adam
   
	    •  Loss: Sparse Categorical Crossentropy (for integer labels)
   
	    •  Metric: Accuracy
	
	 5.	Train the Head
 
	    • Trained the custom head for 10 epochs with a batch size of 64, keeping the base model frozen.

    ✅ Purpose: This approach leverages ResNet50’s powerful pre-trained features, while allowing the custom top layers to adapt specifically to CIFAR-10.
	
5. **Baseline Model evaluation**

   📊 Baseline Model Results (Frozen ResNet50 + Custom Head)

   **Training Performance:**
 
	•	Train Accuracy: 33.7%
	•	Validation Accuracy: 30.6%
	•	Train Loss: 1.8272
	•	Validation Loss: 1.9234

   **Progress Analysis:**
 
	•	✅ Accuracy improved from ~13% (random guessing) to ~33% — showing the model learned basic patterns.
	•	⚠️ Validation accuracy plateaued after epoch 4–5 (~30–32%), suggesting the head reached its learning limit.
	•	⚠️ Validation loss began increasing → early signs of overfitting with frozen base layers.

   **Conclusion & Next Steps:**
 
	•	The frozen ResNet50 base provided useful features, but training only the custom head was insufficient.
	•	To improve:
	  1.	Unfreeze deeper ResNet50 layers for fine-tuning.
	  2.	Apply data augmentation to improve generalization.
	  3.	Train longer with early stopping to prevent overfitting.

	
6. **Fine-Tuning**
   
**🔧 Fine-Tuning the Baseline ResNet50 Model**

Steps Taken

	1.	Unfrozen the Base Model
 
	    •	Allowed deeper ResNet50 layers to be trainable so the model could adapt pretrained features to CIFAR-10.
	 
	2.	Added Two More Hidden Layers + Dropout
 
		•	Purpose of Hidden Layers:
		•	Learn more abstract patterns from ResNet50 output features.
		•	Improve feature interaction for complex class distinctions (e.g., cat vs dog).
		•	Purpose of Dropout:
		•	Reduce overfitting by randomly disabling neurons during training.
	    •	Improve generalization to unseen data.
	 
	3.	Reduced Learning Rate
 
		•	Optimizer: Adam with learning_rate=1e-5 for stable transfer learning.
		•	Loss Function: sparse_categorical_crossentropy for multi-class classification.
		•	Metrics: Accuracy for training & validation monitoring.
  
	4.	Applied Data Augmentation + Early Stopping
 
		•	Data Augmentation Benefits:
		•	Expands training set via random transformations.
		•	Prevents overfitting & improves robustness to real-world variations.
		•	Early Stopping Benefits:
		•	Stops training when validation performance stops improving.
		•	Saves time & restores best model weights automatically.
  
	5.	Increased Epochs
 
		•	Trained for more epochs to allow deeper learning, balanced with early stopping to avoid overfitting.
  
7. **Evaluation after fine-tuning**

After fine-tuning the ResNet50 model with additional hidden layers, dropout regularization, reduced learning rate, data augmentation, and early stopping, the model achieved 71.04% accuracy on the test set.
 
**📊 Model Evaluation & Results**

 **Training vs Validation Accuracy**
 
![Training vs Validation Accuracy](assets/train_val_accuracy.png)

	•	Observation: Accuracy steadily improved for both training and validation sets.
	•	Key Insight: The gap between training and validation accuracy is small, indicating reduced overfitting due to dropout and data augmentation.
	•	Peak Validation Accuracy: ~71%

**Training vs Validation Loss**

![Training vs Validation Loss](assets/train_val_loss.png)

	•	Observation: Loss decreases steadily without significant divergence between training and validation curves.
	•	Key Insight: Model generalized well and avoided severe overfitting.

**Confusion Matrix**

![Confusion Matrix](assets/confusion_matrix.png)

	•	Observation:
	•	High accuracy for frog (class 6), ship (class 8), and automobile (class 1).
	•	Lower performance for cat (class 3) and bird (class 2) — these may share visual similarities with other classes (e.g., dog, deer).
	•	Key Insight: Misclassifications often occur between visually similar classes.

**Classification Report**

![Classification Report](assets/classification_report.png)

	•	Macro Avg F1-Score: 0.704
	•	Highest Precision: Ship (0.8828)
	•	Highest Recall: Frog (0.8920)
	•	Lowest F1: Cat (0.4988) → needs targeted improvement.

**Metrics by Class**

![Metrics by Class](assets/metrics_by_class.png)

	•	Observation: Most classes maintain balanced precision and recall.
	•	Insight: Variations between precision and recall highlight the trade-off between false positives and false negatives per class.
 
![Metrics by Class](assets/all_metric.png)


✅ Final Summary

	•	Final Test Accuracy: 71.04%
	•	Strengths: Strong performance on structured, distinctive classes like automobiles, ships, and frogs.
	•	Weaknesses: Struggles with classes that have high intra-class variation (cats, birds).
	•	Future Work:
	•	Class-specific data augmentation (e.g., more varied cat/bird images).
	•	Fine-tuning additional layers in ResNet50.
	•	Experimenting with learning rate schedules.

8. **Evaluation after further fine-tuning**

**📌 Updated CIFAR-10 Image Classification Model (88.82% Accuracy)**

**🚀 Update Summary**

In this updated version of the CIFAR-10 classification project, two major improvements were implemented based on previous analysis and suggestions:

1. **Resize Images to 96×96**
   
   - The original dataset consists of 32×32 images.
   - Resizing to 96×96 allows the pre-trained ResNet50 model to capture richer spatial and texture features.
   - This step aligns better with the model's original training resolution, boosting feature extraction quality.

3. **Balanced Class-Weighted Loss**
   
   - Applied `class_weight` parameter during training to handle any class imbalance.
   - Helps the model give fair importance to underrepresented classes.
   - Reduces bias towards dominant classes and improves macro average metrics.

---

**📊 Results After Improvements**

- **Final Accuracy:** **88.82%** ✅

  ![Train_Validation_Accuracy](assets/2nd_train_val_accuracy.png)
  ![Train_Validation_loss](assets/2nd_train_val_loss.png)
  ![Classification report](assets/2nd_classification.png)
  
- Significant boost from previous fine-tuned accuracy (**71.04%**).
- Noticeable improvement across all classes in **Precision, Recall, and F1-score**.
- Reduced misclassification between visually similar classes (e.g., cat vs dog, automobile vs truck).

---

## 📂 Colab Notebook

You can run the updated version in Google Colab here:
This part is only the base model with the unfreeze option.

🔗 **[Open in Colab]([YOUR_COLAB_LINK_HERE](https://colab.research.google.com/drive/1ihXbcwJw1KsqEhGQY5ChkOigSSfGl5Lw?usp=sharing))**

---

**🔍 Key Benefits of These Changes**

- **Higher Resolution Input** → More detailed feature maps, better object boundaries detected.
- **Class-Weighted Loss** → Improved fairness in classification results, higher macro F1-score.
- **Better Generalization** → Model performs well on both seen and unseen data.

---

**📌 Next Steps**

- Experiment with **EfficientNetB3** for even better performance with 96×96 input.
- Use **mixup** or **cutmix** augmentation to further improve robustness.
- Try **learning rate scheduling** for optimized convergence.

---
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
