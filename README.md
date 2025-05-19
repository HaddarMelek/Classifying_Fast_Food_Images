# Fast Food Image Classification Project

## Project Overview

This project focuses on classifying images of fast food into multiple categories using deep learning techniques. The goal is to implement and compare three types of models:

1. **Multilayer Perceptron (MLP)**
2. **Convolutional Neural Network (CNN)**
3. **Transfer Learning with Pretrained Models**

The dataset used is the [Fast Food Classification Dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/fast-food-classification-dataset), which contains labeled images of various fast food items.

---

## Project Objectives

- Load and preprocess image data (rescaling, resizing, label encoding).
- Build, train, and evaluate three different models for image classification.
- Apply techniques to reduce overfitting and improve model generalization.
- Compare the performance of the models based on accuracy and loss.
- Visualize training results with accuracy/loss curves and confusion matrices.

---

## Models Implemented

### 1. Multilayer Perceptron (MLP)
- Fully connected neural network with multiple hidden layers.
- Trained on flattened image data.
- Applied regularization techniques such as dropout to reduce overfitting.

### 2. Convolutional Neural Network (CNN)
- Used convolutional layers with max-pooling for feature extraction.
- Added fully connected layers for classification.
- Data augmentation and dropout were applied to improve performance.

### 3. Transfer Learning
- Leveraged pretrained models (MobileNetV2) as feature extractors.
- Tested two fine-tuning approaches:
  - Partial fine-tuning: unfreeze some top layers.
  - Full fine-tuning: train all layers.
- Compared results of both approaches.

---

## Tools and Technologies

- **Python 3**
- **TensorFlow / Keras**
- **NumPy, Pandas**
- **Matplotlib, Seaborn**
- **Scikit-learn**

---

## Dataset Details

- Images organized into training and validation folders by category.
- Dataset link: [Kaggle Fast Food Dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/fast-food-classification-dataset)

---


## Conclusion

The project demonstrates how different neural network architectures perform on the fast food image classification task. Transfer learning with fine-tuning provided the best accuracy and generalization. Techniques like dropout and data augmentation helped reduce overfitting, especially for the CNN model.

---

## Author

*HADDAR Melek*  

