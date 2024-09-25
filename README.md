 # Supervised Learning Project Summary

## Overview
This project focuses on supervised learning techniques to classify images using deep learning models, specifically utilizing a pre-trained ResNet-18 model. The primary goal is to achieve high accuracy in classifying images from a designated dataset.

## Project Structure
1. **Data Preparation**: 
   - Images are loaded and transformed using normalization and resizing techniques to ensure consistency in the input size.

2. **Dataset Loading**:
   - The dataset is split into training and testing sets, organized in a directory structure compatible with PyTorch's `ImageFolder`.

3. **Model Initialization**:
   - A pre-trained ResNet-18 model is employed, with the final layer modified to match the number of classes in the dataset.

4. **Training Process**:
   - The model is trained using the Adam optimizer and CrossEntropy loss function over multiple epochs. 

5. **Evaluation**:
   - The model's performance is assessed on the test set, calculating the accuracy by comparing predicted labels against true labels.

6. **Visualization**:
   - Sample images and their predicted labels are visualized to provide insights into the model's predictions.



# Self-Supervised Learning Project Summary

## Overview
This section of the project implements a self-supervised learning approach using a custom convolutional neural network (CNN) to classify images into permutations. The model is trained on a dataset of jigsaw puzzle images, where the task is to rearrange shuffled image patches back to their original configuration.

## Key Components

### 1. Network Definition
- A CNN is defined using PyTorch, featuring several convolutional layers followed by fully connected layers. The network processes 9 image patches to classify them into 500 permutation classes, effectively solving the jigsaw puzzle problem.

### 2. Data Loading
- The `DataLoader` class loads training and validation datasets from specified directories, facilitating efficient data handling and augmentation.

### 3. Model Training
- The model is trained over 50 epochs using the Adam optimizer and cross-entropy loss. T

### 4. Model Evaluation
- After training, the model's performance is evaluated on a validation set, calculating the overall accuracy and displaying the predictions compared to actual labels.

### 5. Image Visualization
- The first image from the training set is visualized before and after permutation to illustrate the model's capability to identify and order image patches, demonstrating the solution to the jigsaw puzzle task.

### 6. Linear Classifier
- A linear classifier is constructed using the features extracted from the convolutional layers. This classifier is trained and evaluated to predict food categories, utilizing a similar training and validation framework.


## For additional information about the project the paper is available 

