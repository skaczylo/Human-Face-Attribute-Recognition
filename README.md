# Facial Attribute Prediction Project

This project focuses on predicting **age, race, and gender** from a dataset of **23,706 images** of people, each labeled with the corresponding categories.  
It applies **Explainable AI (XAI)** techniques to identify the most relevant features influencing the model's decisions.  
Since the dataset contains images of celebrities, each trained model is assigned a **unique name**.

The goal of this analysis is to develop a **high-performance neural network** for image classification.  
Different neural network architectures are tested to compare results and determine which models provide the best performance.  
**XAI techniques** are also used to extract insights into the most important features driving the classification decisions.

The following models are implemented for age, race, and gender classification from images:

- **Convolutional Neural Networks (CNNs)**  
  CNNs are well-suited for images because they capture **spatial patterns** such as edges, shapes, and textures using convolutional layers.

- **Multi-Layer Perceptrons (MLPs)**  
  MLPs are simpler and faster to train than CNNs when images are **flattened and normalized**, providing a baseline comparison.

- **k-Nearest Neighbors (k-NN)**  
  k-NN does not require training. It stores the dataset and classifies new images based on the **closest examples**, offering a non-parametric perspective.

Comparing these architectures allows evaluating **different approaches** and identifying the strengths and weaknesses of each method.

