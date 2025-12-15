# Facial Attribute Prediction Project

This project focuses on predicting **age, race, and gender** from a dataset of **23,706 images** of people, each labeled with the corresponding categories.  
We are particularly interested in applying **Explainable AI (XAI)** techniques to identify the most relevant features that influence the model's decisions.  
Additionally, since the dataset contains images of celebrities, we have decided to assign a **unique name to each trained model**.

The goal of this analysis is to develop a **neural network capable of high performance** in image classification.  
We propose testing **different neural network architectures** to compare results and determine which models provide the best performance.  
**XAI techniques** will also be applied to extract insights into the most important features driving the classification decisions.

We decided to implement the following models to tackle the problem of age, race, and gender classification from images:

- **Convolutional Neural Networks (CNNs)**  
  CNNs are ideal for working with images because they capture **spatial patterns** such as edges, shapes, and textures using convolutional layers.

- **Multi-Layer Perceptrons (MLPs)**  
  MLPs are simpler and faster to train than CNNs when images are **flattened and normalized**, providing a baseline comparison.

- **k-Nearest Neighbors (k-NN)**  
  k-NN does not require training. It stores the dataset and classifies new images based on the **closest examples**, offering a non-parametric perspective.

By comparing these architectures, we can evaluate **different approaches** and identify the strengths and weaknesses of each method.
