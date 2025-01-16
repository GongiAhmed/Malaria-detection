# Detecting Malaria in Cell Images Using CNN
<img src="https://github.com/GongiAhmed/Malaria-detection/blob/main/Detecting%20Malaria%20using%20CNN/output.png">
This project demonstrates the use of a Convolutional Neural Network (CNN) model to classify microscopic images of blood smears as either containing malaria parasites ("Parasite") or being uninfected ("Uninfected"). The model is trained and evaluated using a dataset derived from the original Kaggle Malaria dataset, but significantly reduced in size to allow for easier and quicker experimentation.

## Overview

Malaria remains a critical global health concern, and accurate detection of the disease is crucial for effective treatment and control. This project explores a machine learning approach, using CNNs, to aid in this process through image analysis of blood smears.

**Key Features:**

-   Implementation of a CNN model using TensorFlow/Keras.
-   Training and evaluation on a reduced version of a malaria cell image dataset.
-   Clear explanation of the problem, approach, and results.
-   Data loading and preparation using TensorFlow.
-   Data optimization using caching, shuffling, and prefetching.
-   Data augmentation techniques including random flipping and rotation for improved model generalization.
-   Model evaluation metrics using loss and accuracy on a validation set.
-   Sample image visualization along with predicted and actual class labels, and confidence scores.
-   The model can be saved and reused later for further prediction and training.

## Dataset

The dataset consists of microscopic images of blood smears categorized into "Parasite" (cells infected with malaria parasites) and "Uninfected" (healthy cells). Due to resource constraints, this project uses a much smaller subset of the original dataset:

### Training Data

*   **Parasite**: 220 images
*   **Uninfected**: 196 images

### Testing Data

*   **Parasite**: 91 images
*   **Uninfected**: 43 images

## Technologies Used

-   **Python:** Core programming language
-   **TensorFlow/Keras:** Deep learning framework for CNN implementation
-   **NumPy, Pandas:** Data manipulation and analysis
-   **Matplotlib:** Data visualization
-   **Kaggle:** Integrated development environment and dataset platform

## Model Architecture

The CNN architecture consists of the following layers:

1.  Resizing and Rescaling Layers: To ensure all images are of the same size and rescaled pixel values
2.  Data Augmentation Layers: To improve the diversity of the data
3.  Convolutional Layers with ReLU activations to learn the features from the images.
4.  Max Pooling Layers to reduce the spatial dimensions and for more robust feature learning
5.  Flatten Layer to transform the output into a 1-D vector.
6.  Dense (fully connected) layers with ReLU activation and a final Dense layer with Sigmoid activation for the classification
\
The model had a total of 29,449,349 parameters, with 9,816,449 trainable parameters.

## Results
<img src="https://github.com/GongiAhmed/Malaria-detection/blob/main/Detecting%20Malaria%20using%20CNN/results.png"> 
After 30 epochs of training, the model achieved a **test accuracy of 88.06%**. The training and validation loss and accuracy were tracked throughout the training process. Sample images are visualized with their actual and predicted classes and confidence scores to show the performance of the trained model.
