# Flower Classification

This repository contains code for classifying flower images using a pre-trained InceptionV3 model, which is fine-tuned with additional convolutional layers and trained on a custom flower dataset. The project covers data preprocessing, augmentation, model training, evaluation, and conversion to a TensorFlow Lite model.

## Dataset

The dataset used in this project includes images of the following flower classes:
- Black Eyed Susan
- Calendula
- California Poppy
- Common Daisy
- Coreopsis
- Dandelion
- Iris
- Rose
- Sunflower
- Tulip

## Setup

To get started, clone the repository:

```bash
git clone https://github.com/ArkZ10/Plant-Classification.git
cd Plant-Classification
```

## Overview
This project explores various models for flower classification using transfer learning with InceptionV3 as the base model. Six different architectures were tested, with the configuration featuring InceptionV3 combined with LeakyReLU showing the best performance in terms of accuracy and efficiency. The trained model is then converted to TensorFlow Lite format for deployment on mobile and edge devices.

## Deployment
The final TensorFlow Lite model (flower_classification_optimized.tflite) is deployed to a mobile application for real-time flower recognition. The lightweight nature of TensorFlow Lite ensures efficient inference on devices with limited computational resources, making it suitable for mobile deployment.

## Results

The model achieves high accuracy in distinguishing between different flower species, as demonstrated in the training and validation accuracy and loss plots included in the notebooks. The LeakyReLU activation in the final model contributes to faster convergence and better generalization compared to other configurations.

## Usage
1. Clone the repository and navigate to the project directory.
2. Install the required packages using.
3. Run the provided scripts or notebooks to preprocess data, train the model, and evaluate it.
4. Use the converted TensorFlow Lite model (flower_classification_optimized.tflite) for deployment on mobile or edge devices.



## Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.






