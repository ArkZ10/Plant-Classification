# Flower Classification

This repository contains code for analyzing six different models for flower image classification using Convolutional Neural Networks (CNNs). The goal is to select the best model for accurately classifying flowers.

### Project Structure
- `Dataset Folder` : Folder containing flower dataset for training the models.
- `Pre-Trained Model` : Folder containing pre-train model like **ResNet50**
- `Test Dataset Folder` : Folder containing flower dataset for testing the models.
- `myApp` : Folder containing files for deploying the flower classification model into an application.
- `Flower_Classification.tflite` : TensorFlow Lite model for flower classification, optimized for mobile and embedded devices.
- `Flower_Classification_Optimized.tflite` : Optimized version of the TFLite model for improved performance and efficiency.
- `Flower_model.ipynb` : Jupyter Notebook containing the entire workflow from data preprocessing to model training and evaluation.
- `Readme.md` : This file, providing an overview of the project.

### Dataset

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

### Data Preprocessing & Augmentation
- Data preprocessing and data augmentation steps include normalization `rescale=1./255.0` and artificial variations:
  - Rotation `rotation_range=40` – Rotates images randomly up to 40 degrees.
  - Shifts `width_shift_range`, `height_shift_range` – Moves images slightly in horizontal/vertical directions.
  - Zoom `zoom_range=0.2` – Zooms in or out on images.
  - Shear `shear_range=0.2` – Applies slanting transformation.

### Model Building
- **Model 1** : This Convolutional Neural Network (CNNs) model begins with a Conv2D layer and followed by a MaxPooling 2D layer
- **Model 2** : This model leverages transfer learning with InceptionV3, a pretrained Convolutional Neural Network (CNNs)
- **Model 3** : This Convolutional Neural Network (CNNs) model begins with a Conv2D layer and followed by a MaxPooling 2D layer and using LeakyRelu as its activation function
- **Model 4** : This model leverages transfer learning with InceptionV3 and uses LeakyRelu as its activation function
- **Model 5** : This model leverages transfer learning with VGG16 as a feature extractor.
- **Model 6** : This model leverages transfer learning with ResNet50, a pretrained Convolutional Neural Network (CNNs)

### Evaluation
- Model 2 & Model 4 (InceptionV3-based models) achieve the highest accuracy, indicating that InceptionV3 is highly effective for flower classification. Model 4 performs slightly better, likely due to the addition of LeakyReLU.
- Model 5 (VGG16-based) also performs well but doesn’t reach the same level as InceptionV3 models, suggesting that VGG16 might not be as optimal for this task.
- Model 1 & Model 3 (custom CNNs) show moderate accuracy improvements. Model 3, which includes LeakyReLU, outperforms Model 1, highlighting the benefits of using LeakyReLU over standard activation functions.
- Model 6 (ResNet50-based) performs the worst, struggling to reach high accuracy. This could be due to overfitting, improper fine-tuning, or incompatibility with the dataset.

## Instructions

To run the notebook (`Flower_model.ipynb`)

1. clone the repository:
   ```bash
   git clone https://github.com/ArkZ10/Plant-Classification.git
   cd Plant-Classification
   ```
2. Open the notebook in Jupyter or Google Colab and execute each cell sequentially.

## Deployment
The final TensorFlow Lite model (flower_classification_optimized.tflite) is deployed to a mobile application for real-time flower recognition. The lightweight nature of TensorFlow Lite ensures efficient inference on devices with limited computational resources, making it suitable for mobile deployment.

## Usage
1. Clone the repository and navigate to the project directory.
2. Install the required packages using.
3. Run the provided scripts or notebooks to preprocess data, train the model, and evaluate it.
4. Use the converted TensorFlow Lite model (flower_classification_optimized.tflite) for deployment on mobile or edge devices.

## Author

[Yeftha Joshua Ezekiel](https://github.com/ArkZ10)



