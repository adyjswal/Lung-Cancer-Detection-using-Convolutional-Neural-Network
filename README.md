# Lung-Cancer-Detection-using-Convolutional-Neural-Network

# Project Overview

This project aims to detect lung cancer from CT scans using machine learning algorithms. 

With early detection, the model can aid healthcare professionals in diagnosis, potentially improving patient outcomes.

# Tech Stack


Python

TensorFlow/Keras or PyTorch for CNN modeling

OpenCV for image preprocessing

scikit-learn for evaluation metrics

Pandas, NumPy for data handling

# Importing Dataset


The dataset which we will use here has been taken from -https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images.

This dataset includes 5000 images for three classes of lung conditions:

Normal Class

Lung Adenocarcinomas

Lung Squamous Cell Carcinomas

These images for each class have been developed from 250 images by performing Data Augmentation on them.

That is why we won’t be using Data Augmentation further on these images.


# Installation

Copy code

Clone the repository:

git clone https://github.com/your-username/lung-cancer-detection.git

# Install dependencies:


Copy code

pip install -r requirements.txt

# Model Pipeline

Data Preprocessing: Lung segmentation and normalization.

Feature Extraction: Radiomics features and/or CNN-based features.

Model Training: Training CNN or other classifiers on preprocessed data.

Prediction: Predicts the likelihood of malignancy.

# Evaluation


Confusion Matrix

AUC-ROC Curve

Precision, Recall, F1-score

# Usage


To run the project:

Preprocess images.

Train the model by running:

Copy code

python train_model.py

Run predictions on new data with:

Copy code

python predict.py

# Results


Include sample images or performance metrics demonstrating the model’s accuracy and reliability.

# Future Work


Potential improvements include experimenting with transfer learning, using more advanced segmentation techniques, and enhancing the user interface.
