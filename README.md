# Diabetes Prediction using Machine Learning

## Overview
This project focuses on predicting diabetes using various machine learning techniques. The goal is to create an accurate predictive model based on patient attributes such as age, gender, BMI, medical history, and more. The dataset used in this project is sourced from Kaggle.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Classification](#classification)
- [Results](#results)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Diabetes is a chronic disease with significant health implications if not managed properly. Early detection is critical for effective management and prevention of complications. This project aims to leverage machine learning algorithms to predict the likelihood of diabetes based on a set of medical and demographic features.

## Dataset
The dataset used for this project includes the following features:
- Gender
- Age
- Hypertension
- Heart Disease
- Smoking History
- BMI (Body Mass Index)
- HbA1c Level
- Blood Glucose Level

The dataset is cleaned, preprocessed, and normalized before applying machine learning models.

## Methodology

### Data Preprocessing
1. **Dataset Cleaning:**
   - Handling missing values, duplicates, and correcting data types.
2. **Feature Analysis and Outlier Removal:**
   - Method 1: Using standard deviation.
   - Method 2: Using Interquartile Range (IQR) method.
3. **Normalization and Feature Transformation:**
   - Normalizing the data for better performance of machine learning models.
4. **Data Correlation:**
   - Analyzing the correlation between features to understand their impact on diabetes prediction.

### Classification
1. **Class Imbalance Handling:**
   - Miscellaneous methods.
   - Oversampling (K-means SMOTE).
   - Undersampling (Edited Nearest Neighbors, Neighborhood Cleaning Rule).
   - Hybrid methods (SMOTE-Tomek, SMOTE-ENN).
2. **Machine Learning Models:**
   - Logistic Regression
   - Decision Tree
   - Support Vector Machine (SVM)
   - Gaussian Naive Bayes
   - Multi-layer Perceptron
   - Random Forest

## Results
The results are evaluated based on different outlier detection methods:
1. **Outlier Detection Method 1:**
   - Standard deviation method results.
2. **Outlier Detection Method 2:**
   - Interquartile Range (IQR) method results.

## Conclusion
The project successfully demonstrates the use of machine learning techniques for diabetes prediction. The preprocessing steps and the handling of class imbalance significantly improve the performance of the models. Future work could involve more advanced feature engineering and the inclusion of additional medical parameters.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd diabetes-prediction
   ```
 3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open and execute the diabetes_prediction.ipynb notebook.

### Requirements
- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib

### Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
