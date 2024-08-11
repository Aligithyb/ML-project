**Activity Classification and Regression Analysis**
This repository contains Python code for classifying activities and performing regression analysis using machine learning models. The analysis is conducted on a dataset from the MHEALTH (Mobile Health) dataset, focusing on various classification algorithms and linear regression.

**Objective**
To develop a comprehensive understanding of various machine learning concepts by applying various machine learning models to a dataset and evaluating their performance.

**Overview**
The provided code performs the following tasks:

**Data Preprocessing: Handles missing values, scales features, and splits the data.**
**Model Training: Trains several classification models including K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Neural Networks, and Logistic Regression.**
**Model Evaluation: Evaluates the models using cross-validation and performance metrics.**
**Regression Analysis: Applies Linear Regression to assess continuous prediction performance.**
Dependencies
pandas
numpy
matplotlib
scikit-learn
Ensure you have these libraries installed. You can install them using pip:

pip install pandas numpy matplotlib scikit-learn
Code Description
1. **Data Preprocessing**
preprocess_data(data): Cleans and prepares the data by handling missing values and scaling features.
preprocess_subset(X, y, subset_size): Creates a subset of the training data and scales it.
**2. Model Training**
Classification Models:

K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Neural Network (MLPClassifier)
Logistic Regression
Hyperparameter Tuning: Uses RandomizedSearchCV to find the best hyperparameters for each model.

**3. Model Evaluation**
Metrics: Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.
Cross-Validation: Computes cross-validation scores for each classification model.
**4. Regression Analysis**
Linear Regression: Evaluates performance using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
Results
Interpretation of Model Predictions:

All models demonstrated high accuracy, precision, recall, and F1 scores, indicating effective classification of activities.
Best-Performing Model:

The SVM model achieved the best metrics across various performance indicators, making it the most suitable model for this dataset.
Strengths and Weaknesses of Each Model:

KNN: Simple and effective with non-linear data but struggles with large datasets and redundant features.
SVM: High accuracy with the ability to find optimal hyperplanes, but requires significant computational resources.
Neural Network: Capable of recognizing complex patterns, requires large datasets.
Logistic Regression: Simple and efficient but can overfit with high-dimensional data.
Insights:

Model performance is influenced by design, complexity, data shape, and feature distribution. SVM's ability to handle high-dimensional data and find optimal hyperplanes was particularly effective.
Conclusion
The SVM model proved to be the best for activity recognition in the MHEALTH dataset, achieving the highest accuracy, precision, recall, and F1 scores. This highlights its suitability for complex activity recognition tasks.

**Usage**
To run the code:

Place your dataset (e.g., mhealth_raw_data.csv) in the same directory.
Execute the script using Python:
python your_script_name.py

Acknowledgements
MHEALTH dataset
Scikit-learn and other libraries used for machine learning and data processing
