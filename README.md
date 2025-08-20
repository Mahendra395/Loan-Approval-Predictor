# Loan Approval Prediction
This project implements various machine learning models to predict loan approval risk. The models are trained on a dataset containing both numerical and categorical features, with a focus on handling class imbalance using SMOTE and ensemble techniques.

## Overview
The goal of this project is to predict whether a loan applicant is at risk (denoted by the Risk_Flag column) using various machine learning models. The models include:<br>
XGBoost<br>
RandomForestClassifier<br>
CatBoost<br>
BalancedRandomForestClassifier<br>
RUSBoostClassifier<br>
Stacking Classifier<br>
The dataset used is heavily imbalanced, and techniques like SMOTE and class weighting are employed to handle this issue.

## Dataset
The dataset contains the following features:<br>
**Numerical:** Various financial and demographic data.<br>
**Categorical:** Information such as profession, gender, and city.<br>
The target variable is Risk_Flag, indicating whether an applicant is at risk of loan default.

## Preprocessing
**Feature Scaling:** Numerical features are scaled using StandardScaler.<br>
**Encoding:** Categorical features are one-hot encoded.<br>
**Dimensionality Reduction:** Truncated SVD is applied to reduce the dimensionality of the data.<br>
**Handling Imbalance:** SMOTE is used to oversample the minority class.<br>

## Modeling
Several models are trained and evaluated:<br>
**XGBoost, RandomForestClassifier, and CatBoost:** Trained with and without SMOTE.<br>
**BalancedRandomForestClassifier and RUSBoostClassifier:** Specifically designed for handling imbalanced data.<br>
**Stacking Classifier:** Combines the predictions of XGBoost, RandomForest, and CatBoost for a final prediction.<br>

## Evaluation
The models are evaluated using:<br>
**Confusion Matrix:** To visualize the performance of each class.<br>
**Classification Report:** Provides precision, recall, f1-score, and support for each class.<br>
**Cross-Validation Accuracy:** Evaluates the model's performance across different folds of the data.



