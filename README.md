# Loan Approval Risk Prediction

This project implements various advanced Machine Learning models to predict loan approval risk. The models are trained on a comprehensive dataset containing both numerical and categorical features. Special focus has been placed on handling severe class imbalance using SMOTE, downsampling, and state-of-the-art ensemble techniques.

## Overview
The primary objective of this project is to accurately predict whether a loan applicant poses a default risk (indicated by the `Risk_Flag` column). To achieve high precision and recall on the minority class, the following models have been implemented and compared:
* XGBoost
* RandomForestClassifier
* CatBoost
* BalancedRandomForestClassifier
* RUSBoostClassifier
* Stacking Classifier (Ensemble)

Since the dataset is heavily imbalanced (skewed towards non-defaulters), specialized techniques like SMOTE, class-weighting, and custom sampling classifiers are utilized to maximize model performance.

## Dataset
The dataset comprises both structural financial data and applicant demographics:
* **Numerical Features:** Income, Age, Experience, Current_Job_Yrs, Current_House_Yrs, etc.
* **Categorical Features:** Profession, Gender, City, State, Marital Status, House_Ownership, and Car_Ownership.
* **Target Variable:** `Risk_Flag` (1 indicates high default risk, 0 indicates low risk/approved).

## Preprocessing & Pipeline
To prepare the data for the algorithms, a robust data preprocessing pipeline was built:
* **Feature Scaling:** Applied `StandardScaler` to ensure numerical variables are on the same scale.
* **Categorical Encoding:** One-Hot Encoding used to convert categorical strings into machine-readable format.
* **Dimensionality Reduction:** Applied `TruncatedSVD` to reduce the high dimensionality caused by one-hot encoding without losing significant variance.
* **Class Imbalance Mitigation:** Utilized Synthetic Minority Over-sampling Technique (`SMOTE`) to intelligently balance the target class distribution.

## Modeling Strategy
The evaluation is split into multiple stages to find the absolute best predictor:
1. **Standard Classifiers:** XGBoost, RandomForest, and CatBoost were trained initially on the raw data, and then fine-tuned after applying `SMOTE`.
2. **Imbalanced-Specific Classifiers:** `BalancedRandomForestClassifier` and `RUSBoostClassifier` were tested to natively handle the skewed weights.
3. **Hybrid Ensemble (Stacking):** A `StackingClassifier` meta-model was created by combining the predictions of the top-performing base estimators (XGBoost, RandomForest, and CatBoost) to optimize final accuracy and F1-score.

## Evaluation Metrics
Models are strictly scrutinized based on metrics that matter for imbalanced data:
* **Confusion Matrix:** To track False Positives and False Negatives closely.
* **Classification Report:** Detailed analysis of Precision, Recall, and F1-Score (specifically focusing on Class 1).
* **Stratified Cross-Validation:** To verify consistency and prevent overfitting across different folds of data.

---
**Author:** mahendrasiyag



