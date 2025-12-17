# Customer Churn Prediction

## Overview
This project predicts whether a telecom customer is likely to churn using machine learning. Customer churn is an important business problem because retaining existing customers is more cost-effective than acquiring new ones. The project demonstrates an end-to-end data science workflow.

## Dataset
**Source:** Telco Customer Churn Dataset (Kaggle)

The dataset contains customer demographics, service usage, billing information, and a churn indicator.

## Project Structure
Customer-Churn/
├── notebook/
│ └── churn.ipynb
├── src/
│ └── churn_model.py
├── data/
│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── plots/
│ └── all the plots 
├── README.md
└── requirements.txt

## Plots
![alt text](plots/Corr_matrix.png)
![alt text](plots/Most_imp_features.png)

## Approach
- Cleaned and preprocessed the dataset
- Converted categorical variables using one-hot encoding
- Handled class imbalance using class weights
- Trained Logistic Regression, Decision Tree, and Random Forest models
- Used probability-based predictions with threshold tuning
- Used 0.4 treshold to ensure more recall
- Evaluated models using precision, recall, F1-score, confusion matrix, and ROC-AUC

## Results
- Logistic Regression achieved the highest ROC-AUC (approximately 0.86)
- Logistic Regression achieved high recall for churn customers (approximately 0.88)
- Random Forest provided more balanced predictions with fewer false positives

Logistic Regression was selected as the best model when the goal is to identify the maximum number of churn customers.

## Key Insights
- Customers with shorter tenure are more likely to churn
- Contract type and monthly charges have a strong impact on churn
- Feature importance from Random Forest highlights the main churn drivers

## Technologies Used
Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn

## How to Run
1. Install dependencies:

2. Run the notebook for analysis:

3. Run the model training script:

## Conclusion
This project demonstrates how machine learning can be applied to predict customer churn while aligning model evaluation with business objectives.
