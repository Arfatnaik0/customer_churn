# import necessary libraries
import numpy as np
import pandas as pd

# import ml libraries
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# load dataset
churn = pd.read_csv('..\\data\\WA_Fn-UseC_-Telco-Customer-Churn.csv')

# data cleaning and preprocessing
churn['TotalCharges']=pd.to_numeric(churn['TotalCharges'],errors='coerce')
churn['TotalCharges']=churn['TotalCharges'].fillna(churn['TotalCharges'].median())
churn.drop('customerID',axis=1,inplace=True)

# replace 'No phone service' and 'No internet service' with 'No'
replace_dict = {'No phone service': 'No', 'No internet service': 'No'}
cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for col in cols:
    churn[col] = churn[col].replace(replace_dict)


# x and y split
x=churn.drop('Churn', axis=1)
y=churn['Churn'].map({'Yes': 1, 'No': 0})

# identify numerical and categorical columns
num_cols=['tenure','MonthlyCharges','TotalCharges']
cat_cols=[column for column in x.columns if column not in num_cols]

# preprocessing pipelines for both numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)

# complete pipeline with logistic regression
model=Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced'))
])

# fit the model
model.fit(x,y)

# save the model
joblib.dump(model,'..\\model\\churn_model.pkl')




