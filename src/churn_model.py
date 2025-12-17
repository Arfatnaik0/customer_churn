# import necessary libraries
import numpy as np
import pandas as pd

# import visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

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


churn=pd.get_dummies(churn,
                     columns=['gender','Partner','Dependents','PhoneService',
                              'MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                              'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
                              'Contract','PaperlessBilling','PaymentMethod','Churn'],
                     drop_first=True,dtype=int)



# import ml libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score

# train test split
x = churn.drop('Churn_Yes',axis=1)
y = churn['Churn_Yes']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# logistic regression model
logmodel=LogisticRegression(class_weight='balanced')
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
logmodel.fit(x_train_scaled,y_train)
y_prob=logmodel.predict_proba(x_test_scaled)[:,1]
y_pred=(y_prob>0.4).astype(int)
# metrics
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(f'ROC-AUC score:{roc_auc_score(y_test,y_prob)}')


# decision tree model
dtree=DecisionTreeClassifier(class_weight='balanced')
dtree.fit(x_train,y_train)
y_prob=dtree.predict_proba(x_test)[:,1]
y_pred=(y_prob>0.4).astype(int)
# metrics
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# random forest model
rfmodel=RandomForestClassifier(class_weight='balanced',n_estimators=200)
rfmodel.fit(x_train,y_train)
y_prob=rfmodel.predict_proba(x_test)[:,1]
y_pred=(y_prob>0.4).astype(int)
# metrics
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# Logistic Regression gives the best results here
# Logistic Regression achieved the highest ROC-AUC (0.86) and recall (0.88),
# making it the most suitable model when the business objective is to identify maximum churn customers.




