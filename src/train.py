from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
def training_knn(df:pd.DataFrame)-> pd.DataFrame:
    df=df.copy()
    X=df.drop(columns=['Churn Label'])
    y=df['Churn Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    features_to_scale = ['Age', 'Population', 'Number of Referrals',
       'Tenure in Months', 'Avg Monthly Long Distance Charges',
       'Avg Monthly GB Download','Monthly Charge', 'Total Charges',
         'Total Refunds','Total Extra Data Charges', 'Total Long Distance Charges', 
           'Satisfaction Score', 'CLTV']
    X_scaled_train=scaler.fit_transform(X_train)
    X_scaled_test=scaler.transform(X_test)
    #now creating first model using KNN
    model_knn = KNeighborsClassifier(n_neighbors=5)
    model_knn=model_knn.fit(X_scaled_train, y_train)
    return model_knn,y_test,X_scaled_test
def training_log_reg(df:pd.DataFrame)-> pd.DataFrame:
     df=df.copy()
     X=df.drop(columns=['Churn Label'])
     y=df['Churn Label']
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
     df.select_dtypes(include=[np.number]).columns
     scaler = StandardScaler()
     features_to_scale = ['Age', 'Population', 'Number of Referrals',
           'Tenure in Months', 'Avg Monthly Long Distance Charges',
           'Avg Monthly GB Download','Monthly Charge', 'Total Charges',
             'Total Refunds','Total Extra Data Charges', 'Total Long Distance Charges', 
               'Satisfaction Score', 'CLTV']
     X_scaled_train=scaler.fit_transform(X_train)
     X_scaled_test=scaler.transform(X_test)
     #now creating first model using Logistic Regression
     model_log_reg = LogisticRegression()
     model_log_reg=model_log_reg.fit(X_scaled_train, y_train)
     return model_log_reg,y_test,X_scaled_test
def training_decision_tree(df:pd.DataFrame)-> pd.DataFrame:
        df=df.copy()
        X=df.drop(columns=['Churn Label'])
        y=df['Churn Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        df.select_dtypes(include=[np.number]).columns
        #since trees dont need to be scaled but i dont wanna change the pipeline so i wont change variable names
        X_scaled_train=X_train
        X_scaled_test=X_test
        #now creating first model using Decision Tree
        model_decision_tree = DecisionTreeClassifier()
        model_decision_tree=model_decision_tree.fit(X_scaled_train, y_train)
        return model_decision_tree,y_test,X_scaled_test
     
   