from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
def training(df:pd.DataFrame)-> pd.DataFrame:
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