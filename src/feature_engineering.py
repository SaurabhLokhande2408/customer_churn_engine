import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder


def feature_eng(df:pd.DataFrame) -> pd.DataFrame:
    df=df.copy()
    #lets do feature engineering and scaling
    le=LabelEncoder()
    columns_to_label_encode = ['Gender', 'Married',
                               'Internet Service','Online Security',
                                 'Device Protection Plan', 'Paperless Billing',
                                 'Under 30','Senior Citizen','Multiple Lines',
                                 'Premium Tech Support','Streaming TV', 
                                 'Streaming Movies','Streaming Music',
                                   'Unlimited Data','Churn Label',
                                   'Online Backup']
    for col in columns_to_label_encode:
        df[col] = le.fit_transform(df[col])
    df.drop(columns=['Dependents','Total Revenue','Customer Status','Under 30'], inplace=True)
    df.drop(columns=['Customer ID','Zip Code', 'Latitude', 'Longitude','Referred a Friend' ],inplace=True)
    #idk how its gonna affect moedel but lets drop rows with missing values
    df.drop(columns=['Churn Category','Churn Reason','Churn Score','City'], inplace=True)
    #features to one hot encode
    columns_to_one_hot_encode = ['Country', 'State','Quarter',
                                  'Offer', 'Phone Service',
                                  'Internet Type','Contract',
                                    'Payment Method']
    df=pd.get_dummies(df, columns=columns_to_one_hot_encode, drop_first=True)
    return df
    
