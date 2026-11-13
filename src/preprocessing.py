import pandas as pd
import warnings
from sklearn.impute import SimpleImputer
warnings.filterwarnings('ignore')

def preprocessing_data(df: pd.DataFrame, test_df: pd.DataFrame = None):
    df = df.copy()

    # Imputation of missing values in Internet Type and Offer features
    impute = SimpleImputer(strategy='constant', fill_value='Unknown')
    df[['Internet Type', 'Offer']] = impute.fit_transform(df[['Internet Type', 'Offer']])

    if test_df is not None:
        test_df = test_df.copy()
        test_df[['Internet Type', 'Offer']] = impute.transform(test_df[['Internet Type', 'Offer']])
        return df, test_df

    return df
