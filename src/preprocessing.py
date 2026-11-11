import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.impute import SimpleImputer
from dataset.data_loading import data_loading
warnings.filterwarnings('ignore')

def preprocessing_data(df: pd.DataFrame) -> pd.DataFrame:
        df=df.copy()
        numeric_features=df.select_dtypes(include=[np.number])
        catergorical_features=df.select_dtypes(include=[object])

        #lets do imputation of missing values in Internet Service feature
        impute=SimpleImputer(strategy='constant',fill_value='Unknown')
        df[['Internet Type']]=impute.fit_transform(df[['Internet Type']])
        df[['Offer']]=impute.fit_transform(df[['Offer']])
        pre_processed_df=df
        return df
