from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def training_knn(df:pd.DataFrame)-> pd.DataFrame:
    df=df.copy()
    X=df.drop(columns=['Churn Label'])
    y=df['Churn Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
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

def training_svm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    X = df.drop(columns=['Churn Label'])
    y = df['Churn Label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    # Scaling features for SVM
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)
    # Creating SVM model
    model_svm = SVC(
    kernel='rbf',          # Uses Radial Basis Function to capture non-linear relationships between features.
    C=1.0,                 # Controls regularization: lower values allow more errors, higher values penalize errors more.
    gamma='scale',         # Automatically calculates the influence range of individual training samples based on feature variance.
    random_state=42,       # Ensures reproducibility of probability estimation.
    probability=True       # Enables probability predictions using predict_proba() instead of only class predictions.
    )
    model_svm.fit(X_scaled_train, y_train)
    return model_svm, y_test, X_scaled_test


def training_naive_bayes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    X = df.drop(columns=['Churn Label'])
    y = df['Churn Label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    model_naive_bayes=GaussianNB()
    model_naive_bayes.fit(X_train,y_train)
    return model_naive_bayes, y_test, X_test

def training_random_forest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    X = df.drop(columns=['Churn Label'])
    y = df['Churn Label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    model_random_forest=RandomForestClassifier(n_estimators=100, random_state=42)
    model_random_forest.fit(X_train,y_train)
    return model_random_forest, y_test, X_test

def training_xgboost(df: pd.DataFrame):
    df = df.copy()

    X = df.drop(columns=['Churn Label'])
    y = df['Churn Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model_xgboost = XGBClassifier(
        n_estimators=200,       # Number of boosting trees.
        learning_rate=0.05,     # Controls contribution of each tree.
        max_depth=5,            # Maximum depth of each tree.
        random_state=42,        # Ensures reproducibility.
        eval_metric='logloss',  # Measures prediction error using log loss.
        verbosity=0       # Suppresses unnecessary training logs.
    )
    

    model_xgboost.fit(X_train, y_train)

    return model_xgboost, y_test, X_test

def training_catboost(df: pd.DataFrame):
    df = df.copy()

    X = df.drop(columns=['Churn Label'])
    y = df['Churn Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model_catboost = CatBoostClassifier(
        iterations=200,         # Number of boosting iterations.
        learning_rate=0.05,     # Controls the contribution of each tree.
        depth=6,                # Maximum depth of each tree.
        random_seed=42,         # Ensures reproducibility.
        verbose=0               # Suppresses training logs.
    )

    model_catboost.fit(X_train, y_train)

    return model_catboost, y_test, X_test

def training_lightgbm(df: pd.DataFrame):
    df = df.copy()

    X = df.drop(columns=['Churn Label'])
    y = df['Churn Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model_lightgbm = LGBMClassifier(
        n_estimators=200,       # Number of boosting trees.
        learning_rate=0.05,     # Controls the contribution of each tree.
        max_depth=-1,           # Allows trees to grow without a fixed depth limit.
        random_state=42,        # Ensures reproducibility.
        verbosity=-1            # Suppresses unnecessary logs.
    )

    model_lightgbm.fit(X_train, y_train)

    return model_lightgbm, y_test, X_test