from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def _prepare_data(X_train, y_train=None, X_test=None, y_test=None):
    if isinstance(X_train, pd.DataFrame) and y_train is None:
        df = X_train.copy()
        X = df.drop(columns=['Churn Label'])
        y = df['Churn Label']
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


def training_knn(X_train, y_train=None, X_test=None, y_test=None):
    X_train, X_test, y_train, y_test = _prepare_data(X_train, y_train, X_test, y_test)
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)

    model_knn = KNeighborsClassifier(n_neighbors=5)
    model_knn.fit(X_scaled_train, y_train)
    return model_knn, X_train, y_train, X_scaled_test, y_test


def training_log_reg(X_train, y_train=None, X_test=None, y_test=None):
    X_train, X_test, y_train, y_test = _prepare_data(X_train, y_train, X_test, y_test)
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)

    model_log_reg = LogisticRegression(random_state=42)
    model_log_reg.fit(X_scaled_train, y_train)
    return model_log_reg, X_train, y_train, X_scaled_test, y_test


def training_decision_tree(X_train, y_train=None, X_test=None, y_test=None):
    X_train, X_test, y_train, y_test = _prepare_data(X_train, y_train, X_test, y_test)
    model_decision_tree = DecisionTreeClassifier(random_state=42)
    model_decision_tree.fit(X_train, y_train)
    return model_decision_tree, X_train, y_train, X_test, y_test


def training_svm(X_train, y_train=None, X_test=None, y_test=None):
    X_train, X_test, y_train, y_test = _prepare_data(X_train, y_train, X_test, y_test)
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)

    model_svm = SVC(
        kernel='rbf',          # Uses Radial Basis Function to capture non-linear relationships
        C=1.0,                 # Controls regularization
        gamma='scale',         # Influence range of individual training samples based on feature variance
        random_state=42,       # Ensures reproducibility
        probability=True       # Enables probability predictions using predict_proba()
    )
    model_svm.fit(X_scaled_train, y_train)
    return model_svm, X_train, y_train, X_scaled_test, y_test


def training_naive_bayes(X_train, y_train=None, X_test=None, y_test=None):
    X_train, X_test, y_train, y_test = _prepare_data(X_train, y_train, X_test, y_test)
    model_naive_bayes = GaussianNB()
    model_naive_bayes.fit(X_train, y_train)
    return model_naive_bayes, X_train, y_train, X_test, y_test


def training_random_forest(X_train, y_train=None, X_test=None, y_test=None):
    X_train, X_test, y_train, y_test = _prepare_data(X_train, y_train, X_test, y_test)
    model_random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    model_random_forest.fit(X_train, y_train)
    return model_random_forest, X_train, y_train, X_test, y_test


def training_xgboost(X_train, y_train=None, X_test=None, y_test=None):
    X_train, X_test, y_train, y_test = _prepare_data(X_train, y_train, X_test, y_test)
    model_xgboost = XGBClassifier(
        n_estimators=200,       # Number of boosting trees
        learning_rate=0.05,     # Controls contribution of each tree
        max_depth=5,            # Maximum depth of each tree
        random_state=42,        # Ensures reproducibility
        eval_metric='logloss',  # Measures prediction error using log loss
        verbosity=0             # Suppresses unnecessary training logs
    )
    model_xgboost.fit(X_train, y_train)
    return model_xgboost, X_train, y_train, X_test, y_test


def training_catboost(X_train, y_train=None, X_test=None, y_test=None):
    X_train, X_test, y_train, y_test = _prepare_data(X_train, y_train, X_test, y_test)
    model_catboost = CatBoostClassifier(
        iterations=200,         # Number of boosting iterations
        learning_rate=0.05,     # Controls contribution of each tree
        depth=6,                # Maximum depth of each tree
        random_seed=42,         # Ensures reproducibility
        verbose=0               # Suppresses training logs
    )
    model_catboost.fit(X_train, y_train)
    return model_catboost, X_train, y_train, X_test, y_test


def training_lightgbm(X_train, y_train=None, X_test=None, y_test=None):
    X_train, X_test, y_train, y_test = _prepare_data(X_train, y_train, X_test, y_test)
    model_lightgbm = LGBMClassifier(
        n_estimators=200,       # Number of boosting trees
        learning_rate=0.05,     # Controls contribution of each tree
        max_depth=-1,           # Allows trees to grow without fixed depth limit
        random_state=42,        # Ensures reproducibility
        verbosity=-1            # Suppresses unnecessary logs
    )
    model_lightgbm.fit(X_train, y_train)
    return model_lightgbm, X_train, y_train, X_test, y_test
