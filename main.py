import sys
from pathlib import Path
from datetime import datetime

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def start_logging():
    log_file = open("logs.txt", "a", encoding="utf-8")

    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print("\n" + "=" * 70)
    print(f"Experiment started: {datetime.now()}")
    print("=" * 70 + "\n")

from sklearn.model_selection import train_test_split
from dataset.data_loading import data_loading   
from src.preprocessing import preprocessing_data
from src.feature_engineering import feature_eng
from src.train import (
    training_knn,
    training_log_reg,
    training_decision_tree,
    training_random_forest,
    training_svm,
    training_naive_bayes,
    training_xgboost,
    training_catboost,
    training_lightgbm
)
from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        f1_score,
        average_precision_score
    )
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from src.predict import predict
from src.cross_validation import cross_valdation_scaled_models, cross_valdation_unscaled_models
from src.evaluation import evaluation_engine
from src.hyperparameter_tuning import (hyperparameter_tuning_xgboost
                                       ,hyperparameter_tuning_catboost
                                       ,hyperparameter_tuning_lightgbm
                                       ,hyperparameter_tuning_random_forest
                                       ,hyperparameter_tuning_logistic_regression)
from warnings import filterwarnings
filterwarnings('ignore')

def main():

    start_logging()
    # loading the data
    df = data_loading()
    # preprocessing data
    df = preprocessing_data(df)

    # train-test split before feature engineering to prevent leakage
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Churn Label'])

    # feature engineering fitted on train, transforming test
    train_df, test_df = feature_eng(train_df, test_df)

    X_train = train_df.drop(columns=['Churn Label'])
    y_train = train_df['Churn Label']
    X_test = test_df.drop(columns=['Churn Label'])
    y_test = test_df['Churn Label']

    # 1. KNN (Scaled)
    model_knn, X_tr, y_tr, X_te, y_te = training_knn(X_train, y_train, X_test, y_test)
    y_pred, y_test_out, y_prob = predict(model_knn, y_te, X_te)
    print("KNN Results:")
    evaluation_engine(y_test_out, y_pred, y_prob)
    print("K Fold Cross Validation for Scaled Models(KNN):")
    scores = cross_valdation_scaled_models(model_knn, X_tr, y_tr)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV score: {scores.mean()}")

    # 2. Logistic Regression (Scaled)
    model_log_reg, X_tr, y_tr, X_te, y_te = training_log_reg(X_train, y_train, X_test, y_test)
    y_pred, y_test_out, y_prob = predict(model_log_reg, y_te, X_te)
    print("Logistic Regression Results:")
    evaluation_engine(y_test_out, y_pred, y_prob)
    print("K Fold Cross Validation for Scaled Models(Logistic Regression):")
    scores = cross_valdation_scaled_models(model_log_reg, X_tr, y_tr)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV score: {scores.mean()}")

    # 3. Decision Tree (Unscaled)
    model_decision_tree, X_tr, y_tr, X_te, y_te = training_decision_tree(X_train, y_train, X_test, y_test)
    y_pred, y_test_out, y_prob = predict(model_decision_tree, y_te, X_te)
    print("Decision Tree Results:")
    evaluation_engine(y_test_out, y_pred, y_prob)
    print("K Fold Cross Validation for Unscaled Models(Decision Tree):")
    scores = cross_valdation_unscaled_models(model_decision_tree, X_tr, y_tr)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV score: {scores.mean()}")

    # 4. Random Forest (Unscaled)
    model_random_forest, X_tr, y_tr, X_te, y_te = training_random_forest(X_train, y_train, X_test, y_test)
    y_pred, y_test_out, y_prob = predict(model_random_forest, y_te, X_te)
    print("Random Forest Results:")
    evaluation_engine(y_test_out, y_pred, y_prob)
    print("K Fold Cross Validation for Unscaled Models(Random Forest):")
    scores = cross_valdation_unscaled_models(model_random_forest, X_tr, y_tr)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV score: {scores.mean()}")

    # 5. SVM (Scaled)
    model_svm, X_tr, y_tr, X_te, y_te = training_svm(X_train, y_train, X_test, y_test)
    y_pred, y_test_out, y_prob = predict(model_svm, y_te, X_te)
    print("SVM Results:")
    evaluation_engine(y_test_out, y_pred, y_prob)
    print("K Fold Cross Validation for Scaled Models(SVM):")
    scores = cross_valdation_scaled_models(model_svm, X_tr, y_tr)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV score: {scores.mean()}")

    # 6. Naive Bayes (Unscaled)
    model_naive_bayes, X_tr, y_tr, X_te, y_te = training_naive_bayes(X_train, y_train, X_test, y_test)
    y_pred, y_test_out, y_prob = predict(model_naive_bayes, y_te, X_te)
    print("Naive Bayes Results:")
    evaluation_engine(y_test_out, y_pred, y_prob)
    print("K Fold Cross Validation for Unscaled Models(Naive Bayes):")
    scores = cross_valdation_unscaled_models(model_naive_bayes, X_tr, y_tr)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV score: {scores.mean()}")

    # 7. XGBoost (Unscaled)
    model_xgb, X_tr, y_tr, X_te, y_te = training_xgboost(X_train, y_train, X_test, y_test)
    y_pred, y_test_out, y_prob = predict(model_xgb, y_te, X_te)
    print("XGBoost Results:")
    evaluation_engine(y_test_out, y_pred, y_prob)
    print("K Fold Cross Validation for Unscaled Models(XGBoost):")
    scores = cross_valdation_unscaled_models(model_xgb, X_tr, y_tr)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV score: {scores.mean()}")

    # 8. CatBoost (Unscaled)
    model_catboost, X_tr, y_tr, X_te, y_te = training_catboost(X_train, y_train, X_test, y_test)
    y_pred, y_test_out, y_prob = predict(model_catboost, y_te, X_te)
    print("CatBoost Results:")
    evaluation_engine(y_test_out, y_pred, y_prob)
    print("K Fold Cross Validation for Unscaled Models(CatBoost):")
    scores = cross_valdation_unscaled_models(model_catboost, X_tr, y_tr)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV score: {scores.mean()}")

    # 9. LightGBM (Unscaled)
    model_lightgbm, X_tr, y_tr, X_te, y_te = training_lightgbm(X_train, y_train, X_test, y_test)
    y_pred, y_test_out, y_prob = predict(model_lightgbm, y_te, X_te)
    print("LightGBM Results:")
    evaluation_engine(y_test_out, y_pred, y_prob)
    print("K Fold Cross Validation for Unscaled Models(LightGBM):")
    scores = cross_valdation_unscaled_models(model_lightgbm, X_tr, y_tr)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV score: {scores.mean()}")



    tuned_probas = {}

    print("Hyperparameter tuning for XGBoost model:")
    best_xgb_model = hyperparameter_tuning_xgboost(model_xgb, X_tr, y_tr)

    y_pred = best_xgb_model.predict(X_test)
    y_proba = best_xgb_model.predict_proba(X_test)[:, 1]
    tuned_probas['XGBoost'] = y_proba

    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("PR-AUC:", average_precision_score(y_test, y_proba))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Hyperparameter tuning for CatBoost model:")
    best_catboost_model = hyperparameter_tuning_catboost(model_catboost, X_tr, y_tr)

    y_pred = best_catboost_model.predict(X_test)
    y_proba = best_catboost_model.predict_proba(X_test)[:, 1]
    tuned_probas['CatBoost'] = y_proba

    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("PR-AUC:", average_precision_score(y_test, y_proba))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Hyperparameter tuning for LightGBM model:")
    best_lightgbm_model = hyperparameter_tuning_lightgbm(model_lightgbm, X_tr, y_tr)
    y_pred = best_lightgbm_model.predict(X_test)
    y_proba = best_lightgbm_model.predict_proba(X_test)[:, 1]
    tuned_probas['LightGBM'] = y_proba

    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("PR-AUC:", average_precision_score(y_test, y_proba))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Hyperparameter tuning for Random Forest model:")
    best_rf_model = hyperparameter_tuning_random_forest(model_random_forest, X_tr, y_tr)
    y_pred = best_rf_model.predict(X_test)
    y_proba = best_rf_model.predict_proba(X_test)[:, 1]
    tuned_probas['Random Forest'] = y_proba

    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("PR-AUC:", average_precision_score(y_test, y_proba))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Hyperparameter tuning for Logistic Regression model:")
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model_log_reg)
    ])
    best_lr_model = hyperparameter_tuning_logistic_regression(lr_pipe, X_tr, y_tr)
    y_pred = best_lr_model.predict(X_test)
    y_proba = best_lr_model.predict_proba(X_test)[:, 1]
    tuned_probas['Logistic Regression'] = y_proba

    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("PR-AUC:", average_precision_score(y_test, y_proba))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Sanity checks
    print("\n" + "=" * 50)
    print("Sanity Checks")
    print("=" * 50)

    # 1. Compare tuned models predicted probabilities against each other
    tuned_names = list(tuned_probas.keys())
    for i in range(len(tuned_names)):
        for j in range(i + 1, len(tuned_names)):
            m1, m2 = tuned_names[i], tuned_names[j]
            identical = np.array_equal(tuned_probas[m1], tuned_probas[m2])
            if identical:
                print(f"WARNING: Tuned models {m1} and {m2} produced identical predicted probabilities!")
            assert not identical, f"Tuned models {m1} and {m2} produced identical predicted probabilities!"

    # 2. Compare each tuned model against the matching untuned model
    untuned_models = {
        'XGBoost': training_xgboost(X_train, y_train, X_test, y_test),
        'CatBoost': training_catboost(X_train, y_train, X_test, y_test),
        'LightGBM': training_lightgbm(X_train, y_train, X_test, y_test),
        'Random Forest': training_random_forest(X_train, y_train, X_test, y_test),
        'Logistic Regression': training_log_reg(X_train, y_train, X_test, y_test),
    }

    for name, (u_model, _, _, u_X_te, u_y_te) in untuned_models.items():
        _, _, u_prob = predict(u_model, u_y_te, u_X_te)
        identical = np.array_equal(tuned_probas[name], u_prob)
        max_diff = np.max(np.abs(tuned_probas[name] - u_prob))
        print(f"Untuned vs Tuned {name}: identical={identical}, max probability diff={max_diff:.6f}")
    
       
   

if __name__ == "__main__":
    main()
