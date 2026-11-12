from dataset.data_loading import data_loading   
from src.preprocessing import preprocessing_data
from src.feature_engineering import feature_eng
from src.train import training_knn,training_log_reg,training_decision_tree,training_random_forest,training_svm,training_naive_bayes
from src.train import training_xgboost,training_catboost,training_lightgbm

from src.predict import predict
from src.evaluation import evaluation_engine
from warnings import filterwarnings
filterwarnings('ignore')

def main():
    #loading the data
    df = data_loading()
    #preprocessing data
    df = preprocessing_data(df)
    #feature engineering
    df=feature_eng(df)
    # training
    model, y_test, X_scaled_test = training_knn(df)
    # prediction
    y_pred, y_test = predict(model, y_test, X_scaled_test)
    # Evaluation
    print("KNN Results:")
    evaluation_engine(y_test, y_pred)

    model, y_test, X_scaled_test = training_log_reg(df)
    y_pred, y_test = predict(model, y_test, X_scaled_test)
    print("Logistic Regression Results:")
    evaluation_engine(y_test, y_pred)

    model, y_test, X_scaled_test = training_decision_tree(df)
    y_pred, y_test = predict(model, y_test, X_scaled_test)
    print("Decision Tree Results:")
    evaluation_engine(y_test, y_pred)

    model, y_test, X_scaled_test = training_random_forest(df)
    y_pred, y_test = predict(model, y_test, X_scaled_test)
    print("Random Forest Results:")
    evaluation_engine(y_test, y_pred)


    model, y_test, X_scaled_test = training_svm(df)
    y_pred, y_test = predict(model, y_test, X_scaled_test)
    print("SVM Results:")
    evaluation_engine(y_test, y_pred)

    model, y_test, X_scaled_test = training_naive_bayes(df)
    y_pred, y_test = predict(model, y_test, X_scaled_test)
    print("Naive Bayes Results:")
    evaluation_engine(y_test, y_pred)

    model, y_test, X_scaled_test = training_xgboost(df)
    y_pred, y_test = predict(model, y_test, X_scaled_test)
    print("XGBoost Results:")
    evaluation_engine(y_test, y_pred)

    model, y_test, X_scaled_test = training_catboost(df)
    y_pred, y_test = predict(model, y_test, X_scaled_test)
    print("CatBoost Results:")
    evaluation_engine(y_test, y_pred)

    model, y_test, X_scaled_test = training_lightgbm(df)
    y_pred, y_test = predict(model, y_test, X_scaled_test)
    print("LightGBM Results:")
    evaluation_engine(y_test, y_pred)


main()


    