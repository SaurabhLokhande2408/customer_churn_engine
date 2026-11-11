from dataset.data_loading import data_loading   
from src.preprocessing import preprocessing_data
from src.feature_engineering import feature_eng
from src.train import training_knn
from src.train import training_log_reg
from src.train import training_decision_tree
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


main()


    