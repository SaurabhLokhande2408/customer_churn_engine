from dataset.data_loading import data_loading   
from src.preprocessing import preprocessing_data
from src.feature_engineering import feature_eng
from src.train import training
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
    #training 
    model_knn, y_test,X_scaled_test = training(df)
    #prediction via multiple models
    
    y_pred, y_test = predict(model_knn, y_test, X_scaled_test)
    #Evaluation
    evaluation_results = evaluation_engine(y_test,y_pred)
    print(evaluation_results)
    
main()


    