#prediction
def predict(model_knn,y_test,X_scaled_test):
    y_pred_knn = model_knn.predict(X_scaled_test)
    return y_pred_knn,y_test
    
