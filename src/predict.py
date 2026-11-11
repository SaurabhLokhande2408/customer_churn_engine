#prediction
def predict(model,y_test,X_scaled_test):
    y_pred = model.predict(X_scaled_test)
    return y_pred,y_test
    
