# prediction
def predict(model, y_test, X_test):
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    return y_pred, y_test, y_prob
