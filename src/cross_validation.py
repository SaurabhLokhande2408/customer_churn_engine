from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def cross_valdation_scaled_models(model, X, y, cv=5):
    # Pass unscaled training data in X and y so that StandardScaler is fitted independently inside each fold
    if isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1')
    return scores


def cross_valdation_unscaled_models(model, X, y, cv=5):
    # Pass unscaled training data in X and y
    if isinstance(cv, int):
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    return scores
