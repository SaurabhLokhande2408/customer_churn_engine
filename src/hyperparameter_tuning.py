#models with high priority CatBoost, XGBoost and LightGBM.
#model with 2nd priority Logistic Regression and Random Forest.



from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import pandas as pd
def best_model_prinitng(random_search):
    results = pd.DataFrame(random_search.cv_results_)
    rank_col = "rank_test_average_precision" if "rank_test_average_precision" in results.columns else "rank_test_score"
    columns = [
        rank_col,
        "mean_test_average_precision" if "mean_test_average_precision" in results.columns else "mean_test_score",
        "mean_test_roc_auc" if "mean_test_roc_auc" in results.columns else "mean_test_score",
        "std_test_average_precision" if "std_test_average_precision" in results.columns else "std_test_score",
        "params"
    ]
    columns = [c for i, c in enumerate(columns) if c in results.columns and c not in columns[:i]]
    if "mean_train_score" in results.columns:
        columns.insert(-1, "mean_train_score")
    elif "mean_train_average_precision" in results.columns:
        columns.insert(-1, "mean_train_average_precision")
    top_10 = (
        results[columns].sort_values(rank_col).head(10)
    )
    print(top_10.to_string(index=False))

def hyperparameter_tuning_catboost(catboost_model, X_train, y_train):
    print(type(catboost_model).__name__)
    param_grid = {
        'iterations': [100, 200, 300, 400, 500],
        'depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 7],
        'border_count': [32, 64, 128],
        'bagging_temperature': [0.0, 0.1, 0.2, 0.3]
    }

    cv_strategy = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        'average_precision': 'average_precision',
        'roc_auc': 'roc_auc'
    }

    random_search = RandomizedSearchCV(
        estimator=catboost_model,
        param_distributions=param_grid,
        n_iter=30,
        scoring=scoring,
        cv=cv_strategy,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        refit='average_precision',
        return_train_score=True
    )
    random_search.fit(X_train, y_train)

    print("Best Parameters:")
    print(random_search.best_params_)

    print("Best CV Average Precision:")
    print(random_search.best_score_)

    best_catboost_model = random_search.best_estimator_
    best_model_prinitng(random_search)
    

    return best_catboost_model

def hyperparameter_tuning_xgboost(xgb_model,X_train, y_train):
    print(type(xgb_model).__name__)
    param_grid= {
            'n_estimators': [200, 300, 500, 700, 1000],

            'max_depth': [3, 4, 5, 6, 7],

            'learning_rate': [0.01, 0.03, 0.05, 0.1],

            'subsample': [0.7, 0.8, 0.9, 1.0],

            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],

            'min_child_weight': [1, 3, 5, 7],

            'reg_alpha': [0, 0.01, 0.1, 1],

            'reg_lambda': [0.1, 1, 5, 10]
                }
    cv_strategy = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        'average_precision': 'average_precision',
        'roc_auc': 'roc_auc'
    }

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=30,
        scoring=scoring,
        cv=cv_strategy,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        refit='average_precision',
        return_train_score=True
    )
    random_search.fit(X_train, y_train)

    print("Best Parameters:")
    print(random_search.best_params_)

    print("Best CV Average Precision:")
    print(random_search.best_score_)

    best_xgb_model = random_search.best_estimator_ 
    best_model_prinitng(random_search)
   
    return best_xgb_model

def hyperparameter_tuning_lightgbm(lgbm_model, X_train, y_train):
    print(type(lgbm_model).__name__)
    param_grid = {
        'n_estimators': [200, 300, 500, 700, 1000],
        'max_depth': [-1, 3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'num_leaves': [31, 50, 100, 200],
        'min_child_samples': [20, 30, 50, 100],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [0.1, 1, 5, 10]
    }

    cv_strategy = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        'average_precision': 'average_precision',
        'roc_auc': 'roc_auc'
    }

    random_search = RandomizedSearchCV(
        estimator=lgbm_model,
        param_distributions=param_grid,
        n_iter=30,
        scoring=scoring,
        cv=cv_strategy,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        refit='average_precision',
        return_train_score=True
    )
    random_search.fit(X_train, y_train)

    print("Best Parameters:")
    print(random_search.best_params_)

    print("Best CV Average Precision:")
    print(random_search.best_score_)

    best_lgbm_model = random_search.best_estimator_
    best_model_prinitng(random_search)
    
    return best_lgbm_model

def hyperparameter_tuning_logistic_regression(lr_model, X_train, y_train):
    print(type(lr_model).__name__)
    param_grid = [
        {
            'model__solver': ['liblinear', 'saga'],
            'model__penalty': ['l1', 'l2'],
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__max_iter': [100, 200, 300, 400, 500]
        },
        {
            'model__solver': ['saga'],
            'model__penalty': ['elasticnet'],
            'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__max_iter': [100, 200, 300, 400, 500]
        }
    ]
    cv_strategy = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )
    
    scoring = {
        'average_precision': 'average_precision',
        'roc_auc': 'roc_auc'
    }

    random_search = RandomizedSearchCV(
        estimator=lr_model,
        param_distributions=param_grid,
        n_iter=30,
        scoring=scoring,
        cv=cv_strategy,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        refit='average_precision',
        return_train_score=True
    )
    random_search.fit(X_train, y_train)

    print("Best Parameters:")
    print(random_search.best_params_)

    print("Best CV Average Precision:")
    print(random_search.best_score_)

    best_lr_model = random_search.best_estimator_
    best_model_prinitng(random_search)
    
    return best_lr_model

def hyperparameter_tuning_random_forest(rf_model, X_train, y_train):
    print(type(rf_model).__name__)
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 3, 4, 5, 6, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    cv_strategy = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        'average_precision': 'average_precision',
        'roc_auc': 'roc_auc'
    }

    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_grid,
        n_iter=30,
        scoring=scoring,
        cv=cv_strategy,
        verbose=2,
        random_state=42,
        n_jobs=-1,
        refit='average_precision',
        return_train_score=True
    )
    random_search.fit(X_train, y_train)

    print("Best Parameters:")
    print(random_search.best_params_)

    print("Best CV Average Precision:")
    print(random_search.best_score_)

    best_rf_model = random_search.best_estimator_
    best_model_prinitng(random_search)
    
    return best_rf_model
