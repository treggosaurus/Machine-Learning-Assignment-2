import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import make_scorer

under_prediction_penalty = 2.0
over_prediction_penalty = 1.0

def asymmetric_mse(y_true_set, y_prediction_set):
    residual = y_true_set - y_prediction_set
    
    penalty = np.where(residual > 0, over_prediction_penalty, under_prediction_penalty)
    
    grad = -2.0 * penalty * residual
    hess = 2.0 * penalty
    
    return grad, hess

def asymmetric_scoring(y_true_set, y_prediction_set):
    residual = y_true_set - y_prediction_set
    
    penalty = np.where(residual > 0, over_prediction_penalty, under_prediction_penalty)
    
    loss = np.mean(penalty * (residual ** 2))
    
    return loss

def add_safe_prediction(model, x_validation_set, y_validation_set, x_testing_set, std_multiplier=1.5):
    validation_predictions = model.predict(x_validation_set)

    residuals = y_validation_set - validation_predictions

    error = np.std(residuals)
    print(f"Validation Residual Std Dev: {error:.2f} MW")

    base_test_predictions = model.predict(x_testing_set)

    safe_test_predictions = base_test_predictions + (std_multiplier * error)

    return safe_test_predictions, base_test_predictions
    
model_scorer = make_scorer(asymmetric_scoring, greater_is_better=False)

    

def train_primary_model(x_training_set, y_training_set):
    base_estimation = XGBRegressor(objective=asymmetric_mse)
    
    parameter_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 8, 10, 15],
        'learning_rate': [0.05, 0.1, 0.15]
    }
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    grid_search = GridSearchCV(
        estimator=base_estimation,
        param_grid=parameter_grid,
        cv=tscv,
        scoring=model_scorer,
        verbose=1
    )

    grid_search.fit(x_training_set, y_training_set)
    
    return grid_search.best_params_

def train_secondary_model(x_training_set, y_training_set, x_validation_set, y_validation_set, parameters):
    model = XGBRegressor(
        objective=asymmetric_mse, 
        n_estimators=parameters['n_estimators'], 
        max_depth=parameters['max_depth'],
        learning_rate=parameters['learning_rate']
    )

    model.fit(
        x_training_set, y_training_set,
        eval_set=[(x_validation_set, y_validation_set)],
        verbose=100
    )

    return model