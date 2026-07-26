import pandas as pd
import numpy as np
from src.testing_validation.model_test import calculate_rmse
from sklearn.model_selection import TimeSeriesSplit

def fit_cv_timeseries_model(model: any, X: pd.DataFrame, y: pd.DataFrame):
    """
    Fits the model on the given data using sklearn TimeSeriesSplit. Can save all the models if 
    neccessary.

    Args:
        model: A model that exposes the .fit() method
        X: Dataframe containing the predictors
        y: The response variable
    
    Returns:
        mean_rmse: int
    """
    tscv = TimeSeriesSplit()
    all_rmses = []
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        temp_X_tr = X.iloc[train_index]
        temp_y_tr = y.iloc[train_index]
        model.fit(temp_X_tr, temp_y_tr)

        temp_X_test = X.iloc[test_index]
        temp_y_test = y.iloc[test_index]

        rmse = calculate_rmse(temp_X_test, temp_y_test, model, None, False)
        all_rmses.append(rmse)

        print(f"Fold {i}, rmse: {rmse}")

    print(np.mean(all_rmses))