import pandas as pd
import pickle
from typing import Any
from sklearn.metrics import root_mean_squared_error
from pathlib import Path
import csv
from src.utils import name_model
import json

def calculate_rmse(X_test: pd.DataFrame, y_true: pd.DataFrame, model_obj: Any = None, model_file_str: str = None, must_log = False, model_name="Unnammed"):
    """
    Calculate the Root Mean Squared Error (RMSE) for a given model against test data,
    optionally logging the result and the model's hyperparameters to a CSV report.

    The model can be supplied either directly as an in-memory object (`model_obj`)
    or loaded from a pickled file on disk (`model_file_str`). Exactly one of these
    should be provided.

    Args:
        X_test (pd.DataFrame): Feature data to generate predictions from.
        y_true (pd.DataFrame): Ground-truth target values corresponding to X_test.
        model_obj (Any, optional): An in-memory model object exposing a `.predict()`
            method (e.g. a fitted sklearn/XGBoost/LightGBM estimator). Defaults to None.
        model_file_str (str, optional): Path to a pickled model file to load from disk.
            Defaults to None.
        must_log (bool, optional): If True, appends the model name, RMSE, and
            hyperparameters as a new row to "../reports/models_rmse.csv". Defaults to False.
        model_name (str, optional): Base name to use when logging a model passed via
            `model_obj` (a timestamp is appended via `name_model`). Ignored if the model
            was loaded from a file, since the filename stem is used instead.
            Defaults to "Unnammed".

    Returns:
        float | None: The computed RMSE value, or None if neither `model_obj` nor
        `model_file_str` was provided.

    Notes:
        - If both `model_obj` and `model_file_str` are provided, `model_obj` takes
          precedence for loading, but `model_file_str`'s check happens first in the
          branching logic below.
        - Hyperparameters are only captured if the model exposes a `get_params()`
          method (standard for sklearn-compatible estimators). Models without this
          method will be logged with an empty params dict.
    """
    model = None

    # Gaurd clause: exit if no model provided
    if model_file_str == None and model_obj == None:
        return None
    # use in-memory model
    elif model_file_str == None:
        model = model_obj
    # load model for pickle file
    elif model_obj == None:
        with open("models/"+ model_file_str, 'rb') as file:
            model = pickle.load(file)
    
    # generate predictions and calcualte rmse
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_true, y_pred)

    if must_log:
        # determine whether or not to use filename (if pickle file) or create custome name for in-memory model
        if model_obj is not None:
            name = name_model(model_name)
        else:
            name = Path(model_file_str).stem

        # extract hyperparameters and dump to json string (NOTE: only works if model has .get_params() function)
        params = model.get_params() if hasattr(model, "get_params") else {}
        params_str = json.dumps(params, default=str)
        
        new_row = [name, rmse, params_str]

        # append to csv file
        with open("reports/models_rmse.csv", mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(new_row)

    return rmse