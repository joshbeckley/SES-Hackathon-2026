"""
Simple test script for calculate_rmse().

Trains a basic RandomForestRegressor on dummy data, then tests:
1. Passing the model directly as model_obj
2. Passing the model via a pickled file (model_file_str)
3. Logging functionality (must_log=True)
"""

import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .model_test import calculate_rmse  # adjust import to match your filename
from pathlib import Path

# --- 1. Create simple dummy data ---
X_train = pd.DataFrame({
    "feature1": [1, 2, 3, 4, 5, 6],
    "feature2": [10, 20, 30, 40, 50, 60],
})
y_train = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])

X_test = pd.DataFrame({
    "feature1": [7, 8],
    "feature2": [70, 80],
})
y_test = pd.Series([7.5, 8.5])

# --- 2. Train a simple model ---
model = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# --- 3. Save model to a pickle file (to test the file-loading path) ---
model_path = "models/test_model.pkl"
model_name = Path(model_path).stem + ".pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

# --- Test 1: pass model directly (model_obj) ---
rmse_obj = calculate_rmse(X_test, y_test, model_obj=model, model_name="rf_test")
print(f"[Test 1] RMSE (model_obj): {rmse_obj}")
assert rmse_obj is not None, "Expected a valid RMSE value, got None"

# --- Test 2: pass model via file path (model_file_str) ---
rmse_file = calculate_rmse(X_test, y_test, model_file_str=model_name)
print(f"[Test 2] RMSE (model_file_str): {rmse_file}")
assert rmse_file is not None, "Expected a valid RMSE value, got None"

# --- Test 3: no model provided at all -> should return None ---
rmse_none = calculate_rmse(X_test, y_test)
print(f"[Test 3] RMSE (no model provided): {rmse_none}")
assert rmse_none is None, "Expected None when no model is provided"

# --- Test 4: logging enabled ---
rmse_logged = calculate_rmse(X_test, y_test, model_obj=model, model_name="rf_logged", must_log=True)
print(f"[Test 4] RMSE (logged): {rmse_logged}")
print("Check ../reports/models_rmse.csv for the new logged row.")

print("\nAll tests completed.")