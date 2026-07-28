import pandas as pd
import numpy as np
import statsmodels.api as sm

# --- Load data ---
daily = pd.read_csv("data/usdzar_clean.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
daily["log_usdzar"] = np.log(daily["usdzar"])

grid = pd.read_csv("data/friday_grid.csv", parse_dates=["target_friday", "actual_date"]).sort_values("actual_date").reset_index(drop=True)

# Fast lookup: date -> row position in the daily series
date_to_idx = {d: i for i, d in enumerate(daily["date"])}

# --- Step 1: for each origin Friday, find its "1 month ahead" target Friday ---
grid["target_date_approx"] = grid["actual_date"] + pd.DateOffset(months=1)

grid_sorted = grid.sort_values("actual_date")
matched = pd.merge_asof(
    grid.sort_values("target_date_approx"),
    grid_sorted[["actual_date", "usdzar"]].rename(
        columns={"actual_date": "future_actual_date", "usdzar": "future_usdzar"}
    ),
    left_on="target_date_approx",
    right_on="future_actual_date",
    direction="nearest",
)

matched = matched[matched["future_actual_date"] > matched["actual_date"]].reset_index(drop=True)

# --- Step 2: walk-forward AR(1) ---
results = []

for _, row in matched.iterrows():
    origin_date = row["actual_date"]
    target_date = row["future_actual_date"]
    actual_future_value = row["future_usdzar"]

    i_idx = date_to_idx[origin_date]
    j_idx = date_to_idx[target_date]
    h = j_idx - i_idx

    train = daily.iloc[: i_idx + 1]

    y = train["log_usdzar"].values
    y_t = y[1:]
    y_lag = y[:-1]

    X = sm.add_constant(y_lag)
    model = sm.OLS(y_t, X).fit()
    c, phi = model.params

    log_y_origin = train["log_usdzar"].iloc[-1]

    if abs(phi - 1) < 1e-8:
        log_forecast = log_y_origin + c * h
    else:
        log_forecast = (phi ** h) * log_y_origin + c * (1 - phi ** h) / (1 - phi)

    forecast_value = np.exp(log_forecast)
    error = forecast_value - actual_future_value

    results.append({
        "origin_date": origin_date,
        "target_date": target_date,
        "horizon_days": h,
        "phi": phi,
        "c": c,
        "forecast": forecast_value,
        "actual": actual_future_value,
        "error": error,
    })

results_df = pd.DataFrame(results)
rmse = np.sqrt(np.mean(results_df["error"] ** 2))

print(results_df.head())
print(results_df.tail())
print("\nNumber of forecasts:", len(results_df))
print("AR(1) RMSE:", round(rmse, 4))

results_df.to_csv("outputs/ar1_results.csv", index=False)