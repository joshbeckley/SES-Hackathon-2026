"""
AR(1) modeling of usd_zar_28 (28-day forward USD/ZAR rate)
------------------------------------------------------------
Fits three variants and compares them against a random-walk benchmark:

  1. AR(1) on levels          y_t = c + phi * y_{t-1} + e_t
  2. AR(1) on returns         r_t = c + phi * r_{t-1} + e_t   (r_t = pct change)
  3. ARX(1)                   y_t = c + phi * y_{t-1} + beta' X_t + e_t
                               (adds contemporaneous macro drivers already in
                               the dataset: fed funds, VIX, CDS spread, etc.)

Usage:
    python3 ar1_usd_zar_28.py --csv winning_dataset_final.csv
"""

import argparse
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


TARGET = "usd_zar_28"

# Candidate exogenous drivers for the ARX model (edit as needed)
EXOG_COLS = [
    "us_fed_funds",
    "us_5y_yield",
    "vix",
    "broad_usd_index",
    "sa_repo_rate",
    "sa_5y_cds_bp",
    "sa_yoy_inflation",
]


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def train_test_split(series: pd.Series, train_frac: float = 0.8):
    split = int(len(series) * train_frac)
    return series.iloc[:split], series.iloc[split:]


def evaluate_walk_forward(train: pd.Series, test: pd.Series, c: float, phi: float):
    """One-step-ahead walk-forward forecast using fixed (c, phi), vs random walk."""
    preds_model, preds_rw = [], []
    history = list(train.values)
    for actual in test.values:
        last = history[-1]
        preds_model.append(c + phi * last)
        preds_rw.append(last)
        history.append(actual)

    preds_model = np.array(preds_model)
    preds_rw = np.array(preds_rw)
    actual_arr = test.values

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    return {
        "rmse_model": rmse(actual_arr, preds_model),
        "mae_model": mae(actual_arr, preds_model),
        "rmse_rw": rmse(actual_arr, preds_rw),
        "mae_rw": mae(actual_arr, preds_rw),
    }


def fit_ar1_levels(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("1) AR(1) on LEVELS:", TARGET)
    print("=" * 60)

    y = df[TARGET].dropna().reset_index(drop=True)
    adf_stat, adf_p, *_ = adfuller(y)
    print(f"ADF test on level series: stat={adf_stat:.3f}, p-value={adf_p:.3f}")
    if adf_p > 0.05:
        print("-> Cannot reject unit root: series behaves like a random walk.")

    model = AutoReg(y, lags=1, old_names=False).fit()
    c, phi = model.params.iloc[0], model.params.iloc[1]
    print(model.summary())

    train, test = train_test_split(y)
    fit_train = AutoReg(train, lags=1, old_names=False).fit()
    metrics = evaluate_walk_forward(
        train, test, fit_train.params.iloc[0], fit_train.params.iloc[1]
    )
    print("\nOut-of-sample (last 20%), 1-step-ahead walk-forward:")
    print(f"  AR(1):        RMSE={metrics['rmse_model']:.4f}  MAE={metrics['mae_model']:.4f}")
    print(f"  Random walk:  RMSE={metrics['rmse_rw']:.4f}  MAE={metrics['mae_rw']:.4f}")

    return {"c": c, "phi": phi, "metrics": metrics}


def fit_ar1_returns(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("2) AR(1) on RETURNS: pct_change of", TARGET)
    print("=" * 60)

    r = df[TARGET].pct_change().dropna().reset_index(drop=True)
    adf_stat, adf_p, *_ = adfuller(r)
    print(f"ADF test on return series: stat={adf_stat:.3f}, p-value={adf_p:.3f}")
    if adf_p <= 0.05:
        print("-> Stationary: AR(1) on returns is well-posed.")

    model = AutoReg(r, lags=1, old_names=False).fit()
    c, phi = model.params.iloc[0], model.params.iloc[1]
    print(model.summary())

    train, test = train_test_split(r)
    fit_train = AutoReg(train, lags=1, old_names=False).fit()
    metrics = evaluate_walk_forward(
        train, test, fit_train.params.iloc[0], fit_train.params.iloc[1]
    )
    print("\nOut-of-sample (last 20%), 1-step-ahead walk-forward (on returns):")
    print(f"  AR(1):        RMSE={metrics['rmse_model']:.6f}  MAE={metrics['mae_model']:.6f}")
    print(f"  Random walk:  RMSE={metrics['rmse_rw']:.6f}  MAE={metrics['mae_rw']:.6f}")
    print("  (random-walk-on-returns benchmark = predicting 0% change)")

    return {"c": c, "phi": phi, "metrics": metrics}


def fit_arx(df: pd.DataFrame, exog_cols=EXOG_COLS):
    print("\n" + "=" * 60)
    print("3) ARX(1): levels + exogenous macro drivers")
    print("=" * 60)

    cols = [TARGET] + exog_cols
    data = df[cols].dropna().reset_index(drop=True)

    y = data[TARGET]
    y_lag = y.shift(1)
    X = data[exog_cols].copy()
    X["y_lag1"] = y_lag
    X = add_constant(X)

    valid = X.dropna().index.intersection(y.dropna().index)
    X_fit = X.loc[valid].iloc[1:]  # drop first row (NaN lag)
    y_fit = y.loc[valid].iloc[1:]

    model = OLS(y_fit, X_fit).fit()
    print(model.summary())

    return model


def main():
    parser = argparse.ArgumentParser(description="Fit AR(1) models to usd_zar_28")
    parser.add_argument(
        "--csv", default="winning_dataset_final.csv", help="Path to input CSV"
    )
    args = parser.parse_args()

    df = load_data(args.csv)

    levels_result = fit_ar1_levels(df)
    returns_result = fit_ar1_returns(df)
    arx_model = fit_arx(df)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Levels AR(1):  phi={levels_result['phi']:.4f}  "
          f"(RMSE {levels_result['metrics']['rmse_model']:.4f} vs "
          f"RW {levels_result['metrics']['rmse_rw']:.4f})")
    print(f"Returns AR(1): phi={returns_result['phi']:.4f}  "
          f"(RMSE {returns_result['metrics']['rmse_model']:.6f} vs "
          f"RW {returns_result['metrics']['rmse_rw']:.6f})")
    print(f"ARX(1) adj. R^2: {arx_model.rsquared_adj:.4f}")


if __name__ == "__main__":
    main()