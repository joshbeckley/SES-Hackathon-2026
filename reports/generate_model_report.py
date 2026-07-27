"""Generate model descriptions, sample statistics, RMSE, and series plots."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

OUTPUT_DIR = ROOT / "outputs" / "model_report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_data():
    train = pd.read_parquet(ROOT / "data" / "train.parquet").sort_values("date").reset_index(drop=True)
    test = pd.read_parquet(ROOT / "data" / "test.parquet").sort_values("date").reset_index(drop=True)
    for frame in (train, test):
        frame["usd_zar_28_movement"] = frame["usd_zar_28"] - frame["usd_zar"]

    data = pd.concat(
        [train.assign(_split="Estimation"), test.assign(_split="Forecast")],
        ignore_index=True,
    ).sort_values("date").reset_index(drop=True)
    data["interest_rate_diff"] = data["sa_repo_rate"] - data["us_fed_funds"]
    data["policy_rate_differential"] = data["interest_rate_diff"]
    data["sa_us_5y_yield_spread"] = data["sa_5y_yield"] - data["us_5y_yield"]
    log_fx = np.log(data["usd_zar"])
    data["fx_log_return_5d"] = log_fx.diff(5)
    data["fx_log_return_21d"] = log_fx.diff(21)
    data["fx_realized_vol_21d"] = log_fx.diff().shift(1).rolling(21).std()
    data["fx_realized_vol_63d"] = log_fx.diff().shift(1).rolling(63).std()
    data["sa_cpi_log_change_21d"] = np.log(data["sa_cpi"]).diff(21)
    data["sa_real_gdp_log_change_63d"] = np.log(data["sa_real_gdp"]).diff(63)
    return (
        data.loc[data["_split"].eq("Estimation")].reset_index(drop=True),
        data.loc[data["_split"].eq("Forecast")].reset_index(drop=True),
    )


MODEL_SPECS = {
    "LightGBM": {
        "description": "Gradient-boosted decision trees that capture nonlinearities and interactions.",
        "features": ["gold_usd_per_oz", "usd_zar_1w_return", "usd_zar_1m_volatility", "interest_rate_diff"],
        "model": lambda: LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15, random_state=42, verbosity=-1, n_jobs=1),
    },
    "XGBoost": {
        "description": "Regularised gradient-boosted trees, using shallow trees and row/column subsampling.",
        "features": ["gold_usd_per_oz", "usd_zar_1w_return", "usd_zar_1m_volatility", "interest_rate_diff"],
        "model": lambda: XGBRegressor(objective="reg:squarederror", n_estimators=500, learning_rate=0.03, max_depth=3, min_child_weight=5, gamma=0.1, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0, random_state=42, n_jobs=-1),
    },
    "Lasso": {
        "description": "Standardised linear regression with L1 shrinkage, which can set coefficients to zero.",
        "features": ["usd_zar", "brent_usd_per_barrel", "interest_rate_diff", "sa_us_5y_yield_spread", "sa_yoy_inflation", "sa_5y_cds_bp", "vix", "broad_usd_index", "sa_cpi", "usd_zar_1w_return", "usd_zar_1m_return", "usd_zar_3m_return", "usd_zar_1m_volatility"],
        "model": lambda: make_pipeline(StandardScaler(), Lasso(alpha=0.1, max_iter=50_000)),
    },
    "MLP": {
        "description": "A three-hidden-layer neural network for nonlinear predictor interactions.",
        "features": ["gold_usd_per_oz_return", "sa_yoy_inflation", "usd_zar_1m_return", "usd_zar_1m_volatility"],
        "model": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128, 64, 32), alpha=0.01, learning_rate="adaptive", max_iter=1000, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=42)),
    },
    "OLS": {
        "description": "Standardised ordinary least squares, imposing a linear additive relationship.",
        "features": ["broad_usd_index", "brent_usd_per_barrel", "sa_cpi", "usd_zar_1m_return", "fx_log_return_5d", "fx_log_return_21d", "sa_cpi_log_change_21d", "sa_real_gdp_log_change_63d"],
        "model": lambda: make_pipeline(StandardScaler(), LinearRegression()),
    },
    "Ridge": {
        "description": "Standardised linear regression with L2 shrinkage to stabilise correlated predictors.",
        "features": ["sa_repo_rate", "sa_real_gdp", "policy_rate_differential", "sa_us_5y_yield_spread", "sa_cpi_log_change_21d"],
        "model": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    },
    "Elastic Net": {
        "description": "Linear regression combining L1 feature selection and L2 coefficient shrinkage.",
        "features": ["sa_real_gdp", "sa_5y_cds_bp", "sa_5y_yield", "policy_rate_differential", "fx_log_return_5d", "fx_realized_vol_21d"],
        "model": lambda: TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), ElasticNet(alpha=0.02, l1_ratio=0.35, max_iter=20_000, random_state=42)), transformer=StandardScaler()),
    },
    "Huber": {
        "description": "Robust linear regression that reduces the influence of unusually large residuals.",
        "features": ["sa_repo_rate", "usd_zar_1m_return", "sa_cpi_log_change_21d"],
        "model": lambda: TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=2_000)), transformer=StandardScaler()),
    },
    "Spline Ridge": {
        "description": "Quadratic spline transformations followed by Ridge, allowing smooth nonlinear effects.",
        "features": ["iron_ore_usd_per_tonne", "sa_repo_rate", "sa_cpi", "sa_yoy_inflation", "usd_zar_1w_return", "usd_zar_1m_return", "sa_5y_yield", "interest_rate_diff", "fx_realized_vol_63d"],
        "model": lambda: make_pipeline(StandardScaler(), SplineTransformer(n_knots=4, degree=2, include_bias=False), Ridge(alpha=25.0)),
    },
}


def fit_ar1(train):
    log_fx = np.log(train["usd_zar"])
    return LinearRegression().fit(log_fx.shift(1).iloc[1:].to_frame(), log_fx.iloc[1:])


def ar1_forecast(model, spot, horizon=28):
    phi, intercept = float(model.coef_[0]), float(model.intercept_)
    log_spot = np.log(np.asarray(spot))
    if np.isclose(phi, 1):
        forecast = log_spot + intercept * horizon
    else:
        forecast = phi**horizon * log_spot + intercept * (1 - phi**horizon) / (1 - phi)
    return np.exp(forecast) - np.asarray(spot)


def plot_series(name, features, train, test):
    combined = pd.concat([train.assign(Period="Estimation"), test.assign(Period="Forecast")])
    columns = ["usd_zar_28_movement", *features]
    fig, axes = plt.subplots(len(columns), 1, figsize=(13, 2.2 * len(columns)), sharex=True)
    for ax, column in zip(axes, columns):
        ax.plot(combined["date"], combined[column], color="#315A7D", linewidth=0.8)
        ax.axvline(test["date"].min(), color="#C44E52", linestyle="--", linewidth=1)
        ax.set_ylabel(column.replace("_", " "), fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].set_title(f"{name}: target and model series\nRed dashed line = start of forecast period")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    filename = name.lower().replace(" ", "_") + "_series.png"
    fig.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def markdown_table(frame, index=False):
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    table = frame.reset_index() if index else frame.copy()
    table.columns = [str(column) for column in table.columns]
    rows = [
        "| " + " | ".join(table.columns) + " |",
        "| " + " | ".join(["---"] * len(table.columns)) + " |",
    ]
    for values in table.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(rows)


def main():
    train, test = prepare_data()
    target = "usd_zar_28_movement"
    ar1 = fit_ar1(train)
    summary_rows, rmse_rows = [], []

    for name, spec in MODEL_SPECS.items():
        features = spec["features"]
        train_valid = train[features + [target]].notna().all(axis=1)
        test_valid = test[features + [target]].notna().all(axis=1)
        model = spec["model"]().fit(train.loc[train_valid, features], train.loc[train_valid, target])
        summary_rows.append({
            "Model": name,
            "Estimation observations": int(train_valid.sum()),
            "Forecast observations": int(test_valid.sum()),
            "Estimation start": train.loc[train_valid, "date"].min().date(),
            "Estimation end": train.loc[train_valid, "date"].max().date(),
            "Forecast start": test.loc[test_valid, "date"].min().date(),
            "Forecast end": test.loc[test_valid, "date"].max().date(),
            "Predictors": len(features),
        })
        for period, frame, valid in [("Estimation", train, train_valid), ("Forecast", test, test_valid)]:
            actual = frame.loc[valid, target]
            rmse_rows.extend([
                {"Model": name, "Period": period, "Forecast method": "Model", "RMSE": root_mean_squared_error(actual, model.predict(frame.loc[valid, features]))},
                {"Model": name, "Period": period, "Forecast method": "AR(1)", "RMSE": root_mean_squared_error(actual, ar1_forecast(ar1, frame.loc[valid, "usd_zar"]))},
            ])
        plot_series(name, features, train, test)

    summary = pd.DataFrame(summary_rows)
    rmse = pd.DataFrame(rmse_rows)
    summary.to_csv(OUTPUT_DIR / "sample_summary.csv", index=False)
    rmse.to_csv(OUTPUT_DIR / "rmse_comparison.csv", index=False)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=False)
    colors = {"Model": "#4C78A8", "AR(1)": "#F58518"}
    for ax, (name, group) in zip(axes.flat, rmse.groupby("Model", sort=False)):
        x = np.arange(2)
        width = 0.36
        for offset, method in [(-width / 2, "Model"), (width / 2, "AR(1)")]:
            values = group[group["Forecast method"].eq(method)].set_index("Period").loc[["Estimation", "Forecast"], "RMSE"]
            bars = ax.bar(x + offset, values, width, label=method, color=colors[method])
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
        ax.set_title(name)
        ax.set_xticks(x, ["Estimation", "Forecast"])
        ax.set_ylabel("RMSE (USD/ZAR)")
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2)
    fig.suptitle("28-day movement RMSE: final models vs AR(1)", y=0.995, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUTPUT_DIR / "rmse_models_vs_ar1.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Final model report",
        "",
        "The target is the 28-observation-ahead USD/ZAR movement, `usd_zar_28 - usd_zar`. "
        "Every model is estimated only on the chronological training period; the 2021–2025 period is an untouched forecast evaluation. "
        "RMSE is in USD/ZAR units, and lower is better. AR(1) is fitted to log USD/ZAR on the estimation sample and iterated 28 steps.",
        "",
        "## Sample summary",
        "",
        markdown_table(summary),
        "",
        "## RMSE comparison",
        "",
        markdown_table(
            rmse.pivot(index=["Model", "Period"], columns="Forecast method", values="RMSE")
            .rename(columns={"Model": "Model RMSE", "AR(1)": "AR(1) RMSE"})
            .round(4),
            index=True,
        ),
        "",
        "![RMSE comparison](../outputs/model_report/rmse_models_vs_ar1.png)",
        "",
        "## Models and series",
        "",
    ]
    for name, spec in MODEL_SPECS.items():
        lines.extend([
            f"### {name}",
            "",
            spec["description"],
            "",
            "**Data used:** " + ", ".join(f"`{feature}`" for feature in spec["features"]) + ".",
            "",
            f"![{name} series](../outputs/model_report/{name.lower().replace(' ', '_')}_series.png)",
            "",
        ])
    (ROOT / "reports" / "final_model_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
