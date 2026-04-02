"""
train_arima.py — ARIMA demand forecasting for M5 retail data.

For each (store_id, item_id) time series:
  1. Tests stationarity with the Augmented Dickey-Fuller test.
  2. Auto-selects the best ARIMA(p, d, q) order by minimising AIC
     over a small grid (avoids heavy pmdarima dependency).
  3. Generates a 12-month ahead forecast with 95% confidence intervals.

Requires: data/processed/processed_m5.csv (run preprocess.py first)
Output  : data/forecast/arima/forecast_{store_id}_{safe_item_id}.csv
          Columns: ds, yhat, yhat_lower, yhat_upper, model_order, month_index
"""

import os
import warnings
import numpy as np
import pandas as pd
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

PROCESSED_PATH   = "data/processed/processed_m5.csv"
FORECAST_DIR     = "data/forecast/arima"
FORECAST_PERIODS = 12   # months ahead
MIN_SERIES_LEN   = 24   # need at least 24 months for reliable ARIMA

# Compact AIC grid — balances speed and model quality
P_RANGE = [0, 1, 2]
D_RANGE = [0, 1]
Q_RANGE = [0, 1, 2]


# ---------------------------------------------------------------------------
# Stationarity & order selection
# ---------------------------------------------------------------------------

def is_stationary(series: pd.Series, alpha: float = 0.05) -> bool:
    """Return True if the ADF test rejects the unit-root null at level alpha."""
    try:
        p_value = adfuller(series.dropna(), autolag="AIC")[1]
        return p_value < alpha
    except Exception:
        return False


def select_d(series: pd.Series) -> int:
    """Choose d = 0 (stationary) or d = 1 (one difference needed)."""
    if is_stationary(series):
        return 0
    if is_stationary(series.diff().dropna()):
        return 1
    return 1  # default


def auto_arima_aic(series: pd.Series, p_values, d_values, q_values):
    """
    Grid-search ARIMA(p, d, q) and return the model with the lowest AIC.
    Skips trivial ARIMA(0, d, 0) models.
    Returns (fitted_model, (p, d, q)) — or (None, None) on total failure.
    """
    best_aic   = np.inf
    best_order = None
    best_fit   = None

    for p, d, q in product(p_values, d_values, q_values):
        if p == 0 and q == 0:
            continue  # trivial white-noise / random-walk — not useful
        try:
            fit = ARIMA(series, order=(p, d, q)).fit()
            if fit.aic < best_aic:
                best_aic   = fit.aic
                best_order = (p, d, q)
                best_fit   = fit
        except Exception:
            continue

    return best_fit, best_order


# ---------------------------------------------------------------------------
# Single-series forecast
# ---------------------------------------------------------------------------

def forecast_one_series(series: pd.Series, periods: int = FORECAST_PERIODS):
    """
    Fit best ARIMA model and return a DataFrame of forecasted values.

    Returns (forecast_df, order) or (None, None) if the series is too
    short or all ARIMA orders fail to converge.
    """
    series = series.dropna()

    if len(series) < MIN_SERIES_LEN:
        return None, None

    # Step 1 — choose d via stationarity test
    d = select_d(series)

    # Step 2 — grid-search p and q
    model, order = auto_arima_aic(series, P_RANGE, [d], Q_RANGE)

    # Fallback if grid search fails entirely
    if model is None:
        try:
            model = ARIMA(series, order=(1, 1, 1)).fit()
            order = (1, 1, 1)
        except Exception:
            return None, None

    # Step 3 — generate forecast with 95% CI
    forecast_obj = model.get_forecast(steps=periods)
    summary      = forecast_obj.summary_frame(alpha=0.05)

    # Build a date index for the forecast horizon
    last_date   = series.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=periods,
        freq="MS",
    )

    result = pd.DataFrame({
        "ds":           future_dates,
        "yhat":         summary["mean"].values,
        "yhat_lower":   summary["mean_ci_lower"].values,
        "yhat_upper":   summary["mean_ci_upper"].values,
        "model_order":  [str(order)] * periods,
        "month_index":  list(range(1, periods + 1)),
    })

    # Sales cannot be negative
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        result[col] = result[col].clip(lower=0)

    return result, order


# ---------------------------------------------------------------------------
# Batch training
# ---------------------------------------------------------------------------

def train_arima_models():
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(
            f"Processed dataset not found at {PROCESSED_PATH}.\n"
            "Run  python src/preprocess.py  first."
        )

    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    os.makedirs(FORECAST_DIR, exist_ok=True)

    groups  = list(df.groupby(["store_id", "item_id"]))
    total   = len(groups)
    success = skipped = failed = 0

    print(f"Fitting ARIMA models for {total} store-item series…")
    print(f"Grid: p∈{P_RANGE}  d∈{D_RANGE}  q∈{Q_RANGE}\n")

    for (store, item), grp in groups:
        grp = grp.sort_values("date").set_index("date")
        series = grp["monthly_sales"].asfreq("MS")

        forecast_df, order = forecast_one_series(series)

        if forecast_df is None:
            skipped += 1
            continue

        safe_item = item.replace("_", "-")
        out_path  = os.path.join(FORECAST_DIR, f"forecast_{store}_{safe_item}.csv")

        try:
            forecast_df.to_csv(out_path, index=False)
            success += 1
        except Exception as e:
            print(f"  Save error for {store}|{item}: {e}")
            failed += 1
            continue

        if success % 25 == 0:
            pct = 100 * (success + skipped + failed) / total
            print(f"  [{success + skipped + failed}/{total}  {pct:.0f}%]  {store} | {item} | ARIMA{order}")

    print(f"\nARIMA complete.")
    print(f"  Success  : {success}")
    print(f"  Skipped (series too short < {MIN_SERIES_LEN} months): {skipped}")
    print(f"  Failed   : {failed}")
    print(f"  Forecasts saved → {FORECAST_DIR}/")


if __name__ == "__main__":
    train_arima_models()