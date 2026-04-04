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
FORECAST_PERIODS = 12
MIN_SERIES_LEN   = 24

P_RANGE = [0, 1, 2]
D_RANGE = [0, 1]
Q_RANGE = [0, 1, 2]


# ---------------------------------------------------------------------------
# Stationarity & order selection
# ---------------------------------------------------------------------------

def is_stationary(series: pd.Series, alpha: float = 0.05) -> bool:

    try:
        p_value = adfuller(series.dropna(), autolag="AIC")[1]
        return p_value < alpha
    except Exception:
        return False


def select_d(series: pd.Series) -> int:
    if is_stationary(series):
        return 0
    if is_stationary(series.diff().dropna()):
        return 1
    return 1  # default


def auto_arima_aic(series: pd.Series, p_values, d_values, q_values):
    
    best_aic   = np.inf
    best_order = None
    best_fit   = None

    for p, d, q in product(p_values, d_values, q_values):
        if p == 0 and q == 0:
            continue
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
    
    series = series.dropna()

    if len(series) < MIN_SERIES_LEN:
        return None, None

    d = select_d(series)

    model, order = auto_arima_aic(series, P_RANGE, [d], Q_RANGE)

    if model is None:
        try:
            model = ARIMA(series, order=(1, 1, 1)).fit()
            order = (1, 1, 1)
        except Exception:
            return None, None

    forecast_obj = model.get_forecast(steps=periods)
    summary      = forecast_obj.summary_frame(alpha=0.05)

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