"""
train_prophet.py — Prophet forecasting model for M5 retail data.

Fits a Facebook Prophet model per (store_id, item_id) series and generates
12-month ahead forecasts with 95% confidence intervals.

Requires: data/processed/processed_m5.csv (run preprocess.py first)
Output  : data/forecast/prophet/forecast_{store_id}_{safe_item_id}.csv
          Columns: ds, yhat, yhat_lower, yhat_upper, month_index
"""

import os
import warnings
import pandas as pd
from prophet import Prophet

warnings.filterwarnings("ignore")

PROCESSED_PATH   = "data/processed/processed_m5.csv"
FORECAST_DIR     = "data/forecast/prophet"
FORECAST_PERIODS = 12   # months ahead
MIN_SERIES_LEN   = 20   # Prophet needs at least this many data points


def safe_item_id(item_id: str) -> str:
    """Convert item_id to a filesystem-safe string (replace _ with -)."""
    return item_id.replace("_", "-")


def train_prophet_models():
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(
            f"Processed dataset not found at {PROCESSED_PATH}.\n"
            "Run  python src/preprocess.py  first."
        )

    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    os.makedirs(FORECAST_DIR, exist_ok=True)

    groups = list(df.groupby(["store_id", "item_id"]))
    print(f"Training Prophet for {len(groups)} store-item series…")
    success = skipped = 0

    for (store, item), grp in groups:
        grp = grp.sort_values("date")

        if len(grp) < MIN_SERIES_LEN:
            skipped += 1
            continue

        # Prophet requires columns named 'ds' and 'y'
        prophet_df = grp[["date", "monthly_sales"]].rename(
            columns={"date": "ds", "monthly_sales": "y"}
        )

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95,
            changepoint_prior_scale=0.05,   # mild — avoid overfitting short series
        )

        # Add holiday and SNAP regressors if available
        if "holiday_days" in grp.columns:
            prophet_df["holiday_days"] = grp["holiday_days"].values
            model.add_regressor("holiday_days")
        if "snap_days" in grp.columns:
            prophet_df["snap_days"] = grp["snap_days"].values
            model.add_regressor("snap_days")

        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=FORECAST_PERIODS, freq="MS")

        # Fill regressors for future periods with their training means
        if "holiday_days" in prophet_df.columns:
            future["holiday_days"] = prophet_df["holiday_days"].mean()
        if "snap_days" in prophet_df.columns:
            future["snap_days"] = prophet_df["snap_days"].mean()

        forecast = model.predict(future)

        # Save forecast horizon only (last FORECAST_PERIODS rows)
        horizon = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(FORECAST_PERIODS).copy()
        horizon["yhat"]        = horizon["yhat"].clip(lower=0)
        horizon["yhat_lower"]  = horizon["yhat_lower"].clip(lower=0)
        horizon["yhat_upper"]  = horizon["yhat_upper"].clip(lower=0)
        horizon["month_index"] = list(range(1, FORECAST_PERIODS + 1))

        out_path = os.path.join(FORECAST_DIR, f"forecast_{store}_{safe_item_id(item)}.csv")
        horizon.to_csv(out_path, index=False)
        success += 1

        if success % 25 == 0:
            print(f"  [{success}/{len(groups)}] {store} | {item}")

    print(f"\nProphet complete  — success: {success}  |  skipped (too short): {skipped}")
    print(f"Forecasts saved → {FORECAST_DIR}/")


if __name__ == "__main__":
    train_prophet_models()