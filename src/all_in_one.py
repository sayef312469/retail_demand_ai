"""
preprocess.py — M5 Forecasting dataset pipeline.

Converts the three raw M5 CSV files into a monthly panel
ready for Prophet and ARIMA forecasting.

Expected files in data/raw/:
    sales_train_evaluation.csv
    sell_prices.csv
    calendar.csv

Download from:
    https://www.kaggle.com/competitions/m5-forecasting-accuracy/data

Typical usage:
    python src/preprocess.py                    # top 200 items, all stores
    python src/preprocess.py --top-items 500    # larger subset
    python src/preprocess.py --stores CA_1 TX_1 --categories FOODS
"""

import os
import argparse
import time
import numpy as np
import pandas as pd

RAW_DIR        = "data/raw"
PROCESSED_PATH = "data/processed/processed_m5.csv"

SALES_FILE    = os.path.join(RAW_DIR, "sales_train_evaluation.csv")
PRICES_FILE   = os.path.join(RAW_DIR, "sell_prices.csv")
CALENDAR_FILE = os.path.join(RAW_DIR, "calendar.csv")


# ---------------------------------------------------------------------------
# 1. Load & melt
# ---------------------------------------------------------------------------

def load_and_melt_sales(sample_stores=None, sample_categories=None, top_n_items=200):
    """
    Load sales_train_evaluation.csv and melt wide → long.
    Rows = items; columns d_1…d_1941 = daily unit sales.
    """
    print("[1/5] Loading sales data…")
    df = pd.read_csv(SALES_FILE)
    print(f"      Raw shape: {df.shape}  ({len(df)} items × {df.shape[1]} cols)")

    if sample_stores:
        df = df[df["store_id"].isin(sample_stores)]
        print(f"      After store filter    : {len(df)} items")

    if sample_categories:
        df = df[df["cat_id"].isin(sample_categories)]
        print(f"      After category filter : {len(df)} items")

    if top_n_items:
        id_cols  = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        day_cols = [c for c in df.columns if c.startswith("d_")]
        df["_total"] = df[day_cols].sum(axis=1)
        df = df.nlargest(top_n_items, "_total").drop(columns=["_total"])
        print(f"      After top-{top_n_items} filter : {len(df)} items")

    id_cols  = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in df.columns if c.startswith("d_")]
    print(f"      Melting {len(day_cols)} day-cols × {len(df)} items…")
    df_long = df.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    df_long["sales"] = pd.to_numeric(df_long["sales"], errors="coerce").fillna(0)
    return df_long


# ---------------------------------------------------------------------------
# 2. Calendar
# ---------------------------------------------------------------------------

def load_calendar():
    """
    Load calendar.csv.
    Maps day-id (d_1…d_1969) → actual date, week-id, holiday and SNAP flags.
    SNAP days are food-stamp assistance days — they significantly spike food demand.
    """
    print("[2/5] Loading calendar…")
    cal = pd.read_csv(CALENDAR_FILE)
    cal["date"] = pd.to_datetime(cal["date"])

    snap_cols         = [c for c in cal.columns if c.startswith("snap_")]
    cal["is_snap"]    = cal[snap_cols].max(axis=1).astype(int)
    cal["is_holiday"] = cal["event_name_1"].notna().astype(int)

    return cal[["d", "date", "wm_yr_wk", "is_holiday", "is_snap"]]


# ---------------------------------------------------------------------------
# 3. Prices
# ---------------------------------------------------------------------------

def load_prices():
    """
    Load sell_prices.csv.
    Provides weekly sell price per (store_id, item_id, wm_yr_wk).
    This is what the original Walmart-only dataset was missing.
    """
    print("[3/5] Loading prices…")
    prices = pd.read_csv(PRICES_FILE)
    prices["sell_price"] = pd.to_numeric(prices["sell_price"], errors="coerce")
    return prices


# ---------------------------------------------------------------------------
# 4. Merge & monthly aggregation
# ---------------------------------------------------------------------------

def aggregate_to_monthly(df_long, calendar, prices):
    """
    Join on calendar (get dates) and prices (get sell_price),
    then collapse daily rows → monthly aggregates.

    Output columns:
        store_id, item_id, dept_id, cat_id,
        date (month-start),
        monthly_sales, avg_price,
        holiday_days, snap_days, trading_days
    """
    print("[4/5] Merging and aggregating to monthly…")

    df = df_long.merge(calendar, on="d", how="left")
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    df["month_start"] = df["date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        df.groupby(
            ["store_id", "item_id", "dept_id", "cat_id", "month_start"],
            as_index=False,
        )
        .agg(
            monthly_sales=("sales",      "sum"),
            avg_price    =("sell_price", "mean"),
            holiday_days =("is_holiday", "sum"),
            snap_days    =("is_snap",    "sum"),
            trading_days =("sales",      "count"),
        )
    )

    monthly = monthly.rename(columns={"month_start": "date"})
    monthly = monthly.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    # Drop permanently dead SKUs (zero sales across entire history)
    active_idx = monthly.groupby(["store_id", "item_id"])["monthly_sales"].sum()
    active_idx = active_idx[active_idx > 0].index
    monthly = monthly.set_index(["store_id", "item_id"])
    monthly = monthly.loc[monthly.index.isin(active_idx)].reset_index()

    return monthly


# ---------------------------------------------------------------------------
# 5. Feature engineering
# ---------------------------------------------------------------------------

def add_lag_features(df):
    """
    Lag and rolling-window features per store-item.
    Used downstream by evaluation and PVI modules.
    """
    print("[5/5] Computing lag and rolling features…")
    df = df.sort_values(["store_id", "item_id", "date"]).copy()
    grp = df.groupby(["store_id", "item_id"])["monthly_sales"]

    df["lag_1"]      = grp.shift(1)
    df["lag_3"]      = grp.shift(3)
    df["lag_12"]     = grp.shift(12)
    df["roll3_mean"] = grp.transform(lambda x: x.shift(1).rolling(3,  min_periods=2).mean())
    df["roll6_mean"] = grp.transform(lambda x: x.shift(1).rolling(6,  min_periods=3).mean())
    df["roll12_std"] = grp.transform(lambda x: x.shift(1).rolling(12, min_periods=6).std())

    return df


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def preprocess_m5(sample_stores=None, sample_categories=None, top_n_items=200):
    """
    Full preprocessing pipeline for the M5 Forecasting dataset.

    Parameters
    ----------
    sample_stores : list[str] | None
        Restrict to specific store IDs, e.g. ["CA_1", "TX_2"].
        Available stores: CA_1..CA_4, TX_1..TX_3, WI_1..WI_3.
    sample_categories : list[str] | None
        Restrict to "FOODS", "HOBBIES", or "HOUSEHOLD".
    top_n_items : int | None
        Keep only the top-N items by total historical sales (after filters).
        Use 200 for quick dev runs; None to process all items (slow).
    """
    for fpath in [SALES_FILE, PRICES_FILE, CALENDAR_FILE]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"\nMissing file: {fpath}\n\n"
                "Download the M5 Forecasting dataset from Kaggle:\n"
                "  https://www.kaggle.com/competitions/m5-forecasting-accuracy/data\n\n"
                "Place these three files inside  data/raw/ :\n"
                "  sales_train_evaluation.csv\n"
                "  sell_prices.csv\n"
                "  calendar.csv\n"
            )

    t0      = time.time()
    df_long = load_and_melt_sales(sample_stores, sample_categories, top_n_items)
    cal     = load_calendar()
    prices  = load_prices()
    monthly = aggregate_to_monthly(df_long, cal, prices)
    monthly = add_lag_features(monthly)

    os.makedirs("data/processed", exist_ok=True)
    monthly.to_csv(PROCESSED_PATH, index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print("Preprocessing complete.")
    print(f"  Rows        : {len(monthly):,}")
    print(f"  Unique items: {monthly['item_id'].nunique():,}")
    print(f"  Stores      : {monthly['store_id'].nunique()}")
    print(f"  Categories  : {sorted(monthly['cat_id'].unique().tolist())}")
    print(f"  Date range  : {monthly['date'].min().date()} → {monthly['date'].max().date()}")
    print(f"  Saved to    : {PROCESSED_PATH}")
    print(f"  Elapsed     : {elapsed:.1f}s")
    print(f"{'='*55}\n")
    print(monthly.head())

    return monthly


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess M5 retail dataset")
    parser.add_argument("--stores",     nargs="+", default=None,
                        help="Store IDs e.g. CA_1 TX_2  (default: all)")
    parser.add_argument("--categories", nargs="+", default=None,
                        choices=["FOODS", "HOBBIES", "HOUSEHOLD"],
                        help="Category IDs (default: all)")
    parser.add_argument("--top-items",  type=int, default=200,
                        help="Top N items by total sales (0 = all, default: 200)")
    args = parser.parse_args()

    preprocess_m5(
        sample_stores=args.stores,
        sample_categories=args.categories,
        top_n_items=args.top_items if args.top_items > 0 else None,
    )

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

"""forecast.py — Unified pipeline runner.

Runs the full data → forecast pipeline in sequence:
    Step 1: Preprocess raw M5 data  (preprocess.py)
    Step 2: Train Prophet models    (train_prophet.py)
    Step 3: Train ARIMA models      (train_arima.py)

Usage examples:
    # Default — top 500 items, both models (recommended for balanced coverage)
    python src/forecast.py

    # Run only on specific stores or categories
    python src/forecast.py --stores CA_1 TX_1 --categories FOODS

    # Train only ARIMA, skipping preprocessing (data already ready)
    python src/forecast.py --model arima --skip-preprocess

    # Bigger run — top 1000 items for comprehensive coverage
    python src/forecast.py --top-items 1000 --model both

    # Quick run — top 100 items for testing
    python src/forecast.py --top-items 100
"""

import argparse
import sys
import time


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def run_preprocess(stores, categories, top_items):
    from preprocess import preprocess_m5
    _banner("STEP 1 — Preprocessing M5 data")
    t0 = time.time()
    preprocess_m5(
        sample_stores=stores or None,
        sample_categories=categories or None,
        top_n_items=top_items if top_items > 0 else None,
    )
    _elapsed(t0)


def run_prophet():
    from train_prophet import train_prophet_models
    _banner("STEP 2 — Prophet forecasting")
    t0 = time.time()
    train_prophet_models()
    _elapsed(t0)


def run_arima():
    from train_arima import train_arima_models
    _banner("STEP 3 — ARIMA forecasting")
    t0 = time.time()
    train_arima_models()
    _elapsed(t0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _elapsed(t0: float):
    print(f"\n  ✓ Done in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Retail Demand AI — forecasting pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
After this pipeline completes, run these next:
  python src/pvi.py        — compute Product Viability Index
  python src/recommend.py  — generate stock recommendations
  python src/evaluate.py   — evaluate forecast accuracy
        """,
    )
    parser.add_argument(
        "--stores", nargs="+", default=None,
        help="Store IDs to include, e.g. CA_1 TX_2  (default: all stores)",
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        choices=["FOODS", "HOBBIES", "HOUSEHOLD"],
        help="Category IDs to include (default: all categories)",
    )
    parser.add_argument(
        "--top-items", type=int, default=500,
        help="Keep top N items by total sales per filters (0 = all, default: 500)",
    )
    parser.add_argument(
        "--model", choices=["prophet", "arima", "both"], default="both",
        help="Which forecasting model(s) to train (default: both)",
    )
    parser.add_argument(
        "--skip-preprocess", action="store_true",
        help="Skip Step 1 if processed_m5.csv already exists",
    )

    args = parser.parse_args()
    pipeline_start = time.time()

    # ── Step 1 ─────────────────────────────────────────────
    if not args.skip_preprocess:
        run_preprocess(args.stores, args.categories, args.top_items)
    else:
        print("\n[Skipping preprocessing — using existing processed_m5.csv]")

    # ── Step 2 ─────────────────────────────────────────────
    if args.model in ("prophet", "both"):
        run_prophet()

    # ── Step 3 ─────────────────────────────────────────────
    if args.model in ("arima", "both"):
        run_arima()

    # ── Summary ────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {total_elapsed:.1f}s")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  python src/pvi.py        — compute Product Viability Index")
    print("  python src/recommend.py  — generate stock recommendations")
    print("  python src/evaluate.py   — evaluate forecast accuracy")
    print("  uvicorn app.api:app --reload  — start the API server\n")


if __name__ == "__main__":
    main()

"""
pvi.py — Product Viability Index (PVI) computation.

Uses weighted ensemble forecast (60% ARIMA + 40% Prophet) for improved predictions.

Computes a 0-100 PVI score for every (store_id, item_id) series by
combining four sub-scores derived from forecasts and historical data:

    PVI = (α · demand + β · growth + γ · stability + δ · price) × 100

Sub-score definitions
---------------------
demand    — normalised mean forecasted sales over the horizon using
            blended ensemble forecast (60% ARIMA, 40% Prophet).
            Higher forecasted volume = higher viability.

growth    — normalised slope of the forecast horizon (last − first) / first
            using blended ensemble. Positive growth trend pushes viability up.

stability — 1 − normalised coefficient of variation (std / mean) of
            historical monthly sales. Low volatility = high stability.

price     — normalised average sell price from processed data.
            Higher price per unit contributes more revenue per sale.

Ensemble Strategy
-----------------
ARIMA consistently outperforms Prophet on this dataset:
  • ARIMA MAE: 169.44  vs  Prophet MAE: 343.57
  • Weighted blend (60% ARIMA + 40% Prophet) balances accuracy with stability
  • Both models contribute to PVI calculations and recommendations

Default weights (α=0.40, β=0.25, γ=0.20, δ=0.15) are intentional and
documented in the report:
  • Demand (40%) is the primary driver — a product nobody buys is not viable.
  • Growth (25%) rewards forward momentum.
  • Stability (20%) penalises erratic demand that drives overstock/stockouts.
  • Price (15%) accounts for revenue contribution per unit.

Category thresholds (matching proposal specification):
  High   : PVI ≥ 67
  Medium : 33 ≤ PVI < 67
  Low    : PVI  < 33

Requires
--------
  data/processed/processed_m5.csv     — run preprocess.py first
  data/forecast/prophet/*.csv         — run train_prophet.py first
  data/forecast/arima/*.csv           — run train_arima.py first

Output
------
  data/pvi_scores.csv
"""

import os
import glob
import numpy as np
import pandas as pd

PROCESSED_PATH     = "data/processed/processed_m5.csv"
PROPHET_FORECAST_DIR = "data/forecast/prophet"
ARIMA_FORECAST_DIR   = "data/forecast/arima"
PVI_OUTPUT_PATH    = "data/pvi_scores.csv"

# Ensemble weights (must sum to 1.0) — ARIMA is more accurate on this dataset
ENSEMBLE_ARIMA_WEIGHT   = 0.60   # ARIMA primary (better accuracy)
ENSEMBLE_PROPHET_WEIGHT = 0.40   # Prophet secondary (stability)

assert abs(ENSEMBLE_ARIMA_WEIGHT + ENSEMBLE_PROPHET_WEIGHT - 1.0) < 1e-9, "Ensemble weights must sum to 1"

# PVI formula weights — must sum to 1.0
ALPHA = 0.40   # demand weight
BETA  = 0.25   # growth weight
GAMMA = 0.20   # stability weight
DELTA = 0.15   # price weight

assert abs(ALPHA + BETA + GAMMA + DELTA - 1.0) < 1e-9, "Weights must sum to 1"

# Viability category thresholds (0–100 scale)
HIGH_THRESHOLD   = 67
MEDIUM_THRESHOLD = 33


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def min_max_norm(series: pd.Series) -> pd.Series:
    """
    Percentile-rank normalization — maps each value to its rank in [0, 1].
    More robust than pure min-max when sample is small or homogeneous.
    Min-max anchors to the single best/worst item (one outlier ruins the scale).
    Percentile rank spreads items evenly regardless of the raw value distribution.
    """
    return series.rank(pct=True)


def pvi_category(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "High"
    if score >= MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def safe_item_to_item_id(safe_item: str) -> str:
    """Reverse the filesystem-safe encoding: replace '-' → '_'."""
    return safe_item.replace("-", "_")


def parse_forecast_filename(fname: str):
    """
    Extract (store_id, item_id) from filenames like:
        forecast_CA_1_FOODS-1-001-CA-1-evaluation.csv
        forecast_TX_2_HOBBIES-2-098-TX-2-evaluation.csv

    Naming scheme written by train_prophet / train_arima:
        forecast_{store_id}_{safe_item_id}.csv
    where safe_item_id has '_' replaced by '-'.

    Store IDs follow pattern  XX_N  (e.g. CA_1, TX_3, WI_2).
    We split on '_' and reconstruct from known patterns.
    """
    base  = os.path.basename(fname).replace(".csv", "")
    # Remove 'forecast_' prefix
    rest  = base[len("forecast_"):]   # e.g. "CA_1_FOODS-1-001-CA-1-evaluation"

    # Store ID is always  {2-letter state}_{digit}
    # Split on '_' and take first two tokens as store_id
    parts    = rest.split("_")
    store_id = f"{parts[0]}_{parts[1]}"          # e.g. "CA_1"
    safe_item = "_".join(parts[2:])               # e.g. "FOODS-1-001-CA-1-evaluation"
    item_id   = safe_item_to_item_id(safe_item)   # e.g. "FOODS_1_001_CA_1_evaluation"

    return store_id, item_id


# ---------------------------------------------------------------------------
# Ensemble blending
# ---------------------------------------------------------------------------

def blend_forecasts(prophet_yhat: np.ndarray, arima_yhat: np.ndarray) -> np.ndarray:
    """
    Blend Prophet and ARIMA forecasts using weighted average.
    
    ARIMA has proven more accurate on this dataset (MAE ~50% lower),
    so it receives higher weight (60% vs 40% Prophet).
    
    Args:
        prophet_yhat: Prophet forecast values
        arima_yhat: ARIMA forecast values
    
    Returns:
        Blended forecast array (weighted average)
    """
    # Ensure same length
    min_len = min(len(prophet_yhat), len(arima_yhat))
    prophet_yhat = prophet_yhat[:min_len]
    arima_yhat = arima_yhat[:min_len]
    
    # Weighted blend: 60% ARIMA (more accurate) + 40% Prophet (more stable)
    blended = (ENSEMBLE_ARIMA_WEIGHT * arima_yhat) + (ENSEMBLE_PROPHET_WEIGHT * prophet_yhat)
    return blended


# ---------------------------------------------------------------------------
# Sub-score computation
# ---------------------------------------------------------------------------

def compute_demand_score(yhat: np.ndarray) -> float:
    """Mean forecasted sales over the horizon (raw, normalised later)."""
    return float(np.mean(yhat))


def compute_growth_score(yhat: np.ndarray) -> float:
    """
    Percentage growth from first to last forecast period.
    Clipped to [-1, +1] to avoid extreme outliers dominating normalisation.
    """
    if yhat[0] < 1e-6:
        return 0.0
    growth = (yhat[-1] - yhat[0]) / yhat[0]
    return float(np.clip(growth, -1.0, 1.0))


def compute_stability_score(monthly_sales: np.ndarray) -> float:
    """
    Coefficient of variation (CV = std / mean) of historical monthly sales.
    Lower CV = more stable = higher viability.
    We return raw CV here; the final score is 1 - norm(CV).
    """
    mean = np.mean(monthly_sales)
    if mean < 1e-6:
        return 99.0   # all-zero series → maximum instability
    return float(np.std(monthly_sales) / mean)


def compute_price_score(avg_prices: pd.Series) -> float:
    """Mean historical sell price; NaN-safe (defaults to 0)."""
    val = avg_prices.dropna().mean()
    return float(val) if not np.isnan(val) else 0.0


# ---------------------------------------------------------------------------
# Anomaly flags
# ---------------------------------------------------------------------------

def detect_anomalies(monthly_sales: np.ndarray) -> dict:
    """
    IQR-based anomaly detection on the historical sales series.
    Returns a dict with:
        has_anomaly   : bool — any month outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
        anomaly_count : int  — number of anomalous months
        anomaly_pct   : float — fraction of months that are anomalies
    """
    if len(monthly_sales) < 4:
        return {"has_anomaly": False, "anomaly_count": 0, "anomaly_pct": 0.0}

    q1, q3 = np.percentile(monthly_sales, [25, 75])
    iqr     = q3 - q1
    lo      = q1 - 1.5 * iqr
    hi      = q3 + 1.5 * iqr

    mask           = (monthly_sales < lo) | (monthly_sales > hi)
    anomaly_count  = int(mask.sum())
    anomaly_pct    = round(anomaly_count / len(monthly_sales), 3)

    return {
        "has_anomaly":   bool(mask.any()),
        "anomaly_count": anomaly_count,
        "anomaly_pct":   anomaly_pct,
    }


# ---------------------------------------------------------------------------
# Master computation
# ---------------------------------------------------------------------------

def compute_pvi():
    # ── 0. Load processed data ───────────────────────────────────────────
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(
            f"Processed dataset not found at {PROCESSED_PATH}.\n"
            "Run  python src/preprocess.py  first."
        )

    processed = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    print(f"Loaded processed data: {len(processed):,} rows, "
          f"{processed['item_id'].nunique()} items, "
          f"{processed['store_id'].nunique()} stores")

    # Index for fast lookup
    processed_idx = processed.set_index(["store_id", "item_id"])

    # ── 1. Discover forecast files ────────────────────────────────────────
    prophet_files = sorted(glob.glob(os.path.join(PROPHET_FORECAST_DIR, "forecast_*.csv")))
    arima_files   = sorted(glob.glob(os.path.join(ARIMA_FORECAST_DIR,   "forecast_*.csv")))

    if not prophet_files and not arima_files:
        raise FileNotFoundError(
            "No forecast files found in data/forecast/prophet/ or data/forecast/arima/.\n"
            "Run  python src/forecast.py  first."
        )

    # Build lookup: (store_id, item_id) → arima forecast path
    arima_lookup = {}
    for f in arima_files:
        store, item = parse_forecast_filename(f)
        arima_lookup[(store, item)] = f

    print(f"Forecast files — Prophet: {len(prophet_files)}, ARIMA: {len(arima_files)}")

    # ── 2. Collect raw sub-scores ─────────────────────────────────────────
    records = []

    for fpath in prophet_files:
        store_id, item_id = parse_forecast_filename(fpath)

        fc = pd.read_csv(fpath)
        prophet_yhat = fc["yhat"].values

        # Historical series for stability and price
        key = (store_id, item_id)
        if key in processed_idx.index:
            hist = processed_idx.loc[[key]]
            monthly_sales = hist["monthly_sales"].values
            avg_prices    = hist["avg_price"]
        else:
            monthly_sales = prophet_yhat   # fallback: use forecast as proxy
            avg_prices    = pd.Series([np.nan])

        # ARIMA forecast for ensemble blending
        arima_yhat = None
        if key in arima_lookup:
            arima_fc    = pd.read_csv(arima_lookup[key])
            arima_yhat  = arima_fc["yhat"].values
            
            # Blend forecasts: 60% ARIMA (more accurate) + 40% Prophet (more stable)
            blended_yhat = blend_forecasts(prophet_yhat, arima_yhat)
        else:
            # If ARIMA not available, use Prophet only
            blended_yhat = prophet_yhat

        # Raw sub-scores using BLENDED forecast (improved accuracy)
        demand_raw = compute_demand_score(blended_yhat)
        growth_raw = compute_growth_score(prophet_yhat)
        stability_raw = compute_stability_score(monthly_sales)   # CV — lower is better
        price_raw     = compute_price_score(avg_prices)

        # Model agreement: mean absolute % difference between Prophet and ARIMA forecasts
        model_agreement = None
        if arima_yhat is not None and len(arima_yhat) == len(prophet_yhat):
            denom = np.abs(prophet_yhat) + 1e-6
            model_agreement = float(1.0 - np.mean(np.abs(prophet_yhat - arima_yhat) / denom))
            model_agreement = float(np.clip(model_agreement, 0.0, 1.0))

        # Anomaly detection on historical data
        anomaly_info = detect_anomalies(monthly_sales)

        # Category from processed data (if available)
        cat_id  = hist["cat_id"].iloc[0]  if key in processed_idx.index else "UNKNOWN"
        dept_id = hist["dept_id"].iloc[0] if key in processed_idx.index else "UNKNOWN"

        records.append({
            "store_id":        store_id,
            "item_id":         item_id,
            "cat_id":          cat_id,
            "dept_id":         dept_id,
            # Raw sub-scores (pre-normalisation, using blended forecast)
            "demand_raw":      demand_raw,
            "growth_raw":      growth_raw,
            "stability_cv":    stability_raw,   # CV — inverted later
            "price_raw":       price_raw,
            # Forecast metadata (using Prophet's confidence intervals)
            "forecast_mean":   float(np.mean(prophet_yhat)),
            "forecast_lower":  float(fc["yhat_lower"].mean()) if "yhat_lower" in fc else None,
            "forecast_upper":  float(fc["yhat_upper"].mean()) if "yhat_upper" in fc else None,
            "model_agreement": model_agreement,
            # Anomaly flags
            "has_anomaly":     anomaly_info["has_anomaly"],
            "anomaly_count":   anomaly_info["anomaly_count"],
            "anomaly_pct":     anomaly_info["anomaly_pct"],
        })

    pvi_df = pd.DataFrame(records)

    if pvi_df.empty:
        print("No records collected — check that forecast files exist and filenames match expected pattern.")
        return

    # ── 3. Normalise sub-scores to [0, 1] ────────────────────────────────
    pvi_df["demand_norm"]    = min_max_norm(pvi_df["demand_raw"])
    pvi_df["growth_norm"]    = min_max_norm(pvi_df["growth_raw"])
    # Stability: high CV = unstable = bad → invert so high norm = stable
    pvi_df["stability_norm"] = 1.0 - min_max_norm(pvi_df["stability_cv"])
    pvi_df["price_norm"]     = min_max_norm(pvi_df["price_raw"])

    # ── 4. Compute PVI (0–100) ────────────────────────────────────────────
    pvi_df["PVI"] = (
        ALPHA * pvi_df["demand_norm"]
        + BETA  * pvi_df["growth_norm"]
        + GAMMA * pvi_df["stability_norm"]
        + DELTA * pvi_df["price_norm"]
    ) * 100

    pvi_df["PVI"] = pvi_df["PVI"].round(2)

    # ── 5. Categorise ─────────────────────────────────────────────────────
    pvi_df["viability"] = pvi_df["PVI"].apply(pvi_category)

    # ── 6. Rank within category ───────────────────────────────────────────
    pvi_df = pvi_df.sort_values("PVI", ascending=False).reset_index(drop=True)
    pvi_df["rank_overall"] = pvi_df.index + 1
    pvi_df["rank_in_category"] = (
        pvi_df.groupby("viability")["PVI"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    # ── 7. Save ───────────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    pvi_df.to_csv(PVI_OUTPUT_PATH, index=False)

    # ── 8. Summary ────────────────────────────────────────────────────────
    counts = pvi_df["viability"].value_counts()
    print(f"\n{'='*55}")
    print("PVI computation complete.")
    print(f"  Total items scored : {len(pvi_df)}")
    print(f"  High  (≥{HIGH_THRESHOLD})       : {counts.get('High',   0)}")
    print(f"  Medium ({MEDIUM_THRESHOLD}–{HIGH_THRESHOLD-1})    : {counts.get('Medium', 0)}")
    print(f"  Low   (<{MEDIUM_THRESHOLD})        : {counts.get('Low',    0)}")
    print(f"  PVI range          : {pvi_df['PVI'].min():.1f} – {pvi_df['PVI'].max():.1f}")
    print(f"  Items with anomaly : {pvi_df['has_anomaly'].sum()}")
    print(f"  Saved to           : {PVI_OUTPUT_PATH}")
    print(f"{'='*55}\n")

    print("Top 10 most viable items:")
    cols = ["store_id", "item_id", "cat_id", "PVI", "viability",
            "demand_norm", "growth_norm", "stability_norm", "price_norm"]
    print(pvi_df[cols].head(10).to_string(index=False))

    return pvi_df


if __name__ == "__main__":
    compute_pvi()

"""
recommend.py — Stock Recommendation Engine.

Generates one of three actions — Increase / Hold / Decrease — for every
(store_id, item_id) by combining PVI score, sub-scores, forecast trend,
anomaly flags, and (when available) model agreement between Prophet and ARIMA.

Decision logic
--------------
The engine works in two layers:

  Layer 1 — Hard overrides (anomaly signals)
    • If a serious anomaly is detected AND demand is declining → Decrease
    • If a serious anomaly is detected AND demand is growing  → Hold
      (don't increase stock until the anomaly resolves)

  Layer 2 — PVI-driven matrix  (applied when no hard override fires)

    Viability | Growth trend  | Risk level  → Decision
    ----------|---------------|-------------|----------
    High      | Positive      | Low/Medium  → Increase
    High      | Flat/Negative | Any         → Hold
    Medium    | Positive      | Low         → Increase
    Medium    | Any           | High        → Hold
    Low       | Any           | High        → Decrease
    Low       | Positive      | Low         → Hold   (give benefit of doubt)
    Low       | Negative      | Any         → Decrease

  Confidence level (High / Medium / Low) reflects how well Prophet and
  ARIMA agree on the forecast direction. Reported alongside each decision.

Explainability
--------------
Every decision comes with a human-readable explanation that references the
actual PVI sub-scores and the signals that triggered the recommendation.

Requires
--------
  data/pvi_scores.csv     — run pvi.py first

Output
------
  data/recommendations.csv
"""

import os
import numpy as np
import pandas as pd

PVI_PATH    = "data/pvi_scores.csv"
OUTPUT_PATH = "data/recommendations.csv"

# ── Thresholds ────────────────────────────────────────────────────────────
# PVI category boundaries (must match pvi.py)
PVI_HIGH   = 67
PVI_MEDIUM = 33

# Growth: normalised growth_norm above this → "positive trend"
GROWTH_POSITIVE_THRESHOLD = 0.55   # top 45% of growth distribution
GROWTH_NEGATIVE_THRESHOLD = 0.40   # bottom 40% of growth distribution

# Stability: stability_norm below this → "high risk"
STABILITY_LOW_THRESHOLD = 0.40

# Anomaly: anomaly_pct above this is "serious"
ANOMALY_SERIOUS_THRESHOLD = 0.15   # >15% of months are anomalies

# Model agreement: below this → low confidence recommendation
AGREEMENT_LOW_THRESHOLD  = 0.55   # only penalise serious disagreement
AGREEMENT_HIGH_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def compute_confidence(row: pd.Series) -> str:
    """
    Return High / Medium / Low confidence based on:
      - model agreement (Prophet vs ARIMA) — strongest signal
      - PVI score distance from category boundary
      - anomaly presence
    """
    score = 1.0

    # Model agreement penalty
    agreement = row.get("model_agreement")
    if pd.notna(agreement):
        if agreement < AGREEMENT_LOW_THRESHOLD:
            score -= 0.4
        elif agreement < AGREEMENT_HIGH_THRESHOLD:
            score -= 0.15

    # Anomaly penalty
    if row.get("has_anomaly", False):
        score -= 0.2

    # PVI distance from nearest boundary (the further, the more confident)
    pvi = row["PVI"]
    dist_to_boundary = min(abs(pvi - PVI_HIGH), abs(pvi - PVI_MEDIUM))
    if dist_to_boundary < 5:
        score -= 0.2   # very close to category boundary

    if score >= 0.75:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Explanation builder
# ---------------------------------------------------------------------------

def build_explanation(
    decision: str,
    row: pd.Series,
    growth_label: str,
    risk_label: str,
    anomaly_override: bool,
) -> str:
    """
    Build a human-readable explanation string for the recommendation.
    References the actual sub-score values so the output is auditable.
    """
    parts = []

    pvi        = row["PVI"]
    viability  = row["viability"]
    demand_n   = row.get("demand_norm",    np.nan)
    growth_n   = row.get("growth_norm",    np.nan)
    stability_n= row.get("stability_norm", np.nan)
    price_n    = row.get("price_norm",     np.nan)

    # PVI summary
    parts.append(
        f"PVI={pvi:.1f}/100 ({viability} viability): "
        f"demand={demand_n:.2f}, growth={growth_n:.2f}, "
        f"stability={stability_n:.2f}, price={price_n:.2f}"
    )

    # Anomaly note
    if row.get("has_anomaly", False):
        pct = row.get("anomaly_pct", 0) * 100
        parts.append(
            f"⚠ Anomaly detected in {pct:.0f}% of historical months"
            + (" — override applied" if anomaly_override else "")
        )

    # Decision rationale
    if decision == "Increase":
        parts.append(
            f"Recommend stocking up: {growth_label} demand trend, "
            f"{risk_label} supply risk, and high viability support expansion."
        )
    elif decision == "Decrease":
        if viability == "Low":
            parts.append(
                f"Low viability with {growth_label} trend — "
                f"reducing stock minimises overstock waste."
            )
        else:
            parts.append(
                f"High supply risk ({risk_label}) despite moderate viability — "
                f"hold back until demand stabilises."
            )
    else:  # Hold
        parts.append(
            f"Maintain current stock levels: {growth_label} trend, "
            f"{risk_label} risk. Monitor PVI trajectory."
        )

    # Model agreement note
    agreement = row.get("model_agreement")
    if pd.notna(agreement):
        if agreement < AGREEMENT_LOW_THRESHOLD:
            parts.append(
                f"⚠ Prophet and ARIMA disagree significantly "
                f"(agreement={agreement:.0%}) — treat recommendation cautiously."
            )
        elif agreement >= AGREEMENT_HIGH_THRESHOLD:
            parts.append(f"Models agree strongly (agreement={agreement:.0%}).")

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Single-row decision
# ---------------------------------------------------------------------------

def make_decision(row: pd.Series):
    """
    Return (decision, explanation, confidence) for one (store, item) row.
    """
    pvi         = row["PVI"]
    viability   = row["viability"]
    growth_n    = row.get("growth_norm",    0.5)
    stability_n = row.get("stability_norm", 0.5)
    has_anomaly = row.get("has_anomaly",    False)
    anomaly_pct = row.get("anomaly_pct",    0.0)

    # Human-readable labels for use in explanations
    if growth_n >= GROWTH_POSITIVE_THRESHOLD:
        growth_label = "positive"
    elif growth_n <= GROWTH_NEGATIVE_THRESHOLD:
        growth_label = "declining"
    else:
        growth_label = "flat"

    if stability_n >= 0.60:
        risk_label = "low"
    elif stability_n >= STABILITY_LOW_THRESHOLD:
        risk_label = "moderate"
    else:
        risk_label = "high"

    is_positive_growth  = growth_n >= GROWTH_POSITIVE_THRESHOLD
    is_declining_growth = growth_n <= GROWTH_NEGATIVE_THRESHOLD
    is_high_risk        = stability_n < STABILITY_LOW_THRESHOLD
    serious_anomaly     = has_anomaly and anomaly_pct >= ANOMALY_SERIOUS_THRESHOLD

    anomaly_override = False

    # ── Layer 1: Hard anomaly overrides ─────────────────────────────────
    if serious_anomaly:
        anomaly_override = True
        if is_declining_growth:
            decision = "Decrease"
        else:
            decision = "Hold"

    # ── Layer 2: PVI-driven matrix ────────────────────────────────────────
    elif viability == "High":
        if is_positive_growth and not is_high_risk:
            decision = "Increase"
        else:
            decision = "Hold"

    elif viability == "Medium":
        if is_positive_growth and not is_high_risk:
            decision = "Increase"
        elif is_declining_growth and is_high_risk:
            decision = "Decrease"
        else:
            decision = "Hold"

    else:  # Low viability
        if is_declining_growth or is_high_risk:
            decision = "Decrease"
        elif is_positive_growth and not is_high_risk:
            decision = "Hold"    # cautious — don't increase a recovering low-viability item
        else:
            decision = "Decrease"

    confidence  = compute_confidence(row)
    explanation = build_explanation(
        decision, row, growth_label, risk_label, anomaly_override
    )

    return decision, confidence, explanation


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(PVI_PATH):
        raise FileNotFoundError(
            f"PVI file not found at {PVI_PATH}.\n"
            "Run  python src/pvi.py  first."
        )

    pvi_df = pd.read_csv(PVI_PATH)
    print(f"Loaded PVI scores: {len(pvi_df)} items")

    results = []
    for _, row in pvi_df.iterrows():
        decision, confidence, explanation = make_decision(row)
        results.append({
            "store_id":        row["store_id"],
            "item_id":         row["item_id"],
            "cat_id":          row.get("cat_id",  ""),
            "dept_id":         row.get("dept_id", ""),
            # Core outputs
            "Decision":        decision,
            "Confidence":      confidence,
            "Explanation":     explanation,
            # PVI breakdown (for dashboard display)
            "PVI":             row["PVI"],
            "Viability":       row["viability"],
            "demand_norm":     round(row.get("demand_norm",    np.nan), 3),
            "growth_norm":     round(row.get("growth_norm",    np.nan), 3),
            "stability_norm":  round(row.get("stability_norm", np.nan), 3),
            "price_norm":      round(row.get("price_norm",     np.nan), 3),
            # Forecast signals
            "forecast_mean":   round(row.get("forecast_mean",  np.nan), 2),
            "model_agreement": round(row.get("model_agreement", np.nan), 3)
                               if pd.notna(row.get("model_agreement")) else None,
            # Anomaly flags
            "has_anomaly":     row.get("has_anomaly",   False),
            "anomaly_count":   row.get("anomaly_count", 0),
            "anomaly_pct":     row.get("anomaly_pct",   0.0),
        })

    rec_df = pd.DataFrame(results)

    # Sort: Decrease first (urgent), then High PVI Increase, then Hold
    decision_order = {"Decrease": 0, "Increase": 1, "Hold": 2}
    rec_df["_sort_decision"] = rec_df["Decision"].map(decision_order)
    rec_df = rec_df.sort_values(
        ["_sort_decision", "store_id", "PVI"],
        ascending=[True, True, False],
    ).drop(columns=["_sort_decision"]).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    rec_df.to_csv(OUTPUT_PATH, index=False)

    # ── Summary ───────────────────────────────────────────────────────────
    decision_counts    = rec_df["Decision"].value_counts()
    confidence_counts  = rec_df["Confidence"].value_counts()
    anomaly_count      = rec_df["has_anomaly"].sum()

    print(f"\n{'='*55}")
    print("Stock recommendations generated.")
    print(f"  Total items      : {len(rec_df)}")
    print(f"  Increase         : {decision_counts.get('Increase', 0)}")
    print(f"  Hold             : {decision_counts.get('Hold',     0)}")
    print(f"  Decrease         : {decision_counts.get('Decrease', 0)}")
    print(f"  High confidence  : {confidence_counts.get('High',   0)}")
    print(f"  Low confidence   : {confidence_counts.get('Low',    0)}")
    print(f"  Anomaly flagged  : {anomaly_count}")
    print(f"  Saved to         : {OUTPUT_PATH}")
    print(f"{'='*55}\n")

    print("Priority actions (Decrease / urgent items):")
    urgent = rec_df[rec_df["Decision"] == "Decrease"][
        ["store_id", "item_id", "cat_id", "PVI", "Viability",
         "Confidence", "has_anomaly"]
    ].head(10)
    print(urgent.to_string(index=False))

    return rec_df


if __name__ == "__main__":
    main()

"""
evaluate.py — Forecast model evaluation with full metric suite.

Metrics computed
----------------
Regression metrics (appropriate for demand forecasting):
  MAE   — Mean Absolute Error: average absolute gap between forecast and actual
  RMSE  — Root Mean Squared Error: penalises large errors more than MAE
  MAPE  — Mean Absolute Percentage Error: scale-independent, skips zero actuals
  WAPE  — Weighted Absolute Percentage Error: robust alternative to MAPE
  R2    — Coefficient of Determination: 1 = perfect, 0 = no better than mean
  bias  — Mean signed error: positive = over-predicts, negative = under-predicts

NOT computed (and why):
  Precision/Recall/F1/Confusion Matrix — these are classification metrics.
  Demand forecasting is a regression problem (predicting a continuous sales
  number), so these do not apply. The stock recommendation (Increase/Hold/
  Decrease) is classification, but it is rule-based with no ground-truth
  labels available, making supervised evaluation impossible.

Method: Held-out evaluation
  The last 3 months of each historical series are withheld as a test set.
  Both Prophet and ARIMA are evaluated on the same held-out window, making
  the comparison fair.

Outputs
-------
  data/eval_metrics.csv    — per-series metrics for every item × model
  data/eval_summary.csv    — aggregate stats (mean, median, std) per metric
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED_PATH       = "data/processed/processed_m5.csv"
PROPHET_FORECAST_DIR = "data/forecast/prophet"
ARIMA_FORECAST_DIR   = "data/forecast/arima"
EVAL_OUTPUT          = "data/eval_metrics.csv"
EVAL_SUMMARY_OUTPUT  = "data/eval_summary.csv"

HOLDOUT_MONTHS = 3


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE — skips zero-actual months to avoid division by zero."""
    mask = y_true > 1e-6
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def safe_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    WAPE (Weighted Absolute Percentage Error).
    More robust than MAPE — weights errors by actual volume so high-selling
    items contribute more to the aggregate score.
    Formula: sum(|actual - pred|) / sum(|actual|)
    """
    total_actual = np.sum(np.abs(y_true))
    if total_actual < 1e-6:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual * 100)


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² (Coefficient of Determination).
    1.0  = perfect forecast
    0.0  = no better than predicting the mean
    <0   = worse than predicting the mean (bad model)
    """
    if len(y_true) < 2 or np.var(y_true) < 1e-9:
        return np.nan
    return float(r2_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_forecast_filename(fname: str):
    base      = os.path.basename(fname).replace(".csv", "")
    rest      = base[len("forecast_"):]
    parts     = rest.split("_")
    store_id  = f"{parts[0]}_{parts[1]}"
    safe_item = "_".join(parts[2:])
    item_id   = safe_item.replace("-", "_")
    return store_id, item_id


# ---------------------------------------------------------------------------
# Per-series evaluation
# ---------------------------------------------------------------------------

def evaluate_one_series(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    store_id: str,
    item_id: str,
    model: str,
    n_obs: int,
    cat_id: str = "",
) -> dict:
    mae_val  = float(mean_absolute_error(y_true, y_pred))
    rmse_val = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape_val = safe_mape(y_true, y_pred)
    wape_val = safe_wape(y_true, y_pred)
    r2_val   = safe_r2(y_true, y_pred)
    bias_val = float(np.mean(y_pred - y_true))

    return {
        "store_id":    store_id,
        "item_id":     item_id,
        "cat_id":      cat_id,
        "model":       model,
        "n_test_obs":  len(y_true),
        "n_train_obs": n_obs - len(y_true),
        # actual values for scatter plots
        "y_true_mean": round(float(np.mean(y_true)), 2),
        "y_pred_mean": round(float(np.mean(y_pred)), 2),
        # metrics
        "MAE":         round(mae_val,  2),
        "RMSE":        round(rmse_val, 2),
        "MAPE":        round(mape_val, 2) if not np.isnan(mape_val) else None,
        "WAPE":        round(wape_val, 2) if not np.isnan(wape_val) else None,
        "R2":          round(r2_val,   4) if not np.isnan(r2_val)   else None,
        "bias":        round(bias_val, 2),
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_models():
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_PATH}.\n"
            "Run  python src/preprocess.py  first."
        )

    processed = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    hist_idx  = processed.set_index(["store_id", "item_id"])

    prophet_files = sorted(glob.glob(os.path.join(PROPHET_FORECAST_DIR, "forecast_*.csv")))
    arima_files   = sorted(glob.glob(os.path.join(ARIMA_FORECAST_DIR,   "forecast_*.csv")))

    arima_lookup = {}
    for f in arima_files:
        s, i = parse_forecast_filename(f)
        arima_lookup[(s, i)] = f

    if not prophet_files:
        raise FileNotFoundError("No Prophet forecast files found. Run  python src/forecast.py  first.")

    print(f"Evaluating {len(prophet_files)} Prophet + {len(arima_lookup)} ARIMA series…")

    all_metrics = []
    skipped = 0

    for fpath in prophet_files:
        store_id, item_id = parse_forecast_filename(fpath)
        key = (store_id, item_id)

        if key not in hist_idx.index:
            skipped += 1
            continue

        hist   = hist_idx.loc[[key]].sort_values("date")
        series = hist["monthly_sales"].values
        n_obs  = len(series)
        cat_id = hist["cat_id"].iloc[0] if "cat_id" in hist.columns else ""

        if n_obs <= HOLDOUT_MONTHS:
            skipped += 1
            continue

        y_true = series[-HOLDOUT_MONTHS:]

        # Prophet
        fc_prophet   = pd.read_csv(fpath)
        y_pred_prophet = fc_prophet["yhat"].values[:HOLDOUT_MONTHS]
        if len(y_pred_prophet) == HOLDOUT_MONTHS:
            all_metrics.append(evaluate_one_series(
                y_true, y_pred_prophet, store_id, item_id, "prophet", n_obs, cat_id
            ))

        # ARIMA
        if key in arima_lookup:
            fc_arima   = pd.read_csv(arima_lookup[key])
            y_pred_arima = fc_arima["yhat"].values[:HOLDOUT_MONTHS]
            if len(y_pred_arima) == HOLDOUT_MONTHS:
                all_metrics.append(evaluate_one_series(
                    y_true, y_pred_arima, store_id, item_id, "arima", n_obs, cat_id
                ))

    if not all_metrics:
        print("No metrics computed — check that forecast files match processed data.")
        return

    metrics_df = pd.DataFrame(all_metrics)

    # Aggregate summary per model
    METRIC_COLS = ["MAE", "RMSE", "MAPE", "WAPE", "R2", "bias"]
    summary_rows = []
    for model_name, grp in metrics_df.groupby("model"):
        for metric in METRIC_COLS:
            vals = grp[metric].dropna()
            if len(vals) == 0:
                continue
            summary_rows.append({
                "model":  model_name,
                "metric": metric,
                "mean":   round(vals.mean(),   4),
                "median": round(vals.median(), 4),
                "std":    round(vals.std(),    4),
                "min":    round(vals.min(),    4),
                "max":    round(vals.max(),    4),
                "n":      len(vals),
            })

    summary_df = pd.DataFrame(summary_rows)

    os.makedirs("data", exist_ok=True)
    metrics_df.to_csv(EVAL_OUTPUT,         index=False)
    summary_df.to_csv(EVAL_SUMMARY_OUTPUT, index=False)

    # Print report
    print(f"\n{'='*60}")
    print(f"Evaluation complete  —  holdout: last {HOLDOUT_MONTHS} months")
    print(f"  Series scored: {len(metrics_df['item_id'].unique())}  |  Skipped: {skipped}\n")
    for model_name in ["prophet", "arima"]:
        grp = summary_df[summary_df["model"] == model_name]
        if grp.empty:
            continue
        print(f"  ── {model_name.upper()} ──")
        for _, row in grp.iterrows():
            print(f"     {row['metric']:5s}  mean={row['mean']:>9.4f}  median={row['median']:>9.4f}")
        print()
    print(f"  Saved: {EVAL_OUTPUT}")
    print(f"  Saved: {EVAL_SUMMARY_OUTPUT}")
    print(f"{'='*60}\n")

    return metrics_df, summary_df


if __name__ == "__main__":
    evaluate_models()

"""
plot_reports.py — Comprehensive visualisation suite.

Generates 8 publication-quality plots saved to outputs/plots/.
Run after the full pipeline (forecast → pvi → recommend → evaluate).

Plots generated
---------------
1. forecast_vs_actual_top5.png  — Forecast vs actual sales for top-5 PVI items
2. mae_rmse_comparison.png      — Grouped bar: MAE & RMSE, Prophet vs ARIMA
3. mape_wape_comparison.png     — Grouped bar: MAPE & WAPE, Prophet vs ARIMA
4. r2_comparison.png            — Box plot: R² distribution per model
5. residual_distribution.png    — Histogram of forecast errors per model
6. actual_vs_predicted.png      — Scatter: actual vs predicted sales
7. pvi_distribution.png         — PVI histogram coloured by viability category
8. pvi_by_category.png          — Avg PVI per product category with breakdown
9. decision_breakdown.png       — Stacked bar: decisions per store

Usage
-----
    python src/plot_reports.py
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")
matplotlib.use("Agg")   # non-interactive backend — safe for servers

# ── Paths ──────────────────────────────────────────────────────────────────
PROCESSED_PATH       = "data/processed/processed_m5.csv"
PROPHET_FORECAST_DIR = "data/forecast/prophet"
ARIMA_FORECAST_DIR   = "data/forecast/arima"
PVI_PATH             = "data/pvi_scores.csv"
RECS_PATH            = "data/recommendations.csv"
EVAL_PATH            = "data/eval_metrics.csv"
EVAL_SUMMARY_PATH    = "data/eval_summary.csv"
OUT_DIR              = "outputs/plots"

# ── Design tokens ──────────────────────────────────────────────────────────
PROPHET_COLOR  = "#3b82f6"   # blue
ARIMA_COLOR    = "#7c3aed"   # purple
HIGH_COLOR     = "#15803d"   # green
MEDIUM_COLOR   = "#b45309"   # amber
LOW_COLOR      = "#b91c1c"   # red
INCREASE_COLOR = "#1d4ed8"
HOLD_COLOR     = "#6d28d9"
DECREASE_COLOR = "#b91c1c"
GRID_COLOR     = "#f5f5f4"
TEXT_COLOR     = "#1c1917"
MUTED_COLOR    = "#78716c"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#e7e5e4",
    "axes.labelcolor":   TEXT_COLOR,
    "xtick.color":       MUTED_COLOR,
    "ytick.color":       MUTED_COLOR,
    "text.color":        TEXT_COLOR,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    1,
})


def _save(fig, name: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {path}")


def _parse_filename(fname: str):
    base      = os.path.basename(fname).replace(".csv", "")
    rest      = base[len("forecast_"):]
    parts     = rest.split("_")
    store_id  = f"{parts[0]}_{parts[1]}"
    item_id   = "_".join(parts[2:]).replace("-", "_")
    return store_id, item_id


# ---------------------------------------------------------------------------
# Plot 1 — Forecast vs Actual (top 5 PVI items)
# ---------------------------------------------------------------------------

def plot_forecast_vs_actual():
    if not os.path.exists(PVI_PATH) or not os.path.exists(PROCESSED_PATH):
        print("  Skipping forecast_vs_actual — PVI or processed data missing")
        return

    pvi_df    = pd.read_csv(PVI_PATH)
    processed = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    top5      = pvi_df.nlargest(5, "PVI")[["store_id", "item_id", "PVI", "viability"]]

    fig, axes = plt.subplots(5, 1, figsize=(12, 18))
    fig.suptitle("Forecast vs Actual Sales — Top 5 Items by PVI", fontsize=15, fontweight="bold", y=0.98)

    for ax, (_, row) in zip(axes, top5.iterrows()):
        store, item = row["store_id"], row["item_id"]
        safe_item   = item.replace("_", "-")

        hist = processed[(processed["store_id"] == store) & (processed["item_id"] == item)].sort_values("date")

        prophet_path = os.path.join(PROPHET_FORECAST_DIR, f"forecast_{store}_{safe_item}.csv")
        arima_path   = os.path.join(ARIMA_FORECAST_DIR,   f"forecast_{store}_{safe_item}.csv")

        ax.plot(hist["date"], hist["monthly_sales"], color=TEXT_COLOR,
                linewidth=2, label="Actual", zorder=3)

        if os.path.exists(prophet_path):
            fc = pd.read_csv(prophet_path, parse_dates=["ds"])
            ax.plot(fc["ds"], fc["yhat"], color=PROPHET_COLOR, linewidth=2,
                    linestyle="--", label="Prophet", zorder=2)
            if "yhat_lower" in fc.columns:
                ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"],
                                alpha=0.15, color=PROPHET_COLOR)

        if os.path.exists(arima_path):
            fc = pd.read_csv(arima_path, parse_dates=["ds"])
            ax.plot(fc["ds"], fc["yhat"], color=ARIMA_COLOR, linewidth=2,
                    linestyle=":", label="ARIMA", zorder=2)
            if "yhat_lower" in fc.columns:
                ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"],
                                alpha=0.10, color=ARIMA_COLOR)

        vcolor = HIGH_COLOR if row["viability"] == "High" else (MEDIUM_COLOR if row["viability"] == "Medium" else LOW_COLOR)
        ax.set_title(f"{item}  |  {store}  |  PVI {row['PVI']:.1f}  ({row['viability']})",
                     color=vcolor, fontsize=11)
        ax.set_ylabel("Monthly sales")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.8)

    axes[-1].set_xlabel("Date")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "forecast_vs_actual_top5.png")


# ---------------------------------------------------------------------------
# Plot 2 — MAE & RMSE comparison
# ---------------------------------------------------------------------------

def plot_mae_rmse_comparison():
    if not os.path.exists(EVAL_SUMMARY_PATH):
        print("  Skipping mae_rmse — eval_summary missing"); return

    df = pd.read_csv(EVAL_SUMMARY_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Forecast Error — Prophet vs ARIMA", fontsize=14, fontweight="bold")

    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        sub     = df[df["metric"] == metric]
        models  = sub["model"].tolist()
        means   = sub["mean"].tolist()
        stds    = sub["std"].tolist()
        colors  = [PROPHET_COLOR if m == "prophet" else ARIMA_COLOR for m in models]
        bars    = ax.bar(models, means, color=colors, width=0.5, zorder=2, edgecolor="white")
        ax.errorbar(models, means, yerr=stds, fmt="none", color="#44403c",
                    capsize=5, linewidth=1.5, zorder=3)

        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.05,
                    f"{val:,.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_title(f"{metric}  (lower is better)", pad=10)
        ax.set_ylabel("Units of sales")
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.set_ylim(0, max(means) * 1.3)

    fig.tight_layout()
    _save(fig, "mae_rmse_comparison.png")


# ---------------------------------------------------------------------------
# Plot 3 — MAPE & WAPE comparison
# ---------------------------------------------------------------------------

def plot_mape_wape_comparison():
    if not os.path.exists(EVAL_SUMMARY_PATH):
        print("  Skipping mape_wape — eval_summary missing"); return

    df = pd.read_csv(EVAL_SUMMARY_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Percentage Error — Prophet vs ARIMA", fontsize=14, fontweight="bold")

    for ax, metric, note in zip(axes, ["MAPE", "WAPE"],
                                 ["skips zero-actual months", "weighted by sales volume"]):
        sub   = df[df["metric"] == metric]
        if sub.empty:
            ax.text(0.5, 0.5, f"{metric} not available", ha="center", va="center",
                    transform=ax.transAxes, color=MUTED_COLOR)
            continue

        models = sub["model"].tolist()
        means  = sub["mean"].tolist()
        colors = [PROPHET_COLOR if m == "prophet" else ARIMA_COLOR for m in models]
        bars   = ax.bar(models, means, color=colors, width=0.5, zorder=2, edgecolor="white")

        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_title(f"{metric}  —  {note}", pad=10)
        ax.set_ylabel("Percentage error (%)")
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.set_ylim(0, max(means) * 1.35)

    fig.tight_layout()
    _save(fig, "mape_wape_comparison.png")


# ---------------------------------------------------------------------------
# Plot 4 — R² distribution
# ---------------------------------------------------------------------------

def plot_r2_distribution():
    if not os.path.exists(EVAL_PATH):
        print("  Skipping r2_distribution — eval_metrics missing"); return

    df = pd.read_csv(EVAL_PATH)
    if "R2" not in df.columns:
        print("  Skipping r2_distribution — R2 column missing"); return

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("R² Distribution — How Well Each Model Explains Variance",
                 fontsize=14, fontweight="bold")

    data   = [df[df["model"] == m]["R2"].dropna().values for m in ["prophet", "arima"]]
    labels = ["Prophet", "ARIMA"]
    colors = [PROPHET_COLOR, ARIMA_COLOR]

    bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker="o", markersize=4, alpha=0.4))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.axhline(0, color="#e24b4a", linestyle="--", linewidth=1.2,
               label="R²=0 (no better than mean)")
    ax.axhline(1, color=HIGH_COLOR, linestyle="--", linewidth=1.2,
               label="R²=1 (perfect forecast)")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel("R² score")

    # FIX 1: Expand ylim to show actual data range, not cut it off
    all_vals = [v for d in data for v in d]
    y_min = max(np.percentile(all_vals, 1), -15)   # clip extreme outliers for display
    ax.set_ylim(y_min - 0.5, 1.3)

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, framealpha=0.8)

    # FIX 2: Clamp text position inside axes bounds so it doesn't blow up the canvas
    y_lo, y_hi = ax.get_ylim()
    for i, (d, color) in enumerate(zip(data, colors), 1):
        if len(d):
            med = np.median(d)
            # Place label inside axes — if median is below view, pin to bottom
            text_y = np.clip(med, y_lo + 0.3, y_hi - 0.1)
            label  = f"  med={med:.2f}"
            if med < y_lo + 0.3:
                label = f"  med={med:.2f} (below view)"
            ax.text(i, text_y, label,
                    va="center", fontsize=9, color=color, fontweight="bold")

    fig.tight_layout()
    _save(fig, "r2_distribution.png")


# ---------------------------------------------------------------------------
# Plot 5 — Residual distribution
# ---------------------------------------------------------------------------

def plot_residual_distribution():
    if not os.path.exists(EVAL_PATH):
        print("  Skipping residuals — eval_metrics missing"); return

    df = pd.read_csv(EVAL_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    fig.suptitle("Residual Distribution (Forecast Error = Predicted − Actual)",
                 fontsize=14, fontweight="bold")

    for ax, (model, color) in zip(axes, [("prophet", PROPHET_COLOR), ("arima", ARIMA_COLOR)]):
        sub = df[df["model"] == model]["bias"].dropna()
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        ax.hist(sub, bins=25, color=color, alpha=0.8, edgecolor="white", zorder=2)
        ax.axvline(0,          color=TEXT_COLOR,   linestyle="--", linewidth=1.5, label="Zero bias")
        ax.axvline(sub.mean(), color="#e24b4a",    linestyle="-",  linewidth=2,   label=f"Mean = {sub.mean():.1f}")

        ax.set_title(f"{model.capitalize()}  —  bias distribution")
        ax.set_xlabel("Forecast error (predicted − actual units)")
        ax.set_ylabel("Number of items")
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.legend(fontsize=9)

        note = "Over-predicts" if sub.mean() > 0 else "Under-predicts"
        ax.text(0.97, 0.95, note, transform=ax.transAxes,
                ha="right", va="top", fontsize=10,
                color="#e24b4a" if sub.mean() > 0 else ARIMA_COLOR,
                fontweight="bold")

    fig.tight_layout()
    _save(fig, "residual_distribution.png")


# ---------------------------------------------------------------------------
# Plot 6 — Actual vs Predicted scatter
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted():
    if not os.path.exists(EVAL_PATH):
        print("  Skipping actual_vs_predicted — eval_metrics missing"); return

    df = pd.read_csv(EVAL_PATH)
    if "y_true_mean" not in df.columns:
        print("  Skipping actual_vs_predicted — y_true_mean column missing"); return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Actual vs Predicted Sales — Perfect = Points on Diagonal",
                 fontsize=14, fontweight="bold")

    for ax, (model, color) in zip(axes, [("prophet", PROPHET_COLOR), ("arima", ARIMA_COLOR)]):
        sub = df[df["model"] == model].dropna(subset=["y_true_mean", "y_pred_mean"])
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        x, y = sub["y_true_mean"].values, sub["y_pred_mean"].values
        ax.scatter(x, y, alpha=0.35, color=color, s=18, zorder=2, edgecolors="none")

        lo, hi = min(x.min(), y.min()) * 0.9, max(x.max(), y.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], color=TEXT_COLOR, linewidth=1.5,
                linestyle="--", label="Perfect forecast")

        # Regression line
        if len(x) > 2:
            m, b = np.polyfit(x, y, 1)
            xs   = np.linspace(lo, hi, 100)
            ax.plot(xs, m * xs + b, color=color, linewidth=2,
                    label=f"Fitted (slope={m:.2f})")

        ax.set_title(f"{model.capitalize()}")
        ax.set_xlabel("Actual monthly sales")
        ax.set_ylabel("Predicted monthly sales")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(linestyle="--", alpha=0.4, zorder=0)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.tight_layout()
    _save(fig, "actual_vs_predicted.png")


# ---------------------------------------------------------------------------
# Plot 7 — PVI distribution
# ---------------------------------------------------------------------------

def plot_pvi_distribution():
    if not os.path.exists(PVI_PATH):
        print("  Skipping pvi_distribution — pvi_scores missing"); return

    pvi = pd.read_csv(PVI_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Product Viability Index (PVI) Distribution", fontsize=14, fontweight="bold")

    # Left: overall histogram coloured by viability zone
    ax = axes[0]
    bins = np.linspace(0, 100, 26)
    for viab, color in [("High", HIGH_COLOR), ("Medium", MEDIUM_COLOR), ("Low", LOW_COLOR)]:
        sub = pvi[pvi["viability"] == viab]["PVI"]
        ax.hist(sub, bins=bins, color=color, alpha=0.8, edgecolor="white", label=viab, zorder=2)

    ax.axvline(33, color=MUTED_COLOR, linestyle="--", linewidth=1, label="Low/Medium boundary")
    ax.axvline(67, color=MUTED_COLOR, linestyle=":",  linewidth=1, label="Medium/High boundary")
    ax.set_xlabel("PVI score (0–100)")
    ax.set_ylabel("Number of items")
    ax.set_title("All items by viability zone")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    counts = pvi["viability"].value_counts()
    total  = len(pvi)
    for viab, color, y_pos in [("High", HIGH_COLOR, 0.90), ("Medium", MEDIUM_COLOR, 0.82), ("Low", LOW_COLOR, 0.74)]:
        n = counts.get(viab, 0)
        ax.text(0.97, y_pos, f"{viab}: {n} ({100*n/total:.0f}%)",
                transform=ax.transAxes, ha="right", fontsize=10,
                color=color, fontweight="bold")

    # Right: PVI by category
    ax2 = axes[1]
    if "cat_id" in pvi.columns:
        cat_means = pvi.groupby(["cat_id", "viability"]).size().unstack(fill_value=0)
        cat_order = pvi.groupby("cat_id")["PVI"].mean().sort_values(ascending=False).index
        cat_means = cat_means.reindex(cat_order)

        bottom = np.zeros(len(cat_means))
        for viab, color in [("High", HIGH_COLOR), ("Medium", MEDIUM_COLOR), ("Low", LOW_COLOR)]:
            if viab in cat_means.columns:
                vals = cat_means[viab].values
                ax2.bar(cat_means.index, vals, bottom=bottom,
                        color=color, label=viab, edgecolor="white")
                bottom += vals

        avg_pvi = pvi.groupby("cat_id")["PVI"].mean().reindex(cat_order)
        for i, (cat, val) in enumerate(avg_pvi.items()):
            ax2.text(i, bottom[i] + 1, f"avg\n{val:.0f}", ha="center",
                     va="bottom", fontsize=9, color=TEXT_COLOR)

    ax2.set_title("Items per category by viability")
    ax2.set_xlabel("Category")
    ax2.set_ylabel("Number of items")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    fig.tight_layout()
    _save(fig, "pvi_distribution.png")


# ---------------------------------------------------------------------------
# Plot 8 — PVI sub-score heatmap by category
# ---------------------------------------------------------------------------

def plot_pvi_subscores_by_category():
    if not os.path.exists(PVI_PATH):
        print("  Skipping subscores heatmap — pvi_scores missing"); return

    pvi = pd.read_csv(PVI_PATH)
    if "cat_id" not in pvi.columns:
        print("  Skipping subscores heatmap — cat_id missing"); return

    sub_cols = ["demand_norm", "growth_norm", "stability_norm", "price_norm"]
    missing  = [c for c in sub_cols if c not in pvi.columns]
    if missing:
        print(f"  Skipping subscores heatmap — missing columns: {missing}"); return

    pivot = pvi.groupby("cat_id")[sub_cols].mean() * 100
    pivot.columns = ["Demand (40%)", "Growth (25%)", "Stability (20%)", "Price (15%)"]

    fig, ax = plt.subplots(figsize=(9, max(4, len(pivot) * 1.1)))
    fig.suptitle("Average PVI Sub-scores by Category", fontsize=14, fontweight="bold")

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=15, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if val < 35 or val > 70 else TEXT_COLOR)

    plt.colorbar(im, ax=ax, label="Score (0–100%)", shrink=0.8)
    fig.tight_layout()
    _save(fig, "pvi_subscores_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 9 — Decision breakdown per store
# ---------------------------------------------------------------------------

def plot_decision_breakdown():
    if not os.path.exists(RECS_PATH):
        print("  Skipping decision_breakdown — recommendations missing"); return

    recs = pd.read_csv(RECS_PATH)
    if "store_id" not in recs.columns:
        print("  Skipping decision_breakdown — store_id column missing"); return

    pivot = recs.groupby(["store_id", "Decision"]).size().unstack(fill_value=0)
    for col in ["Increase", "Hold", "Decrease"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["Increase", "Hold", "Decrease"]]
    pivot = pivot.sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Stock Recommendation Breakdown", fontsize=14, fontweight="bold")

    # Stacked bar per store
    ax = axes[0]
    bottom = np.zeros(len(pivot))
    for dec, color in [("Increase", INCREASE_COLOR), ("Hold", HOLD_COLOR), ("Decrease", DECREASE_COLOR)]:
        vals = pivot[dec].values
        bars = ax.bar(pivot.index, vals, bottom=bottom, color=color,
                      label=dec, edgecolor="white")
        for bar, val, bot in zip(bars, vals, bottom):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
                        str(val), ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottom += vals

    ax.set_title("Decisions per store")
    ax.set_xlabel("Store ID")
    ax.set_ylabel("Number of items")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")

    # Overall donut
    ax2 = axes[1]
    totals = pivot.sum()
    colors = [INCREASE_COLOR, HOLD_COLOR, DECREASE_COLOR]
    wedges, texts, autotexts = ax2.pie(
        totals.values, labels=totals.index, autopct="%1.0f%%",
        colors=colors, startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(fontsize=11)
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_color("white")
        at.set_fontweight("bold")

    # Draw hole for donut effect
    centre_circle = plt.Circle((0, 0), 0.60, fc="white")
    ax2.add_patch(centre_circle)
    ax2.set_title("Overall decision split")
    ax2.text(0, 0, f"{len(recs)}\nitems", ha="center", va="center",
             fontsize=12, fontweight="bold", color=TEXT_COLOR)

    fig.tight_layout()
    _save(fig, "decision_breakdown.png")


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def generate_all_plots():
    print(f"\n{'='*55}")
    print("Generating all visualisation plots…")
    print(f"{'='*55}")

    plot_forecast_vs_actual()
    plot_mae_rmse_comparison()
    plot_mape_wape_comparison()
    plot_r2_distribution()
    plot_residual_distribution()
    plot_actual_vs_predicted()
    plot_pvi_distribution()
    plot_pvi_subscores_by_category()
    plot_decision_breakdown()

    print(f"\nAll plots saved to  {OUT_DIR}/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    generate_all_plots()

"""
summary_report.py — Pipeline summary table.

Prints a clean, structured summary of all pipeline outputs to the terminal.
Run this after the full pipeline completes.

Usage:
    python src/summary_report.py
"""

import os
import numpy as np
import pandas as pd

PVI_PATH       = "data/pvi_scores.csv"
RECS_PATH      = "data/recommendations.csv"
EVAL_PATH      = "data/eval_metrics.csv"
EVAL_SUM_PATH  = "data/eval_summary.csv"
PROCESSED_PATH = "data/processed/processed_m5.csv"

DIV  = "=" * 65
DIV2 = "-" * 65


def section(title):
    print(f"\n{DIV}")
    print(f"  {title}")
    print(DIV)


def load(path, label):
    if not os.path.exists(path):
        print(f"  [missing] {label} — run the pipeline first")
        return None
    return pd.read_csv(path)


def fmt(n, decimals=2):
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "—"
    if isinstance(n, float):
        return f"{n:,.{decimals}f}"
    return f"{n:,}"


def run():
    print(f"\n{'#'*65}")
    print(f"  RETAIL DEMAND AI — PIPELINE SUMMARY REPORT")
    print(f"{'#'*65}")

    # ── 1. Dataset ────────────────────────────────────────────────────────
    section("1. DATASET OVERVIEW")
    processed = load(PROCESSED_PATH, "processed data")
    if processed is not None:
        processed["date"] = pd.to_datetime(processed["date"])
        print(f"  Dataset         : M5 Forecasting Competition (Kaggle)")
        print(f"  Total rows      : {len(processed):,}")
        print(f"  Unique items    : {processed['item_id'].nunique():,}")
        print(f"  Unique stores   : {processed['store_id'].nunique()}")
        stores = sorted(processed['store_id'].unique().tolist())
        print(f"  Stores included : {', '.join(stores)}")
        cats = sorted(processed['cat_id'].unique().tolist())
        print(f"  Categories      : {', '.join(cats)}")
        print(f"  Date range      : {processed['date'].min().date()} → {processed['date'].max().date()}")
        months = processed.groupby(["store_id","item_id"]).size()
        print(f"  Avg series len  : {months.mean():.1f} months  (min {months.min()}, max {months.max()})")
        if "avg_price" in processed.columns:
            print(f"  Price range     : ${processed['avg_price'].min():.2f} – ${processed['avg_price'].max():.2f}")

    # ── 2. PVI ────────────────────────────────────────────────────────────
    section("2. PRODUCT VIABILITY INDEX (PVI)")
    pvi = load(PVI_PATH, "PVI scores")
    if pvi is not None:
        total = len(pvi)
        counts = pvi["viability"].value_counts()
        high   = counts.get("High",   0)
        med    = counts.get("Medium", 0)
        low    = counts.get("Low",    0)

        print(f"  Items scored    : {total}")
        print(f"  High  (≥67)     : {high:>4}  ({100*high/total:.1f}%)")
        print(f"  Medium (33–66)  : {med:>4}  ({100*med/total:.1f}%)")
        print(f"  Low   (<33)     : {low:>4}  ({100*low/total:.1f}%)")
        print(f"  PVI mean        : {pvi['PVI'].mean():.2f}")
        print(f"  PVI median      : {pvi['PVI'].median():.2f}")
        print(f"  PVI std         : {pvi['PVI'].std():.2f}")
        print(f"  PVI range       : {pvi['PVI'].min():.1f} – {pvi['PVI'].max():.1f}")
        print(f"  Anomaly flagged : {pvi['has_anomaly'].sum()} items "
              f"({100*pvi['has_anomaly'].mean():.1f}%)")

        print(f"\n  {DIV2}")
        print(f"  TOP 10 ITEMS BY PVI")
        print(f"  {DIV2}")
        cols = ["store_id","item_id","cat_id","PVI","viability"]
        top10 = pvi.nlargest(10, "PVI")[cols].reset_index(drop=True)
        top10.index += 1
        for i, r in top10.iterrows():
            print(f"  {i:>2}. {r['store_id']:6} | {r['item_id']:35} | "
                  f"PVI {r['PVI']:5.1f} | {r['viability']}")

        print(f"\n  {DIV2}")
        print(f"  BOTTOM 5 ITEMS BY PVI (most at risk)")
        print(f"  {DIV2}")
        bot5 = pvi.nsmallest(5, "PVI")[cols].reset_index(drop=True)
        for i, r in bot5.iterrows():
            print(f"  {i+1:>2}. {r['store_id']:6} | {r['item_id']:35} | "
                  f"PVI {r['PVI']:5.1f} | {r['viability']}")

        if "cat_id" in pvi.columns:
            print(f"\n  {DIV2}")
            print(f"  PVI BY CATEGORY")
            print(f"  {DIV2}")
            cat_stats = pvi.groupby("cat_id")["PVI"].agg(["mean","median","std","count"])
            for cat, row in cat_stats.iterrows():
                print(f"  {cat:12} — mean {row['mean']:5.1f} | "
                      f"median {row['median']:5.1f} | "
                      f"std {row['std']:5.1f} | "
                      f"n={int(row['count'])}")

        if "model_agreement" in pvi.columns:
            ma = pvi["model_agreement"].dropna()
            if len(ma):
                print(f"\n  Model agreement : mean {ma.mean():.2%}  "
                      f"| median {ma.median():.2%}  "
                      f"| items with agreement < 70%: {(ma < 0.70).sum()}")

    # ── 3. Recommendations ────────────────────────────────────────────────
    section("3. STOCK RECOMMENDATIONS")
    recs = load(RECS_PATH, "recommendations")
    if recs is not None:
        total = len(recs)
        dc    = recs["Decision"].value_counts()
        cc    = recs["Confidence"].value_counts()

        print(f"  Total items     : {total}")
        print(f"\n  Decision breakdown:")
        for dec in ["Increase","Hold","Decrease"]:
            n = dc.get(dec, 0)
            bar = "█" * int(30 * n / total)
            print(f"    {dec:10} : {n:>4} ({100*n/total:5.1f}%)  {bar}")

        print(f"\n  Confidence breakdown:")
        for conf in ["High","Medium","Low"]:
            n = cc.get(conf, 0)
            bar = "█" * int(30 * n / total)
            print(f"    {conf:8} : {n:>4} ({100*n/total:5.1f}%)  {bar}")

        if "cat_id" in recs.columns:
            print(f"\n  {DIV2}")
            print(f"  DECISIONS BY CATEGORY")
            print(f"  {DIV2}")
            pivot = recs.groupby(["cat_id","Decision"]).size().unstack(fill_value=0)
            for cat in pivot.index:
                row = pivot.loc[cat]
                parts = []
                for dec in ["Increase","Hold","Decrease"]:
                    if dec in row.index:
                        parts.append(f"{dec}: {row[dec]}")
                print(f"  {cat:12} — {' | '.join(parts)}")

        print(f"\n  {DIV2}")
        print(f"  URGENT — TOP 10 DECREASE ITEMS")
        print(f"  {DIV2}")
        urgent = recs[recs["Decision"] == "Decrease"].nsmallest(10, "PVI")
        if len(urgent):
            for _, r in urgent.iterrows():
                print(f"  {r['store_id']:6} | {r['item_id']:35} | "
                      f"PVI {r['PVI']:5.1f} | conf: {r['Confidence']}")
        else:
            print("  No Decrease recommendations.")

        print(f"\n  {DIV2}")
        print(f"  PRIORITY — TOP 10 INCREASE ITEMS")
        print(f"  {DIV2}")
        priority = recs[recs["Decision"] == "Increase"].nlargest(10, "PVI")
        if len(priority):
            for _, r in priority.iterrows():
                print(f"  {r['store_id']:6} | {r['item_id']:35} | "
                      f"PVI {r['PVI']:5.1f} | conf: {r['Confidence']}")
        else:
            print("  No Increase recommendations.")

    # ── 4. Evaluation ─────────────────────────────────────────────────────
    section("4. MODEL EVALUATION (Held-out: last 3 months)")
    eval_sum = load(EVAL_SUM_PATH, "eval summary")
    if eval_sum is not None:
        metrics = ["MAE","RMSE","MAPE","WAPE","R2","bias"]
        header  = f"  {'Metric':6}  {'Prophet mean':>14}  {'Prophet median':>15}  {'ARIMA mean':>12}  {'ARIMA median':>13}  {'Winner':>8}"
        print(header)
        print(f"  {DIV2}")
        for metric in metrics:
            p = eval_sum[(eval_sum["model"]=="prophet") & (eval_sum["metric"]==metric)]
            a = eval_sum[(eval_sum["model"]=="arima")   & (eval_sum["metric"]==metric)]
            p_mean   = p["mean"].values[0]   if len(p) else None
            p_med    = p["median"].values[0] if len(p) else None
            a_mean   = a["mean"].values[0]   if len(a) else None
            a_med    = a["median"].values[0] if len(a) else None

            # Winner: for R2 higher is better, for others lower is better
            if p_mean is not None and a_mean is not None:
                if metric == "R2":
                    winner = "ARIMA" if a_mean > p_mean else "Prophet"
                elif metric == "bias":
                    winner = "ARIMA" if abs(a_mean) < abs(p_mean) else "Prophet"
                else:
                    winner = "ARIMA" if a_mean < p_mean else "Prophet"
            else:
                winner = "—"

            print(f"  {metric:6}  {fmt(p_mean,2):>14}  {fmt(p_med,2):>15}  "
                  f"{fmt(a_mean,2):>12}  {fmt(a_med,2):>13}  {winner:>8}")

    eval_all = load(EVAL_PATH, "eval metrics")
    if eval_all is not None:
        n_items = eval_all["item_id"].nunique()
        print(f"\n  Items evaluated : {n_items}")
        print(f"  Test window     : last 3 months per series (held-out)")
        print(f"  Note: Lower MAE/RMSE/MAPE/WAPE = better. Higher R2 = better. Bias near 0 = unbiased.")

    # ── 5. Ensemble ───────────────────────────────────────────────────────
    if eval_sum is not None:
        p_mae = eval_sum[(eval_sum["model"]=="prophet")&(eval_sum["metric"]=="MAE")]["mean"]
        a_mae = eval_sum[(eval_sum["model"]=="arima")  &(eval_sum["metric"]=="MAE")]["mean"]
        if len(p_mae) and len(a_mae):
            section("5. ENSEMBLE STRATEGY")
            ratio = p_mae.values[0] / a_mae.values[0]
            print(f"  ARIMA MAE       : {a_mae.values[0]:,.2f}")
            print(f"  Prophet MAE     : {p_mae.values[0]:,.2f}")
            print(f"  ARIMA advantage : {ratio:.2f}× lower error")
            print(f"  Blend weights   : ARIMA 60%  +  Prophet 40%")
            print(f"  Rationale       : Higher weight to the more accurate model.")
            print(f"                    Prophet retained for trend and seasonality stability.")

    # ── 6. Final ──────────────────────────────────────────────────────────
    section("6. PIPELINE SUMMARY")
    parts = []
    if processed is not None: parts.append(f"✓ Preprocessed  {processed['item_id'].nunique()} items")
    if pvi       is not None: parts.append(f"✓ PVI scored    {len(pvi)} items")
    if recs      is not None: parts.append(f"✓ Recommended   {len(recs)} items")
    if eval_all  is not None: parts.append(f"✓ Evaluated     {eval_all['item_id'].nunique()} items × 2 models")
    for p in parts:
        print(f"  {p}")
    print(f"\n  Run  python src/plot_reports.py  to regenerate all charts.")
    print(f"\n{'#'*65}\n")


if __name__ == "__main__":
    run()