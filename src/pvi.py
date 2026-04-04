import os
import glob
import numpy as np
import pandas as pd

PROCESSED_PATH     = "data/processed/processed_m5.csv"
PROPHET_FORECAST_DIR = "data/forecast/prophet"
ARIMA_FORECAST_DIR   = "data/forecast/arima"
PVI_OUTPUT_PATH    = "data/pvi_scores.csv"

ENSEMBLE_ARIMA_WEIGHT   = 0.60
ENSEMBLE_PROPHET_WEIGHT = 0.40

assert abs(ENSEMBLE_ARIMA_WEIGHT + ENSEMBLE_PROPHET_WEIGHT - 1.0) < 1e-9, "Ensemble weights must sum to 1"

ALPHA = 0.40
BETA  = 0.25
GAMMA = 0.20
DELTA = 0.15

assert abs(ALPHA + BETA + GAMMA + DELTA - 1.0) < 1e-9, "Weights must sum to 1"

HIGH_THRESHOLD   = 67
MEDIUM_THRESHOLD = 33


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def min_max_norm(series: pd.Series) -> pd.Series:
    return series.rank(pct=True)


def pvi_category(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "High"
    if score >= MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def safe_item_to_item_id(safe_item: str) -> str:
    return safe_item.replace("-", "_")


def parse_forecast_filename(fname: str):

    base  = os.path.basename(fname).replace(".csv", "")
    
    rest  = base[len("forecast_"):]

    parts    = rest.split("_")
    store_id = f"{parts[0]}_{parts[1]}"
    safe_item = "_".join(parts[2:])
    item_id   = safe_item_to_item_id(safe_item)

    return store_id, item_id


# ---------------------------------------------------------------------------
# Ensemble blending
# ---------------------------------------------------------------------------

def blend_forecasts(prophet_yhat: np.ndarray, arima_yhat: np.ndarray) -> np.ndarray:

    min_len = min(len(prophet_yhat), len(arima_yhat))
    prophet_yhat = prophet_yhat[:min_len]
    arima_yhat = arima_yhat[:min_len]

    blended = (ENSEMBLE_ARIMA_WEIGHT * arima_yhat) + (ENSEMBLE_PROPHET_WEIGHT * prophet_yhat)
    return blended


# ---------------------------------------------------------------------------
# Sub-score computation
# ---------------------------------------------------------------------------

def compute_demand_score(yhat: np.ndarray) -> float:
    return float(np.mean(yhat))


def compute_growth_score(yhat: np.ndarray) -> float:

    if yhat[0] < 1e-6:
        return 0.0
    growth = (yhat[-1] - yhat[0]) / yhat[0]
    return float(np.clip(growth, -1.0, 1.0))


def compute_stability_score(monthly_sales: np.ndarray) -> float:

    mean = np.mean(monthly_sales)
    if mean < 1e-6:
        return 99.0
    return float(np.std(monthly_sales) / mean)


def compute_price_score(avg_prices: pd.Series) -> float:
    val = avg_prices.dropna().mean()
    return float(val) if not np.isnan(val) else 0.0


# ---------------------------------------------------------------------------
# Anomaly flags
# ---------------------------------------------------------------------------

def detect_anomalies(monthly_sales: np.ndarray) -> dict:

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
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(
            f"Processed dataset not found at {PROCESSED_PATH}.\n"
            "Run  python src/preprocess.py  first."
        )

    processed = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    print(f"Loaded processed data: {len(processed):,} rows, "
          f"{processed['item_id'].nunique()} items, "
          f"{processed['store_id'].nunique()} stores")

    processed_idx = processed.set_index(["store_id", "item_id"])

    prophet_files = sorted(glob.glob(os.path.join(PROPHET_FORECAST_DIR, "forecast_*.csv")))
    arima_files   = sorted(glob.glob(os.path.join(ARIMA_FORECAST_DIR,   "forecast_*.csv")))

    if not prophet_files and not arima_files:
        raise FileNotFoundError(
            "No forecast files found in data/forecast/prophet/ or data/forecast/arima/.\n"
            "Run  python src/forecast.py  first."
        )

    arima_lookup = {}
    for f in arima_files:
        store, item = parse_forecast_filename(f)
        arima_lookup[(store, item)] = f

    print(f"Forecast files — Prophet: {len(prophet_files)}, ARIMA: {len(arima_files)}")

    records = []

    for fpath in prophet_files:
        store_id, item_id = parse_forecast_filename(fpath)

        fc = pd.read_csv(fpath)
        prophet_yhat = fc["yhat"].values

        key = (store_id, item_id)
        if key in processed_idx.index:
            hist = processed_idx.loc[[key]]
            monthly_sales = hist["monthly_sales"].values
            avg_prices    = hist["avg_price"]
        else:
            monthly_sales = prophet_yhat
            avg_prices    = pd.Series([np.nan])

        arima_yhat = None
        if key in arima_lookup:
            arima_fc    = pd.read_csv(arima_lookup[key])
            arima_yhat  = arima_fc["yhat"].values
            blended_yhat = blend_forecasts(prophet_yhat, arima_yhat)
        else:
            blended_yhat = prophet_yhat

        demand_raw = compute_demand_score(blended_yhat)
        growth_raw = compute_growth_score(prophet_yhat)
        stability_raw = compute_stability_score(monthly_sales)
        price_raw     = compute_price_score(avg_prices)

        model_agreement = None
        if arima_yhat is not None and len(arima_yhat) == len(prophet_yhat):
            denom = np.abs(prophet_yhat) + 1e-6
            model_agreement = float(1.0 - np.mean(np.abs(prophet_yhat - arima_yhat) / denom))
            model_agreement = float(np.clip(model_agreement, 0.0, 1.0))

        anomaly_info = detect_anomalies(monthly_sales)

        cat_id  = hist["cat_id"].iloc[0]  if key in processed_idx.index else "UNKNOWN"
        dept_id = hist["dept_id"].iloc[0] if key in processed_idx.index else "UNKNOWN"

        records.append({
            "store_id":        store_id,
            "item_id":         item_id,
            "cat_id":          cat_id,
            "dept_id":         dept_id,
            "demand_raw":      demand_raw,
            "growth_raw":      growth_raw,
            "stability_cv":    stability_raw,
            "price_raw":       price_raw,
            "forecast_mean":   float(np.mean(prophet_yhat)),
            "forecast_lower":  float(fc["yhat_lower"].mean()) if "yhat_lower" in fc else None,
            "forecast_upper":  float(fc["yhat_upper"].mean()) if "yhat_upper" in fc else None,
            "model_agreement": model_agreement,
            "has_anomaly":     anomaly_info["has_anomaly"],
            "anomaly_count":   anomaly_info["anomaly_count"],
            "anomaly_pct":     anomaly_info["anomaly_pct"],
        })

    pvi_df = pd.DataFrame(records)

    if pvi_df.empty:
        print("No records collected — check that forecast files exist and filenames match expected pattern.")
        return

    pvi_df["demand_norm"]    = min_max_norm(pvi_df["demand_raw"])
    pvi_df["growth_norm"]    = min_max_norm(pvi_df["growth_raw"])
    pvi_df["stability_norm"] = 1.0 - min_max_norm(pvi_df["stability_cv"])
    pvi_df["price_norm"]     = min_max_norm(pvi_df["price_raw"])

    pvi_df["PVI"] = (
        ALPHA * pvi_df["demand_norm"]
        + BETA  * pvi_df["growth_norm"]
        + GAMMA * pvi_df["stability_norm"]
        + DELTA * pvi_df["price_norm"]
    ) * 100

    pvi_df["PVI"] = pvi_df["PVI"].round(2)

    pvi_df["viability"] = pvi_df["PVI"].apply(pvi_category)

    pvi_df = pvi_df.sort_values("PVI", ascending=False).reset_index(drop=True)
    pvi_df["rank_overall"] = pvi_df.index + 1
    pvi_df["rank_in_category"] = (
        pvi_df.groupby("viability")["PVI"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    os.makedirs("data", exist_ok=True)
    pvi_df.to_csv(PVI_OUTPUT_PATH, index=False)

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