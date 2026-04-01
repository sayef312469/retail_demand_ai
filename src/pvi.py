"""
pvi.py — Product Viability Index (PVI) computation.

Computes a 0-100 PVI score for every (store_id, item_id) series by
combining four sub-scores derived from forecasts and historical data:

    PVI = (α · demand + β · growth + γ · stability + δ · price) × 100

Sub-score definitions
---------------------
demand    — normalised mean forecasted sales over the 3-month horizon.
            Higher forecasted volume = higher viability.

growth    — normalised slope of the forecast horizon (last − first) / first.
            Positive growth trend pushes viability up.

stability — 1 − normalised coefficient of variation (std / mean) of
            historical monthly sales. Low volatility = high stability.

price     — normalised average sell price from processed data.
            Higher price per unit contributes more revenue per sale.

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
  data/forecast/arima/*.csv           — run train_arima.py first (optional)

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
    """Min-max normalise a Series to [0, 1]. Returns 0.5 if constant."""
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


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
        yhat = fc["yhat"].values

        # Historical series for stability and price
        key = (store_id, item_id)
        if key in processed_idx.index:
            hist = processed_idx.loc[[key]]
            monthly_sales = hist["monthly_sales"].values
            avg_prices    = hist["avg_price"]
        else:
            monthly_sales = yhat   # fallback: use forecast as proxy
            avg_prices    = pd.Series([np.nan])

        # ARIMA forecast for model-agreement confidence
        arima_yhat = None
        if key in arima_lookup:
            arima_fc    = pd.read_csv(arima_lookup[key])
            arima_yhat  = arima_fc["yhat"].values

        # Raw sub-scores (normalised in batch below)
        demand_raw    = compute_demand_score(yhat)
        growth_raw    = compute_growth_score(yhat)
        stability_raw = compute_stability_score(monthly_sales)   # CV — lower is better
        price_raw     = compute_price_score(avg_prices)

        # Model agreement: mean absolute % difference between Prophet and ARIMA forecasts
        if arima_yhat is not None and len(arima_yhat) == len(yhat):
            denom = np.abs(yhat) + 1e-6
            model_agreement = float(1.0 - np.mean(np.abs(yhat - arima_yhat) / denom))
            model_agreement = float(np.clip(model_agreement, 0.0, 1.0))
        else:
            model_agreement = None

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
            # Raw sub-scores (pre-normalisation)
            "demand_raw":      demand_raw,
            "growth_raw":      growth_raw,
            "stability_cv":    stability_raw,   # CV — inverted later
            "price_raw":       price_raw,
            # Forecast metadata
            "forecast_mean":   float(np.mean(yhat)),
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