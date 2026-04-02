"""
api.py — FastAPI backend for the Retail Demand AI system.

New in this version
-------------------
  GET  /plots                 — list all available plot images
  GET  /plots/{filename}      — serve a plot PNG (via StaticFiles)
  POST /upload                — upload CSV + optionally run forecast on it
  POST /upload/forecast       — run mini Prophet forecast on uploaded file
"""

import os
import io
import glob
import numpy as np
import pandas as pd
import warnings
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

app = FastAPI(
    title="Retail Demand AI API",
    description="Demand forecasting, PVI scoring and stock recommendations.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve plot PNGs as static files at /plots/<filename>
PLOTS_DIR = "outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
app.mount("/plots", StaticFiles(directory=PLOTS_DIR), name="plots")

# ── File paths ────────────────────────────────────────────────────────────
PROCESSED_PATH        = "data/processed/processed_m5.csv"
PROPHET_FORECAST_DIR  = "data/forecast/prophet"
ARIMA_FORECAST_DIR    = "data/forecast/arima"
PVI_PATH              = "data/pvi_scores.csv"
RECS_PATH             = "data/recommendations.csv"
EVAL_PATH             = "data/eval_metrics.csv"
EVAL_SUMMARY_PATH     = "data/eval_summary.csv"
UPLOAD_DIR            = "data/uploads"

# ── In-memory cache ───────────────────────────────────────────────────────
_cache: dict = {}

def _load(path: str, key: str) -> pd.DataFrame:
    if key not in _cache:
        if not os.path.exists(path):
            return None
        _cache[key] = pd.read_csv(path)
    return _cache[key]

def _invalidate(key: str):
    _cache.pop(key, None)

def _require(df, path: str) -> pd.DataFrame:
    if df is None:
        raise HTTPException(status_code=503,
            detail=f"Data file not found: {path}. Run the pipeline first.")
    return df

def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj

def _forecast_path(store_id: str, item_id: str, model: str) -> str:
    safe_item = item_id.replace("_", "-")
    d = PROPHET_FORECAST_DIR if model == "prophet" else ARIMA_FORECAST_DIR
    return os.path.join(d, f"forecast_{store_id}_{safe_item}.csv")


# ── PVI & Recommendation Helpers ─────────────────────────────────────────

# PVI formula weights
PVI_ALPHA = 0.40   # demand weight
PVI_BETA  = 0.25   # growth weight
PVI_GAMMA = 0.20   # stability weight
PVI_DELTA = 0.15   # price weight

# Thresholds for PVI categories
PVI_HIGH   = 67
PVI_MEDIUM = 33

# Thresholds for recommendation logic
GROWTH_POSITIVE_THRESHOLD = 0.55
GROWTH_NEGATIVE_THRESHOLD = 0.40
STABILITY_LOW_THRESHOLD   = 0.40
ANOMALY_SERIOUS_THRESHOLD = 0.15
AGREEMENT_LOW_THRESHOLD   = 0.70
AGREEMENT_HIGH_THRESHOLD  = 0.90


def _min_max_norm(series: pd.Series) -> pd.Series:
    """Min-max normalise a Series to [0, 1]. Returns 0.5 if constant."""
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def _compute_demand_score(yhat: np.ndarray) -> float:
    """Mean forecasted sales over the horizon."""
    return float(np.mean(yhat))


def _compute_growth_score(yhat: np.ndarray) -> float:
    """Percentage growth from first to last forecast period."""
    if yhat[0] < 1e-6:
        return 0.0
    growth = (yhat[-1] - yhat[0]) / yhat[0]
    return float(np.clip(growth, -1.0, 1.0))


def _compute_stability_score(monthly_sales: np.ndarray) -> float:
    """Coefficient of variation (CV = std / mean) of historical monthly sales."""
    mean = np.mean(monthly_sales)
    if mean < 1e-6:
        return 99.0
    return float(np.std(monthly_sales) / mean)


def _detect_anomalies(monthly_sales: np.ndarray) -> dict:
    """IQR-based anomaly detection on historical sales series."""
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


def _pvi_category(score: float) -> str:
    """Categorize PVI score to High/Medium/Low."""
    if score >= PVI_HIGH:
        return "High"
    if score >= PVI_MEDIUM:
        return "Medium"
    return "Low"


def _compute_pvi(demand_norm: float, growth_norm: float, stability_norm: float, price_norm: float) -> float:
    """Compute final PVI score (0-100)."""
    return (
        PVI_ALPHA * demand_norm
        + PVI_BETA * growth_norm
        + PVI_GAMMA * stability_norm
        + PVI_DELTA * price_norm
    ) * 100


def _compute_confidence(pvi: float, viability: str, has_anomaly: bool, anomaly_pct: float, model_agreement: float = None) -> str:
    """Return High / Medium / Low confidence."""
    score = 1.0
    
    # Model agreement penalty
    if model_agreement is not None and not np.isnan(model_agreement):
        if model_agreement < AGREEMENT_LOW_THRESHOLD:
            score -= 0.4
        elif model_agreement < AGREEMENT_HIGH_THRESHOLD:
            score -= 0.15
    
    # Anomaly penalty
    if has_anomaly:
        score -= 0.2
    
    # PVI distance from nearest boundary
    dist_to_boundary = min(abs(pvi - PVI_HIGH), abs(pvi - PVI_MEDIUM))
    if dist_to_boundary < 5:
        score -= 0.2
    
    if score >= 0.75:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"


def _build_explanation(
    decision: str,
    pvi: float,
    viability: str,
    demand_norm: float,
    growth_norm: float,
    stability_norm: float,
    price_norm: float,
    has_anomaly: bool,
    anomaly_pct: float,
    growth_label: str,
    risk_label: str,
    anomaly_override: bool,
) -> str:
    """Build human-readable explanation for the recommendation."""
    parts = []
    
    # PVI summary
    parts.append(
        f"PVI={pvi:.1f}/100 ({viability} viability): "
        f"demand={demand_norm:.2f}, growth={growth_norm:.2f}, "
        f"stability={stability_norm:.2f}, price={price_norm:.2f}"
    )
    
    # Anomaly note
    if has_anomaly:
        pct = anomaly_pct * 100
        parts.append(
            f"⚠ Anomaly detected in {pct:.0f}% of historical months"
            + (" — override applied" if anomaly_override else "")
        )
    
    # Decision rationale
    if decision == "Increase":
        parts.append(
            f"Recommend stocking up: {growth_label} demand trend, "
            f"{risk_label} supply risk, and {viability.lower()} viability support expansion."
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
    
    return " | ".join(parts)


def _make_decision(
    pvi: float,
    viability: str,
    demand_norm: float,
    growth_norm: float,
    stability_norm: float,
    price_norm: float,
    has_anomaly: bool,
    anomaly_pct: float,
) -> tuple:
    """Return (decision, confidence, explanation)."""
    
    # Growth label
    if growth_norm >= GROWTH_POSITIVE_THRESHOLD:
        growth_label = "positive"
    elif growth_norm <= GROWTH_NEGATIVE_THRESHOLD:
        growth_label = "declining"
    else:
        growth_label = "flat"
    
    # Risk label
    if stability_norm >= 0.60:
        risk_label = "low"
    elif stability_norm >= STABILITY_LOW_THRESHOLD:
        risk_label = "moderate"
    else:
        risk_label = "high"
    
    is_positive_growth  = growth_norm >= GROWTH_POSITIVE_THRESHOLD
    is_declining_growth = growth_norm <= GROWTH_NEGATIVE_THRESHOLD
    is_high_risk        = stability_norm < STABILITY_LOW_THRESHOLD
    serious_anomaly     = has_anomaly and anomaly_pct >= ANOMALY_SERIOUS_THRESHOLD
    
    anomaly_override = False
    
    # Layer 1: Hard anomaly overrides
    if serious_anomaly:
        anomaly_override = True
        if is_declining_growth:
            decision = "Decrease"
        else:
            decision = "Hold"
    
    # Layer 2: PVI-driven matrix
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
            decision = "Hold"
        else:
            decision = "Decrease"
    
    confidence  = _compute_confidence(pvi, viability, has_anomaly, anomaly_pct)
    explanation = _build_explanation(
        decision, pvi, viability, demand_norm, growth_norm, stability_norm, price_norm,
        has_anomaly, anomaly_pct, growth_label, risk_label, anomaly_override
    )
    
    return decision, confidence, explanation


# ── System ────────────────────────────────────────────────────────────────

# ── ARIMA Helpers ────────────────────────────────────────────────────────

def _is_stationary(series: pd.Series, alpha: float = 0.05) -> bool:
    """Return True if the ADF test rejects the unit-root null at level alpha."""
    try:
        p_value = adfuller(series.dropna(), autolag="AIC")[1]
        return p_value < alpha
    except Exception:
        return False


def _select_d(series: pd.Series) -> int:
    """Choose d = 0 (stationary) or d = 1 (one difference needed)."""
    if _is_stationary(series):
        return 0
    if _is_stationary(series.diff().dropna()):
        return 1
    return 1  # default


def _auto_arima_aic(series: pd.Series, p_values, d_values, q_values):
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
            continue  # trivial model — not useful
        try:
            fit = ARIMA(series, order=(p, d, q)).fit()
            if fit.aic < best_aic:
                best_aic   = fit.aic
                best_order = (p, d, q)
                best_fit   = fit
        except Exception:
            continue

    return best_fit, best_order


def _forecast_arima(series: pd.Series, periods: int = 3):
    """
    Fit best ARIMA model and return a DataFrame of forecasted values.
    Returns (forecast_df, order) or (None, None) if series is too short or all orders fail.
    """
    series = series.dropna()

    if len(series) < 10:  # Minimum for quick upload forecasts (vs 24 for batch)
        return None, None

    # Step 1 — choose d via stationarity test
    d = _select_d(series)

    # Step 2 — grid-search p and q
    p_values = [0, 1, 2]
    q_values = [0, 1, 2]
    model, order = _auto_arima_aic(series, p_values, [d], q_values)

    # Fallback if grid search fails
    if model is None:
        try:
            model = ARIMA(series, order=(1, 1, 1)).fit()
            order = (1, 1, 1)
        except Exception:
            return None, None

    # Step 3 — generate forecast with 95% CI
    try:
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
            "ds":          future_dates,
            "yhat":        summary["mean"].values,
            "yhat_lower":  summary["mean_ci_lower"].values,
            "yhat_upper":  summary["mean_ci_upper"].values,
            "model_order": [str(order)] * periods,
        })

        # Sales cannot be negative
        for col in ["yhat", "yhat_lower", "yhat_upper"]:
            result[col] = result[col].clip(lower=0)

        return result, order
    except Exception:
        return None, None


# ── System ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "files": {
        "processed":   os.path.exists(PROCESSED_PATH),
        "pvi":         os.path.exists(PVI_PATH),
        "recommend":   os.path.exists(RECS_PATH),
        "eval":        os.path.exists(EVAL_PATH),
        "prophet_dir": os.path.isdir(PROPHET_FORECAST_DIR),
        "arima_dir":   os.path.isdir(ARIMA_FORECAST_DIR),
        "plots_dir":   os.path.isdir(PLOTS_DIR),
    }}


@app.get("/summary", tags=["Dashboard"])
def get_summary():
    pvi_df  = _load(PVI_PATH,  "pvi")
    rec_df  = _load(RECS_PATH, "recs")
    eval_df = _load(EVAL_SUMMARY_PATH, "eval_summary")
    _require(pvi_df,  PVI_PATH)
    _require(rec_df,  RECS_PATH)

    viability_counts = pvi_df["viability"].value_counts().to_dict()
    decision_counts  = rec_df["Decision"].value_counts().to_dict()
    anomaly_count    = int(rec_df["has_anomaly"].sum()) if "has_anomaly" in rec_df.columns else 0

    model_perf = {}
    if eval_df is not None:
        for metric in ["MAE", "RMSE", "MAPE", "WAPE", "R2"]:
            rows = eval_df[eval_df["metric"] == metric]
            for _, row in rows.iterrows():
                if row["model"] not in model_perf:
                    model_perf[row["model"]] = {}
                model_perf[row["model"]][f"{metric}_mean"]   = row["mean"]
                model_perf[row["model"]][f"{metric}_median"] = row["median"]

    # Distribution metrics for better coverage visibility
    items_per_store = pvi_df["store_id"].value_counts().to_dict()
    items_per_category = pvi_df["cat_id"].value_counts().to_dict() if "cat_id" in pvi_df.columns else {}
    store_list = sorted(pvi_df["store_id"].unique().tolist())
    category_list = sorted(pvi_df["cat_id"].unique().tolist()) if "cat_id" in pvi_df.columns else []

    return _clean({
        "total_items":            len(pvi_df),
        "total_stores":           int(pvi_df["store_id"].nunique()),
        "total_categories":       int(pvi_df["cat_id"].nunique()) if "cat_id" in pvi_df.columns else 0,
        "viability":              viability_counts,
        "decisions":              decision_counts,
        "anomaly_count":          anomaly_count,
        "pvi_mean":               round(float(pvi_df["PVI"].mean()), 1),
        "pvi_median":             round(float(pvi_df["PVI"].median()), 1),
        "model_performance":      model_perf,
        # Distribution metrics
        "distribution": {
            "items_per_store":     items_per_store,
            "items_per_category":  items_per_category,
            "stores":              store_list,
            "categories":          category_list,
        }
    })


# ── Data ──────────────────────────────────────────────────────────────────

@app.get("/stores", tags=["Data"])
def list_stores():
    pvi_df = _require(_load(PVI_PATH, "pvi"), PVI_PATH)
    return {"stores": sorted(pvi_df["store_id"].unique().tolist())}


@app.get("/items", tags=["Data"])
def list_items(
    store_id: str = Query(...),
    category: Optional[str] = Query(None),
    viability: Optional[str] = Query(None),
):
    pvi_df = _require(_load(PVI_PATH, "pvi"), PVI_PATH)
    df = pvi_df[pvi_df["store_id"] == store_id]
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Store '{store_id}' not found.")
    if category:  df = df[df["cat_id"]    == category.upper()]
    if viability: df = df[df["viability"] == viability.capitalize()]
    cols   = ["item_id", "cat_id", "dept_id", "PVI", "viability"]
    result = df[cols].sort_values("PVI", ascending=False).to_dict(orient="records")
    return {"store_id": store_id, "count": len(result), "items": _clean(result)}


# ── Forecast ──────────────────────────────────────────────────────────────

@app.get("/forecast/{store_id}/{item_id}", tags=["Forecast"])
def get_forecast(store_id: str, item_id: str,
                 model: str = Query("both"),
                 months: int = Query(12, ge=1, le=12)):
    """
    Get forecast for a store-item combination.
    
    Parameters:
    - store_id: Store ID (e.g., "CA_1")
    - item_id: Item ID (e.g., "FOODS_1_001")
    - model: "prophet", "arima", or "both"
    - months: Forecast horizon to return (1-12, default 12 for full forecast). Returns first N months.
    """
    result = {"store_id": store_id, "item_id": item_id, "months_requested": months}
    processed = _load(PROCESSED_PATH, "processed")
    if processed is not None:
        hist = processed[
            (processed["store_id"] == store_id) &
            (processed["item_id"]  == item_id)
        ].sort_values("date")
        if not hist.empty:
            result["actuals"] = {
                "dates": hist["date"].astype(str).tolist(),
                "sales": hist["monthly_sales"].round(2).tolist(),
            }
    if model in ("prophet", "both"):
        fpath = _forecast_path(store_id, item_id, "prophet")
        if os.path.exists(fpath):
            fc = pd.read_csv(fpath)
            # Slice to requested months if month_index column exists, otherwise take first N rows
            if "month_index" in fc.columns:
                fc = fc[fc["month_index"] <= months]
            else:
                fc = fc.head(months)
            result["prophet"] = {
                "dates":      fc["ds"].astype(str).tolist(),
                "yhat":       fc["yhat"].round(2).tolist(),
                "yhat_lower": fc["yhat_lower"].round(2).tolist() if "yhat_lower" in fc else [],
                "yhat_upper": fc["yhat_upper"].round(2).tolist() if "yhat_upper" in fc else [],
            }
            result["months_available_prophet"] = len(pd.read_csv(fpath))
    if model in ("arima", "both"):
        fpath = _forecast_path(store_id, item_id, "arima")
        if os.path.exists(fpath):
            fc = pd.read_csv(fpath)
            # Slice to requested months if month_index column exists, otherwise take first N rows
            if "month_index" in fc.columns:
                fc = fc[fc["month_index"] <= months]
            else:
                fc = fc.head(months)
            result["arima"] = {
                "dates":       fc["ds"].astype(str).tolist(),
                "yhat":        fc["yhat"].round(2).tolist(),
                "yhat_lower":  fc["yhat_lower"].round(2).tolist() if "yhat_lower" in fc else [],
                "yhat_upper":  fc["yhat_upper"].round(2).tolist() if "yhat_upper" in fc else [],
                "model_order": fc["model_order"].iloc[0] if "model_order" in fc else None,
            }
            result["months_available_arima"] = len(pd.read_csv(fpath))
    
    # Compute blended forecast (weighted ensemble: 60% ARIMA + 40% Prophet)
    # ARIMA is more accurate on this dataset (MAE 50% lower)
    if model in ("both") and "prophet" in result and "arima" in result:
        prophet_yhat = np.array(result["prophet"]["yhat"])
        arima_yhat = np.array(result["arima"]["yhat"])
        prophet_lower = np.array(result["prophet"]["yhat_lower"]) if result["prophet"]["yhat_lower"] else prophet_yhat
        prophet_upper = np.array(result["prophet"]["yhat_upper"]) if result["prophet"]["yhat_upper"] else prophet_yhat
        arima_lower = np.array(result["arima"]["yhat_lower"]) if result["arima"]["yhat_lower"] else arima_yhat
        arima_upper = np.array(result["arima"]["yhat_upper"]) if result["arima"]["yhat_upper"] else arima_yhat
        
        # Weighted blend: 60% ARIMA (more accurate) + 40% Prophet (more stable)
        blended_yhat = 0.60 * arima_yhat + 0.40 * prophet_yhat
        blended_lower = 0.60 * arima_lower + 0.40 * prophet_lower
        blended_upper = 0.60 * arima_upper + 0.40 * prophet_upper
        
        result["ensemble"] = {
            "dates":      result["prophet"]["dates"],
            "yhat":       np.round(blended_yhat, 2).tolist(),
            "yhat_lower": np.round(blended_lower, 2).tolist(),
            "yhat_upper": np.round(blended_upper, 2).tolist(),
        }
    
    if len(result) == 4:  # Only store_id, item_id, months_requested, and one of the availability fields
        raise HTTPException(status_code=404, detail=f"No forecasts for {store_id}/{item_id}")
    return _clean(result)


# ── PVI ───────────────────────────────────────────────────────────────────

@app.get("/pvi/{store_id}/{item_id}", tags=["PVI"])
def get_pvi(store_id: str, item_id: str):
    pvi_df = _require(_load(PVI_PATH, "pvi"), PVI_PATH)
    row = pvi_df[(pvi_df["store_id"] == store_id) & (pvi_df["item_id"] == item_id)]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"No PVI for {store_id}/{item_id}")
    return _clean(row.iloc[0].to_dict())


@app.get("/pvi", tags=["PVI"])
def list_pvi(store_id: Optional[str] = Query(None),
             viability: Optional[str] = Query(None),
             category:  Optional[str] = Query(None),
             limit: int = Query(50, ge=1, le=500),
             offset: int = Query(0, ge=0)):
    pvi_df = _require(_load(PVI_PATH, "pvi"), PVI_PATH)
    df = pvi_df.copy()
    if store_id:  df = df[df["store_id"]  == store_id]
    if viability: df = df[df["viability"] == viability.capitalize()]
    if category:  df = df[df["cat_id"]    == category.upper()]
    total = len(df)
    df    = df.sort_values("PVI", ascending=False).iloc[offset:offset + limit]
    return _clean({"total": total, "offset": offset, "limit": limit,
                   "items": df.to_dict(orient="records")})


# ── Recommendations ───────────────────────────────────────────────────────

@app.get("/recommendation/{store_id}/{item_id}", tags=["Recommendations"])
def get_recommendation(store_id: str, item_id: str):
    rec_df = _require(_load(RECS_PATH, "recs"), RECS_PATH)
    row = rec_df[(rec_df["store_id"] == store_id) & (rec_df["item_id"] == item_id)]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"No recommendation for {store_id}/{item_id}")
    return _clean(row.iloc[0].to_dict())


@app.get("/recommendations", tags=["Recommendations"])
def list_recommendations(
    store_id:   Optional[str]  = Query(None),
    decision:   Optional[str]  = Query(None),
    viability:  Optional[str]  = Query(None),
    confidence: Optional[str]  = Query(None),
    anomaly:    Optional[bool] = Query(None),
    limit: int  = Query(50, ge=1, le=10000),
    offset: int = Query(0, ge=0),
):
    rec_df = _require(_load(RECS_PATH, "recs"), RECS_PATH)
    df = rec_df.copy()
    if store_id:   df = df[df["store_id"]   == store_id]
    if decision:   df = df[df["Decision"]   == decision.capitalize()]
    if viability:  df = df[df["Viability"]  == viability.capitalize()]
    if confidence: df = df[df["Confidence"] == confidence.capitalize()]
    if anomaly is not None and "has_anomaly" in df.columns:
        df = df[df["has_anomaly"] == anomaly]
    total = len(df)
    df    = df.sort_values(["Decision", "PVI"], ascending=[True, False]).iloc[offset:offset + limit]
    return _clean({"total": total, "offset": offset, "limit": limit,
                   "items": df.to_dict(orient="records")})


# ── Evaluation ────────────────────────────────────────────────────────────

@app.get("/eval", tags=["Evaluation"])
def get_eval_summary():
    summary_df = _load(EVAL_SUMMARY_PATH, "eval_summary")
    if summary_df is None:
        raise HTTPException(status_code=503, detail="Run src/evaluate.py first.")
    return _clean({"summary": summary_df.to_dict(orient="records")})


@app.get("/eval/{store_id}/{item_id}", tags=["Evaluation"])
def get_eval_item(store_id: str, item_id: str):
    eval_df = _load(EVAL_PATH, "eval")
    if eval_df is None:
        raise HTTPException(status_code=503, detail="Run src/evaluate.py first.")
    rows = eval_df[(eval_df["store_id"] == store_id) & (eval_df["item_id"] == item_id)]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No eval data for {store_id}/{item_id}")
    return _clean({"metrics": rows.to_dict(orient="records")})


# ── Plots ─────────────────────────────────────────────────────────────────

@app.get("/plots", tags=["Plots"])
def list_plots():
    """
    Return all available plot filenames.
    Images are served at  GET /plots/{filename}.
    """
    if not os.path.isdir(PLOTS_DIR):
        return {"plots": []}
    files = sorted([
        f for f in os.listdir(PLOTS_DIR)
        if f.endswith(".png")
    ])
    return {
        "plots": [
            {"filename": f, "url": f"/plots/{f}",
             "title": f.replace("_", " ").replace(".png", "").title()}
            for f in files
        ]
    }


# ── Upload ────────────────────────────────────────────────────────────────

@app.post("/upload", tags=["Upload"])
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a custom sales CSV for preview and optional forecasting.

    What this actually does
    -----------------------
    This endpoint accepts your own historical sales data, saves it,
    and returns a column summary and a data preview.

    To run a forecast on uploaded data, call POST /upload/forecast
    after uploading — it will run a quick Prophet model on the file
    and return 3-month ahead predictions.

    Expected CSV columns (flexible — code auto-detects):
        date / Date / week_start_date   — date of each sales row
        sales / Weekly_Sales / Sales    — sales figure
        store_id / Store                — store identifier
        item_id / product / SKU         — product identifier (optional)
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = os.path.basename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, safe_name)

    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")

    # Detect column types
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    date_cols    = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]

    return _clean({
        "filename":     safe_name,
        "saved_to":     save_path,
        "rows":         len(df),
        "columns":      df.columns.tolist(),
        "numeric_cols": numeric_cols,
        "date_cols":    date_cols,
        "preview":      df.head(5).to_dict(orient="records"),
        "next_step":    "POST /upload/forecast to run Prophet on this file",
    })


@app.post("/upload/forecast", tags=["Upload"])
async def forecast_uploaded_csv(
    file: UploadFile = File(...),
    date_col:  str = Query("date",  description="Name of the date column"),
    sales_col: str = Query("sales", description="Name of the sales column"),
    periods:   int = Query(3,       description="Months ahead to forecast", ge=1, le=12),
    model:     str = Query("both",  description="Forecasting model: 'prophet', 'arima', or 'both'"),
):
    """
    Upload your own sales CSV and get Prophet and/or ARIMA forecasts back instantly.

    This is the actual 'upload a product's history → get a forecast' feature.

    How it works
    ------------
    1. You upload a CSV with at least a date column and a sales column.
    2. The API fits Prophet and/or ARIMA models to your entire series.
    3. It returns the historical data + forecasts for the next N months,
       with 95% confidence intervals.

    Your CSV must have at least these two columns (names are flexible):
        date    — e.g. 2022-01-01, 01/01/2022 — monthly or weekly dates
        sales   — numeric sales figures

    Example curl
    ------------
    curl -X POST "http://localhost:8000/upload/forecast?date_col=date&sales_col=sales&periods=3&model=both" \\
         -F "file=@my_product_sales.csv"
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted.")

    if model not in ("prophet", "arima", "both"):
        raise HTTPException(status_code=422, detail="model must be 'prophet', 'arima', or 'both'")

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")

    # Flexible column detection
    def find_col(df, candidates):
        cols_lower = {c.lower(): c for c in df.columns}
        for c in candidates:
            if c in df.columns:        return c
            if c.lower() in cols_lower: return cols_lower[c.lower()]
        return None

    resolved_date  = find_col(df, [date_col,  "date", "Date", "DATE", "week_start_date", "month"])
    resolved_sales = find_col(df, [sales_col, "sales", "Sales", "Weekly_Sales", "weekly_sales", "quantity"])

    if resolved_date is None:
        raise HTTPException(status_code=422,
            detail=f"Date column not found. Tried: {date_col}, date, Date, week_start_date. "
                   f"Available columns: {df.columns.tolist()}")
    if resolved_sales is None:
        raise HTTPException(status_code=422,
            detail=f"Sales column not found. Tried: {sales_col}, sales, Weekly_Sales. "
                   f"Available columns: {df.columns.tolist()}")

    df[resolved_date]  = pd.to_datetime(df[resolved_date], errors="coerce")
    df[resolved_sales] = pd.to_numeric(df[resolved_sales], errors="coerce")
    df = df.dropna(subset=[resolved_date, resolved_sales])
    df = df[df[resolved_sales] >= 0]

    if len(df) < 10:
        raise HTTPException(status_code=422,
            detail=f"Need at least 10 valid rows after cleaning. Got {len(df)}.")

    # Aggregate to monthly if needed
    df["month"] = df[resolved_date].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month")[resolved_sales].sum().reset_index()
    monthly.columns = ["ds", "y"]

    hist_dates = monthly["ds"].dt.strftime("%Y-%m-%d").tolist()
    hist_sales = monthly["y"].round(2).tolist()

    result = {
        "filename":         file.filename,
        "rows_used":        len(monthly),
        "date_col_used":    resolved_date,
        "sales_col_used":   resolved_sales,
        "forecast_periods": periods,
        "actuals": {
            "dates": hist_dates,
            "sales": hist_sales,
        },
    }

    # ── Prophet Forecast ─────────────────────────────────────────────
    if model in ("prophet", "both"):
        try:
            from prophet import Prophet
            prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                interval_width=0.95,
            )
            prophet_model.fit(monthly)
            future   = prophet_model.make_future_dataframe(periods=periods, freq="MS")
            forecast = prophet_model.predict(future)
            horizon  = forecast.tail(periods)

            result["prophet"] = {
                "dates":      horizon["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "yhat":       horizon["yhat"].clip(lower=0).round(2).tolist(),
                "yhat_lower": horizon["yhat_lower"].clip(lower=0).round(2).tolist(),
                "yhat_upper": horizon["yhat_upper"].clip(lower=0).round(2).tolist(),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prophet fitting failed: {e}")

    # ── ARIMA Forecast ───────────────────────────────────────────────
    if model in ("arima", "both"):
        # Prepare monthly series with date index for ARIMA
        monthly_indexed = monthly.copy()
        monthly_indexed["ds"] = pd.to_datetime(monthly_indexed["ds"])
        monthly_indexed = monthly_indexed.set_index("ds")
        series = monthly_indexed["y"].asfreq("MS")

        forecast_df, order = _forecast_arima(series, periods=periods)

        if forecast_df is not None:
            result["arima"] = {
                "dates":       forecast_df["ds"].dt.strftime("%Y-%m-%d").tolist(),
                "yhat":        forecast_df["yhat"].round(2).tolist(),
                "yhat_lower":  forecast_df["yhat_lower"].round(2).tolist(),
                "yhat_upper":  forecast_df["yhat_upper"].round(2).tolist(),
                "model_order": forecast_df["model_order"].iloc[0] if len(forecast_df) > 0 else None,
            }
        else:
            result["arima"] = {"error": "ARIMA model failed to converge on this data"}

    # ── PVI & Recommendation Computation ──────────────────────────────
    try:
        monthly_sales = monthly["y"].values
        
        # Get forecast yhat (prefer Prophet if available, else ARIMA)
        if "prophet" in result:
            yhat = np.array(result["prophet"]["yhat"])
        elif "arima" in result and "yhat" in result["arima"]:
            yhat = np.array(result["arima"]["yhat"])
        else:
            yhat = np.array([np.mean(monthly_sales)] * periods)
        
        # Compute raw sub-scores
        demand_raw    = _compute_demand_score(yhat)
        growth_raw    = _compute_growth_score(yhat)
        stability_raw = _compute_stability_score(monthly_sales)
        price_raw     = 0.5  # Default price normalization (no price data from CSV)
        
        # Normalize to [0, 1]
        # For normalization across single items, use reasonable bounds
        demand_norm    = min(demand_raw / max(demand_raw, 1e-6), 1.0)  # cap at 1.0
        growth_norm    = (growth_raw + 1.0) / 2.0  # Convert [-1, 1] to [0, 1]
        stability_norm = 1.0 - min(stability_raw / max(stability_raw, 1.0), 1.0)  # Invert CV
        price_norm     = 0.5  # Default
        
        # Compute PVI
        pvi = _compute_pvi(demand_norm, growth_norm, stability_norm, price_norm)
        pvi = round(float(pvi), 2)
        viability = _pvi_category(pvi)
        
        # Detect anomalies
        anomaly_info = _detect_anomalies(monthly_sales)
        
        # Make recommendation
        decision, confidence, explanation = _make_decision(
            pvi, viability, demand_norm, growth_norm, stability_norm, price_norm,
            anomaly_info["has_anomaly"], anomaly_info["anomaly_pct"]
        )
        
        result["recommendation"] = {
            "pvi_score":  pvi,
            "viability":  viability,
            "decision":   decision,
            "confidence": confidence,
            "has_anomaly": anomaly_info["has_anomaly"],
            "anomaly_pct": round(anomaly_info["anomaly_pct"] * 100, 1),
            "explanation": explanation,
            "sub_scores": {
                "demand":    round(float(demand_norm), 3),
                "growth":    round(float(growth_norm), 3),
                "stability": round(float(stability_norm), 3),
                "price":     round(float(price_norm), 3),
            }
        }
    except Exception as e:
        # If PVI computation fails, include error but don't fail the whole request
        result["recommendation"] = {"error": f"Recommendation computation failed: {str(e)}"}

    result["note"] = (
        "Forecasts are based solely on the patterns in your uploaded data. "
        "More historical data generally improves forecast quality."
    )

    return _clean(result)