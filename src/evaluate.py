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
    mask = y_true > 1e-6
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def safe_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    
    total_actual = np.sum(np.abs(y_true))
    if total_actual < 1e-6:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual * 100)


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:

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
        "y_true_mean": round(float(np.mean(y_true)), 2),
        "y_pred_mean": round(float(np.mean(y_pred)), 2),
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

        fc_prophet   = pd.read_csv(fpath)
        y_pred_prophet = fc_prophet["yhat"].values[:HOLDOUT_MONTHS]
        if len(y_pred_prophet) == HOLDOUT_MONTHS:
            all_metrics.append(evaluate_one_series(
                y_true, y_pred_prophet, store_id, item_id, "prophet", n_obs, cat_id
            ))

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