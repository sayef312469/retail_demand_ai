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
