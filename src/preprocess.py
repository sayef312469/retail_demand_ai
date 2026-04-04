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
    
    print("[3/5] Loading prices…")
    prices = pd.read_csv(PRICES_FILE)
    prices["sell_price"] = pd.to_numeric(prices["sell_price"], errors="coerce")
    return prices


# ---------------------------------------------------------------------------
# 4. Merge & monthly aggregation
# ---------------------------------------------------------------------------

def aggregate_to_monthly(df_long, calendar, prices):
    
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

    active_idx = monthly.groupby(["store_id", "item_id"])["monthly_sales"].sum()
    active_idx = active_idx[active_idx > 0].index
    monthly = monthly.set_index(["store_id", "item_id"])
    monthly = monthly.loc[monthly.index.isin(active_idx)].reset_index()

    return monthly


# ---------------------------------------------------------------------------
# 5. Feature engineering
# ---------------------------------------------------------------------------

def add_lag_features(df):
    
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