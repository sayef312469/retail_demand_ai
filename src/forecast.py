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
After this pipeline completes, next to run:
  python src/pvi.py        — compute Product Viability Index
  python src/recommend.py  — generate stock recommendations
  python src/evaluate.py   — evaluate forecast accuracy
  python src/plot_reports.py — generate visual reports
  python src/summary_report.py  — generate text summary report
  uvicorn app.api:app --reload  — start the API server
  cd frontend && npm start      — start the React dashboard (in separate terminal)
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

    if not args.skip_preprocess:
        run_preprocess(args.stores, args.categories, args.top_items)
    else:
        print("\n[Skipping preprocessing — using existing processed_m5.csv]")

    if args.model in ("prophet", "both"):
        run_prophet()

    if args.model in ("arima", "both"):
        run_arima()

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